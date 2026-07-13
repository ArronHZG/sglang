# 在 Docker 中使用 RDMA 完整教程

> 结合 sglang / mooncake PD disaggregation 场景，从原理到部署全流程。
> 解决的核心问题：`Failed to create completion queue: Cannot allocate memory`（memlock 不足）、网卡选错、容器内 RDMA 设备不可见等。

---

## 〇、背景知识：RDMA 与 ulimit

在进入部署细节前，先建立两个核心概念。理解了它们，后面的每一个参数就不再是死记硬背。

### 1. RDMA 是什么，为什么它对容器这么挑剔

**RDMA（Remote Direct Memory Access，远程直接内存访问）** 是一种让一台机器的网卡**直接读写另一台机器内存**的技术，全程**绕过 CPU、绕过操作系统内核协议栈**。

对比传统网络（TCP/Socket）：

```
传统 TCP 收发一次数据:
  应用内存 → 内核 socket 缓冲区 → 网卡 → ... → 网卡 → 内核 → 应用内存
  （多次内存拷贝，CPU 参与每一次拷贝和上下文切换）

RDMA 收发一次数据:
  应用内存 ────────────────────────────────→ 对端应用内存
  （网卡硬件直接搬，零拷贝，CPU 不参与，内核不参与）
```

RDMA 的代价是**严格的硬件依赖**：

- 需要**专用网卡**：InfiniBand HCA 或 RoCE（RDMA over Converged Ethernet）网卡，常见是 Mellanox/NVIDIA 的 `mlx5_*` 系列。普通以太网卡不行。
- 需要**内核驱动**：`ib_core`、`mlx5_ib`、`rdma_cm` 等模块，提供 `/dev/infiniband/uverbs*` 设备节点。
- 需要**用户态库**：`libibverbs` + 厂商 provider（如 `libmlx5`），用户态程序通过 verbs API 直接驱动网卡。
- 需要**锁住物理内存**：这是最关键的一点——见下文。

**为什么 RDMA 要锁内存？** 因为网卡是硬件 DMA 引擎，它按**物理地址**直接搬数据，不走 CPU 和页表。但操作系统为了内存利用率，会随时把虚拟内存页换出到磁盘（swap）、或移动物理页位置。如果网卡正在 DMA 一段内存，而这期间 OS 把那页换走了，数据就会写到错误的地方、甚至损坏系统。所以 RDMA 注册内存（memory region, MR）时，必须把那段虚拟内存**钉死在物理 RAM 里**（pin/lock），保证它的物理地址在传输完成前不变。这就是 `mlock()`/`mlockall()` 系统调用，也是后面所有麻烦的根源。

RDMA 的两种常见形态：

| 类型     | 全称                          | 物理网络                | 典型网卡       |
|----------|-------------------------------|-------------------------|----------------|
| **IB**   | InfiniBand                    | 专用 IB 交换机/线缆     | ConnectX 系列  |
| **RoCE** | RDMA over Converged Ethernet  | 普通以太网（需无损配置）| mlx5 RoCE 口   |

RoCE 走以太网但要求**无损网络**（PFC + ECN），且网卡要配 IP。本文档里 `mlx5_0 "without network device"` 的报错，就是因为选了没配 IP 的 RoCE 口。

**在 LLM 推理里 RDMA 用在哪？** 主要是**跨节点的 KV cache 传输**（PD disaggregation 把 prefill 的 KV 传给 decode）、**多节点 TP 的 NCCL 通信**、**分布式权重加载**。这些场景下几 GB 到几十 GB 的张量要低延迟搬运，RDMA 比普通 TCP 快一个量级，CPU 占用几乎为零。SGLang 的 Mooncake 和 NIXL 传输引擎底层都是 RDMA。

---

### 2. ulimit 是什么，为什么 memlock 会卡死 RDMA

**`ulimit`** 是 Linux 对**单个进程/会话**设置的资源使用上限，由 shell 内置命令 `ulimit` 查看/修改。它分两类：

- **soft limit**：当前生效的上限。进程自己可以随时调高（但不超过 hard）。
- **hard limit**：上限的上限。普通用户只能调低，**只有 root 能调高**。

常用项（`ulimit -a` 可看全部）：

| 选项             | 含义                       | 典型默认 | RDMA 相关      |
|------------------|----------------------------|----------|----------------|
| `-l` (memlock)   | 可锁住的物理内存上限（KB） | **64**   | ⭐ **RDMA 致命** |
| `-s` (stack)     | 栈大小（KB）               | 8192     | 大模型推理需调大 |
| `-n` (nofile)    | 最大打开文件数             | 1024     | NCCL/连接数     |
| `-u` (nproc)     | 最大进程/线程数            | 不定     | 多线程推理       |

**`memlock` 默认 64KB 是 RDMA 的头号杀手。** RDMA 注册一段 KV cache（动辄几百 MB～几 GB）需要 `mlock` 这么大的物理内存，而 ulimit 只允许 64KB，于是 `mlock` 失败、连带 `ibv_create_cq`（completion queue 也占 locked memory）也失败，内核返回 `errno=ENOMEM`（12），表现为：

```
E... rdma_context.cpp:131] Failed to create completion queue: Cannot allocate memory [12]
```

这就是你前面遇到的崩溃。**修复只有一条路：把 memlock 提到 unlimited。**

#### ulimit 的三条铁律（理解了就不会再踩坑）

**铁律一：soft 只能在 hard 范围内调。**
```bash
$ ulimit -l        # soft = 64
$ ulimit -Hl       # hard = 64
$ ulimit -l unlimited
bash: ulimit: max locked memory: cannot modify limit: Operation not permitted
# 失败！因为 hard 也是 64，普通用户提不动 hard
```
所以**普通用户在容器里 `ulimit -l unlimited` 是无效的**——除非容器启动时 hard 已经被设成 unlimited。

**铁律二：hard 由"谁创建进程"决定，子进程继承。**
- 登录 shell：由 PAM 在登录时根据 `/etc/security/limits.conf` 设置。
- Docker 容器：由 docker daemon 在创建容器时根据 `--ulimit` 或 `daemon.json` 设置，**容器内 root 也改不了 hard**。
- systemd 服务：由 unit 文件的 `LimitMEMLOCK=` 设置。

这就是为什么文档反复强调：**ulimit 必须在容器启动时由外部注入，Dockerfile/entrypoint 里的 `ulimit` 命令提不了 hard。**

**铁律三：修改 limits.conf 必须重新登录才生效。**
`/etc/security/limits.conf` 通过 PAM 的 `pam_limits` 模块在**会话建立时**读取一次。改完文件，当前已登录的 shell 不会变，必须 `exit` 重连或新开 ssh。Docker 容器则要重建。

#### 三种设置 memlock unlimited 的位置

```bash
# 1) 单个 docker run（推荐，最显式）
docker run --ulimit memlock=-1:-1 ...

# 2) docker-compose.yml
ulimits:
  memlock: { soft: -1, hard: -1 }

# 3) 宿主机全局默认（所有容器生效）
# /etc/docker/daemon.json
{ "default-ulimits": { "memlock": { "name": "memlock", "soft": -1, "hard": -1 } } }

# 4) 裸机/登录会话（非容器）
# /etc/security/limits.conf
*  soft  memlock  unlimited
*  hard  memlock  unlimited
# 然后重新登录
```

> `memlock` 设 `unlimited` 是 **RDMA / GPU + NCCL / 大模型推理部署的标配**，所有跑大模型的服务器都这么设。它的"风险"只是理论上用户可以锁住大量内存导致系统换页——在专用推理机器上无所谓。基本不存在安全顾虑。

#### 一个排查口诀

遇到 RDMA `Cannot allocate memory` / `ENOMEM` / `mlock failed` 类报错，**第一步永远是 `ulimit -l`**。如果输出不是 `unlimited`，根因就找到了 90%。

---

## 一、核心原理：RDMA 需要容器拿到什么

RDMA（InfiniBand / RoCE）跑起来要满足以下条件，缺一不可：

| 需求                   | 为什么需要                                  | 怎么给容器                  |
|------------------------|---------------------------------------------|-----------------------------|
| **RDMA 设备节点**      | 用户态 verbs API 要访问 `/dev/infiniband/uverbs*` | `--device` 挂载             |
| **锁内存权限**         | 注册 MR 要 pin 住物理内存                   | `--cap-add=IPC_LOCK`        |
| **memlock ulimit 无限**| 内核对 locked memory 的上限                 | `--ulimit memlock=-1:-1`    |
| **RDMA 用户态库**      | libibverbs + 厂商 provider（如 mlx5）       | 镜像内安装                  |
| **大共享内存**         | NCCL / mooncake 跨进程 IPC                  | `--ipc=host` 或 `--shm-size`|
| **网络可达**           | RoCE 走 IP，要对端能通                      | `--network=host`（最简单）  |

**关键认知**：前三个（设备节点、IPC_LOCK、memlock ulimit）都是**运行时**由 docker daemon 注入的，**Dockerfile 里设不了**。Dockerfile 只负责第四条（装库）。这是最常见的误区——很多人试图在 Dockerfile 或 entrypoint 里 `ulimit -l unlimited`，但那只能把 soft 调到 hard 以内，而 hard 在容器启动时就由 daemon 定死了（详见第〇节"ulimit 三条铁律"）。

---

## 二、宿主机准备（一次性，确认 RDMA 本身没问题）

在**宿主机**上（不是容器内）执行：

```bash
# 1. 看 RDMA 内核驱动是否加载
lsmod | grep -E 'ib_core|mlx5_ib|rdma_cm'
# 没有就加载：
# sudo modprobe ib_core
# sudo modprobe mlx5_ib
# sudo modprobe rdma_cm

# 2. 看 RDMA 设备
ibstat
# 期望看到多块 mlx5_*，且 Port state: Active

# 3. 看 RoCE 网卡有没有配 IP（RoCE 必需，IB 不需要）
ip -o addr show

# 4. 看设备节点
ls -l /dev/infiniband/
# 应有 uverbs0, uverbs1, ..., rdma_cm, 可能还有 umad*, issm*
```

如果宿主机上 `ibstat` 都看不到设备，那是驱动/硬件层面的问题，先解决宿主机，容器里不可能凭空出来。

---

## 三、如何判断机器有没有 RDMA、该用哪块卡

这是部署前最关键的一步。机器上往往有多块 `mlx5_*` 网卡（有的连存储、有的连管理网、只有部分是给 GPU/RDMA 用的），**选错卡**就会出现"设备能发现、但传不动数据"或 `without network device` 的怪现象。本节给一套从头到尾的判断流程。

### 1. 判断机器到底有没有 RDMA

三条命令任一命中即说明硬件/驱动在：

```bash
# 看内核模块 —— 最底层
lsmod | grep -E 'ib_core|mlx5_ib|rdma_cm|irdma'
# 有输出 = RDMA 内核栈已加载

# 看 verbs 设备 —— 用户态可见性
ibv_devinfo
# 列出 hca_id: mlx5_0, mlx5_1 ... transport: InfiniBand/Ethernet

# 看设备节点 —— 容器要挂载的东西
ls /dev/infiniband/
# uverbs0 uverbs1 ... rdma_cm
```

- 三个都没输出 → **这台机器没有可用 RDMA**，要么没装 OFED 驱动，要么网卡不支持。容器里更不可能有，直接走 TCP（`MOONCAKE_PROTOCOL=tcp`）。
- `lsmod` 有但 `ibv_devinfo` 无 → 用户态库没装或版本不匹配，装 `rdma-core` / MOFED。
- 都有 → 进下一步选卡。

### 2. 列出所有 RDMA 卡并区分 IB / RoCE

```bash
ibstat
```

输出示例（截取）：

```
CA 'mlx5_0'
  CA type: MT4125
  Number of ports: 1
  Port 1:
    State: Active           ← 关键：要 Active
    Physical state: LinkUp
    Link layer: Ethernet    ← Ethernet = RoCE；InfiniBand = IB

CA 'mlx5_1'
  Port 1:
    State: Down             ← 没接线/没启用，跳过
    Link layer: InfiniBand
```

判断要点：

| 字段            | 含义                                   | 怎么用                           |
|-----------------|----------------------------------------|----------------------------------|
| `State`         | 端口协议层状态                         | **必须 `Active`**，`Down`/`Initializing` 跳过 |
| `Link layer`    | `InfiniBand` 或 `Ethernet`             | 决定是否需要配 IP（见下）         |
| `Physical state`| 物理链路                               | 应为 `LinkUp`                    |

### 3. RoCE 卡还要确认有没有配 IP

IB 卡靠 GID 寻址，不需要 IP；**RoCE 卡必须配 IP**，否则 mooncake 会报 `without network device`。把 `ibstat` 里 Link layer=Ethernet 的卡名（如 `mlx5_0`）对应到系统网卡：

```bash
# 看 RDMA 卡对应的 netdev 名字和 IP
ip -o -d link show          # 找 roce/ib 的接口
ip -o addr show             # 看哪些接口配了 IP

# 更直接：查每块 RDMA 设备绑的网卡名
cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/ndevs/0  # 输出如 eth1
ip addr show eth1           # 看 eth1 有没有 IP
```

- 有 IP 且能 ping 通对端同样配置的卡 → 可用。
- 没有 IP → 要么 `ip addr add <ip>/<mask> dev <iface>` 现配，要么换一块有 IP 的卡。

### 4. GPU 和 NIC 的 NUMA 配对（性能关键，多卡机器必看）

一块机器插多张 GPU 和多块网卡时，**GPU 和网卡分属不同 NUMA 节点会导致跨 NUMA 访问，RDMA 带宽掉一半**。让每张 GPU 用它同 NUMA 的网卡：

```bash
nvidia-smi topo -m
```

输出示例：

```
        GPU0  GPU1  GPU2  GPU3  mlx5_0  mlx5_1  mlx5_2  mlx5_3
GPU0     X    NV1   NV1   NV1   SYS    SYS     PIX     PIX
GPU1    NV1    X    NV1   NV1   SYS    SYS     PIX     PIX
GPU2    NV1   NV1    X    NV1   PIX    PIX     SYS     SYS
GPU3    NV1   NV1   NV1    X    PIX    PIX     SYS     SYS
mlx5_0  SYS   SYS   PIX   PIX    X      PIX     SYS     SYS
mlx5_1  SYS   SYS   PIX   PIX   PIX     X       SYS     SYS
mlx5_2  PIX   PIX   SYS   SYS   SYS    SYS      X       PIX
mlx5_3  PIX   PIX   SYS   SYS   SYS    SYS     PIX      X
```

亲和性等级（从近到远）：`PIX`（同 PCIe 桥）> `NV1`（同 NVLink）> `SYS`（跨 NUMA，最远）。

读法：`GPU2` 这一行，`mlx5_0` 是 `PIX`（同 PCIe，最佳），`mlx5_2` 是 `SYS`（跨 NUMA，最差）。所以 **GPU2 应该配 `mlx5_0`**，而不是 `mlx5_2`。

配对原则：**对你打算用的每张 GPU，在 `nvidia-smi topo -m` 里找亲和性最高（PIX > PHB > NODE > SYS）的那块 RDMA 网卡。**

### 5. 实测连通性和带宽（两台机器都做）

选好卡之后，启动前先验证两台机器之间这块卡真的能通：

```bash
# A 机（服务端）
ibv_rc_pingpong -d mlx5_1

# B 机（客户端），用 A 机该 RoCE 网卡的 IP
ibv_rc_pingpong -d mlx5_1 <A机IP>
# 期望：两边各打印一堆字节并退出，无报错

# 测带宽（更接近真实 KV 传输场景）
# A 机
ib_send_bw -d mlx5_1
# B 机
ib_send_bw -d mlx5_1 <A机IP>
```

- 报 `Couldn't resolve` / `Connection refused` → 多半是 RoCE 没配 IP 或防火墙。
- 报 `Failed to allocate` / ENOMEM → 回到第〇节查 `ulimit -l`。
- 跑通带宽接近网卡标称（如 100/200/400 GbE 的 80%+）→ RDMA 链路健康，可以上 sglang。

### 6. 把选好的卡填进启动参数

单卡或所有 GPU 共用一块卡：

```bash
--disaggregation-ib-device mlx5_1
```

每张 GPU 用各自网卡（推荐，配合第 4 步的 NUMA 配对）——key 是**物理 GPU id**：

```bash
--disaggregation-ib-device '{"0":"mlx5_0","1":"mlx5_1","2":"mlx5_2","3":"mlx5_3"}'
```

> SGLang 解析这个 JSON 的代码在 `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py:15-64`，key 必须是 GPU 物理编号（即 `nvidia-smi` 看到的 id，不是 `CUDA_VISIBLE_DEVICES` 截断后的逻辑 rank）。

### 小结流程图

```
lsmod/ibv_devinfo 有 RDMA?
  ├─ 否 → 装 OFED/rdma-core，或放弃用 TCP
  └─ 是 → ibstat 选 State:Active 的卡
           ├─ Link layer=Ethernet? → 确认配了 IP（ip addr）
           └─ nvidia-smi topo -m → 选与目标 GPU 亲和性最高的卡
                └─ ibv_rc_pingpong 实测两机连通
                     └─ 填 --disaggregation-ib-device，进容器部署
```

---

## 四、最小验证：一个 RDMA 测试容器

先不碰 sglang，用一个干净镜像验证 RDMA 通路是否打通：

```bash
docker run --rm -it \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  --cap-add=IPC_LOCK \
  --device=/dev/infiniband \
  --gpus all \
  rdma/test:latest bash
```

容器内验证：

```bash
# 1. memlock 真的无限了？
ulimit -l
# 期望: unlimited

# 2. 设备节点在不在？
ls /dev/infiniband/

# 3. RDMA 设备可见？
ibstat
ibv_devinfo

# 4. 实际能否注册内存（这是 memlock 卡住的那一步）
# 用 ibv_rc_pingpong 测，两个节点分别跑：
ibv_rc_pingpong -d mlx5_1            # 一端
ibv_rc_pingpong -d mlx5_1 <对端IP>   # 另一端
```

**这一步通了，才说明 docker + RDMA 配置正确。** sglang/mooncake 在这套配置上一定能跑（剩下的只是网卡选哪个的问题，见第三节）。

---

## 五、完整 `docker run`（sglang prefill worker 示例）

把 sglang 启动命令搬到容器里，加上 RDMA 必需的运行时参数：

```bash
docker run -d \
  --name sglang-prefill \
  --network=host \
  --ipc=host \
  --gpus '"device=1,2,3,4"' \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864 \
  --cap-add=IPC_LOCK \
  --device=/dev/infiniband \
  --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES=1,2,3,4 \
  -v /user/yanhui/ckpts:/models:ro \
  -v $(pwd)/logs:/workspace/logs \
  -w /workspace \
  your-sglang-image:latest \
  python -m sglang.launch_server \
    --model-path /models/minicpm5/39a5b/job_510790.step_13000 \
    --trust-remote-code \
    --host 0.0.0.0 --port 30002 --tp 4 \
    --mem-fraction-static 0.85 --enable-metrics --skip-server-warmup \
    --max-running-requests 256 --chunked-prefill-size 8192 \
    --disaggregation-mode prefill \
    --disaggregation-ib-device mlx5_1 \
    --tool-call-parser minicpm4_xml
```

**参数逐条解释**（重点）：

- `--network=host` —— RoCE 流量要直接走宿主机网卡，bridge 网络对 RDMA 几乎不可用。**几乎不可省。**
- `--ipc=host` —— 让容器共享宿主机的 IPC namespace，NCCL/mooncake 跨进程共享内存才够大。
- `--gpus '"device=1,2,3,4"'` —— NVIDIA container toolkit 的写法，指定物理 GPU 1-4。注意引号嵌套。
- `--ulimit memlock=-1:-1` —— **soft:hard 都设 unlimited**。`-1` 是 unlimited 的写法。这条是 `Cannot allocate memory` 崩溃的直接修复。
- `--ulimit stack=67108864` —— 64MB 栈，CUDA/PyTorch 大模型推理常需要。
- `--cap-add=IPC_LOCK` —— 容器内进程被允许调用 `mlock`/`mlockall`。没这条，即使 memlock ulimit 是 unlimited，注册 MR 也会 EPERM。
- `--device=/dev/infiniband` —— 整个目录挂进去，含所有 `uverbs*` 和 `rdma_cm`。也可以精确到 `--device=/dev/infiniband/uverbs1`。
- `--shm-size=16g` —— 如果不用 `--ipc=host`，就要靠这个扩大 `/dev/shm`。

> ⚠️ `--device=/dev/infiniband` 必须配合 `--cap-add=IPC_LOCK` 和 `--ulimit memlock`，**三者缺一**都会以不同方式失败（CQ 创建失败 / EPERM / ENOMEM）。

---

## 六、`docker-compose.yml` 版本（推荐，可版本化）

```yaml
services:
  sglang-prefill:
    image: your-sglang-image:latest
    network_mode: host
    ipc: host
    shm_size: 16g
    ulimits:
      memlock: { soft: -1, hard: -1 }
      stack:   { soft: 67108864, hard: 67108864 }
    cap_add:
      - IPC_LOCK
    devices:
      - /dev/infiniband
    environment:
      - CUDA_VISIBLE_DEVICES=1,2,3,4
    volumes:
      - /user/yanhui/ckpts:/models:ro
      - ./logs:/workspace/logs
    working_dir: /workspace
    command: >
      python -m sglang.launch_server
      --model-path /models/minicpm5/39a5b/job_510790.step_13000
      --trust-remote-code
      --host 0.0.0.0 --port 30002 --tp 4
      --mem-fraction-static 0.85 --enable-metrics --skip-server-warmup
      --max-running-requests 256 --chunked-prefill-size 8192
      --disaggregation-mode prefill
      --disaggregation-ib-device mlx5_1
      --tool-call-parser minicpm4_xml
    restart: unless-stopped
```

启动：`docker compose up -d`，看日志：`docker compose logs -f sglang-prefill`。

---

## 七、宿主机全局默认（一劳永逸）

不想每次都写 ulimit，改宿主机的 `/etc/docker/daemon.json`：

```json
{
  "default-ulimits": {
    "memlock": { "name": "memlock", "soft": -1, "hard": -1 },
    "stack":   { "name": "stack",   "soft": 67108864, "hard": 67108864 }
  }
}
```

```bash
sudo systemctl restart docker
```

之后所有容器默认 memlock unlimited。**`--cap-add=IPC_LOCK` 和 `--device` 仍需每次指定**（这两个没法全局默认，也不该全局默认，安全考虑）。

---

## 八、Dockerfile 该装什么

RDMA 相关的库进镜像，运行时参数进 compose/run。一个 RDMA-ready 的 Dockerfile 片段：

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# RDMA 用户态库
RUN apt-get update && apt-get install -y --no-install-recommends \
    ibverbs-utils \
    libibverbs-dev \
    librdmacm-dev \
    infiniband-diags \
    rdma-core \
    && rm -rf /var/lib/apt/lists/*

# 厂商 provider（Mellanox/NVIDIA）
# 如果基础镜像没带，装 mlx5 的 OFED 用户态：
# RUN apt-get install -y libmlx5-1 libmlx5-dev

# Python 依赖 + sglang + mooncake
RUN pip install --no-cache-dir \
    sglang[all] \
    mooncake-transfer-engine

# 注意：不要在 Dockerfile 里写 ulimit —— 没用
# 不要在 ENTRYPOINT 里 ulimit -l unlimited —— hard 没提上去会被内核拒绝

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

`entrypoint.sh` 可以做的事（**不是**提 ulimit）：

```bash
#!/bin/bash
# 确认运行时环境，便于早期失败定位
echo "memlock: $(ulimit -l)"     # 应为 unlimited，否则启动参数没配对
ls /dev/infiniband/ || { echo "RDMA devices missing!"; exit 1; }
exec "$@"
```

---

## 九、常见坑速查

| 症状                                                | 原因                          | 修复                                                |
|-----------------------------------------------------|-------------------------------|-----------------------------------------------------|
| `Failed to create completion queue: Cannot allocate memory` | memlock 不足          | `--ulimit memlock=-1:-1`                            |
| `Permission denied` 注册 MR                         | 没有 IPC_LOCK cap             | `--cap-add=IPC_LOCK`                                |
| 容器内 `ibstat` 看不到设备                          | 设备没挂载                    | `--device=/dev/infiniband`                          |
| 设备在但 `ibv_open_device` 失败                     | provider 库没装               | 镜像装 `libmlx5`/`rdma-core`                        |
| RoCE 能发现设备但 ping 不通对端                     | 容器网络隔离                  | `--network=host`                                    |
| NCCL 报 `no IB devices found`                       | 同上几条任一 + shm 太小       | 加 `--ipc=host` + `--shm-size`                      |
| `mlx5_0: without network device`                    | 选了没配 IP 的卡              | `--disaggregation-ib-device` 指定 Active 的卡（第三节）|
| 改了 `daemon.json` 没生效                           | 没重启 docker                 | `sudo systemctl restart docker`                     |
| `ulimit -l unlimited` 报错 / 不生效                 | 容器启动时 hard 已被定死      | 重建容器加 `--ulimit memlock=-1:-1`，或改 `daemon.json` |
| KV 传输带宽远低于网卡标称                           | GPU 与网卡跨 NUMA             | 按第三节 `nvidia-smi topo -m` 重新配对               |

---

## 十、验证清单（跑 sglang 前过一遍）

在容器内执行，全部通过再起 sglang：

```bash
[ "$(ulimit -l)" = "unlimited" ] && echo "✓ memlock" || echo "✗ memlock"
[ -e /dev/infiniband/rdma_cm ] && echo "✓ rdma_cm" || echo "✗ devices"
ibstat | grep -q Active && echo "✓ ibstat Active" || echo "✗ no active port"
python -c "from mooncake.engine import TransferEngine; print('✓ mooncake')" || echo "✗ mooncake"
```

---

## 十一、最小行动路径

1. 按第三节判断机器有 RDMA、选好对应的卡（NUMA 配对）。
2. 确认宿主机 `ibstat` 该卡 `State: Active`，RoCE 卡有 IP。
3. 用**第四节**的最小测试容器先验证 `ulimit -l` 是 `unlimited` 且 `ibstat` 可见、`ibv_rc_pingpong` 能通。
4. 通了之后，把 sglang prefill 命令套到**第五节**的 `docker run`（或第六节 compose）里。
5. decode worker 用同样一套运行时参数，`--disaggregation-mode decode`，端口和 `--disaggregation-ib-device` 按实际配。
6. router 用普通容器即可（它不做 RDMA），不需要 `--device` / `IPC_LOCK` / memlock。

---

## 附：PD Disaggregation 三容器完整示例

prefill + decode + router 的 `docker-compose.yml`，包含 bootstrap 端口、KV 传输、健康检查：

```yaml
services:
  sglang-prefill:
    image: your-sglang-image:latest
    network_mode: host
    ipc: host
    shm_size: 16g
    ulimits:
      memlock: { soft: -1, hard: -1 }
      stack:   { soft: 67108864, hard: 67108864 }
    cap_add:
      - IPC_LOCK
    devices:
      - /dev/infiniband
    environment:
      - CUDA_VISIBLE_DEVICES=1,2,3,4
    volumes:
      - /user/yanhui/ckpts:/models:ro
      - ./logs/prefill:/workspace/logs
    working_dir: /workspace
    command: >
      python -m sglang.launch_server
      --model-path /models/minicpm5/39a5b/job_510790.step_13000
      --trust-remote-code
      --host 0.0.0.0 --port 30002 --tp 4
      --mem-fraction-static 0.85 --enable-metrics --skip-server-warmup
      --max-running-requests 256 --chunked-prefill-size 8192
      --disaggregation-mode prefill
      --disaggregation-ib-device mlx5_1
      --disaggregation-bootstrap-port 8998
      --tool-call-parser minicpm4_xml
    restart: unless-stopped

  sglang-decode:
    image: your-sglang-image:latest
    network_mode: host
    ipc: host
    shm_size: 16g
    ulimits:
      memlock: { soft: -1, hard: -1 }
      stack:   { soft: 67108864, hard: 67108864 }
    cap_add:
      - IPC_LOCK
    devices:
      - /dev/infiniband
    environment:
      - CUDA_VISIBLE_DEVICES=5,6,7,8
    volumes:
      - /user/yanhui/ckpts:/models:ro
      - ./logs/decode:/workspace/logs
    working_dir: /workspace
    command: >
      python -m sglang.launch_server
      --model-path /models/minicpm5/39a5b/job_510790.step_13000
      --trust-remote-code
      --host 0.0.0.0 --port 30003 --tp 4
      --mem-fraction-static 0.85 --enable-metrics --skip-server-warmup
      --max-running-requests 256
      --disaggregation-mode decode
      --disaggregation-ib-device mlx5_5
      --disaggregation-bootstrap-port 8999
      --tool-call-parser minicpm4_xml
    restart: unless-stopped

  sglang-router:
    image: your-sglang-router-image:latest
    network_mode: host
    command: >
      python -m sglang_router.launch_router
      --pd-disaggregation
      --prefill http://127.0.0.1:30002
      --decode http://127.0.0.1:30003
      --host 0.0.0.0 --port 8000
      --prometheus-host 0.0.0.0 --prometheus-port 29000
      --disable-health-check
      --disable-circuit-breaker
      --disable-retries
      --log-level warn
      --log-dir /workspace/router_logs
    volumes:
      - ./logs/router:/workspace/router_logs
    restart: unless-stopped
    depends_on:
      - sglang-prefill
      - sglang-decode
```

> 注意：router 容器**不需要** RDMA 相关参数（`--device`/`IPC_LOCK`/`memlock`），它只做 HTTP 转发和调度。RDMA 全部发生在 prefill ↔ decode 之间。
