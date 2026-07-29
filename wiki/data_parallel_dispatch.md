# SGLang `--dp` 请求分发机制梳理

> 本文基于 SGLang 当前主干代码，梳理当 server 以 `--dp`（`--data-parallel-size`，副本级数据并行）启动时，请求如何从 HTTP 入口被分发到某个 DP worker（Scheduler 副本）。所有结论均标注 `文件:行号`，便于对照源码。
>
> 注意：本文的 `--dp` 指 **副本级数据并行**（每个 DP rank 一份完整模型副本），与 `--enable-dp-attention`（在单个 TP 组内对 attention 做数据并行切分）是两套不同机制。后者详见 [`dp_attention.md`](./dp_attention.md)。

---

## 1. 概述

`--dp N` 启动后，SGLang 会在 **N 个 DP rank** 上各起一组完整模型副本（每组是一个 TP/PP group，内部走张量/流水线并行），并额外起一个 `DataParallelController` 进程统一接管所有外部请求，再按某种负载均衡策略把每个请求路由到某一个 rank。

核心问题：**TokenizerManager 本身不感知 DP**。它仍按单副本逻辑把请求发往 `scheduler_input_ipc_name` 这个 IPC 端点；只是当 `dp_size > 1` 时，该端点的 PULL 端被 Controller 接管，由 Controller 做二次分发。

---

## 2. CLI 参数

`python/sglang/srt/server_args.py:845-866`

```python
dp_size: A[
    int,
    Arg(
        help="The data parallelism size.",
        aliases=["--data-parallel-size"],
    ),
] = 1

load_balance_method: A[
    str,
    Arg(
        help="The load balancing strategy for data parallelism.",
        choices=["auto", "round_robin", "follow_bootstrap_room",
                 "total_requests", "total_tokens"],
    ),
] = "auto"
```

- `--dp` / `--data-parallel-size`：DP 副本数，默认 `1`。
- `--load-balance-method`：分发策略，默认 `auto`。

### 2.1 `auto` 的实际解析

`server_args.py` 的 `_handle_load_balance_method`（post-init 时调用）：

```python
if self.load_balance_method == "auto":
    self.load_balance_method = (
        "follow_bootstrap_room"
        if self.disaggregation_mode == "prefill"
        else "round_robin"
    )
```

即：

- **普通（非分离式）`--dp` server** → `auto` 解析为 **`round_robin`**。
- **PD-prefill 实例** → `auto` 解析为 **`follow_bootstrap_room`**（保证同一 `bootstrap_room` 的 prefill/decode 命中同一 rank）。
- 想要按负载均衡，需显式指定 `total_requests` 或 `total_tokens`。

---

## 3. DP Worker 的创建

仅当 `dp_size > 1` 时才拉起 Controller，入口在 `python/sglang/srt/entrypoints/engine.py:667` 的 `run_data_parallel_controller_process`，对应类 **`DataParallelController`**（`python/sglang/srt/managers/data_parallel_controller.py:129`）。

`launch_dp_schedulers`（`data_parallel_controller.py:279`）按 rank 循环创建副本：

```python
for dp_rank in range(server_args.dp_size):
    # 每个 rank 起一组 TP group（scheduler 进程）
    ...
    base_gpu_id += tp_size * pp_size * gpu_id_step   # 每个副本占用独立 GPU
```

- 每个 rank 通过 `launch_tensor_parallel_group`（line 487）→ `mp.Process(target=run_scheduler_process_func, ...)`（line 574）拉起对应 Scheduler 进程；`base_gpu_id` 按 `tp_size * pp_size * gpu_id_step` 步进，保证各副本落到不同 GPU。
- node 0 为每个 rank 创建一个 **`zmq.PUSH`** socket，绑定到该 rank 的 `scheduler_input_ipc_name`，存入 `self.workers[dp_rank]`（line 309-314）。`self.workers` 即分发器最终 push 的目标数组。
- 若设置了 `--enable-dp-attention`，则走另一条路径 `launch_dp_attention_schedulers`（line 456）。

---

## 4. 端到端分发流程

```
HTTP 请求
  │
  ▼
TokenizerManager._send_one_request                 (tokenizer_manager.py:1331)
  │  send_to_scheduler → zmq.PUSH 到 scheduler_input_ipc_name  (line 388-408)
  ▼
DataParallelController 拦截（dp_size>1 时它持有该 IPC 的 PULL 端）
  recv_from_tokenizer = zmq.PULL                   (data_parallel_controller.py:149-151)
  │  event_loop 非阻塞 recv → _request_dispatcher  (line 654, 265)
  ▼
按策略选 rank → self.workers[target]               (line 612/628/639/645)
  │  sock_send → 对应 rank 的 PUSH socket
  ▼
该 rank 的 Scheduler 处理
```

关键细节：

1. **HTTP → TokenizerManager**：`TokenizerManager._send_one_request`（`tokenizer_manager.py:1331`）调用 `send_to_scheduler`（line 408），向 `port_args.scheduler_input_ipc_name`（line 388-389）的 `zmq.PUSH` 发送。
2. **Controller 拦截**：`dp_size > 1` 时，`DataParallelController` 拥有该 IPC 名的 PULL 端（`data_parallel_controller.py:149-151`）。TokenizerManager 无感知，仍按单副本方式发送，但接收方变成了 Controller。
3. **事件循环 → 类型分发**：`event_loop`（line 654）非阻塞 recv 每个请求，交给 `self._request_dispatcher`。`TypeBasedDispatcher`（`init_dispatcher`，line 265）按类型路由：
   - `TokenizedGenerateReqInput` / `TokenizedEmbeddingReqInput` → `dispatching_with_trace`（line 239）
   - Batched 变体 → `dispatch_batch_generate` / `dispatch_batch_embedding`
   - 控制 / profile 消息 → **广播到所有 worker**
4. **策略分发**：`dispatching_with_trace` 调 `self.dispatching(req)`，构造时（line 155-161）由 `dispatch_lookup` 绑定到具体策略方法。

---

## 5. 分发策略

四种策略都先经过 `maybe_external_dp_rank_routing`（line 605）：若 `req.routed_dp_rank is not None`（外部显式指定），直接发往该 rank，跳过策略。否则按规则选 rank，最后 `sock_send(self.workers[target], req)`。

| 策略 | 方法（行号） | 逻辑 |
|---|---|---|
| `round_robin` | `round_robin_scheduler` (612) | 计数器取模 `dp_size`，跳过 inactive rank |
| `follow_bootstrap_room` | `follow_bootstrap_room_scheduler` (628) | `req.bootstrap_room % len(workers)`，一致性哈希（PD 场景保证同 room 命中同 rank） |
| `total_requests` | `total_requests_scheduler` (639) | 选 `running + waiting` 请求数最少的 rank |
| `total_tokens` | `total_tokens_scheduler` (645) | 选总 token 最少的 rank，平手时再比请求数；会估算 `len(req.input_ids)` |

默认（普通 server）：`auto` → **`round_robin`**。

---

## 6. 负载感知的数据来源

`total_requests` / `total_tokens` 不是本地估算，而是各 rank 实时上报的负载，经共享内存读取。

### 6.1 `DPBudget` 与选择逻辑

`DPBudget`（`data_parallel_controller.py:93-126`）持有每个 rank 的 `total_requests[]` / `total_tokens[]`。`dispatch()`（line 111）用 `argmin` 选 rank，并对选中 rank **+1 投机计数**（line 124）：

```python
# 选中后立即给该 rank 的计数 +1，避免下一次（还没刷新到新数据前）又选它，
# 从而让突发流量在两次刷新之间被摊开到多个 rank。
```

### 6.2 负载刷新

- `refresh_load_budget`（line 222）在每次负载感知分发前调用（`refresh_load_budget_on_dispatch`，line 162），但**节流到每 20ms 一次**（line 222-237 注释解释：防止突发全压一个 rank）。
- 刷新时 `dp_budget.update_budget(self.load_snapshot_reader.read_all())`；`update_budget`（line 100）设 `total_requests = num_running_reqs + num_waiting_reqs`、`total_tokens = num_total_tokens`。

### 6.3 负载快照传输（`python/sglang/srt/managers/load_snapshot.py`）

- 每个 rank-0 Scheduler 通过 `Scheduler.publish_load_snapshot`（`scheduler.py:670-683`）发布 `LoadSnapshot`（running/waiting reqs、tokens、吞吐等），写入端在 `create_load_snapshot_writer`（`scheduler.py:643`）创建。
- **单机默认走 `/dev/shm` mmap**：`ShmLoadSnapshotWriter`/`ShmLoadSnapshotReader`（load_snapshot.py:381, 492）。Controller 关联一个 `ShmLoadSnapshotReader`（`data_parallel_controller.py:169`），`read_all()`（load_snapshot.py:571）返回每个 rank 的最新槽位。
- **多节点 DP-attention 回退到 ZMQ PUSH/PULL**（`ZmqShmLoadSnapshotReader`）。`zmq_reader_owner`（load_snapshot.py:106）决定 socket 归属：仅当启用负载感知策略时由 `DataParallelController` 持有（line 133-134）；否则归 TokenizerManager（仅供 `/v1/loads` 端点使用）。

---

## 7. 总结

| 维度 | 说明 |
|---|---|
| 分发位置 | `DataParallelController.dispatching()`，`python/sglang/srt/managers/data_parallel_controller.py`；具体策略方法 line 612 / 628 / 639 / 645；负载选择数学在 `DPBudget.dispatch`（line 111） |
| 策略选项 | `round_robin` / `follow_bootstrap_room` / `total_requests` / `total_tokens`（外加 `auto`） |
| 默认 | 普通 `--dp` server → `auto` → **`round_robin`**；PD-prefill → `follow_bootstrap_room` |
| 拦截机制 | `dp_size > 1` 时 Controller 接管 `scheduler_input_ipc_name` 的 PULL 端，TokenizerManager 无感知 |
| 负载感知输入 | 各 rank 的 `num_running_reqs + num_waiting_reqs` 与 `num_total_tokens`，经 `/dev/shm`（多节点经 ZMQ）上报，每 20ms 刷新，间隙用投机 +1 平摊 |

**实践建议**：

- 默认 `--dp` 是**轮询**分发，请求量均匀时够用。
- 若各请求长度差异大、副本间负载不均，可显式加 `--load-balance-method total_requests`（按请求数）或 `total_tokens`（按 token 数）做负载均衡。
- 负载感知策略依赖各 rank 上报负载快照，单机走共享内存开销极小；多节点 DP-attention 自动回退到 ZMQ。

---

## 附：关键文件索引

| 文件 | 作用 |
|---|---|
| `python/sglang/srt/server_args.py` | `--dp`、`--load-balance-method` 参数定义与 `auto` 解析 |
| `python/sglang/srt/managers/data_parallel_controller.py` | `DataParallelController`：拦截请求、按策略分发、维护 `DPBudget` |
| `python/sglang/srt/managers/load_snapshot.py` | 负载快照的读写与传输（`/dev/shm` / ZMQ） |
| `python/sglang/srt/managers/tokenizer_manager.py` | HTTP→TokenizerManager→`send_to_scheduler`（无感知 DP） |
| `python/sglang/srt/managers/scheduler.py` | 各 rank Scheduler 发布 `LoadSnapshot` |
| `python/sglang/srt/entrypoints/engine.py` | `dp_size > 1` 时拉起 Controller 进程的入口 |
