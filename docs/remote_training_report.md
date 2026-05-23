# SpecForge 训推分离（Remote Training）技术报告

## 1. 项目概述

### 1.1 背景

SpecForge 是一个投机解码（Speculative Decoding）Draft Model 在线训练框架，支持 Eagle3 和 DFlash 两种投机解码方案。训练过程中，Draft Model 需要 Target Model（如 Qwen3-30B-A3B-FP8）的推理输出作为训练信号。

传统方案将 Target Model 与 Draft Model 训练放在同一进程/同一 GPU 上，导致：
- 显存竞争：30B 模型 FP8 权重 + KV Cache 占用大量显存，挤压训练显存
- 计算串行：Target 推理与 Draft 训练无法并行，浪费 GPU 算力
- 扩展受限：无法独立扩展推理和训练资源

### 1.2 训推分离方案

将 Target Model 部署为独立推理服务，训练端通过网络异步获取目标模型数据：

```
┌──────────────────────────┐          ┌──────────────────────────┐
│   Training Client        │          │   Target Model Server    │
│   Draft Model + Optimizer│          │   SGLang Target Model    │
│                          │  HTTP    │                          │
│  requests / metadata     │ ───────► │  forward / setup APIs    │
│                          │ ◄─────── │                          │
│  large CUDA tensors      │ ═NCCL══► │  hidden states / target_p│
└──────────────────────────┘          └──────────────────────────┘
```

- **HTTP 控制面**：请求调度、metadata 传输、健康检查、配置同步
- **NCCL 数据面**：GPU-to-GPU 大张量传输（支持同机 NVLink / 跨机 RDMA）
- **Wire format fallback**：NCCL 不可用时的紧凑二进制格式备选

## 2. 架构详解

### 2.1 Server 端

`launch_target_server.py` 通过 `torchrun` 启动，支持 TP>1 多 rank：

- **Rank 0**：运行 HTTP server + 处理请求 + NCCL send
- **Rank 1+**：参与 TP forward，通过 `broadcast_object_list` 同步请求

请求处理流程：
1. 接收 HTTP POST（input_ids, attention_mask 等 metadata）
2. 广播请求到所有 TP rank
3. 执行 target model forward（SGLang backend）
4. 计算 `target_p`（softmax + optional top-k 压缩）
5. 通过 NCCL send 返回 hidden_states / target_p 给 client

### 2.2 Client 端

`remote_target_client.py` 作为训练脚本的 target model backend：

- 首次请求时通过 POST `/init_nccl` 初始化 NCCL 数据通道
- 通过 POST `/setup` 获取模型配置（hidden_size, vocab_size 等）
- 每步训练通过 POST `/generate` 发送请求 + NCCL recv 接收结果
- 支持 TP>1 训练：仅 rank 0 发送请求，结果 broadcast 到其他 rank

### 2.3 NCCL Transport

`_nccl_transport.py` 实现专用 NCCL 传输层：

- Server = rank 0, Client = rank 1 组成 2-process NCCL group
- TCP rendezvous 通过 `torch.distributed.TCPStore` 建立连接
- 支持同机（NVLink）和跨机（RDMA/RoCE）传输
- `SPECFORGE_NCCL_PORT` 控制 rendezvous 端口（默认 HTTP port + 100）
- 安全退出：`pg.abort()` + 注销避免 `destroy_process_group` 挂起

### 2.4 Prefetch 流水线

```
Timeline (depth=2, 2 servers round-robin):
─────────────────────────────────────────────────────────────
Server A: [req1]────────[req3]────────[req5]────────
Server B:     [req2]────────[req4]────────[req6]────
Client:   [train1][train2][train3][train4][train5]──
                  ↑ req1 ready    ↑ req3 ready
```

- `fill_prefetch_queue()` 维护最多 `depth` 个 in-flight 请求
- 多服务器通过 `itertools.cycle` round-robin 分发
- 训练步开始时 `future.result()` 获取已完成的 prefetch 结果
- 当 server 延迟 < 训练步时间时，server 完全被 overlap

### 2.5 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SPECFORGE_ENABLE_NCCL` | `1` | 启用 NCCL 传输（`0` 降级为 wire format） |
| `SPECFORGE_NCCL_PORT` | HTTP port + 100 | NCCL TCP rendezvous 端口 |
| `SPECFORGE_TOPK` | `0` | Server 端 target_p top-k 压缩（`0` 为全分布） |
| `SPECFORGE_TARGET_DTYPE` | `fp32` | target_p 计算精度 |
| `SPECFORGE_GPU_ID` | auto | 指定 GPU 设备 ID |

## 3. 使用方法

### 3.1 启动 Target Server

```bash
# 单卡 server（推荐）
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master-port 29500 \
  scripts/launch_target_server.py \
  --model-path /path/to/Qwen3-30B-A3B-FP8 \
  --mode eagle3 \
  --port 8001 \
  --tp-size 1 \
  --mem-fraction-static 0.4 \
  --trust-remote-code \
  --attention-backend flashinfer

# TP=2 server（无额外速度收益，不推荐）
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master-port 29500 \
  scripts/launch_target_server.py \
  --model-path /path/to/Qwen3-30B-A3B-FP8 \
  --mode eagle3 \
  --port 8001 \
  --tp-size 2 \
  --mem-fraction-static 0.35 \
  --trust-remote-code \
  --attention-backend flashinfer
```

Server 启动后会打印 `listening on 0.0.0.0:8001`，可通过 `curl http://<host>:8001/health` 验证就绪。

### 3.2 启动训练（单 server）

```bash
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.run --nproc_per_node=1 --master-port 29600 \
  scripts/train_eagle3.py \
  --target-model-path /path/to/Qwen3-30B-A3B-FP8 \
  --target-model-backend remote \
  --remote-url http://<server-host>:8001 \
  --target-prefetch-depth 1 \
  --train-data-path /path/to/data.jsonl \
  --max-length 10240 \
  --batch-size 1 \
  --tp-size 1 \
  --trust-remote-code \
  --is-preformatted \
  --output-dir /path/to/output
```

### 3.3 启动训练（双 server，推荐）

在两台机器上各启动一个 server，训练端指定多个 URL：

```bash
# Machine A: 启动 server
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master-port 29500 \
  scripts/launch_target_server.py \
  --model-path /path/to/model --mode eagle3 --port 8001 --tp-size 1 \
  --mem-fraction-static 0.4 --trust-remote-code --attention-backend flashinfer

# Machine B: 启动 server
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master-port 29500 \
  scripts/launch_target_server.py \
  --model-path /path/to/model --mode eagle3 --port 8001 --tp-size 1 \
  --mem-fraction-static 0.4 --trust-remote-code --attention-backend flashinfer

# 训练端（任一机器）
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.run --nproc_per_node=1 --master-port 29600 \
  scripts/train_eagle3.py \
  --target-model-path /path/to/model \
  --target-model-backend remote \
  --remote-urls "http://machineA:8001,http://machineB:8001" \
  --target-prefetch-depth 2 \
  --train-data-path /path/to/data.jsonl \
  --max-length 10240 --batch-size 1 --tp-size 1 \
  --trust-remote-code --is-preformatted \
  --output-dir /path/to/output
```

### 3.4 DFlash 模式

DFlash 使用方式相同，将 `--mode eagle3` 改为 `--mode dflash`，训练脚本改为 `train_dflash.py`：

```bash
# Server
scripts/launch_target_server.py --mode dflash --port 8002 ...

# Training
scripts/train_dflash.py --target-model-backend remote --remote-url http://host:8002 ...
```

### 3.5 跨机 RDMA 配置

跨机部署时，NCCL 自动使用 RDMA（如果可用）。可通过环境变量优化：

```bash
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0        # 网络接口
export NCCL_IB_HCA=mlx5_bond_0         # IB HCA 设备
export NCCL_IB_GID_INDEX=3             # RoCE GID index
```

### 3.6 关键参数说明

| 参数 | 位置 | 说明 |
|------|------|------|
| `--target-model-backend remote` | 训练端 | 启用 remote backend |
| `--remote-url` | 训练端 | 单 server URL |
| `--remote-urls` | 训练端 | 多 server URL（逗号分隔） |
| `--target-prefetch-depth` | 训练端 | 预取队列深度（推荐=server数量） |
| `--remote-timeout` | 训练端 | HTTP 请求超时（秒，默认 120） |
| `--mem-fraction-static` | Server | SGLang KV cache 显存比例（TP=1 用 0.4，TP=2 用 0.35） |
| `--attention-backend` | Server | 注意力后端（推荐 flashinfer） |
| `--nccl-port` | Server | NCCL rendezvous 端口（默认 HTTP port + 100） |
| `--host` | Server | 绑定地址（跨机必须 0.0.0.0） |

## 4. 文件索引

| 文件 | 职责 |
|------|------|
| `scripts/launch_target_server.py` | Server 启动器（HTTP + NCCL + TP 多 rank） |
| `scripts/train_eagle3.py` | Eagle3 训练脚本（支持 remote backend + prefetch） |
| `scripts/train_dflash.py` | DFlash 训练脚本（同上） |
| `specforge/args.py` | `RemoteBackendArgs` / `SGLangBackendArgs` 参数定义 |
| `specforge/modeling/target/remote_target_client.py` | Client（HTTP + NCCL recv + TP broadcast + prefetch） |
| `specforge/modeling/target/remote_target_server.py` | Server（forward + target_p + NCCL send） |
| `specforge/modeling/target/_nccl_transport.py` | NCCL 传输层（rendezvous + send/recv + teardown） |
| `specforge/modeling/target/_tensor_wire.py` | 二进制 wire format fallback |
| `specforge/distributed.py` | 分布式初始化 + `SPECFORGE_GPU_ID` 支持 |
| `specforge/modeling/target/eagle3_target_model.py` | Eagle3 target model wrapper |
| `specforge/modeling/target/dflash_target_model.py` | DFlash target model wrapper |
| `specforge/modeling/target/sglang_backend/` | SGLang backend 适配层 |
