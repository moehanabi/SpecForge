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

## 2. 代码变更详解（bc815ec 以来）

### 2.1 `bc815ec` — feat: train-inference disaggregation（初始实现）

核心提交，引入训推分离的完整架构：

**新增文件：**
- `scripts/launch_target_server.py` — Target Model 服务端启动器
- `specforge/modeling/target/remote_target_client.py` — 训练端 remote backend 客户端
- `specforge/modeling/target/remote_target_server.py` — 服务端请求处理与模型推理
- `specforge/modeling/target/_nccl_transport.py` — NCCL 传输层（TCP rendezvous + send/recv）
- `specforge/modeling/target/_tensor_wire.py` — 二进制 wire format（NCCL fallback）
- `specforge/modeling/target/_shm_transport.py` — 共享内存传输（后续移除）
- `specforge/args.py` — `RemoteBackendArgs` 参数定义

**修改文件：**
- `scripts/train_eagle3.py` — 添加 `--target-model-backend remote` 支持
- `scripts/train_dflash.py` — 同上
- `specforge/distributed.py` — 添加 `SPECFORGE_GPU_ID` 支持
- `specforge/core/eagle3.py` — 适配 remote target model 接口

### 2.2 `e3cde67` — refactor: optimize codes

- 移除 `_shm_transport.py`（共享内存方案），统一使用 NCCL + wire format
- 精简 `remote_target_client.py`，移除冗余的 POSIX SHM 层
- 简化 NCCL transport 测试

### 2.3 `36c7c3d` — fix: support cross machine remote

- 修复跨机场景：client 不再限制 NCCL 仅在 localhost 使用
- `_get_server_host()` 从 remote URL 提取实际 host/IP
- Server 端 wildcard bind（`0.0.0.0`/`::`）映射到 TCPStore 监听所有接口
- 修复 NCCL rendezvous 在跨机 TCP 连接时的地址解析

### 2.4 `bb1dd12` — fix: Exit correctly

- 修复进程退出时 NCCL PG 清理挂起问题
- `NCCLTransport.destroy()` 调用 `pg.abort()` 并从 PyTorch `_world` 注册表注销
- 训练脚本添加 `target_model.close()` 确保资源释放
- `launch_target_server.py` 在 SIGTERM 时从独立线程调用 `HTTPServer.shutdown()`
- 添加 "Training process cleanup complete." 标记用于验证正常退出

### 2.5 `f0a476f` — fix: support new version sglang

- 适配新版 SGLang API 变更
- 更新 `sglang_backend/patch.py` 中的 monkey-patch 逻辑
- 修复 `model_runner.py` 中的接口兼容性

### 2.6 `ee79f7b` — fix: modify padding handle to lower mem use

- 优化 server 端 padding 处理，降低动态显存占用
- 修复 `eagle3_target_model.py` 中的 padding 逻辑
- 允许 `--mem-fraction-static` 设置更高值而不 OOM

### 2.7 `22f70be` — feat: Support backend setting

- Server 端添加 `--attention-backend` 参数（支持 flashinfer/fa3 等）
- 允许独立控制 server 的 attention backend 选择

### 2.8 `c6ace65` — feat: Support target prefetch（最新）

- 实现 target prefetch 流水线：训练端异步预取下一批 target 数据
- 添加 `--target-prefetch-depth` 参数控制预取队列深度
- 支持多服务器 round-robin 负载均衡（`--remote-urls` 逗号分隔）
- 训练端维护 prefetch queue，在训练计算期间异步发起下一次请求
- 实现 `_AsyncTargetHandle` 封装异步请求生命周期

## 3. 架构详解

### 3.1 Server 端

`launch_target_server.py` 通过 `torchrun` 启动，支持 TP>1 多 rank：

- **Rank 0**：运行 HTTP server + 处理请求 + NCCL send
- **Rank 1+**：参与 TP forward，通过 `broadcast_object_list` 同步请求

请求处理流程：
1. 接收 HTTP POST（input_ids, attention_mask 等 metadata）
2. 广播请求到所有 TP rank
3. 执行 target model forward（SGLang backend）
4. 计算 `target_p`（softmax + optional top-k 压缩）
5. 通过 NCCL send 返回 hidden_states / target_p 给 client

### 3.2 Client 端

`remote_target_client.py` 作为训练脚本的 target model backend：

- 首次请求时通过 POST `/init_nccl` 初始化 NCCL 数据通道
- 通过 POST `/setup` 获取模型配置（hidden_size, vocab_size 等）
- 每步训练通过 POST `/generate` 发送请求 + NCCL recv 接收结果
- 支持 TP>1 训练：仅 rank 0 发送请求，结果 broadcast 到其他 rank

### 3.3 NCCL Transport

`_nccl_transport.py` 实现专用 NCCL 传输层：

- Server = rank 0, Client = rank 1 组成 2-process NCCL group
- TCP rendezvous 通过 `torch.distributed.TCPStore` 建立连接
- 支持同机（NVLink）和跨机（RDMA/RoCE）传输
- `SPECFORGE_NCCL_PORT` 控制 rendezvous 端口（默认 HTTP port + 100）
- 安全退出：`pg.abort()` + 注销避免 `destroy_process_group` 挂起

### 3.4 Prefetch 流水线

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

### 3.5 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SPECFORGE_ENABLE_NCCL` | `1` | 启用 NCCL 传输（`0` 降级为 wire format） |
| `SPECFORGE_NCCL_PORT` | HTTP port + 100 | NCCL TCP rendezvous 端口 |
| `SPECFORGE_TOPK` | `0` | Server 端 target_p top-k 压缩（`0` 为全分布） |
| `SPECFORGE_TARGET_DTYPE` | `fp32` | target_p 计算精度 |
| `SPECFORGE_GPU_ID` | auto | 指定 GPU 设备 ID |

## 4. 精度验证

### 4.1 DFlash

| 配置 | 比较点 | 结果 |
|------|--------|------|
| TP=1 remote (single/multi) | 100 steps | **bit-perfect**（100% 精确匹配） |
| TP=2 remote (single/multi) | 100 steps | **bit-perfect**（100% 精确匹配） |

DFlash 仅使用 intermediate hidden_states（不经过 lm_head），无浮点非确定性。

### 4.2 Eagle3

| 配置 | 比较点 | 最大相对误差 | 说明 |
|------|--------|-------------|------|
| Train TP=1, Server TP=1 | 700 | 0.032% | 接近 bit-perfect |
| Train TP=2, Server TP=2 | 700 | 0.277% | TP all-reduce 引入误差 |
| Train TP=1, Server TP=2 | 700 | 0.705% | Server TP 不匹配放大误差 |

误差来源：FP8 cuBLAS GEMM 在 `lm_head` 中的非确定性，经 softmax 放大后体现在 target_p 上。所有配置误差 <1%，训练收敛不受影响。

## 5. 性能测试

### 5.1 测试环境

- **模型**：Qwen3-30B-A3B-FP8（Target）+ 1-layer Draft Model
- **硬件**：B200 × 8 / 机（两台机器，RDMA 互联）
- **参数**：seq_len=10240, batch_size=1, ttt_length=7
- **测试**：100 步，skip 前 5 步取平均
- **Prefetch**：depth=2，双服务器 round-robin

### 5.2 全配置速度对比

| 配置 | 训练TP | 服务器TP | 服务器数 | 步时 | vs TP1 baseline | 总GPU |
|------|--------|----------|----------|------|-----------------|-------|
| baseline TP=1 | 1 | — | 本地 | 0.378s | — | 1 |
| baseline TP=2 | 2 | — | 本地 | 0.470s | +24.3% | 2 |
| remote TP1+srv1 single | 1 | 1 | 1 | 0.287s | -24.1% | 2 |
| **remote TP1+srv1 multi** | **1** | **1** | **2** | **0.265s** | **-29.9%** | **3** |
| remote TP1+srv2 single | 1 | 2 | 1 | 0.288s | -23.8% | 3 |
| remote TP1+srv2 multi | 1 | 2 | 2 | 0.267s | -29.4% | 5 |
| remote TP2+srv2 single | 2 | 2 | 1 | 0.353s | -6.6% | 4 |
| remote TP2+srv2 multi | 2 | 2 | 2 | 0.320s | -15.3% | 6 |

### 5.3 GPU 时间分解（最优配置：Train TP=1, depth=2, multi）

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Draft forward（7 次 TTT 迭代） | 120.5ms | 45% |
| Backward | 131.4ms | 49% |
| Optimizer step | 7.5ms | 3% |
| Python overhead | 8ms | 3% |
| **总计** | **265ms** | 100% |

Server 延迟（~180ms/请求）完全被 prefetch overlap 隐藏（ready_gap=21ms）。

### 5.4 Batch scaling

| Batch | 步时 | 每样本时间 | vs batch=1 |
|-------|------|-----------|-----------|
| 1 | 265ms | 265ms | — |
| 4 | 988ms | 247ms | -7% |

seq=10240 时 GPU 已计算饱和，增大 batch 线性增加计算量，无法摊薄固定开销。

### 5.5 已验证无效的加速方案

| 方案 | 结果 | 原因 |
|------|------|------|
| `torch.compile` 整体 draft model | 0% | TTT loop 阻止 graph fusion |
| 训练端 TP=2 | -24% 退化 | Draft model 仅 FSDP，TP 增加通信开销 |
| Server TP=2 | 0% | Server 已被 prefetch 完全 overlap |
| Batch=4 | 仅 7% | GPU 已计算饱和 |
| 优化 server 端 target_p | 0% | Server 不在 critical path |

### 5.6 可探索的加速方向

| 方案 | 预期收益 | 代价 |
|------|----------|------|
| 减少 ttt_length（7→5） | ~28% | 需验证训练质量 |
| 减少 ttt_length（7→4） | ~43% | 需验证收敛性 |
| 减少 max-length | 与序列长度成正比 | 取决于数据分布 |
| FlashAttention for draft | 待测 | 需适配 backend |

## 6. 使用方法

### 6.1 启动 Target Server

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

### 6.2 启动训练（单 server）

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

### 6.3 启动训练（双 server，推荐）

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

### 6.4 DFlash 模式

DFlash 使用方式相同，将 `--mode eagle3` 改为 `--mode dflash`，训练脚本改为 `train_dflash.py`：

```bash
# Server
scripts/launch_target_server.py --mode dflash --port 8002 ...

# Training
scripts/train_dflash.py --target-model-backend remote --remote-url http://host:8002 ...
```

### 6.5 跨机 RDMA 配置

跨机部署时，NCCL 自动使用 RDMA（如果可用）。可通过环境变量优化：

```bash
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0        # 网络接口
export NCCL_IB_HCA=mlx5_bond_0         # IB HCA 设备
export NCCL_IB_GID_INDEX=3             # RoCE GID index
```

### 6.6 关键参数说明

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

### 6.7 精度验证模式

用于验证训推分离精度一致性：

```bash
# Server 端关闭 top-k 压缩，使用 fp32
SPECFORGE_TOPK=0 SPECFORGE_TARGET_DTYPE=fp32 python -m torch.distributed.run ... \
  scripts/launch_target_server.py ...

# 训练端同样设置
SPECFORGE_TOPK=0 SPECFORGE_TARGET_DTYPE=fp32 python -m torch.distributed.run ... \
  scripts/train_eagle3.py --target-model-backend remote ...
```

## 7. 最优配置推荐

```
Train TP=1 + Server TP=1 × 2台 + prefetch depth=2
```

| 指标 | 值 |
|------|-----|
| 步时 | 0.265s（比本地 baseline 快 30%） |
| 精度 | max_rel 0.032%（接近 bit-perfect） |
| 资源 | 训练 1 GPU + 服务器 2 GPU = 总共 3 GPU |
| 吞吐 | 3.77 samples/s（单卡训练） |

## 8. 文件索引

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
