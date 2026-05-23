# 训推分离全场景基准测试报告 (跨机版)

**生成时间**: 2026-05-27 21:04
**总耗时**: 51 min
**分支**: remote_train_main_github
**模型**: Qwen3-30B-A3B-Instruct-2507-FP8
**每实验步数**: 100
**数据**: preformatted, max_length=12288, warmup_ratio=0

## 测试环境

| 项目 | 说明 |
|------|------|
| Server 机器 | h-860 (10.95.103.24), 8x NVIDIA Hopper 80GB |
| Client 机器 | h-728 (10.48.51.15), 8x NVIDIA Hopper 80GB |
| 机间互联 | RoCE v2 RDMA, 388 Gb/s (mlx5_2~mlx5_9) |
| NCCL 配置 | IB_HCA=mlx5_3..9, GID_INDEX=3, SOCKET_IFNAME=bond0 |

## 总览

| 指标 | 值 |
|------|-----|
| 总实验数 | 28 |
| 成功 | 28 |
| 失败 | 0 |
| Baseline | 4 (B1-B4) |
| 单机训推分离 | 12 (S1-S12) |
| 双机训推分离 | 12 (D1-D12) |

## Baseline 测试结果 (sglang backend, 训推不分离)

| 实验 | 模型 | TP | 步数 | Loss | Avg Iter (s) | 备注 |
|------|------|-----|------|------|-------------|------|
| B1 | DFlash | 1 | 100 | 6.4594 | 0.263 | success |
| B2 | DFlash | 2 | 100 | 6.4624 | 0.215 | success |
| B3 | EAGLE3 | 1 | 100 | 0.1999 | 0.533 | success |
| B4 | EAGLE3 | 2 | 100 | 0.1588 | 0.610 | success |

## 单机训推分离测试结果 (跨机，训练 x 1，推理 x 1)

### DFlash

| 实验 | TP | Depth | 步数 | Loss | Avg Iter (s) | 加速比 vs Baseline | 加速比 vs depth=0 |
|------|-----|-------|------|------|-------------|-------------------|-------------------|
| S1 | 1 | 0 | 100 | 6.4594 | 0.256 | 1.03x | - |
| S2 | 1 | 1 | 100 | 6.4594 | 0.188 | 1.39x | 1.36x |
| S3 | 1 | 2 | 100 | 6.4594 | 0.189 | 1.39x | 1.36x |
| S4 | 2 | 0 | 100 | 6.4624 | 0.202 | 1.06x | - |
| S5 | 2 | 1 | 100 | 6.4624 | 0.132 | 1.63x | 1.53x |
| S6 | 2 | 2 | 100 | 6.4624 | 0.132 | 1.63x | 1.53x |

### EAGLE3

| 实验 | TP | Depth | 步数 | Loss | Avg Iter (s) | 加速比 vs Baseline | 加速比 vs depth=0 |
|------|-----|-------|------|------|-------------|-------------------|-------------------|
| S7 | 1 | 0 | 100 | 0.1999 | 0.521 | 1.02x | - |
| S8 | 1 | 1 | 100 | 0.1999 | 0.342 | 1.56x | 1.52x |
| S9 | 1 | 2 | 100 | 0.1999 | 0.323 | 1.65x | 1.61x |
| S10 | 2 | 0 | 100 | 0.2006 | 0.456 | 1.34x | - |
| S11 | 2 | 1 | 100 | 0.2006 | 0.325 | 1.88x | 1.41x |
| S12 | 2 | 2 | 100 | 0.2006 | 0.326 | 1.87x | 1.40x |

## 双机训推分离测试结果 (跨机，训练 x 1，推理 x 2)

### DFlash

| 实验 | TP | Depth | 步数 | Loss | Avg Iter (s) | 加速比 vs Baseline | 加速比 vs depth=0 |
|------|-----|-------|------|------|-------------|-------------------|-------------------|
| D1 | 1 | 0 | 100 | 6.4594 | 0.255 | 1.03x | - |
| D2 | 1 | 1 | 100 | 6.4594 | 0.184 | 1.43x | 1.39x |
| D3 | 1 | 2 | 100 | 6.4594 | 0.111 | 2.37x | 2.30x |
| D4 | 2 | 0 | 100 | 6.4624 | 0.205 | 1.05x | - |
| D5 | 2 | 1 | 100 | 6.4624 | 0.134 | 1.61x | 1.53x |
| D6 | 2 | 2 | 100 | 6.4624 | 0.094 | 2.28x | 2.17x |

### EAGLE3

| 实验 | TP | Depth | 步数 | Loss | Avg Iter (s) | 加速比 vs Baseline | 加速比 vs depth=0 |
|------|-----|-------|------|------|-------------|-------------------|-------------------|
| D7 | 1 | 0 | 100 | 0.1999 | 0.472 | 1.13x | - |
| D8 | 1 | 1 | 100 | 0.1999 | 0.283 | 1.88x | 1.67x |
| D9 | 1 | 2 | 100 | 0.1999 | 0.273 | 1.95x | 1.73x |
| D10 | 2 | 0 | 100 | 0.2006 | 0.454 | 1.34x | - |
| D11 | 2 | 1 | 100 | 0.2006 | 0.291 | 2.09x | 1.56x |
| D12 | 2 | 2 | 100 | 0.2006 | 0.288 | 2.12x | 1.58x |

## 对比分析

### 1. Baseline vs 训推分离 (DFlash)

| 配置 | Baseline | 训推分离 depth=0 | depth=1 | depth=2 |
|------|----------|-----------------|---------|---------|
| TP=1 单机 | 0.263s | 0.256s (1.03x) | 0.188s (1.39x) | 0.189s (1.39x) |
| TP=2 单机 | 0.215s | 0.202s (1.06x) | 0.132s (1.63x) | 0.132s (1.63x) |
| TP=1 推理 x 2 | 0.263s | 0.255s (1.03x) | 0.184s (1.43x) | **0.111s (2.37x)** |
| TP=2 双机 | 0.215s | 0.205s (1.05x) | 0.134s (1.61x) | **0.094s (2.28x)** |

**关键发现**:
- 训推分离本身 (depth=0) 比 baseline 快 3-6%，因为 target model 不再与 draft model 争抢 GPU 资源
- depth=1 预取带来额外 36-53% 加速（vs depth=0）
- **双机 depth=2 实现最高加速比 2.37x** — 两台 server 交替预取，完全隐藏 target forward 延迟
- 单机 depth=2 与 depth=1 无差异（只有一个 server，无法并行预取）

### 2. Baseline vs 训推分离 (EAGLE3)

| 配置 | Baseline | 训推分离 depth=0 | depth=1 | depth=2 |
|------|----------|-----------------|---------|---------|
| TP=1 单机 | 0.533s | 0.521s (1.02x) | 0.342s (1.56x) | 0.323s (1.65x) |
| TP=2 单机 | 0.610s | 0.456s (1.34x) | 0.325s (1.88x) | 0.326s (1.87x) |
| TP=1 双机 | 0.533s | 0.472s (1.13x) | 0.283s (1.88x) | 0.273s (1.95x) |
| TP=2 双机 | 0.610s | 0.454s (1.34x) | 0.291s (2.09x) | **0.288s (2.12x)** |

**关键发现**:
- EAGLE3 TP=2 baseline (0.610s) 比 TP=1 (0.533s) 慢，因为 sglang backend TP=2 通信开销大
- 训推分离后 TP=2 反而更快（draft model 单卡训练不受 TP 通信影响，且 TP=2 target forward 更快）
- depth=1 已是性价比最高的配置（1.56-2.09x 加速）
- depth=2 在 EAGLE3 上额外收益有限（draft forward 耗时长，已部分隐藏 target forward）

### 3. 单机 vs 双机

| 模型 | TP | Depth | 单机 Iter (s) | 双机 Iter (s) | 双机优势 |
|------|-----|-------|---------------|---------------|---------|
| DFlash | 1 | 0 | 0.256 | 0.255 | ~0% |
| DFlash | 1 | 1 | 0.188 | 0.184 | 2% |
| DFlash | 1 | 2 | 0.189 | 0.111 | **41%** |
| DFlash | 2 | 0 | 0.202 | 0.205 | ~0% |
| DFlash | 2 | 1 | 0.132 | 0.134 | ~0% |
| DFlash | 2 | 2 | 0.132 | 0.094 | **29%** |
| EAGLE3 | 1 | 0 | 0.521 | 0.472 | 9% |
| EAGLE3 | 1 | 1 | 0.342 | 0.283 | **17%** |
| EAGLE3 | 1 | 2 | 0.323 | 0.273 | **15%** |
| EAGLE3 | 2 | 0 | 0.456 | 0.454 | ~0% |
| EAGLE3 | 2 | 1 | 0.325 | 0.291 | **10%** |
| EAGLE3 | 2 | 2 | 0.326 | 0.288 | **12%** |

**关键发现**:
- **DFlash depth=2 双机有显著优势** (29-41%)：两个 server 可以真正 pipeline，每个处理一个 prefetch 请求
- EAGLE3 双机在 depth≥1 时有 10-17% 优势：双 server 分担 target forward 负载
- depth≤1 时双机对 DFlash 几乎无优势（单 server 已足够快，瓶颈不在 target forward）
- EAGLE3 因 draft forward 耗时长（含多层 hidden state 提取），双机在 depth=1 就有 17% 优势

### 4. Prefetch Depth 收益分析

| 模型 | 场景 | depth=0→1 加速 | depth=1→2 额外加速 | 说明 |
|------|------|---------------|-------------------|------|
| DFlash | 单机 TP=1 | +36% | +0% | 单 server 无法并行预取 |
| DFlash | 单机 TP=2 | +53% | +0% | 同上 |
| DFlash | 双机 TP=1 | +39% | +66% | 双 server pipeline 生效 |
| DFlash | 双机 TP=2 | +53% | +43% | 同上 |
| EAGLE3 | 单机 TP=1 | +52% | +6% | draft forward 长，depth=2 有限额外收益 |
| EAGLE3 | 单机 TP=2 | +41% | -0% | 同上 |
| EAGLE3 | 双机 TP=1 | +67% | +4% | draft forward 是瓶颈，双 server 帮助有限 |
| EAGLE3 | 双机 TP=2 | +56% | +1% | 同上 |

**结论**: DFlash 因 draft forward 极快（~0.02s），target forward 是绝对瓶颈，双机 depth=2 能完全隐藏。EAGLE3 因 draft forward 较慢（~0.3s），即使 target forward 被完全隐藏，总 iter_time 仍受 draft forward 限制。

## 精度验证

### Loss 一致性

| 模型 | 场景 | TP=1 Loss | TP=2 Loss | 说明 |
|------|------|-----------|-----------|------|
| DFlash | Baseline (sglang) | 6.4594 | 6.4624 | TP 浮点差异 0.05% |
| DFlash | Remote depth=0 | 6.4594 | 6.4624 | 与 baseline 完全一致 |
| DFlash | Remote depth=1 | 6.4594 | 6.4624 | 与 depth=0 完全一致 |
| DFlash | Remote depth=2 | 6.4594 | 6.4624 | 与 depth=0 完全一致 |
| EAGLE3 | Baseline (sglang) | 0.1999 | 0.1588 | TP=2 改变训练语义（见下方） |
| EAGLE3 | Remote depth=0 | 0.1999 | 0.2006 | TP 浮点差异 0.3% |
| EAGLE3 | Remote depth=1 | 0.1999 | 0.2006 | 与 depth=0 完全一致 |
| EAGLE3 | Remote depth=2 | 0.1999 | 0.2006 | 与 depth=0 完全一致 |

**精度结论**:
1. **不同 prefetch depth 对精度无影响** — 纯 pipeline 优化，不改变计算路径
2. **跨机 RDMA 传输无精度损失** — GPU tensor 直传，无序列化/反序列化
3. **DFlash: baseline loss 与 remote loss 完全一致** (6.4594/6.4624)，训推分离无精度损失
4. **单机 vs 双机精度完全一致** — DFlash: 6.4594/6.4624, EAGLE3: 0.1999/0.2006

### EAGLE3 TP=2 Baseline vs Remote Loss 差异说明

Baseline TP=2 loss (0.1588) 低于 Remote TP=2 loss (0.2006) 的原因：

**Baseline 使用 `nproc_per_node=2`** 启动两个 GPU 进程，分布式布局为：
- `dp_size = world_size // tp_size = 2 // 2 = 1`
- Target model: TP=2 张量并行
- Draft model: FSDP on WORLD group (`ShardingStrategy.SHARD_GRAD_OP`)

**EAGLE3 TP=2 baseline 的实际行为**：
1. `target_batch_size = tp_size * batch_size`，dataloader 产出 2 倍 batch
2. Target 模型（TP=2）协作处理整个 2 倍 batch
3. 通过 `get_dp_data_shard_from_tp()` 按 tp_rank 切分 → 每个 rank 取**不同的**半 batch 训练 draft
4. FSDP 在 WORLD group 上做 gradient averaging → **draft 模型本质上是数据并行训练**
5. 因此 TP=2 改变了 draft 的优化轨迹（2 倍有效 batch + gradient averaging），loss 收敛到更低值

**Remote 模式**：Client 始终 `nproc_per_node=1`，draft 模型单卡训练，无 FSDP，无数据切分。TP 只影响 server 端 target forward 的并行方式，不改变 client 端 draft 训练语义。

**DFlash 为什么没有这个差异**：DFlash baseline TP=2 中两个 rank 看到**相同 batch**（通过 `dp_size=1` 的 dataloader），FSDP 同步的梯度本来就相同，训练语义不变。

## 最佳配置推荐

| 场景 | 推荐配置 | 期望加速 (vs sglang baseline) | Iter Time |
|------|---------|-------------------------------|-----------|
| DFlash 单机 | TP=2, depth=1 | **1.63x** | 0.132s |
| DFlash 双机 | TP=2, depth=2 | **2.28x** | 0.094s |
| EAGLE3 单机 | TP=2, depth=1 | **1.88x** | 0.325s |
| EAGLE3 双机 | TP=2, depth=1 | **2.09x** | 0.291s |

**选型建议**:
- 只有 1 台额外 GPU 机器 → 单机训推分离 + depth=1 即可获得 1.4-1.9x 加速
- 有 2 台额外 GPU 机器 → DFlash 用 depth=2 可达 2.3x，EAGLE3 用 depth=1 即可达 2.1x
- EAGLE3 双机 depth=2 vs depth=1 差异仅 1%，不值得多占一台 server

## 环境信息

- GPU: 2 × 8x NVIDIA Hopper (80GB HBM3)
- 网络: RoCE v2 RDMA, 实测 388 Gb/s (ib_write_bw)
- 模型: Qwen3-30B-A3B-Instruct-2507-FP8 (Qwen3-MoE, 30B activated params, ~30GB FP8)
- 数据: 103998 samples, preformatted, chat_template=qwen
- Max Length: 12288 tokens
- Batch Size: 1
- Warmup Ratio: 0 (LR=6e-4 from step 1)
- Server mem-fraction-static: DFlash=0.7, EAGLE3=0.5
- NCCL: IB_HCA=mlx5_3..9, IB_GID_INDEX=3, SOCKET_IFNAME=bond0, IB_TIMEOUT=22

## 技术说明

1. **真实跨机**: 物理分离，通过 RoCE v2 RDMA 通信
2. **双机测试**: 两台机器各运行一个 Server, Client 通过 round-robin 调度
3. **NCCL transport**: 跨机 GPU tensor 传输使用 RoCE v2 RDMA (400Gbps mlx5 网卡)
4. **Prefetch 原理**: 在 draft model 训练时提前发起下一个 sample 的 target forward，实现 pipeline overlap
5. **双机优势**: depth=2 时在两台 server 上交替预取，充分隐藏 target forward 延迟
