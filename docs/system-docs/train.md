# train.py — 可改面（Agent 唯一编辑文件）

`train.py` 包含完整 GPT 模型、MuonAdamW 优化器与训练循环。Agent 的自由度都在这里；下面是当前代码事实与可调旋钮。所有行号指向目标 commit。

> 注意：下方「超参（可直接编辑）」一节的值即 Agent 当前基线；改模型结构/超参只需直接改这些顶层常量，无需命令行参数（见 `train.py:429` 注释）。

## 运行时生效的模型配置

`GPTConfig` 的 dataclass 默认值（12 层 / dim 768 / 6 头 / vocab 32768）只是占位；真正生效的配置由 `build_model_config(DEPTH)`（`train.py:469`）按 `DEPTH` 推导：

- `model_dim = ((DEPTH * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM`，`num_heads = model_dim // HEAD_DIM`
- 因此在基线 `DEPTH=8` 下：`model_dim = 512`（8×64，已是 128 的倍数）、`num_heads = 4`、`n_kv_head = 4`、`n_layer = 8`、`sequence_len = 2048`（取自 `MAX_SEQ_LEN`）、`vocab_size` 由分词器决定（8192）、`window_pattern = "SSSL"`。

## 模型结构要点

- **注意力**：GQA（kv 头数 ≤ q 头数）、RoPE（base 10000，预计算到 `sequence_len*10`）、q/k 走 RMSNorm 后做 FlashAttention-3（`train.py:21` 按 GPU 能力选 Hopper 用 `varunneal/flash-attention-3`，否则 `kernels-community/flash-attn3`），因果 + 滑窗。
- **滑窗模式** `window_pattern`：字符 `L`=满窗口（=context），`S`=半窗口；最后一层强制满窗口（`train.py:195`）。基线 `"SSSL"` = 前 3 层半窗、第 4 层满窗，循环。
- **MLP**：squared ReLU，即 `relu(x).square()`，宽度 4×dim（`train.py:99`）。
- **残差**：pre-norm，块内 `x = x + attn(norm(x))` 然后 `x = x + mlp(norm(x))`（`train.py:118`）。
- **Value Embedding（ResFormer）**：交错层（含最后一层）持有 value embedding，并以输入相关的 per-head gate 混入；`has_ve` 决定哪些层启用（`train.py:47`）。
- **每层层标量**：`resid_lambdas`（残差权重，init 1.0）与 `x0_lambdas`（init 0.1），forward 中 `x = resid_lambdas[i]*x + x0_lambdas[i]*x0`（`train.py:277`）。
- **LM head**：独立 `lm_head`，logits 做 softcap=15 的 `tanh`（`train.py:282`）；embeddings 与 value embeddings 初始化为 bf16。
- 权重初始化：wte `normal(0,1)`、lm_head `normal(0,0.001)`、矩阵 uniform、门控权重 0（`train.py:150`）。

## 优化器 MuonAdamW

组合优化器（`train.py:356`）：
- **Muon** 作用于 2D 矩阵参数（按 shape 分组），含 Polar-Express 正交化（`ns_steps=5`）与 NorMuon 方差缩减、cautious 权重衰减。
- **AdamW** 作用于其余参数：lm_head、token embeddings、value embeddings、残差/x0 标量。
- 实现用 `@torch.compile` 融合内核（`adamw_step_fused` / `muon_step_fused`），0-D CPU 张量避免重编译。
- LR 按 `1/sqrt(model_dim/768)` 缩放（`train.py:248`）。

`setup_optimizer` 默认（仅在该函数签名内）：`unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8,0.95), scalar_lr=0.5`。

## 超参（可直接编辑，顶层常量）

位于 `train.py:432` 起。当前值：

| 常量 | 当前值 | 含义 |
| --- | --- | --- |
| `ASPECT_RATIO` | 64 | `model_dim = DEPTH × 64`（再向上取整到 HEAD_DIM） |
| `HEAD_DIM` | 128 | 注意力头维度目标 |
| `WINDOW_PATTERN` | `"SSSL"` | 滑窗模式（L=满，S=半） |
| `TOTAL_BATCH_SIZE` | `2**19`（≈524K tokens） | 每优化步总 token 数 |
| `DEVICE_BATCH_SIZE` | 128 | 每卡 batch（OOM 时调小） |
| `EMBEDDING_LR` | 0.6 | token embedding LR（Adam） |
| `UNEMBEDDING_LR` | 0.004 | lm_head LR（Adam） |
| `MATRIX_LR` | 0.04 | 矩阵参数 LR（Muon） |
| `SCALAR_LR` | 0.5 | 每层标量 LR（Adam） |
| `WEIGHT_DECAY` | 0.2 | Muon 的 cautious 权重衰减 |
| `ADAM_BETAS` | (0.8, 0.95) | Adam beta1/beta2 |
| `WARMUP_RATIO` | 0.0 | 时间预算中用于 LR warmup 的比例 |
| `WARMDOWN_RATIO` | 0.5 | 时间预算中用于 LR warmdown 的比例 |
| `FINAL_LR_FRAC` | 0.0 | 末段 LR 相对初值的比例 |
| `DEPTH` | 8 | transformer 层数（驱动模型规模） |

梯度累积步数 `grad_accum_steps = TOTAL_BATCH_SIZE / (DEVICE_BATCH_SIZE × MAX_SEQ_LEN)`，并断言整除（基线 = 2）。

## 训练循环与调度

- 进度 `progress = total_training_time / TIME_BUDGET`（训练时间，前 10 步不计入以排除编译）。
- **LR 倍率** `get_lr_multiplier`：warmup 段线性升到 1，中间恒定 1，warmdown 段线性降到 `FINAL_LR_FRAC`（`train.py:518`）。
- **Muon 动量** `get_muon_momentum`：前 300 步从 0.85 升到 0.95（`train.py:527`）。
- **权重衰减** `get_weight_decay`：随进度从 `WEIGHT_DECAY` 线性衰减到 0（`train.py:531`）。
- **快速失败**：`NaN` 或 `train_loss > 100` → 打印 `FAIL` 并 `exit(1)`（`train.py:570`）。
- **结束条件**：`step > 10 且 total_training_time >= TIME_BUDGET` 时 break（`train.py:603`）。
- 硬件假设：bf16 autocast；MFU 按 `H100_BF16_PEAK_FLOPS = 989.5e12` 计算（`train.py:463`）。
- 结束评估：用 `DEVICE_BATCH_SIZE` 调 `evaluate_bpb`（`train.py:613`）。

## 改动的边界与建议

- 公平游戏：结构、超参、优化器、batch、模型规模都可在 `train.py` 内调；VRAM 适度增加可接受。
- 不可触碰：`prepare.py`、评估函数、依赖清单。改这些会让实验失去可比性且破坏契约。
- 简洁优先：相对 0 收益但让代码复杂化的改动不值得；删代码且效果持平/更好是胜利。