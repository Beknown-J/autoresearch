# prepare.py — 固定契约（只读）

`prepare.py` 提供固定常量、一次性数据准备、分词器、运行时数据加载器与评估函数。它**只读**，Agent 与实验不得修改（改动会破坏实验可比性与事实标准指标）。

## 固定常量

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `MAX_SEQ_LEN` | 2048 | 上下文长度 |
| `TIME_BUDGET` | 300（秒） | 训练时间预算（5 分钟） |
| `EVAL_TOKENS` | `40 × 524288` | 验证评估使用的 token 数 |
| `VOCAB_SIZE` | 8192 | BPE 词表大小（训练时取 `vocab_size - 4` 个可合并词位） |
| `MAX_SHARD` | 6542 | 数据分片总数（最后一片 `shard_06542`） |
| `VAL_SHARD` | 6542 | 固定验证分片（= `MAX_SHARD`） |
| `SPECIAL_TOKENS` | 4 个 `<|reserved_i|>` | 特殊 token，其中 `<|reserved_0|>` 为 BOS |
| `BASE_URL` | climbmix-400b-shuffle（HF） | 训练数据来源 |

## 数据准备

- `download_data`：从 HuggingFace 下载 parquet 分片（默认 10 片训练 + 固定验证片），8 进程并行、带重试；数据落在 `~/.cache/autoresearch/data`（`prepare.py:38`）。
- `train_tokenizer`：用 `rustbpe` 训练 BPE，导出为 tiktoken pickle（`tokenizer.pkl`）；额外保存 `token_bytes.pt` 用于 BPB 评估（每个 token 的 UTF-8 字节长度，特殊 token 记为 0）。训练前做 roundtrip 健全性断言。
- 用法：`python prepare.py`（完整）、`python prepare.py --num-shards 8`（仅下载 8 片）。

## 分词器与数据加载器

- `Tokenizer`：对 `prepare.py` 导出的 `tiktoken.Encoding` 的轻封装，`from_directory()` 反序列化；提供 `encode/decode`、`get_vocab_size`、`get_bos_token_id`。
- `get_token_bytes`：读 `token_bytes.pt`（每个 token 的字节数，特殊 token=0），BPB 计算用。
- `make_dataloader`（train/val）：**BOS 对齐 + best-fit 打包**的无限迭代器，100% 利用率（无 padding）；每行以 BOS 起头，文档 best-fit 装入，装不下时裁最短文档填满。验证加载器固定在 `VAL_SHARD`。

## 评估函数（事实标准）`evaluate_bpb`

`train.py` 的唯一指标来源，**不可改**（`prepare.py:343`）：

- 对固定 `MAX_SEQ_LEN` 的验证集跑 `EVAL_TOKENS // (batch_size × MAX_SEQ_LEN)` 步。
- 对每个 token 的交叉熵（nats）求和，再对目标字节长度求和，最后 `nats/byte → /log(2) → bits/byte`。
- **特殊 token（字节长度 0）在两个求和中都被排除**，因此指标与词表大小无关，可公平比较架构改动。

> Agent 只能让模型在该指标上更低；任何改 `evaluate_bpb` 或 `prepare.py` 常量的行为都是违规。