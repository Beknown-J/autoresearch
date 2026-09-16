# autoresearch 系统文档

> 目标 commit：`7890a85d686308084e2e43c98950d12a0d00ccef`（master tip）
> 本文档基于仓库真实代码取证，解释系统是什么、怎么用、有哪些规则与约束，不逐文件复述源码。

## 这是什么

autoresearch 是一项「让 LLM Agent 自主做 LLM 预训练研究」的实验：在一个小型但真实的单 GPU 训练框架上，Agent 改代码 → 训练 5 分钟 → 看指标是否变好 → 保留或丢弃，循环往复。代码是从 [nanochat](https://github.com/karpathy/nanochat) 精简化剥出的单 GPU 实现。

核心规则（详见 [experiment-protocol.md](./experiment-protocol.md)）：
- **固定时间预算**：每次实验训练满 5 分钟（墙钟时间，不含启动/编译），与硬件无关。
- **唯一指标**：`val_bpb`（验证集 bits per byte，越低越好），对词表大小无关，便于跨架构公平比较。
- **Agent 只改一个文件**：`train.py`。`prepare.py` 只读，评估函数 `evaluate_bpb` 是事实标准，不可改。

## 主要使用者

- **人类研究者**：搭建实验（分支 `autoresearch/<tag>`）、维护 `program.md`、查看结果。
- **AI Agent（研究者）**：在分支上反复迭代 `train.py`，并把每次实验结果写入 `results.tsv`。

## 文件地图

| 文件 | 角色 | 可改？ |
| --- | --- | --- |
| `prepare.py` | 固定常量、数据下载、BPE 分词器训练、数据加载器、评估函数 `evaluate_bpb` | 否（只读） |
| `train.py` | GPT 模型、MuonAdamW 优化器、训练循环；Agent 唯一可改文件 | 是（仅此文件） |
| `program.md` | Agent 的基线指令；人类迭代这份「研究组织代码」 | 人类可改 |
| `analysis.ipynb` | 离线分析 `results.tsv`、绘制 `progress.png` | 辅助 |
| `pyproject.toml` / `.python-version` | 依赖与 Python 版本（3.10） | 否（不可新增依赖） |
| `results.tsv` | 实验记录（commit/val_bpb/memory_gb/status/description），git 忽略 | Agent 写，不提交 |

## 快速开始

前置：单张 NVIDIA GPU（H100 验证过）、Python 3.10+、[uv](https://docs.astral.sh/uv/)。

```bash
uv sync                      # 安装依赖
uv run prepare.py            # 一次性下载数据 + 训练分词器（~2 min）
uv run train.py              # 手动跑一次实验（~5 min）
```

进入自主研究模式：让 Agent 读 `program.md` 并在 `autoresearch/<tag>` 分支上循环实验（详见 [experiment-protocol.md](./experiment-protocol.md)）。

## 关键约束（Agent 行为边界）

1. 只能改动 `train.py`（模型结构、超参、优化器、batch、训练循环等）。
2. **不能**改 `prepare.py`；**不能**新增/添加依赖（只能用 `pyproject.toml` 已有的）；**不能**改评估函数 `evaluate_bpb`。
3. VRAM 是软约束：为明显收益小幅增加可接受，但不要爆涨。
4. 简洁准则：同等效果下更简单更好；删代码得到相等或更好的结果就是胜利。

## 文档索引

- [train.md](./train.md) — `train.py` 可改面：模型结构、MuonAdamW、训练循环、全部超参当前值。
- [prepare.md](./prepare.md) — `prepare.py` 固定契约：常量、数据、分词器、数据加载器、`evaluate_bpb`。
- [experiment-protocol.md](./experiment-protocol.md) — 自主实验循环、`results.tsv` 格式、指标与打印输出、分支与时间预算执行。