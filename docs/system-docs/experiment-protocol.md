# 自主实验协议

目标 commit 的 `program.md` 定义了 Agent 的自主研究循环。下面是代码与 `program.md` 共同确认的事实。

## 一次实验是什么

单 GPU、单次 `uv run train.py`：固定 5 分钟（墙钟训练时间，不含启动/编译），跑完打印指标摘要。指标只看 `val_bpb`（越低越好）。

## 实验循环

Agent 在专用分支（如 `autoresearch/mar5`）上无限循环（`program.md`）：

1. 看 git 状态（当前分支/commit）。
2. 直接在 `train.py` 上做实验性改动。
3. `git commit`。
4. 运行：`uv run train.py > run.log 2>&1`（重定向全部输出，勿用 tee）。
5. 读结果：`grep "^val_bpb:\|^peak_vram_mb:" run.log`。
6. 若 grep 为空 = 崩溃：`tail -n 50 run.log` 看栈；小错误可修后重跑，思路本身坏则跳过并记为 `crash`。
7. 把结果写入 `results.tsv`（**不提交该文件**，保持 untracked）。
8. `val_bpb` 更低 → 保留 commit（前进分支）。
9. 持平或更差 → `git reset` 回起点。

**永不停止**：循环开始后不向人类请求是否继续，自主运行直到被人工中断。单次实验约 5 分钟，约 12 次/小时。

**超时**：>10 分钟视为失败（丢弃并回退）。

## 时间预算的执行（代码侧）

`train.py` 以 `total_training_time` 累加每步耗时（前 10 步排除，避开编译），当 `step > 10 且 total_training_time >= TIME_BUDGET(300s)` 时 break（`train.py:578`、`:603`）。因此实验可比、且平台无关（都是 5 分钟）。代价：结果不可跨不同算力平台横向比较。

## results.tsv 格式

Tab 分隔（**不是逗号**），表头一行 + 5 列（`program.md`）：

```
commit\tval_bpb\tmemory_gb\tstatus\tdescription
```

- `commit`：短 hash（7 字符）。
- `val_bpb`：达到的 val_bpb（如 `1.234567`）；崩溃记 `0.000000`。
- `memory_gb`：峰值显存（GB，`peak_vram_mb/1024`，保留 1 位）；崩溃记 `0.0`。
- `status`：`keep` / `discard` / `crash`。
- `description`：本次实验做了什么的简短说明。

示例：

```
a1b2c3d\t0.997900\t44.0\tkeep\tbaseline
d4e5f6g\t0.000000\t0.0\tcrash\tdouble model width (OOM)
```

该文件被 `.gitignore` 忽略，不纳入版本控制。

## 打印输出与指标

训练结束打印（`train.py:621`）：

```
---
val_bpb:          0.997900
training_seconds: 300.1      # 墙钟训练时间（排除启动/编译）
total_seconds:    325.9      # 含启动（t_end - t_start）
peak_vram_mb:     45060.2
mfu_percent:      39.80       # 稳态 MFU（按 H100 BF16 峰值）
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

- **`val_bpb`**：bits per byte，源自 `evaluate_bpb`（`prepare.py:343`），词表无关。
- **`training_seconds`**：累计训练时间（不含前 10 步）。
- **`peak_vram_mb`**：`torch.cuda.max_memory_allocated`。
- **`mfu_percent`**：相对 `H100_BF16_PEAK_FLOPS=989.5e12` 的稳态 MFU（`train.py:618`）。

## 结果分析

`analysis.ipynb` 离线读 `results.tsv`（5 列同上），统计实验数、各状态计数、keep 率、绘制 val_bpb 随实验推进的「前沿」曲线并输出 `progress.png`（README 的配图）。仅用于复盘，不参与训练。

## 分支约定

- 新实验：`git checkout -b autoresearch/<tag>`（tag 如 `mar5`，须不存在）。
- 训练/迭代只在 `autoresearch/*` 分支进行；master 为基线（当前 commit `7890a85d686308084e2e43c98950d12a0d00ccef`）。