# Offline RL：State-Action Coverage 与 Dataset Size 的性能决定权研究

## 1. 项目简介

本项目研究 **Offline Reinforcement Learning** 中的一个核心问题：

> **在 Offline RL 中，state-action coverage 是否比 dataset size 更能决定策略的性能上限？**

### 核心研究设计

在离散 gridworld 环境（EnvA_v2，30×30 四走廊迷宫）上构造四类数据集：

| 数据集 | 数据量 | Coverage 宽度 |
|--------|--------|--------------|
| small-wide | 5 万条 | 宽（~21% SA 覆盖率） |
| small-narrow | 5 万条 | 窄（~6% SA 覆盖率） |
| large-wide | 20 万条 | 宽（~21% SA 覆盖率） |
| large-narrow | 20 万条 | 窄（~6% SA 覆盖率） |

**核心对照**：small-wide vs large-narrow（数据量更少但 coverage 更广 vs 数据量更大但 coverage 更窄）。

对照算法：BC（行为克隆）、CQL（保守 Q-learning）、IQL（隐式 Q-learning），每组 20 个训练种子。

### 主结论

**coverage 是决定 Offline RL 性能上限的主要因素，dataset size 的独立效应较弱。**

| 算法 | Small-Wide | Large-Narrow | 差值（SW−LN） |
|------|-----------|--------------|-------------|
| BC   | 0.3265 | 0.2700 | **+0.057** |
| CQL  | 0.3970 | 0.2700 | **+0.127** |
| IQL  | 0.3970 | 0.2700 | **+0.127** |

三个算法均显示 small-wide > large-narrow。固定 narrow coverage 时，将数据量从 5 万增加到 20 万，CQL/IQL 性能提升 Δ ≈ 0.002–0.005，BC 在 narrow coverage 下也无提升（Δ=0）。

### 辅助验证

- **Quality sweep**：数据质量在超过随机门槛后对性能的边际贡献接近于零（BC、IQL），证明 coverage 操纵是主效应变量。
- **EnvB/C validation**：单路径环境（瓶颈迷宫、钥匙-门迷宫）中 wide/narrow 数据集 SA 覆盖率几乎相同，gap = 0，划定了 coverage 效应的适用边界。
- **Hopper D4RL benchmark**：验证 BC/IQL/TD3+BC 实现与文献一致，作为外部参照点。

---

## 2. 仓库结构

```
RL_Final_Project/
├── envs/                          # 核心环境实现
│   ├── gridworld_envs.py          # 三个 gridworld 环境定义
│   └── __init__.py
│
├── scripts/                       # 所有可执行脚本
│   ├── final_analysis_and_plots.py        # 最终分析：生成全部表格、图、报告
│   ├── run_envA_v2_main_experiment.py     # 主实验：BC/CQL × EnvA_v2 × 4 数据集
│   ├── run_envA_v2_iql_main.py            # IQL 主实验：EnvA_v2 × 4 数据集
│   ├── run_envA_v2_quality_sweep.py       # BC/CQL quality sweep × 5 档
│   ├── run_envA_v2_iql_quality_sweep.py   # IQL quality sweep × 5 档
│   ├── run_envbc_validation.py            # BC/CQL EnvB/C 跨环境验证
│   ├── run_envbc_iql_validation.py        # IQL EnvB/C 跨环境验证
│   ├── run_envA_v2_mechanism_analysis.py  # SA coverage 机制分析
│   ├── run_hopper_benchmark.py            # Hopper D4RL 连续控制 benchmark
│   ├── generate_envA_v2_final_datasets.py # 生成 EnvA_v2 冻结数据集
│   ├── build_envA_v2_behavior_pool.py     # 训练行为策略池（DQN）
│   └── audit_final_datasets.py            # 数据集审计工具
│
├── tests/                         # 核心功能测试
│   ├── test_phase1_envs.py                # 基础环境可用性测试
│   ├── test_envA_v2_structure.py          # EnvA_v2 结构验证
│   ├── test_envA_v2_final_datasets.py     # 冻结数据集完整性测试
│   ├── test_envA_v2_main_experiment.py    # 主实验管线 smoke test
│   ├── test_envA_v2_iql_main.py           # IQL 主实验 smoke test
│   ├── test_envA_v2_quality_sweep.py      # quality sweep smoke test
│   ├── test_envA_v2_iql_quality_sweep.py  # IQL quality sweep smoke test
│   ├── test_envbc_validation.py           # EnvB/C 验证 smoke test
│   ├── test_envbc_iql_validation.py       # IQL EnvB/C 验证 smoke test
│   ├── test_envA_v2_mechanism_analysis.py # 机制分析 smoke test
│   └── test_hopper_benchmark.py           # Hopper benchmark smoke test
│
├── artifacts/
│   ├── final_datasets/            # 冻结的正式实验数据集（.npz）
│   ├── final_results/             # 最终结果汇总表（4 个 CSV）
│   ├── behavior_pool/             # 行为策略池 checkpoint（.pt）
│   ├── analysis/                  # 机制分析中间数据
│   ├── training_main/             # BC/CQL 主实验 summary CSV
│   ├── training_iql/              # IQL 系列 summary CSV
│   ├── training_quality/          # Quality sweep summary CSV
│   ├── training_validation/       # EnvB/C 验证 summary CSV
│   └── training_benchmark/        # Hopper benchmark summary CSV
│
├── figures/final/                 # 最终 6 张核心图（.png）
├── reports/
│   └── final_project_results.md  # 最终分析报告（含全部结论与解释）
├── docs/
│   ├── EXP_PROTOCOL.md            # 实验协议（研究问题、数据集设计、算法范围）
│   └── PROJECT_SCOPE.md           # 项目边界与必做项定义
├── requirements.txt
└── README.md
```

---

## 3. 关键文件作用说明

### 环境（`envs/`）

- **`envs/gridworld_envs.py`**：定义三个离散 gridworld 环境：
  - `EnvA_v2`：30×30 四走廊迷宫（主实验环境），670 个可达状态，2680 个 SA 对，观测为 900 维 one-hot 编码。
  - `EnvB`：15×15 双瓶颈迷宫（跨环境验证，单路径结构）。
  - `EnvC`：15×15 钥匙-门迷宫（跨环境验证，单路径结构）。
  - 同文件还定义 `HORIZON`（每轮最长步数）和 `N_ACTIONS`（动作空间大小=4）。

### 核心脚本（`scripts/`）

- **`final_analysis_and_plots.py`**：主分析入口。读取所有 summary CSV，生成 4 张结果汇总表（`artifacts/final_results/`）、6 张核心图（`figures/final/`）和最终报告（`reports/final_project_results.md`）。**只需运行这一个脚本即可重新生成所有分析输出。**

- **`run_envA_v2_main_experiment.py`**：EnvA_v2 主实验（BC + CQL × 4 数据集 × 20 seeds），结果写入 `artifacts/training_main/envA_v2_main_summary.csv`。

- **`run_envA_v2_iql_main.py`**：EnvA_v2 IQL 主实验（4 数据集 × 20 seeds），结果写入 `artifacts/training_iql/envA_v2_iql_main_summary.csv`。

- **`run_envA_v2_quality_sweep.py`**：BC/CQL quality sweep（5 质量档 × 20 seeds），结果写入 `artifacts/training_quality/envA_v2_quality_summary.csv`。

- **`run_envA_v2_iql_quality_sweep.py`**：IQL quality sweep（5 质量档 × 20 seeds），结果写入 `artifacts/training_iql/envA_v2_iql_quality_sweep_summary.csv`。

- **`run_envbc_validation.py`**：BC/CQL 在 EnvB/EnvC 上的跨环境验证（各 2 数据集 × 20 seeds），结果写入 `artifacts/training_validation/envbc_validation_summary.csv`。

- **`run_envbc_iql_validation.py`**：IQL 在 EnvB/EnvC 上的跨环境验证，结果写入 `artifacts/training_iql/envbc_iql_validation_summary.csv`。

- **`run_envA_v2_mechanism_analysis.py`**：计算每个 training run 的 SA coverage 指标和 OOD action rate，结果写入 `artifacts/analysis/`。

- **`run_hopper_benchmark.py`**：Hopper-v2 D4RL benchmark（BC/CQL/IQL/TD3+BC × 3 数据集 × 5 seeds），使用 d3rlpy，结果写入 `artifacts/training_benchmark/hopper_benchmark_summary.csv`。

- **`generate_envA_v2_final_datasets.py`**：基于行为策略池生成 EnvA_v2 的 9 个冻结数据集（4 主实验 + 5 quality 档），写入 `artifacts/final_datasets/`。

- **`build_envA_v2_behavior_pool.py`**：用 DQN 训练 8×3 行为策略（expert/medium/suboptimal 各 8 个 seed），checkpoint 写入 `artifacts/behavior_pool/`。

- **`audit_final_datasets.py`**：验证冻结数据集的 SA coverage、数据量、转移质量，生成审计报告。

### 测试（`tests/`）

- **`test_phase1_envs.py`**：基础验证三个环境可以正常 step/reset、返回合法 obs 和 reward。
- **`test_envA_v2_structure.py`**：验证 EnvA_v2 可达状态数、SA 覆盖率上限、地图形状等结构属性。
- **`test_envA_v2_final_datasets.py`**：验证 9 个冻结数据集的形状、数据量和 SA coverage 是否符合设计规格。
- **`test_envA_v2_main_experiment.py`**：主实验脚本的端到端 smoke test（少量 seeds，快速验证管线可通）。
- **`test_envA_v2_iql_main.py`**：IQL 主实验管线 smoke test。
- **`test_envA_v2_quality_sweep.py` / `test_envA_v2_iql_quality_sweep.py`**：quality sweep 管线 smoke test。
- **`test_envbc_validation.py` / `test_envbc_iql_validation.py`**：EnvB/C 验证管线 smoke test。
- **`test_envA_v2_mechanism_analysis.py`**：机制分析管线 smoke test。
- **`test_hopper_benchmark.py`**：Hopper benchmark 管线 smoke test。

### 数据与结果（`artifacts/`）

- **`artifacts/final_datasets/`**：9 个冻结的正式 `.npz` 数据集（EnvA_v2 四格主实验 + 5 quality 档 + EnvB/C 验证共 13 文件），是全部实验的数据基础。
- **`artifacts/final_results/`**：最终 4 张汇总表：
  - `final_discrete_results_master_table.csv`：主实验结果（BC/CQL/IQL × 4 数据集，含 mean/std/95% CI）
  - `final_quality_results_table.csv`：quality sweep 结果
  - `final_validation_results_table.csv`：EnvB/C 跨环境验证结果
  - `final_benchmark_results_table.csv`：Hopper benchmark 结果
- **`artifacts/behavior_pool/`**：40 个行为策略 `.pt` checkpoint，用于重新生成数据集。
- **`artifacts/analysis/`**：机制分析输出，包含 seed 级别的 SA coverage 和 OOD rate 指标。
- **`artifacts/training_*/`**：各实验阶段的 summary CSV（原始 per-run 结果汇总，`final_analysis_and_plots.py` 的输入源）。

### 图与报告

- **`figures/final/`**：6 张核心图：
  - `fig1_main_coverage_vs_size.png`：四格矩阵总结果图（coverage × size 双因素）
  - `fig2_core_smallwide_vs_largenarrow.png`：核心对照图（small high-coverage vs large low-coverage）
  - `fig3_quality_modulation.png`：quality 调制结果图
  - `fig4_envbc_validation.png`：EnvB/C 跨环境验证图
  - `fig5_mechanism_summary.png`：SA coverage 机制分析图
  - `fig6_benchmark_validation.png`：Hopper benchmark 趋势图

- **`reports/final_project_results.md`**：最终分析报告。包含完整的主结论、数据表、辅助验证解释、机制分析、局限性说明。**建议优先阅读此文件。**

### 协议文档（`docs/`）

- **`docs/EXP_PROTOCOL.md`**：实验协议，定义研究问题、数据集设计规格（size/coverage 正交矩阵）、算法范围。
- **`docs/PROJECT_SCOPE.md`**：项目边界，定义必做项与非必做项。

---

## 4. 环境与依赖

**Python 版本**：3.10 或以上（开发环境 3.12.2）

**安装依赖**：

```bash
pip install -r requirements.txt
```

**关键依赖**：

| 库 | 用途 |
|----|------|
| `torch >= 2.0` | BC/CQL/IQL 神经网络训练 |
| `numpy >= 1.24` | 数组操作、数据集处理 |
| `matplotlib >= 3.7` | 图表生成 |
| `scipy >= 1.10` | 统计检验（95% 置信区间） |
| `d3rlpy >= 2.0` | Hopper benchmark（连续控制算法） |
| `gymnasium >= 0.29` | Hopper-v2 环境（benchmark 用） |
| `h5py >= 3.8` | D4RL HDF5 数据文件读取（benchmark 用） |

> 说明：`d3rlpy`、`gymnasium`、`h5py` 仅 Hopper benchmark（`run_hopper_benchmark.py`）需要。若只关心离散主线实验，可不安装这三个库。

---

## 5. 复现步骤

### Option A：只查看最终结果（无需运行任何代码）

所有最终结果均已预计算并保存在仓库中：

1. 阅读 `reports/final_project_results.md` — 完整分析报告（推荐首先看这里）
2. 查看 `artifacts/final_results/` — 4 张汇总 CSV 表
3. 查看 `figures/final/` — 6 张核心图

### Option B：重新生成分析输出（表格 + 图 + 报告）

不需要重跑任何实验，只从已有 summary CSV 重新生成所有分析产物：

```bash
# Step 1: 安装依赖
pip install torch numpy matplotlib scipy

# Step 2: 运行最终分析脚本
python scripts/final_analysis_and_plots.py
```

输出写入 `artifacts/final_results/`、`figures/final/`、`reports/final_project_results.md`。

### Option C：重跑主实验（需要 GPU，耗时较长）

> **注意**：重跑实验需要完整 GPU 环境，EnvA_v2 主实验约需数小时。
> 所有训练脚本均支持断点续跑（逐 run append 模式）。

**Step 1**：确认冻结数据集已存在

```bash
ls artifacts/final_datasets/
# 应看到 13 个 .npz 文件
```

若数据集缺失，可重新生成（需要行为策略池 checkpoint）：

```bash
python scripts/build_envA_v2_behavior_pool.py   # 训练行为策略池
python scripts/generate_envA_v2_final_datasets.py  # 生成数据集
```

**Step 2**：运行 BC/CQL 主实验

```bash
python scripts/run_envA_v2_main_experiment.py
```

**Step 3**：运行 IQL 主实验

```bash
python scripts/run_envA_v2_iql_main.py
```

**Step 4**：运行 quality sweep

```bash
python scripts/run_envA_v2_quality_sweep.py
python scripts/run_envA_v2_iql_quality_sweep.py
```

**Step 5**：运行 EnvB/C 跨环境验证

```bash
python scripts/run_envbc_validation.py
python scripts/run_envbc_iql_validation.py
```

**Step 6**：运行机制分析

```bash
python scripts/run_envA_v2_mechanism_analysis.py
```

**Step 7**（可选）：运行 Hopper D4RL benchmark

```bash
# 需要额外依赖：d3rlpy, gymnasium, h5py
# 需要下载 Hopper D4RL 数据集（运行时自动下载，或手动放入 artifacts/training_benchmark/d4rl_cache/）
python scripts/run_hopper_benchmark.py
```

**Step 8**：重新生成最终分析输出

```bash
python scripts/final_analysis_and_plots.py
```

---

## 6. 关键脚本命令速查

```bash
# 最终分析（最重要）——从已有 summary CSV 生成所有表格、图、报告
python scripts/final_analysis_and_plots.py

# BC/CQL 主实验（EnvA_v2 × 4 数据集 × 20 seeds）
python scripts/run_envA_v2_main_experiment.py

# IQL 主实验（EnvA_v2 × 4 数据集 × 20 seeds）
python scripts/run_envA_v2_iql_main.py

# BC/CQL quality sweep（5 质量档 × 20 seeds）
python scripts/run_envA_v2_quality_sweep.py

# IQL quality sweep（5 质量档 × 20 seeds）
python scripts/run_envA_v2_iql_quality_sweep.py

# BC/CQL EnvB/C 跨环境验证（各 2 数据集 × 20 seeds）
python scripts/run_envbc_validation.py

# IQL EnvB/C 跨环境验证
python scripts/run_envbc_iql_validation.py

# SA coverage 机制分析
python scripts/run_envA_v2_mechanism_analysis.py

# Hopper benchmark（需 d3rlpy/gymnasium）
python scripts/run_hopper_benchmark.py

# 运行核心测试（不运行完整实验，快速 smoke test）
python -m pytest tests/ -v
```

---

## 7. 最终输出说明

| 文件 | 位置 | 内容 |
|------|------|------|
| 最终分析报告 | `reports/final_project_results.md` | 完整研究结论、数据、解释、局限性 |
| 主实验结果表 | `artifacts/final_results/final_discrete_results_master_table.csv` | BC/CQL/IQL × 4 数据集，mean/std/95% CI |
| Quality sweep 结果表 | `artifacts/final_results/final_quality_results_table.csv` | 5 质量档 × 3 算法 |
| 跨环境验证表 | `artifacts/final_results/final_validation_results_table.csv` | EnvB/C × 3 算法 |
| Benchmark 结果表 | `artifacts/final_results/final_benchmark_results_table.csv` | Hopper × 4 算法 |
| 核心对照图 | `figures/final/fig2_core_smallwide_vs_largenarrow.png` | small-wide vs large-narrow 主结论图 |
| 四格矩阵图 | `figures/final/fig1_main_coverage_vs_size.png` | coverage × size 双因素结果 |
| Quality 调制图 | `figures/final/fig3_quality_modulation.png` | 数据质量门槛效应 |
| EnvB/C 验证图 | `figures/final/fig4_envbc_validation.png` | 跨环境 zero-gap 结果 |
| 机制分析图 | `figures/final/fig5_mechanism_summary.png` | SA coverage 与性能的对应关系 |
| Benchmark 图 | `figures/final/fig6_benchmark_validation.png` | Hopper D4RL 趋势参照 |

**建议阅读顺序**：`reports/final_project_results.md` → `fig2` → `fig1` → `fig3` → `fig4` → `fig5` → `fig6`

---

## 8. 当前问题与局限

### 环境结构导致的局限

1. **EnvB/C 单路径结构**：EnvB（双瓶颈）和 EnvC（钥匙-门）都是单路径结构，任何策略必须经过相同关键节点，导致 wide/narrow 数据集的 SA 覆盖率几乎相同（EnvB ≈ 99%）。wide vs narrow 的 gap 为 0，无法形成有效 coverage 对照。**结论：coverage 效应需要环境中存在多条可选路径。EnvB/C 未能实现跨环境泛化验证。**

2. **环境规模偏小**：EnvA_v2 共 2680 个 SA 对，即使 narrow 数据集也覆盖了测试轨迹的大多数状态，导致 OOD action rate 在所有条件下均为 0.000。无法直接观测分布偏移（distribution shift）效应，"coverage 约束 OOD 风险"的机制路径在此环境中不可见。

3. **离散动作空间限制**：IQL 和 TD3+BC 的连续控制版本未纳入离散主线实验（离散环境不适用连续控制算法的原始实现）。

### Benchmark 配置导致的局限

4. **CQL 连续控制配置未针对 D4RL 调优**：Hopper benchmark 中 CQL 的 `cql_alpha` 等超参数使用默认值（`batch_size=256`，未按 D4RL 标准调参），导致 normalized score 仅 1–3，远低于文献报告的 ~58。**此异常不影响离散主线结论**，因为离散 CQL 的配置是独立调优的。

5. **Benchmark 仅 5 个种子**：统计置信度低于离散主线（20 seeds），Hopper 结果仅作趋势参照，不作为精确基准。

### 结论的解释边界

- **主结论适用范围**：coverage 效应在 EnvA_v2（多路径离散 gridworld）上得到三算法 20-seed 验证。能否推广至连续控制环境或更大规模环境，需进一步实验。
- **BC 的 size 效应**：BC 在 wide coverage 条件下存在 Δ=+0.0745 的 size 效应，说明 size 并非完全无效——其效应大小依赖算法和 coverage 条件的交互。

---

## 9. 公开版说明

本仓库为项目最终公开整理版。已移除以下内容：

- 所有中间开发阶段文件（archive/）
- 所有训练 checkpoint（训练主实验 `.pt` 文件，共约 2.2 GB）
- 内部开发协议文件（clean restart 计划、patch 链协议等）
- 原始 D4RL HDF5 缓存文件（可运行时自动下载）
- 调试脚本、诊断脚本、sanity 脚本、pilot 脚本
- Python 缓存文件（`__pycache__`、`.pyc`、`.pytest_cache`）

保留的是主线实验所需的最小必要内容：环境代码、训练脚本、测试文件、冻结数据集、summary CSV、最终图表与报告。
