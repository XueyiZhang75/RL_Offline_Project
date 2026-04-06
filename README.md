# Offline RL: Does State-Action Coverage Matter More Than Dataset Size?

A study of the relative impact of **state-action coverage** versus **dataset size** on the performance ceiling of Offline Reinforcement Learning algorithms.

---

## 1. Project Overview

### Research Question

> **In Offline RL, does state-action (SA) coverage determine the policy performance ceiling more than dataset size?**

The core comparison is between two dataset conditions on a discrete gridworld (EnvA_v2, 30×30 four-corridor maze):

| Condition | Transitions | SA Coverage |
|-----------|------------|-------------|
| small-wide | 50k | ~21% of all SA pairs |
| small-narrow | 50k | ~6% of all SA pairs |
| large-wide | 200k | ~21% of all SA pairs |
| large-narrow | 200k | ~6% of all SA pairs |

**Primary contrast**: small-wide vs large-narrow — smaller data with broader coverage versus larger data with narrower coverage.

Three algorithms are evaluated on this contrast: BC (Behavior Cloning), CQL (Conservative Q-Learning), and IQL (Implicit Q-Learning), each with 20 training seeds.

### Core Conclusion

**SA coverage is the primary determinant of the Offline RL performance ceiling; the independent effect of dataset size is weak.**

| Algorithm | Small-Wide | Large-Narrow | Gap (SW−LN) |
|-----------|-----------|--------------|-------------|
| BC  | 0.3265 | 0.2700 | **+0.057** |
| CQL | 0.3970 | 0.2700 | **+0.127** |
| IQL | 0.3970 | 0.2700 | **+0.127** |

All three algorithms show small-wide > large-narrow. When coverage is held fixed at the narrow level, increasing dataset size from 50k to 200k yields no improvement (Δ = 0 for all three algorithms).

### Primary vs Auxiliary Evidence

- **Primary**: EnvA_v2 main experiment — 4-condition factorial (coverage × size), BC/CQL/IQL, 20 seeds each
- **Auxiliary (quality sweep)**: 5 data quality levels on EnvA_v2 — shows that quality has a threshold effect; above the random floor, further quality improvements yield negligible gains
- **Auxiliary (EnvB/C validation)**: Cross-environment check on single-path mazes — shows zero coverage gap, delimiting where the coverage effect applies
- **Auxiliary (Hopper D4RL benchmark)**: Continuous control reference — validates that BC/IQL/TD3+BC implementations match published baselines

---

## 2. Repository Structure

```
RL_Final_Project/
├── envs/                          # Environment implementations
│   ├── gridworld_envs.py
│   └── __init__.py
│
├── scripts/                       # All executable scripts
│   ├── final_analysis_and_plots.py        # Main analysis entry point
│   ├── run_envA_v2_main_experiment.py     # BC/CQL main experiment
│   ├── run_envA_v2_iql_main.py            # IQL main experiment
│   ├── run_envA_v2_quality_sweep.py       # BC/CQL quality sweep
│   ├── run_envA_v2_iql_quality_sweep.py   # IQL quality sweep
│   ├── run_envbc_validation.py            # BC/CQL cross-environment validation
│   ├── run_envbc_iql_validation.py        # IQL cross-environment validation
│   ├── run_envA_v2_mechanism_analysis.py  # SA coverage mechanism analysis
│   ├── run_hopper_benchmark.py            # Hopper D4RL benchmark
│   ├── generate_envA_v2_final_datasets.py # Generate frozen datasets
│   ├── build_envA_v2_behavior_pool.py     # Train behavior policy pool (DQN)
│   └── audit_final_datasets.py            # Dataset audit tool
│
├── tests/                         # Core functional tests
│   ├── test_phase1_envs.py
│   ├── test_envA_v2_structure.py
│   ├── test_envA_v2_final_datasets.py
│   ├── test_envA_v2_main_experiment.py
│   ├── test_envA_v2_iql_main.py
│   ├── test_envA_v2_quality_sweep.py
│   ├── test_envA_v2_iql_quality_sweep.py
│   ├── test_envbc_validation.py
│   ├── test_envbc_iql_validation.py
│   ├── test_envA_v2_mechanism_analysis.py
│   └── test_hopper_benchmark.py
│
├── artifacts/
│   ├── final_datasets/            # Frozen official experiment datasets (.npz)
│   ├── final_results/             # Final summary tables (4 CSVs)
│   ├── behavior_pool/             # Behavior policy checkpoints (.pt)
│   ├── analysis/                  # Mechanism analysis intermediate data
│   ├── training_main/             # BC/CQL main experiment summary CSV
│   ├── training_iql/              # IQL series summary CSVs
│   ├── training_quality/          # Quality sweep summary CSV
│   ├── training_validation/       # EnvB/C validation summary CSV
│   └── training_benchmark/        # Hopper benchmark summary CSVs
│
├── figures/final/                 # All 6 final figures (.png)
├── reports/
│   └── final_project_results.md  # Final analysis report
├── requirements.txt
└── README.md
```

---

## 3. File-by-File Role Explanation

### Environment (`envs/`)

**`envs/gridworld_envs.py`** — Defines all three discrete gridworld environments:
- `EnvA_v2`: 30×30 four-corridor maze, the primary experiment environment. Has 670 reachable states and 2,680 SA pairs. Observations are 900-dimensional one-hot vectors (flattened 30×30 grid).
- `EnvB`: 15×15 double-bottleneck maze used for cross-environment validation. Single-path structure.
- `EnvC`: 15×15 key-door maze used for cross-environment validation. Single-path structure.
- Also exports `HORIZON` (max steps per episode) and `N_ACTIONS` (= 4: up/down/left/right).

### Scripts (`scripts/`)

**`scripts/final_analysis_and_plots.py`** — The primary analysis entry point. Reads all experiment summary CSVs, produces 4 result tables (`artifacts/final_results/`), 6 figures (`figures/final/`), and the final report (`reports/final_project_results.md`). Running this single script regenerates all analysis outputs without rerunning any experiments.

**`scripts/run_envA_v2_main_experiment.py`** — Runs the BC and CQL main experiment on EnvA_v2 across all 4 dataset conditions × 20 seeds. Output: `artifacts/training_main/envA_v2_main_summary.csv`.

**`scripts/run_envA_v2_iql_main.py`** — Runs the IQL main experiment on EnvA_v2 across all 4 dataset conditions × 20 seeds. Output: `artifacts/training_iql/envA_v2_iql_main_summary.csv`.

**`scripts/run_envA_v2_quality_sweep.py`** — Runs BC and CQL across 5 data quality levels (random / suboptimal / medium / expert / mixed) × 20 seeds. Output: `artifacts/training_quality/envA_v2_quality_summary.csv`.

**`scripts/run_envA_v2_iql_quality_sweep.py`** — Runs IQL across the same 5 quality levels × 20 seeds. Output: `artifacts/training_iql/envA_v2_iql_quality_sweep_summary.csv`.

**`scripts/run_envbc_validation.py`** — Runs BC and CQL on EnvB and EnvC (2 dataset conditions × 20 seeds each). Output: `artifacts/training_validation/envbc_validation_summary.csv`.

**`scripts/run_envbc_iql_validation.py`** — Runs IQL on EnvB and EnvC cross-environment validation. Output: `artifacts/training_iql/envbc_iql_validation_summary.csv`.

**`scripts/run_envA_v2_mechanism_analysis.py`** — Computes per-run SA coverage metrics and OOD action rates for each experiment run, linking coverage to performance. Output: `artifacts/analysis/`.

**`scripts/run_hopper_benchmark.py`** — Runs BC, CQL, IQL, and TD3+BC on Hopper-v2 (3 D4RL dataset splits × 5 seeds) using d3rlpy. Output: `artifacts/training_benchmark/hopper_benchmark_summary.csv`.

**`scripts/generate_envA_v2_final_datasets.py`** — Uses the trained behavior policy pool to generate all 9 frozen EnvA_v2 datasets (4 main experiment conditions + 5 quality sweep variants). Output: `artifacts/final_datasets/`.

**`scripts/build_envA_v2_behavior_pool.py`** — Trains 24 DQN behavior policies (3 quality levels × 8 seeds) for EnvA_v2. Output: `artifacts/behavior_pool/`.

**`scripts/audit_final_datasets.py`** — Validates frozen dataset SA coverage, transition count, and quality against design specifications.

### Tests (`tests/`)

**`tests/test_phase1_envs.py`** — Verifies all three environments accept valid actions, return correct observation shapes, and terminate correctly.

**`tests/test_envA_v2_structure.py`** — Validates EnvA_v2 structural properties: number of reachable states, SA pair count, corridor topology.

**`tests/test_envA_v2_final_datasets.py`** — Checks all 9 frozen `.npz` datasets for shape, transition count, and SA coverage against design specs.

**`tests/test_envA_v2_main_experiment.py`** — End-to-end smoke test for the BC/CQL main experiment pipeline (small seed count, fast).

**`tests/test_envA_v2_iql_main.py`** — End-to-end smoke test for the IQL main experiment pipeline.

**`tests/test_envA_v2_quality_sweep.py`** / **`test_envA_v2_iql_quality_sweep.py`** — Smoke tests for the quality sweep pipeline (BC/CQL and IQL respectively).

**`tests/test_envbc_validation.py`** / **`test_envbc_iql_validation.py`** — Smoke tests for the EnvB/C cross-environment validation pipeline.

**`tests/test_envA_v2_mechanism_analysis.py`** — Smoke test for the mechanism analysis pipeline.

**`tests/test_hopper_benchmark.py`** — Smoke test for the Hopper benchmark pipeline.

### Data and Results (`artifacts/`)

**`artifacts/final_datasets/`** — 13 frozen `.npz` dataset files: 4 main experiment conditions for EnvA_v2, 5 quality sweep variants for EnvA_v2, and 2 conditions each for EnvB and EnvC. These are the inputs to all training experiments.

**`artifacts/final_results/`** — 4 final summary tables:
- `final_discrete_results_master_table.csv`: Main experiment results (BC/CQL/IQL × 4 conditions, with mean/std/95% CI)
- `final_quality_results_table.csv`: Quality sweep results
- `final_validation_results_table.csv`: EnvB/C cross-environment validation results
- `final_benchmark_results_table.csv`: Hopper D4RL benchmark results

**`artifacts/behavior_pool/`** — 40 trained DQN policy checkpoints used to generate the frozen datasets. Also includes `behavior_policy_catalog.csv` and `envA_v2_controller_eval.csv`.

**`artifacts/analysis/`** — Output from mechanism analysis: `envA_v2_mechanism_summary.csv` and per-seed metrics `envA_v2_mechanism_seed_metrics.csv`.

**`artifacts/training_*/`** — Per-experiment summary CSVs produced by the training scripts. These are the source data for `final_analysis_and_plots.py`.

### Figures (`figures/final/`)

- **`fig1_main_coverage_vs_size.png`** — Four-condition matrix: mean return across all coverage × size combinations for BC, CQL, and IQL.
- **`fig2_core_smallwide_vs_largenarrow.png`** — Core contrast bar chart: small-wide vs large-narrow across all three algorithms.
- **`fig3_quality_modulation.png`** — Quality sweep results showing the threshold effect and quality insensitivity above the random floor.
- **`fig4_envbc_validation.png`** — EnvB/C cross-environment validation showing zero wide/narrow gap due to single-path structure.
- **`fig5_mechanism_summary.png`** — SA coverage vs mean return scatter, illustrating the mechanism linking coverage to performance.
- **`fig6_benchmark_validation.png`** — Hopper D4RL benchmark results for external reference validation.

### Report (`reports/`)

**`reports/final_project_results.md`** — Full final analysis report. Contains the complete research conclusions, data tables, interpretation of all experiments, mechanism analysis, and limitations. **Recommended first read.**

---

## 4. Environment and Dependencies

**Python**: 3.10 or higher (developed on 3.12.2)

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Key libraries:**

| Library | Purpose |
|---------|---------|
| `torch >= 2.0` | Neural network training for BC / CQL / IQL |
| `numpy >= 1.24` | Array operations, dataset handling |
| `matplotlib >= 3.7` | Figure generation |
| `scipy >= 1.10` | Statistical tests (95% confidence intervals) |
| `d3rlpy >= 2.0` | Continuous control algorithms for Hopper benchmark |
| `gymnasium >= 0.29` | Hopper-v2 environment (benchmark only) |
| `h5py >= 3.8` | D4RL HDF5 dataset loading (benchmark only) |

> Note: `d3rlpy`, `gymnasium`, and `h5py` are only required for the Hopper benchmark script (`run_hopper_benchmark.py`). For the discrete gridworld main line, only `torch`, `numpy`, `matplotlib`, and `scipy` are needed.

---

## 5. Step-by-Step Reproduction

### Option A — Inspect final results (no code required)

All results are precomputed and included in the repository.

- **Step 1**: Read `reports/final_project_results.md` for the complete analysis report (recommended starting point).
- **Step 2**: Open `artifacts/final_results/` to inspect the 4 summary CSV tables.
- **Step 3**: Open `figures/final/` to view the 6 final figures.

### Option B — Regenerate all analysis outputs (tables + figures + report)

This does not rerun any experiments. It reads existing summary CSVs and reproduces all outputs.

```bash
# Step 1: Install core dependencies
pip install torch numpy matplotlib scipy

# Step 2: Run the final analysis script
python scripts/final_analysis_and_plots.py
```

Outputs are written to `artifacts/final_results/`, `figures/final/`, and `reports/final_project_results.md`.

### Option C — Rerun experiments (GPU required; time-intensive)

> **Warning**: Full experiment reruns require a GPU and take several hours for the main experiment alone. All training scripts support resumable append mode (each seed is written individually; completed seeds are skipped on re-run).

**Step 1**: Verify frozen datasets exist.

```bash
ls artifacts/final_datasets/
# Should show 13 .npz files
```

If datasets are missing, regenerate them (requires behavior pool checkpoints):

```bash
python scripts/build_envA_v2_behavior_pool.py    # Train DQN behavior policies
python scripts/generate_envA_v2_final_datasets.py # Generate frozen datasets
```

**Step 2**: Run the BC/CQL main experiment.

```bash
python scripts/run_envA_v2_main_experiment.py
```

**Step 3**: Run the IQL main experiment.

```bash
python scripts/run_envA_v2_iql_main.py
```

**Step 4**: Run quality sweep experiments.

```bash
python scripts/run_envA_v2_quality_sweep.py
python scripts/run_envA_v2_iql_quality_sweep.py
```

**Step 5**: Run cross-environment validation.

```bash
python scripts/run_envbc_validation.py
python scripts/run_envbc_iql_validation.py
```

**Step 6**: Run mechanism analysis.

```bash
python scripts/run_envA_v2_mechanism_analysis.py
```

**Step 7 (optional)**: Run Hopper D4RL benchmark.

```bash
# Requires: d3rlpy, gymnasium, h5py
# D4RL dataset files are downloaded automatically on first run
python scripts/run_hopper_benchmark.py
```

**Step 8**: Regenerate final analysis outputs.

```bash
python scripts/final_analysis_and_plots.py
```

---

## 6. Key Script Commands

```bash
# Regenerate all tables, figures, and the final report from existing CSVs
python scripts/final_analysis_and_plots.py

# BC/CQL main experiment — EnvA_v2, 4 conditions × 20 seeds
python scripts/run_envA_v2_main_experiment.py

# IQL main experiment — EnvA_v2, 4 conditions × 20 seeds
python scripts/run_envA_v2_iql_main.py

# BC/CQL quality sweep — 5 quality levels × 20 seeds
python scripts/run_envA_v2_quality_sweep.py

# IQL quality sweep — 5 quality levels × 20 seeds
python scripts/run_envA_v2_iql_quality_sweep.py

# BC/CQL cross-environment validation — EnvB + EnvC, 2 conditions × 20 seeds
python scripts/run_envbc_validation.py

# IQL cross-environment validation — EnvB + EnvC
python scripts/run_envbc_iql_validation.py

# SA coverage mechanism analysis
python scripts/run_envA_v2_mechanism_analysis.py

# Hopper D4RL benchmark (requires d3rlpy / gymnasium)
python scripts/run_hopper_benchmark.py

# Run all smoke tests
python -m pytest tests/ -v
```

---

## 7. Final Outputs

| File | Location | Description |
|------|----------|-------------|
| Final analysis report | `reports/final_project_results.md` | Complete conclusions, tables, interpretation |
| Main experiment table | `artifacts/final_results/final_discrete_results_master_table.csv` | BC/CQL/IQL × 4 conditions, mean/std/95% CI |
| Quality sweep table | `artifacts/final_results/final_quality_results_table.csv` | 5 quality levels × 3 algorithms |
| Validation table | `artifacts/final_results/final_validation_results_table.csv` | EnvB/C × 3 algorithms |
| Benchmark table | `artifacts/final_results/final_benchmark_results_table.csv` | Hopper × 4 algorithms |
| Core contrast figure | `figures/final/fig2_core_smallwide_vs_largenarrow.png` | Primary conclusion visualization |
| Four-condition matrix | `figures/final/fig1_main_coverage_vs_size.png` | Coverage × size double-factor results |
| Quality modulation figure | `figures/final/fig3_quality_modulation.png` | Quality threshold effect |
| Cross-env validation figure | `figures/final/fig4_envbc_validation.png` | EnvB/C zero-gap result |
| Mechanism figure | `figures/final/fig5_mechanism_summary.png` | SA coverage vs performance |
| Benchmark figure | `figures/final/fig6_benchmark_validation.png` | Hopper D4RL reference trends |

**Recommended reading order**: `reports/final_project_results.md` > `fig2` > `fig1` > `fig3` > `fig4` > `fig5` > `fig6`

---

## 8. Current Issues and Limitations

### Limitations from Environment Structure

**1. EnvB/C single-path structure invalidates cross-environment generalization.**
EnvB (double-bottleneck) and EnvC (key-door) both have single-path layouts where any policy must pass through the same critical nodes. This makes the SA coverage of wide and narrow datasets nearly identical (EnvB ≈ 99% coverage). The wide vs narrow gap is 0.000 for all algorithms on both environments. The cross-environment validation does not provide a meaningful coverage contrast; it only confirms that the training pipeline runs correctly on other environments and delimits where the coverage effect applies (multi-path environments).

**2. Environment scale too small to observe distribution shift.**
EnvA_v2 has only 2,680 SA pairs. Even the narrow dataset covers most states encountered during evaluation trajectories. OOD (out-of-distribution) action rate is 0.000 under all experimental conditions. The mechanism by which coverage constrains performance cannot be attributed to distribution shift in this setting; coverage acts instead by limiting the diversity of accessible state-action pairs during learning.

**3. Discrete action space excludes continuous control algorithms from the main line.**
IQL and TD3+BC in their original continuous-control formulations are not directly applicable to the discrete gridworld. The Hopper benchmark serves as a separate validation track rather than as part of the coverage study.

### Limitations from Benchmark Configuration

**4. CQL is not properly tuned for the D4RL Hopper benchmark.**
The Hopper benchmark uses default CQL hyperparameters (`batch_size=256`, `cql_alpha` not tuned for D4RL). This produces normalized scores of 1–3, far below the published value of ~58. This anomaly does not affect the discrete main-line conclusions — the discrete CQL configuration was tuned independently — but the benchmark CQL results should not be used as a reference.

**5. Hopper benchmark uses only 5 seeds.**
Statistical confidence is lower than the discrete main line (20 seeds). Hopper results are provided as directional reference only.

### Interpretation Boundaries

**6. Main conclusion is specific to multi-path discrete environments.**
The coverage advantage of small-wide over large-narrow is validated on EnvA_v2 (four-corridor, 30×30 discrete gridworld). Whether this finding generalizes to continuous control, higher-dimensional state spaces, or environments with different topology requires further study.

**7. BC shows a non-negligible size effect under wide coverage.**
BC achieves a gain of Δ=+0.0745 when moving from small-wide to large-wide, indicating that dataset size is not irrelevant for BC when coverage is already broad. The conclusion that size matters less than coverage holds directionally but the magnitude of the size effect is algorithm-dependent.

