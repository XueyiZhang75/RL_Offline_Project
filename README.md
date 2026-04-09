# Offline RL: Does State-Action Coverage Matter More Than Dataset Size?

A study of the relative impact of **state-action (SA) coverage** versus **dataset size**
on the performance ceiling of Offline Reinforcement Learning algorithms.

> **Course delivery version** — all experiments frozen, results reproducible.
> See `docs/FINAL_PROJECT_STATUS_CHECK.md` for authoritative completion status.

---

## Research Question

> **In Offline RL, does state-action (SA) coverage determine the policy performance ceiling
> more decisively than dataset size?**

The core comparison is between two dataset conditions on a 30×30 four-corridor discrete
gridworld (EnvA_v2): a *small dataset with broad SA coverage* versus a *large dataset with
narrow SA coverage*.

---

## Core Conclusion

**SA coverage is the primary determinant of the Offline RL performance ceiling; the
independent effect of dataset size is weak.**

| Algorithm | Small-Wide (50k, ~21% SA) | Large-Narrow (200k, ~6% SA) | Gap (SW−LN) |
|-----------|--------------------------|------------------------------|-------------|
| BC  | 0.3265 | 0.2700 | **+0.057** |
| CQL | 0.3970 | 0.2700 | **+0.127** |
| IQL | 0.3970 | 0.2700 | **+0.127** |

All three algorithms confirm: 4× less data with broader SA coverage outperforms a larger
dataset with narrow coverage. When SA coverage is held fixed at the narrow level, increasing
dataset size from 50k to 200k yields **exactly Δ = 0** for all algorithms.

**Mechanism**: SA coverage determines which behavioral strategy families are learnable from
the data, not merely which states are reachable. OOD action rate = 0 in all conditions;
the effect is *behavioral diversity*, not distribution-shift prevention.

---

## Evidence Structure

### Primary Evidence — EnvA_v2 (fully closed)
- 2×2 factorial design: coverage (wide ~21% / narrow ~6%) × size (50k / 200k)
- BC, CQL, IQL — 20 training seeds each
- Statistical closure: Claims 1.1 and 1.2 formally closed (p < 1e-7)
- See `reports/final_project_results.md` §2–3

### Cross-Environment Validation (rebuilt multi-path environments)

| Environment | BC gap | IQL gap | CQL gap | Status |
|-------------|--------|---------|---------|--------|
| EnvA_v2 (4-corridor, primary) | +0.057 | +0.127 | +0.127 | **3/3 CONFIRMED** |
| EnvB_v2 (3-corridor, rebuilt) | +0.020 | +0.026 | +0.025 | **3/3 CONFIRMED** |
| EnvC_v2 (key-door, rebuilt) | +0.020 | +0.024 | +0.002* | **BC+IQL confirmed; CQL mixed** |

*EnvC_v2 CQL gap is positive but negligible (CI overlapping, 1/20 seeds). CQL's conservative
Q-penalty suppresses path-diversity discovery in staged multi-phase environments.

Original single-path EnvB/EnvC showed gap = 0 (structural saturation / above threshold).
This delimits the conditions under which coverage effects appear.
See `docs/VALIDATION_STATUS_ADDENDUM.md` for full details.

### Auxiliary Evidence
- **Quality sweep**: 5 data quality levels × BC/CQL/IQL × 20 seeds. Random = universal
  failure floor. **Caveat**: SA coverage varies across quality bins (coverage confound).
- **Hopper D4RL benchmark**: BC/IQL/TD3+BC match published baselines; included as
  implementation validation only (**appendix**). CQL anomaly (hyperparameter issue)
  documented and excluded from claims.

---

## Quick Start / Reproduction

### Environment setup
```bash
pip install -r requirements.txt
```

### Run all tests (smoke, ~2 minutes)
```bash
python -m pytest tests/ -x -q
```

### Reproduce primary results from frozen datasets
```bash
# BC/CQL main experiment (20 seeds per condition)
python scripts/run_envA_v2_main_experiment.py

# IQL main experiment
python scripts/run_envA_v2_iql_main.py

# Regenerate all figures and analysis tables
python scripts/final_analysis_and_plots.py
python scripts/generate_final_figure_suite.py
```

### Reproduce validation environments
```bash
# EnvB_v2 formal validation (3 algorithms × 20 seeds)
python scripts/run_envB_v2_bc_formal_validation.py
python scripts/run_envB_v2_iql_formal_validation.py
python scripts/run_envB_v2_cql_formal_validation.py

# EnvC_v2 formal validation
python scripts/run_envC_v2_bc_formal_validation.py
python scripts/run_envC_v2_iql_formal_validation.py
python scripts/run_envC_v2_cql_formal_validation.py
```

### Reproduce frozen datasets from scratch (optional)
```bash
python scripts/build_envA_v2_behavior_pool.py      # ~15 min
python scripts/generate_envA_v2_final_datasets.py  # ~5 min
python scripts/build_envB_v2_datasets.py           # ~2 min
python scripts/build_envC_v2_datasets.py           # ~2 min
```

---

## Repository Layout

```
RL_Final_Project/
│
├── envs/                          # Environment implementations
│   ├── gridworld_envs.py          # All discrete gridworlds (EnvA/B/C/A_v2/B_v2/C_v2)
│   └── __init__.py
│
├── scripts/                       # All executable scripts
│   ├── run_envA_v2_main_experiment.py     # BC/CQL primary experiment
│   ├── run_envA_v2_iql_main.py            # IQL primary experiment
│   ├── run_envA_v2_quality_sweep.py       # Quality sweep BC/CQL
│   ├── run_envA_v2_iql_quality_sweep.py   # Quality sweep IQL
│   ├── run_envA_v2_mechanism_analysis.py  # SA coverage mechanism analysis
│   ├── run_envbc_validation.py            # Original EnvB/C cross-env validation
│   ├── run_envbc_iql_validation.py        # Original EnvB/C IQL validation
│   ├── run_envB_v2_*_formal_validation.py # EnvB_v2 rebuilt validation (BC/IQL/CQL)
│   ├── run_envC_v2_*_formal_validation.py # EnvC_v2 rebuilt validation (BC/IQL/CQL)
│   ├── run_hopper_benchmark.py            # Hopper D4RL benchmark
│   ├── build_envB_v2_datasets.py          # EnvB_v2 dataset generation
│   ├── build_envC_v2_datasets.py          # EnvC_v2 dataset generation
│   ├── generate_envA_v2_final_datasets.py # Primary frozen dataset generation
│   ├── build_envA_v2_behavior_pool.py     # Behavior policy pool
│   ├── generate_statistical_closure.py   # Hypothesis test table
│   ├── final_analysis_and_plots.py        # Analysis entry point
│   ├── generate_final_figure_suite.py     # All 12 canonical figures
│   ├── audit_final_datasets.py            # Dataset verification
│   ├── run_envA_v2_sanity.py              # [shared lib] BC/CQL training infrastructure
│   ├── run_envA_v2_iql_sanity.py          # [shared lib] IQL training infrastructure
│   └── verify_envA_v2_proxy_gate.py       # [shared lib] Corridor controllers
│
├── tests/                         # Test suite (15 test files)
│
├── docs/                          # Key project documents
│   ├── FINAL_PROJECT_STATUS_CHECK.md    # Authoritative completion status
│   ├── FINAL_SUBMISSION_BRIEF.md        # One-page research brief
│   ├── FINAL_CANONICAL_CLAIMS.md        # Canonical claim text blocks
│   ├── PRESENTATION_OUTLINE.md          # 14-slide presentation guide
│   ├── FIGURE_MANIFEST.md               # Figure catalogue
│   ├── VALIDATION_STATUS_ADDENDUM.md    # EnvB_v2/EnvC_v2 status
│   ├── CLAIM_HIERARCHY.md               # Claim boundaries and evidence tiers
│   ├── CLAIM_SUPPORT_MATRIX.md          # Claims mapped to statistics
│   └── SUBMISSION_PROTOCOL_V2.md        # Submission-readiness decisions
│
├── reports/
│   └── final_project_results.md         # Full research report (10 sections)
│
├── artifacts/
│   ├── final_results/             # 5 result tables + hypothesis test CSV
│   ├── final_datasets/            # 9 frozen primary NPZ datasets
│   ├── envB_v2_datasets/          # 2 EnvB_v2 NPZ datasets + manifest
│   ├── envC_v2_datasets/          # 2 EnvC_v2 NPZ datasets + manifest
│   ├── analysis/                  # Mechanism analysis CSVs
│   ├── training_main/             # BC/CQL main experiment result CSV
│   ├── training_iql/              # IQL result CSVs
│   ├── training_quality/          # Quality sweep result CSV
│   ├── training_validation_v2/    # EnvB_v2/EnvC_v2 formal result CSVs
│   ├── training_benchmark/        # Hopper benchmark result CSV
│   └── behavior_pool/             # Pre-trained behavior policies (1.9 MB)
│
└── figures/final/
    ├── mainline/                  # M1–M7: primary evidence figures
    ├── auxiliary/                 # A1–A5 + supplement: auxiliary figures
    └── fig1_*.png … fig6_*.png   # Backward-compatible aliases
```

---

## Figure Roadmap

### Mainline Figures (`figures/final/mainline/`)

| File | Contents | Report § |
|------|---------|---------|
| fig01_envA_factorial_overview.png | All 4 conditions × 3 algorithms | §2 |
| fig02_envA_primary_contrast_sw_vs_ln.png | SW vs LN primary contrast (+0.057–+0.127) | §2.1 |
| fig03_envA_size_effect_decomposition.png | SN→LN Δ=0; SW→LW algorithm-dependent | §2.2–2.3 |
| fig04_statistical_closure_matrix.png | 4×3 p-value / verdict matrix | §3 |
| fig05_mechanism_behavioral_diversity.png | SA coverage vs return; OOD=0; mechanism | §7 |
| fig06_rebuilt_validation_gap_summary.png | EnvB_v2/EnvC_v2 gap bars with status | §5 |
| fig07_rebuilt_route_family_convergence.png | Route-family stacked bars; CQL mixed shown | §5 |

### Auxiliary Figures (`figures/final/auxiliary/`)

| File | Contents | Report § |
|------|---------|---------|
| fig08_original_envbc_boundary_conditions.png | Original EnvB/C zero-gap boundary | §4 |
| fig09_quality_sweep_with_coverage_caveat.png | Quality sweep + coverage confound | §6 |
| fig10_dataset_audit_overview.png | All datasets: SA coverage vs size | Appendix |
| fig11_visitation_heatmaps.png | Transition-level visit heatmaps (3×2) | Appendix |
| fig11b_visitation_heatmaps_postdoor.png | EnvC_v2 post-door supplement | Appendix |
| fig12_hopper_benchmark_appendix.png | Hopper benchmark (5-seed pilot, appendix) | §8 |

---

## Key Documents for Reviewers

| Document | Purpose |
|----------|---------|
| `reports/final_project_results.md` | Full 10-section research report |
| `docs/FINAL_SUBMISSION_BRIEF.md` | 1-page abstract + claims + caveats + Q&A |
| `docs/FINAL_PROJECT_STATUS_CHECK.md` | Authoritative status: what is and isn't closed |
| `docs/PRESENTATION_OUTLINE.md` | 14-slide course presentation outline |
| `docs/CLAIM_HIERARCHY.md` | Exact claim boundaries (what can/cannot be stated) |

---

## Caveats and Honest Limitations

1. **Small discrete gridworlds only**: All primary experiments use deterministic ≤30-cell
   environments. OOD rate = 0 in all conditions — scalability to larger settings is untested.
2. **EnvC_v2 CQL shows mixed evidence**: BC+IQL confirm coverage effect (+0.020/+0.024,
   non-overlapping CIs); CQL gap = +0.002 (negligible, CI overlapping). CQL's conservative
   Q-penalty suppresses path-diversity in staged multi-phase environments.
3. **Quality sweep has coverage confound**: SA coverage varies across quality bins; quality
   and coverage effects are not cleanly separated.
4. **Hopper benchmark CQL anomaly**: CQL normalized scores are inconsistent
   (3.17 / 26.65 / 1.32 vs. published ~58 / 46 / 91). Likely D4RL hyperparameter
   misconfiguration. CQL benchmark results excluded from all claims.
5. **BC/IQL Claim 1.3 (size effect) is directional only**: p = 0.093 / 0.119, not significant
   at α = 0.05 with n = 20.
