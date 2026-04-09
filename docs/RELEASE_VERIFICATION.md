# RELEASE_VERIFICATION.md
# Release Package Health Check

> Date: 2026-04-08
> Branch: submission-release
> Remote: https://github.com/XueyiZhang75/RL_Offline_Project.git

---

## 1. What Is Included in This Release

### Source code
- `envs/gridworld_envs.py` — all environments (EnvA/B/C, EnvA_v2/B_v2/C_v2)
- 24 scripts for experiments, data generation, validation, and figure production
- 15 test files covering all environments and datasets

### Frozen result artifacts
- `artifacts/final_results/` — 5 result tables + hypothesis test summary (6 CSVs)
- `artifacts/final_datasets/` — 9 frozen NPZ datasets (~33 MB, primary experiment)
- `artifacts/envB_v2_datasets/` — 2 NPZ + manifest (EnvB_v2 validation, ~217 KB)
- `artifacts/envC_v2_datasets/` — 2 NPZ + manifest (EnvC_v2 validation, ~169 KB)
- `artifacts/analysis/` — mechanism analysis CSVs
- `artifacts/training_main/`, `training_iql/`, `training_quality/` — result CSVs
- `artifacts/training_validation_v2/` — 11 formal validation CSVs (EnvB_v2 + EnvC_v2)
- `artifacts/training_benchmark/hopper_benchmark_summary.csv`
- `artifacts/behavior_pool/` — behavior policy checkpoints (~1.9 MB)

### Figures
- `figures/final/mainline/` — 7 mainline figures (M1–M7)
- `figures/final/auxiliary/` — 6 auxiliary figures (A1–A5 + fig11b supplement)
- `figures/final/fig1_*.png`–`fig6_*.png` — 6 backward-compatible aliases

### Documentation
- `docs/FINAL_PROJECT_STATUS_CHECK.md` — authoritative completion status
- `docs/FINAL_SUBMISSION_BRIEF.md` — one-page brief with claims, caveats, Q&A
- `docs/FINAL_CANONICAL_CLAIMS.md` — canonical claim text blocks (3 formats)
- `docs/PRESENTATION_OUTLINE.md` — 14-slide presentation outline
- `docs/FIGURE_MANIFEST.md` — full figure catalogue
- `docs/VALIDATION_STATUS_ADDENDUM.md` — EnvB_v2/EnvC_v2 rebuilt validation status
- `docs/CLAIM_HIERARCHY.md` — claim boundaries (Tier 1 / Tier 2 / Cannot claim)
- `docs/CLAIM_SUPPORT_MATRIX.md` — claims mapped to statistical evidence
- `docs/SUBMISSION_PROTOCOL_V2.md` — submission-readiness decisions
- `docs/RELEASE_VERIFICATION.md` — this file
- `reports/final_project_results.md` — full 10-section research report

---

## 2. What Is Intentionally Not Included

### Checkpoint binaries (not needed; regenerable)
- `artifacts/training_validation_v2/envB_v2_*_checkpoints/` — 27 MB each, 6 dirs (~162 MB)
- `artifacts/training_main/*.pt`, `training_iql/*.pt`, `training_quality/*.pt` — already gitignored

### Internal pilot and audit artifacts
- `artifacts/pilot_envbc_v2/` — pilot mock-up outputs, intermediate gate decisions
- `artifacts/training_validation/` — original single-path EnvB/C results (superseded by `training_validation_v2/`)

### Internal process documents (35 files)
All `docs/ENVB_V2_*.md`, `docs/ENVC_V2_*.md`, `docs/ENVBC_*.md`,
`docs/EXP_PROTOCOL.md`, `docs/PROJECT_SCOPE.md`, `docs/RESEARCH_EVOLUTION_NOTE.md`,
`docs/STATISTICAL_CLOSURE_*.md`, `docs/LOCAL_DOC_ALIGNMENT_CHECK.md`,
`docs/DOC_ALIGNMENT_CHANGELOG.md`, `docs/FINAL_SYNC_CHANGELOG.md`,
`docs/FIGURE_POLISH_CHANGELOG.md`, `docs/FIGURE_FINAL_MICROFIX_NOTE.md`,
`docs/FINAL_DOC_CONSISTENCY_CHECK.md`, `docs/RELEASE_PACKAGE_MANIFEST.md`

These document the iterative development and audit process; not needed for result
verification. They are preserved locally.

### Internal iterative scripts (10 files)
Pilot mockups, pre-formal audits, v4 fairness audits, smoke tests, and figure polish
scripts. The final formal validation scripts are included; the development trail is not.

---

## 3. Reproduction Entry Points

| Goal | Command |
|------|---------|
| Run all structural tests | `python -m pytest tests/ -q` |
| Primary BC/CQL experiment | `python scripts/run_envA_v2_main_experiment.py` |
| Primary IQL experiment | `python scripts/run_envA_v2_iql_main.py` |
| Quality sweep | `python scripts/run_envA_v2_quality_sweep.py` |
| Mechanism analysis | `python scripts/run_envA_v2_mechanism_analysis.py` |
| EnvB_v2 BC validation (20 seeds) | `python scripts/run_envB_v2_bc_formal_validation.py` |
| EnvC_v2 CQL validation (20 seeds) | `python scripts/run_envC_v2_cql_formal_validation.py` |
| Regenerate all figures | `python scripts/generate_final_figure_suite.py` |
| Regenerate statistical tests | `python scripts/generate_statistical_closure.py` |
| Verify datasets | `python scripts/audit_final_datasets.py` |
| Run final analysis | `python scripts/final_analysis_and_plots.py` |

---

## 4. Health Check Results

### Test suite (184 tests)
```
tests/test_phase1_envs.py       ✓
tests/test_envA_v2_structure.py ✓
tests/test_envB_v2_structure.py ✓
tests/test_envC_v2_structure.py ✓
... (184/184 passed, 0 failures)
```

### Script imports
```
envs.gridworld_envs (EnvA_v2, EnvB_v2, EnvC_v2)  ✓
scripts.build_envB_v2_datasets                     ✓
scripts.build_envC_v2_datasets                     ✓
```

### Key figures
```
figures/final/mainline/fig01_envA_factorial_overview.png    ✓ (71 KB)
figures/final/mainline/fig06_rebuilt_validation_gap_summary ✓ (85 KB)
figures/final/auxiliary/fig11_visitation_heatmaps.png       ✓ (256 KB)
figures/final/fig1_main_coverage_vs_size.png                ✓ (71 KB, compat alias)
```

### Key artifacts
```
artifacts/final_results/final_discrete_results_master_table.csv  ✓ (1 KB)
artifacts/final_results/final_hypothesis_test_summary.csv         ✓ (4 KB)
artifacts/final_datasets/envA_v2_small_wide_medium.npz            ✓ (2151 KB)
artifacts/envB_v2_datasets/envB_v2_small_wide.npz                 ✓ (63 KB)
artifacts/envC_v2_datasets/envC_v2_small_wide.npz                 ✓ (41 KB)
artifacts/training_validation_v2/envB_v2_bc_formal_summary.csv    ✓ (7 KB)
artifacts/training_validation_v2/envC_v2_cql_formal_summary.csv   ✓ (8 KB)
```

### Key documentation
```
docs/FINAL_PROJECT_STATUS_CHECK.md   ✓
docs/FINAL_SUBMISSION_BRIEF.md       ✓
docs/VALIDATION_STATUS_ADDENDUM.md   ✓
docs/FIGURE_MANIFEST.md              ✓
docs/CLAIM_HIERARCHY.md              ✓
reports/final_project_results.md     ✓
```

---

## 5. Consistency Check

**EnvC_v2 CQL status**: All documents, figure labels, and the README correctly state
"BC + IQL confirmed; CQL mixed evidence (gap=+0.002, CI overlapping)." No document
claims EnvC_v2 as "3/3 confirmed."

**Figure ↔ manuscript alignment**: All figure paths in `docs/FIGURE_MANIFEST.md` and
`docs/PRESENTATION_OUTLINE.md` reference files that exist in the release package.

**Numerical consistency**: All delta values in figures (+0.057, +0.127, +0.127)
match the authoritative values in `reports/final_project_results.md`.

---

## 6. Delivery Standard Assessment

**YES — this package is ready to be given to an instructor for review.**

The instructor can:
1. Read the research brief (`docs/FINAL_SUBMISSION_BRIEF.md`) for a 1-page overview
2. Read the full report (`reports/final_project_results.md`) for complete findings
3. Browse the figures in `figures/final/` for visual evidence
4. Run `python -m pytest tests/ -q` to verify environment implementations
5. Re-run any formal validation script to verify reported results
6. Re-generate figures from frozen datasets without re-running training

The package does not expose internal audit trail, pilot iterations, checkpoint binaries,
or process documents. All claim boundaries are accurately stated, including the EnvC_v2
CQL mixed evidence finding.
