# CLAIM_SUPPORT_MATRIX.md
# Claim-to-Statistics Mapping

> Maps each Tier 1 and Tier 2 claim from CLAIM_HIERARCHY.md to its specific row(s)
> in `artifacts/final_results/final_hypothesis_test_summary.csv`.
> All statistics computed from existing experiment CSVs. No new training.
> Date: 2026-04-06

---

## Legend

| Column | Meaning |
|--------|---------|
| Comparison | Row ID in final_hypothesis_test_summary.csv |
| Primary test | The designated primary statistical test for this cell |
| p-value | Primary test p-value |
| d | Cohen's d (pooled); see caveats below |
| Effect label | negligible / small / medium / large per |d| thresholds 0.2/0.5/0.8 |
| Verdict | CLOSED = formally statistically supported; DIRECTIONAL = supported by CI but formal test has caveat |

**Cohen's d caveats**:
- BC C1: d = 0.26 is computed with inflated pooled SD (bimodal distribution). Treat as lower-bound estimate.
- CQL/IQL C1: d ≈ 18 reflects near-zero variance, not absolute practical effect scale. Treat as "extremely large."
- C3: d = 0 because groups are identical (degenerate, not an effect size estimate).

---

## Tier 1 Claims

### Claim 1.1 — Coverage Dominance (small-wide > large-narrow)

*"In the EnvA_v2 four-corridor gridworld, SA coverage is the primary determinant of Offline RL performance ceiling. Small-wide outperforms large-narrow across all three algorithms."*

| Algorithm | Comparison | Primary test | p-value | d | Effect | Verdict |
|-----------|-----------|-------------|---------|---|--------|---------|
| BC | sw_vs_ln | **Mann-Whitney U** | 1.13e-07 | 0.26 (small; bimodal caveat) | small | **CLOSED** |
| CQL | sw_vs_ln | Welch t-test | 7.47e-23 | 18.35 (large; near-zero SD caveat) | large | **CLOSED** |
| IQL | sw_vs_ln | Welch t-test | 7.47e-23 | 18.35 (large; near-zero SD caveat) | large | **CLOSED** |

**Overall verdict**: CLOSED. All three algorithms show p << 0.001. BC uses MWU as primary test due to bimodal distribution (1 failure seed); MWU detects correct stochastic ordering (p = 1.13e-07). CQL and IQL show extremely tight distributions with large effect sizes.

---

### Claim 1.2 — Size Ineffectiveness Under Narrow Coverage

*"When SA coverage is fixed at the narrow level, increasing dataset size from 50k to 200k yields no improvement for any algorithm."*

| Algorithm | Comparison | Primary test | p-value | d | Effect | Verdict |
|-----------|-----------|-------------|---------|---|--------|---------|
| BC | sn_vs_ln | Exact equality | N/A | 0.000 | negligible | **CLOSED** |
| CQL | sn_vs_ln | Exact equality | N/A | 0.000 | negligible | **CLOSED** |
| IQL | sn_vs_ln | Exact equality | N/A | 0.000 | negligible | **CLOSED** |

**Overall verdict**: CLOSED. All 20 seeds in both small-narrow and large-narrow are identical (0.2700) for all three algorithms. This is not a statistical inference — it is exact equality. No test is applicable or needed. The size effect is precisely zero.

---

### Claim 1.3 — Size Effect Is Algorithm-Dependent Under Wide Coverage

*"Under wide coverage, CQL and IQL show negligible size effect (Δ < 0.01); BC shows a modest positive effect (Δ = +0.074)."*

| Algorithm | Comparison | Primary test | p-value | d | Effect | Verdict |
|-----------|-----------|-------------|---------|---|--------|---------|
| BC | sw_vs_lw | Mann-Whitney U | 9.32e-02 | −0.34 (Δ = +0.074 abs) | small | **DIRECTIONAL** |
| CQL | sw_vs_lw | Welch t-test | 5.31e-01 | −0.20 | negligible | **CLOSED (null)** |
| IQL | sw_vs_lw | Welch t-test | 1.19e-01 | −0.50 | medium | **DIRECTIONAL** |

**Notes**:
- Sign convention: A = small-wide, B = large-wide, so d < 0 means large-wide > small-wide.
- BC: p = 0.093 (not significant at α=0.05). The Δ = +0.074 trend is present but not formally confirmed. DIRECTIONAL.
- CQL: p = 0.531, d = −0.20. Null result. CLOSED (the null — no meaningful size effect).
- IQL: p = 0.119, d = −0.50. Not significant but moderate effect size. DIRECTIONAL. Requires more seeds to confirm.

**Overall verdict**: PARTIALLY CLOSED. Per-algorithm breakdown:
- **BC**: DIRECTIONAL — p = 0.093, trend present but not significant at α = 0.05.
- **CQL**: CLOSED (null) — p = 0.531, d = −0.20; no meaningful size effect confirmed.
- **IQL**: DIRECTIONAL — p = 0.119, d = −0.50 (medium effect); not significant at α = 0.05; more seeds would be needed to confirm.

The claim that "CQL shows negligible size effect" is formally closed. The claim that "IQL shows negligible size effect" is NOT closed — a medium effect size (d = −0.50) with a borderline p-value warrants treating IQL as directional, not null-confirmed.

---

### Claim 1.4 — Results Are Directionally Robust

*"Main results are based on 20 seeds per condition; 95% CIs are narrow for CQL/IQL; BC has higher variance due to bimodal convergence."*

This claim is a description of the data distribution, not a hypothesis test. It is supported by the following from the statistics table:

| Condition | Algorithm | std | 95% CI width |
|-----------|-----------|-----|-------------|
| small-wide | BC | 0.3124 | wide (bimodal) |
| small-wide | CQL | 0.0098 | narrow |
| small-wide | IQL | 0.0098 | narrow |
| large-narrow | all | 0.0000 | zero (constant) |

**Verdict**: DESCRIPTIVE. No formal test required. The distributional characteristics are factually accurate.

**Statistical closure note**: The formal hypothesis test summary table is now generated. The remaining gap for Claim 1.4 is effect size and p-value summary (now available in `final_hypothesis_test_summary.csv`). The claim can now be stated with specific statistics.

---

## Tier 2 Claims

### Claim 2.1 — Random Data Is a Hard Failure Floor

*"Random data causes complete failure for all algorithms regardless of coverage."*

This claim is from the quality sweep, not the main experiment. No row in `final_hypothesis_test_summary.csv` directly addresses it (the stats table covers main experiment comparisons only). The evidence is:
- All three algorithms: random bin mean = −1.000, std = 0.000, success rate = 0.000

**Verdict**: CLOSED from raw data inspection. No formal test needed — the result is exact and unambiguous. The quality sweep CSV (`final_quality_results_table.csv`) is the supporting artifact.

---

### Claim 2.2 — SA Coverage Metric Directly Tracks Performance

*"Normalized SA coverage predicts policy performance; wide datasets (~21%) yield higher returns than narrow (~6%)."*

This claim is supported by the C4 (wide-pooled vs narrow-pooled) rows, which collapse both size conditions per coverage type:

| Algorithm | Comparison | Primary test | p-value | d | Effect | Verdict |
|-----------|-----------|-------------|---------|---|--------|---------|
| BC | wide_vs_narrow | Mann-Whitney U | 1.84e-15 | 0.60 | medium | **CLOSED** |
| CQL | wide_vs_narrow | Welch t-test | 3.38e-45 | 18.24 | large | **CLOSED** |
| IQL | wide_vs_narrow | Welch t-test | 4.52e-45 | 18.11 | large | **CLOSED** |

Additionally supported by mechanism analysis (`artifacts/analysis/envA_v2_mechanism_summary.csv`).

**Overall verdict**: CLOSED. The pooled wide vs narrow contrast is extremely significant for all algorithms.

---

### Claim 2.3 — Coverage Effect Generalizes Across Multi-Path Environments

> **UPDATED 2026-04-08.** See VALIDATION_STATUS_ADDENDUM.md for full details.

**Boundary condition evidence (original single-path EnvB/C)**:
- gap = 0.000 for all algorithms and environments — exact zero by structural necessity

**Directional replication evidence (rebuilt multi-path environments)**:
- EnvB_v2 (3-corridor, 270 states): BC=+0.020, IQL=+0.026, CQL=+0.025; IQL/CQL CIs non-overlapping. 3/3 algorithms confirmed.
- EnvC_v2 (key-door, 269 ext. states): BC=+0.020 (CI non-overlap), IQL=+0.024 (CI non-overlap), CQL=+0.002 (negligible, CI overlapping, 1/20 seeds — mixed evidence). CQL formal 20-seed completed 2026-04-08.

**Updated verdict**: CONFIRMED for EnvB_v2 (3/3 algorithms). EnvC_v2: BC + IQL strongly confirmed (both CI non-overlapping); CQL mixed evidence (gap negligible, conservative Q-penalty suppresses staged-route discovery). EnvC_v2 = **2/3 strong + CQL mixed** — NOT 3/3 confirmed.

---

## Summary: Statistical Closure Status

| Claim | Status | Strongest evidence |
|-------|--------|-------------------|
| 1.1 Coverage dominance (all algos) | **CLOSED** | C1: BC MWU p=1.13e-07; CQL/IQL t p<1e-22 |
| 1.2 Size=0 under narrow coverage | **CLOSED** | C3: exact equality, all seeds identical |
| 1.3 BC size effect under wide coverage | **DIRECTIONAL** | C2 BC: MWU p=0.093 (trend, not confirmed) |
| 1.3 CQL size null under wide coverage | **CLOSED (null)** | C2 CQL: t p=0.531 |
| 1.3 IQL size effect (trend) | **DIRECTIONAL** | C2 IQL: t p=0.119 |
| 1.4 Distributional description | **DESCRIPTIVE** | Statistics table provides exact values |
| 2.1 Random = failure floor | **CLOSED** | Raw quality results: exact zero success rate |
| 2.2 SA coverage tracks performance | **CLOSED** | C4: BC MWU p=1.84e-15; CQL/IQL p<1e-44 |
| 2.3 Coverage generalizes across multi-path envs | **CONFIRMED (EnvB_v2, 3/3); BC+IQL confirmed, CQL mixed (EnvC_v2)** | EnvB_v2 3/3 confirmed; EnvC_v2 BC+IQL CI non-overlap; CQL gap=+0.002 mixed (2026-04-08) |
