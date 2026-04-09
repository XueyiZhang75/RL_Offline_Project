# SUBMISSION_PROTOCOL_V2.md
# Project Submission Protocol — Version 2 (Post-Execution Refreeze)

> This document supersedes the implicit submission assumptions in EXP_PROTOCOL.md.
> It reflects the actual executed state of the project and defines what is complete,
> what is pending, and what the submission-ready scope is.
> Date: 2026-04-06 (last major sync: 2026-04-08 — EnvB_v2/EnvC_v2 rebuild results incorporated)
>
> Note: Some alignment/closure check documents (STATISTICAL_CLOSURE_CONSISTENCY_CHECK.md,
> LOCAL_DOC_ALIGNMENT_CHECK.md, DOC_ALIGNMENT_CHANGELOG.md) are historical snapshots from
> pre-rebuild stages and should not be interpreted as the latest cross-environment status record.
> For current status, see FINAL_PROJECT_STATUS_CHECK.md and VALIDATION_STATUS_ADDENDUM.md.

---

## 1. Submission-Ready Components (Frozen)

These components are complete, verified, and must not be modified before submission.

### 1.1 Primary Research Track — EnvA_v2 Discrete Main Line

| Component | Status | Files |
|-----------|--------|-------|
| EnvA_v2 environment | Complete, frozen | `envs/gridworld_envs.py` |
| Frozen datasets (4 main + 5 quality + 4 envBC) | Complete, frozen | `artifacts/final_datasets/*.npz` |
| BC/CQL main experiment (4 conditions × 20 seeds) | Complete | `artifacts/training_main/envA_v2_main_summary.csv` |
| IQL main experiment (4 conditions × 20 seeds) | Complete | `artifacts/training_iql/envA_v2_iql_main_summary.csv` |
| Quality sweep BC/CQL (5 levels × 20 seeds) | Complete | `artifacts/training_quality/envA_v2_quality_summary.csv` |
| Quality sweep IQL (5 levels × 20 seeds) | Complete | `artifacts/training_iql/envA_v2_iql_quality_sweep_summary.csv` |
| Mechanism analysis | Complete | `artifacts/analysis/*.csv` |
| Final result tables (4 CSVs) | Complete | `artifacts/final_results/*.csv` |
| Final figures (6 PNGs) | Complete | `figures/final/*.png` |
| Final analysis report | Complete | `reports/final_project_results.md` |

**Algorithm scope (frozen)**: BC / CQL / IQL.
All three algorithms are treated as primary evidence for the main conclusion.
No further algorithm additions are in scope for submission.

**Core conclusion (frozen)**:
> State-action coverage is the primary determinant of Offline RL performance ceiling
> in multi-path discrete environments. Small-wide datasets outperform large-narrow
> datasets across BC, CQL, and IQL. Dataset size shows weak independent effect:
> fixed narrow coverage + increased size yields Δ = 0; BC shows a size effect
> only under wide coverage (Δ = +0.074), while CQL/IQL show Δ < 0.01.

### 1.2 Auxiliary Track — Quality Sweep

Status: **Complete**. Results are included in submission as **auxiliary evidence with caveat**.

Role: Demonstrates that random data produces a hard failure floor, and that BC/IQL are
relatively insensitive to quality above that floor. CQL shows higher sensitivity, especially
at the suboptimal level.

**Caveat — quality datasets are not quality-only isolations**: The quality sweep datasets
have meaningfully different SA coverage values across quality levels (from manifest):

| Quality bin | norm_SA_coverage |
|-------------|-----------------|
| random | ~0.382 |
| suboptimal | ~0.208 |
| medium | ~0.208 |
| expert | ~0.152 |
| mixed | ~0.425 |

Coverage is not held constant across quality bins. The observed performance differences
therefore reflect a combination of data quality and dataset coverage, not quality alone.
This does not invalidate the threshold finding (random = failure regardless of coverage),
but it means the quality sweep cannot be presented as clean evidence of quality-only effects.

Permitted claims from this track:
- Random data is a universal failure floor for all three algorithms.
- Above the random floor, BC and IQL show small performance variation (Δ < 0.010 from suboptimal to expert), though this is partly explained by near-identical coverage for those bins.
- CQL is more sensitive to low-quality data (suboptimal: mean 0.051, std 0.622).

Not permitted: "This sweep cleanly isolates the effect of data quality on Offline RL performance."

### 1.3 Auxiliary Track — Hopper D4RL Benchmark

Status: **Pilot only (5 seeds). CQL severely underperforming and inconsistent. Included as appendix.**

- BC (20–47), IQL (29–31), TD3+BC (22–63) results are broadly consistent with published baselines.
- CQL is severely underperforming and **inconsistently** so across splits: hopper-medium (3.17), hopper-medium-replay (26.65), hopper-medium-expert (1.32). The inconsistency pattern suggests hyperparameter sensitivity rather than a uniform implementation failure.
- Included in submission as `appendix / auxiliary pilot`. Not cited as evidence for main conclusion.
- CQL results must be **excluded from any comparative claim** until the hyperparameter issue is debugged.
- See §2.2 for required action.

---

## 2. Components with Known Issues Requiring Action Before Strong Claims

### 2.1 EnvB/C Cross-Environment Validation

> **UPDATED 2026-04-08** — EnvB_v2 and EnvC_v2 rebuilt as multi-path environments and
> formally validated. See `docs/VALIDATION_STATUS_ADDENDUM.md` for full details.

**Original single-path EnvB/C**: gap = 0.000 for all conditions (structural saturation in
EnvB; above support threshold in EnvC). These results remain valid as boundary conditions
showing WHY multi-path structure is necessary.

**Rebuilt multi-path environments** (see `VALIDATION_STATUS_ADDENDUM.md`):

- **EnvB_v2** (three-corridor, 270 states): small-wide > large-narrow confirmed across all
  three algorithms (BC +0.020, IQL +0.026, CQL +0.025). CIs non-overlapping for IQL and CQL.
  Narrow policies lock to single corridor; wide policies discover shorter paths in 9–11/20 seeds.
  **Status: CONFIRMED (3/3 algorithms).**

- **EnvC_v2** (key-door staged, 269 extended states): BC (+0.020) and IQL (+0.024) both
  confirmed, CIs non-overlapping. CQL formal 20-seed run (2026-04-08): gap=+0.002 (negligible,
  CI overlapping, 1/20 seeds) — mixed evidence. CQL's conservative Q-penalty suppresses
  path-diversity in the multi-phase key-door structure.
  **Status: DIRECTIONAL CONFIRMED (2/3 strong; CQL mixed evidence — NOT 3/3 confirmed).**

**For submission**: Present original EnvB/C results as the boundary condition motivating
the rebuild, then present rebuilt results as directional cross-environment validation.

**Permitted claim**: "The coverage effect replicates in two structurally distinct rebuilt
multi-path discrete environments (EnvB_v2, EnvC_v2), at smaller effect sizes consistent
with simpler path structure."

**Caveats required**:
- Effect sizes smaller than EnvA_v2 (0.020–0.026 vs 0.057–0.127)
- EnvC_v2 CQL formal 20-seed completed (2026-04-08); evidence is mixed (gap=+0.002, CI overlapping, 1/20 seeds); report as "BC + IQL strong confirmed; CQL mixed" — NOT "3/3 confirmed"
- Effect requires multi-path structure; does not apply to single-path environments

### 2.2 Hopper Benchmark CQL Anomaly

**Current status**: CQL normalized scores are severely underperforming and inconsistent — hopper-medium (3.17), hopper-medium-replay (26.65), hopper-medium-expert (1.32).
**What this means**: The pattern is not uniform failure. Medium-replay is closer to plausible; medium and medium-expert are near-zero. This suggests D4RL hyperparameter sensitivity (primarily `cql_alpha`, batch size, and training length) rather than a fundamental implementation error.

**For submission**: Explicitly acknowledge the anomaly in the benchmark section. Note that it does not affect discrete main-line conclusions.

**Required action before upgrading to 20 seeds**:
1. Debug step: identify D4RL-appropriate CQL hyperparameters (start with `cql_alpha = 5.0`, larger batch, longer training per published CQL D4RL configs).
2. Validation: run 3-seed diagnostic on Hopper-medium. Target: normalized score ≥ 40.
3. Decision: if CQL recovers, upgrade all benchmark runs to 20 seeds. If CQL cannot be fixed within scope, maintain 5-seed pilot and explicitly note the limitation.

---

## 2.3 Statistical Closure — Completed

**Current status**: Core pairwise statistical closure is now completed.

The standalone hypothesis test table has been generated:
`artifacts/final_results/final_hypothesis_test_summary.csv` — 12 rows (4 comparisons × 3 algorithms),
containing Cohen's d, Welch t-test, Mann-Whitney U, and primary test designations.

**Claim closure summary**:

| Claim | Status |
|-------|--------|
| 1.1 Coverage dominance (sw > ln, all algos) | **CLOSED** — BC MWU p=1.13e-07; CQL/IQL t p=7.47e-23 |
| 1.2 Size = 0 under narrow coverage | **CLOSED** — exact equality, all seeds identical |
| 1.3 CQL wide-size null | **CLOSED (null)** — t p=0.531 |
| 1.3 BC wide-size effect | **DIRECTIONAL** — MWU p=0.093 (trend present, not α=0.05 significant) |
| 1.3 IQL wide-size effect | **DIRECTIONAL** — t p=0.119, d=−0.50 (not α=0.05 significant) |

**What remains open**: BC and IQL Claim 1.3 are directional only. Closing them to "confirmed" would require additional seeds (out of current scope). This limitation is accurately documented and does not affect the primary submission claim (Claim 1.1 and 1.2 are formally closed).

**What this means for submission phrasing**:
- "Coverage dominance is statistically supported (p < 1e-06 for all algorithms)" — permitted
- "Size effect is zero under narrow coverage (exact equality, all seeds)" — permitted
- "CQL shows no meaningful size effect under wide coverage (p = 0.531)" — permitted
- "BC and IQL show a directional size trend under wide coverage (p ≈ 0.09–0.12, not significant)" — permitted

---

## 3. Component Status Matrix

| Component | Submission-ready? | Action required |
|-----------|-------------------|-----------------|
| EnvA_v2 main experiment (BC/CQL/IQL × 4 datasets × 20 seeds) | **YES** | None |
| Quality sweep (all algorithms × 5 levels × 20 seeds) | **YES** | None |
| Mechanism analysis | **YES** | None |
| Final tables + figures + report | **YES** | None |
| EnvB/C validation (original single-path results) | **Partial** — include as boundary condition | Required caveat on single-path structure |
| Hopper benchmark (5 seeds, CQL anomalous) | **Partial** — include as appendix pilot | Debug CQL before upgrading seeds |
| EnvB_v2 rebuilt (3-corridor, 270 states) | **CONFIRMED (3/3 algos)** | Include as cross-env validation (with effect-size caveat) |
| EnvC_v2 rebuilt (key-door, 269 ext. states) | **BC+IQL confirmed; CQL mixed** | All 3 algos run; CQL gap=+0.002 (negligible, CI overlap) — report as "2/3 strong + CQL mixed" |
| Hopper benchmark at 20 seeds with working CQL | **NOT DONE** | Conditional on CQL debug |

---

## 4. Frozen Protocol Rules (Carry-Forward from EXP_PROTOCOL.md)

These rules from the original protocol remain in force:

1. **EnvB/C must not be expanded into full 4-condition factorial experiments.** Even after rebuild, their scope is: small-wide vs large-narrow, 20 seeds, 3 algorithms.

2. **Benchmark results must not be used to support the main conclusion.** They are external trend validation only.

3. **The main research question is unchanged**: Does SA coverage determine Offline RL performance ceiling more than dataset size?

4. **The primary environment is EnvA_v2.** All main-line claims are specific to this environment's properties (30×30 four-corridor, ~670 states, multi-path structure).

5. **No new algorithms, environments, or dataset conditions may be added without explicit protocol revision.**

---

## 5. Submission Package Definition

The submission package consists of:

**Required (complete)**:
- `reports/final_project_results.md` — primary report
- `artifacts/final_results/*.csv` — 4 result tables
- `figures/final/*.png` — 6 figures
- All source code in `envs/`, `scripts/`, `tests/`
- `artifacts/final_datasets/*.npz` — frozen datasets
- `README.md`

**Included with caveats**:
- `artifacts/training_benchmark/hopper_benchmark_summary.csv` — labeled as 5-seed pilot; CQL results excluded from any comparative claim
- `artifacts/training_validation/envbc_validation_summary.csv` — presented as boundary condition evidence only

**Not included in submission claims**:
- Any cross-environment generalization statement beyond "coverage effects require multi-path environments"
- Any CQL D4RL benchmark comparison claim
