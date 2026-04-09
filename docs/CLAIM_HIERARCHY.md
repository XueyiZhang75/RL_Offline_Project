# CLAIM_HIERARCHY.md
# Research Claim Hierarchy — What Can Be Said, At What Strength, From What Evidence

> This document defines the exact claims this project can support, the evidence
> tier each claim belongs to, and the conditions under which each claim's strength
> can be upgraded. Claims outside this hierarchy must not appear in the submission.
> Date: 2026-04-06

---

## 1. Primary Claims (Tier 1 — Directly Supported by Main Evidence)

These claims are supported by EnvA_v2 main experiment data: 3 algorithms × 4 dataset
conditions × 20 seeds each. They can be stated without qualification in the main body.

---

**Claim 1.1 — Coverage Dominance**

> In the EnvA_v2 four-corridor gridworld, state-action coverage is the primary determinant
> of Offline RL policy performance ceiling. A smaller dataset with broader SA coverage
> (small-wide, 50k transitions, ~21% SA coverage) consistently outperforms a larger dataset
> with narrower SA coverage (large-narrow, 200k transitions, ~6% SA coverage) across all
> three evaluated algorithms.

Evidence:
- BC: small-wide (0.3265) > large-narrow (0.2700), Δ = +0.057
- CQL: small-wide (0.3970) > large-narrow (0.2700), Δ = +0.127
- IQL: small-wide (0.3970) > large-narrow (0.2700), Δ = +0.127
- 20 seeds per condition; all gaps positive; direction consistent across algorithms

Strength: **Strong.** Three independent algorithms. 20 seeds each. Zero exceptions.

---

**Claim 1.2 — Size Ineffectiveness Under Narrow Coverage**

> When SA coverage is held fixed at the narrow level (~6%), increasing dataset size from 50k
> to 200k yields no performance improvement for any of the three algorithms.

Evidence:
- BC: small-narrow (0.2700) → large-narrow (0.2700), Δ = 0.000
- CQL: 0.2700 → 0.2700, Δ = 0.000
- IQL: 0.2700 → 0.2700, Δ = 0.000

Strength: **Strong.** Exact zero for all algorithms. Mechanism: narrow dataset already at the
coverage ceiling for this environment — more transitions of the same narrow distribution
add no new SA information.

---

**Claim 1.3 — Size Effect Is Algorithm-Dependent Under Wide Coverage**

> When SA coverage is held at the wide level (~21%), increasing dataset size from 50k to 200k
> has negligible effect for CQL and IQL, but a modest positive effect for BC.

Evidence:
- BC: small-wide (0.3265) → large-wide (0.4010), Δ = +0.074
- CQL: small-wide (0.3970) → large-wide (0.3990), Δ = +0.002
- IQL: small-wide (0.3970) → large-wide (0.4020), Δ = +0.005

Strength: **Moderate for BC, strong for CQL/IQL.** The BC size effect is real but small;
the CQL/IQL insensitivity is effectively zero within noise bounds.
Qualified phrasing required: "size effect is weak on average, but BC shows a non-negligible
size effect specifically under wide coverage."

---

**Claim 1.4 — Results Are Statistically Documented**

> Main experiment results are based on 20 training seeds per condition. 95% confidence
> intervals are narrow for CQL and IQL (std ≈ 0.010); BC shows higher variance
> (std ≈ 0.31 for small-wide) due to multi-modal convergence behavior.

Note: BC's high variance in the small-wide condition is genuine — some seeds converge to
better solutions, others do not. This does not invalidate the mean comparison but should
be noted when reporting CI bounds for BC.

**Statistical closure status**: Core pairwise statistical closure is now completed.
The standalone hypothesis test table (`artifacts/final_results/final_hypothesis_test_summary.csv`)
has been generated and contains Cohen's d, p-values, and primary test designations for all
four comparisons × three algorithms (12 rows). Summary:

- **Claim 1.1** (coverage dominance): formally CLOSED — BC MWU p=1.13e-07; CQL/IQL t p=7.47e-23
- **Claim 1.2** (narrow size = 0): formally CLOSED — exact equality confirmed for all algorithms
- **Claim 1.3** (size effect under wide coverage): PARTIALLY CLOSED
  - CQL: CLOSED (null) — t p=0.531; no size effect
  - BC: DIRECTIONAL — MWU p=0.093; trend present, not significant at α=0.05
  - IQL: DIRECTIONAL — t p=0.119, d=−0.50; borderline, not significant at α=0.05

For the specific p-values and effect sizes, see `artifacts/final_results/final_hypothesis_test_summary.csv`
and `docs/CLAIM_SUPPORT_MATRIX.md`.

---

## 2. Secondary Claims (Tier 2 — Supported by Auxiliary Evidence)

These claims are supported by quality sweep and mechanism analysis data. They are valid
but should be presented as supporting/contextual evidence, not primary conclusions.

---

**Claim 2.1 — Random Data Is a Hard Failure Floor; Above It, BC/IQL Show Low Quality Sensitivity**

> Random data causes complete failure for all three algorithms. Above the random floor,
> BC and IQL show small performance variation across quality levels. CQL is more sensitive
> to low-quality data.

**Important caveat**: This is not a clean quality-only isolation. Dataset SA coverage
varies substantially across quality bins:

| Quality | norm_SA_coverage | BC | CQL | IQL |
|---------|-----------------|-----|-----|-----|
| random | ~0.382 | -1.000 | -1.000 | -1.000 |
| suboptimal | ~0.208 | 0.393 | 0.051 | 0.402 |
| medium | ~0.208 | 0.400 | 0.332 | 0.402 |
| expert | ~0.152 | 0.396 | 0.389 | 0.400 |
| mixed | ~0.425 | 0.393 | 0.393 | 0.393 |

Coverage is not controlled across quality bins. The near-identical performance of BC/IQL
for suboptimal through expert is partly explained by near-identical coverage for those bins
(~0.208 vs ~0.152 — a smaller gap than wide vs narrow in the main experiment). The random
bin's failure is unambiguous regardless of coverage level.

Strength:
- **Strong**: Random = failure floor, robust across all algorithms and independent of coverage.
- **Moderate with caveat**: BC/IQL quality insensitivity above the floor — coverage is not fully controlled, so quality and coverage effects are partially confounded.
- **Moderate**: CQL sensitivity to low-quality data (suboptimal: mean 0.051, std 0.622) — real effect, but magnitude also reflects coverage confound.

Permitted phrasing: "The quality sweep suggests a threshold effect, with random data as a universal failure floor. Above this floor, BC and IQL show limited sensitivity under the current dataset construction, though quality and coverage effects are not fully separated."

Not permitted: "This sweep demonstrates a clean quality-only effect on Offline RL performance."

---

**Claim 2.2 — SA Coverage Metric Directly Tracks Performance**

> The normalized SA coverage metric of each dataset predicts policy performance:
> wide datasets (~21% SA coverage) consistently yield higher returns than narrow
> datasets (~6% SA coverage), independent of dataset size.

Evidence: Mechanism analysis (`artifacts/analysis/envA_v2_mechanism_summary.csv`)
confirms that `dataset_norm_sa_cov` is the strongest predictor of `run_avg_return`.

Constraint: OOD action rate = 0.000 in all conditions. The mechanism is not "distribution
shift prevention" (as in standard offline RL theory) but rather "limiting the diversity of
accessible SA pairs during learning." This is a scale-specific finding — the environment
is small enough that even narrow coverage avoids OOD states in evaluation.

---

**Claim 2.3 — Coverage Effect Generalizes Across Structurally Distinct Multi-Path Environments**

> **UPDATED 2026-04-08** (was: "boundary condition only"). See `VALIDATION_STATUS_ADDENDUM.md`.

> The SA coverage effect observed in EnvA_v2 replicates in two structurally distinct rebuilt
> multi-path validation environments. Both require (a) a meaningful SA coverage gap and
> (b) that gap falling below the effective support threshold. Original single-path EnvB/C
> results confirm that environments lacking these structural properties show zero effect.

**Evidence — Original single-path EnvB/C (boundary condition)**:
- EnvB: SA coverage identical (~99.3%) for both conditions → gap = 0 by structural necessity
- EnvC: ~10pp coverage gap exists but performance gap = 0.000 → both above support threshold
These results remain valid and constrain the claim's scope.

**Evidence — Rebuilt multi-path environments (new)**:
- **EnvB_v2** (3-corridor, 270 states, SA gap 0.156): BC +0.020, IQL +0.026, CQL +0.025.
  All gaps positive; IQL and CQL CIs non-overlapping. Narrow: 100% locked to corridor A.
  Wide: 9–11/20 seeds discover shorter corridor B. **Status: CONFIRMED (3/3 algorithms).**

- **EnvC_v2** (key-door staged, 269 ext. states, SA gap 0.241): BC +0.020, IQL +0.024.
  Both CIs non-overlapping. Narrow: 100% LU path (return 0.66). Wide: 10–12/20 seeds
  find shorter RU/RD path (return 0.70). CQL formal 20-seed run: gap=+0.002 (negligible, CI overlap, 1/20 seeds) — mixed evidence.
  **Status: DIRECTIONAL CONFIRMED (2/3 algorithms).**

**Permitted phrasing**:
> "The coverage effect replicates in structurally distinct multi-path discrete environments
> at effect sizes consistent with simpler path structures (BC/IQL/CQL gaps of 0.020–0.026
> in EnvB_v2; BC/IQL gaps of 0.020–0.024 in EnvC_v2), compared to 0.057–0.127 in EnvA_v2.
> The effect requires multi-path structure and a coverage gap below the effective support threshold."

**Caveats required**:
- Effect sizes differ by environment (0.020–0.026 validation vs 0.057–0.127 primary)
- EnvC_v2 CQL formal 20-seed completed (2026-04-08): gap=+0.002 (negligible, CI overlapping, 1/20 seeds) — mixed evidence; CQL conservative penalty suppresses staged-route discovery
- Claim scope: discrete gridworld environments with multi-path structure

Strength: **Moderate-Strong** for EnvA_v2+EnvB_v2 combined (BC+IQL+CQL in both);
**Moderate** for EnvC_v2 (BC+IQL only CQL mixed evidence).

---

## 3. Claims That Cannot Currently Be Made

These claims would require evidence not yet obtained or would overstate current results.

---

**Cannot claim**: "Coverage effects generalize to all environments unconditionally."
Reason: Effect requires multi-path structure and a coverage gap below the effective support
threshold. Original single-path EnvB/C confirm that these conditions are not always met.
**What CAN now be claimed** (as of 2026-04-08): Coverage effects replicate in rebuilt
multi-path EnvB_v2 (BC/IQL/CQL all confirmed, 3/3) and EnvC_v2 (BC/IQL confirmed, 2/3;
CQL shows positive but negligible gap — mixed evidence, not a third-algorithm confirmation).
See Claim 2.3 update and VALIDATION_STATUS_ADDENDUM.md.

---

**Cannot claim**: "CQL underperforms on D4RL benchmarks."
Reason: CQL's D4RL results are severely underperforming and inconsistently so across
splits — hopper-medium (3.17), hopper-medium-replay (26.65), hopper-medium-expert (1.32).
The inconsistency (medium-replay is closer to plausible; medium and medium-expert are
near-zero) suggests D4RL hyperparameter misconfiguration (`cql_alpha`, batch size,
training length), not an inherent algorithm limitation. Until CQL is properly configured
and confirmed, no comparative claim about its D4RL behavior is permitted.

---

**Cannot claim**: "Results generalize to continuous control settings."
Reason: The Hopper benchmark is a 5-seed pilot with a known CQL defect. BC/IQL/TD3+BC
results are consistent with published ranges, but this validates implementation only —
not coverage-vs-size conclusions in continuous settings.

---

**Cannot claim**: "Coverage effects hold across all data quality levels."
Reason: The main experiment fixes quality at "medium." A crossed quality × coverage design
was never run — the quality sweep varies quality bins but does not systematically control
coverage across them (coverage also varies across bins; see Claim 2.1 caveat). Therefore,
no claim about the interaction between quality and coverage can be made from current data.

---

## 4. Claim Upgrade Conditions

These conditions define when a currently-constrained claim can be upgraded.

| Constrained claim | Required action | Upgrade condition |
|------------------|-----------------|-------------------|
| "Coverage generalizes across environments" | **PARTIALLY MET** (2026-04-08): EnvB_v2 3/3 confirmed; EnvC_v2 2/3 strong + CQL mixed (+0.002). See VALIDATION_STATUS_ADDENDUM.md. | CQL mixed evidence in EnvC_v2 — novel finding about CQL sensitivity in multi-phase envs |
| "CQL performs competitively on D4RL" | Debug CQL hyperparameters; confirm 3-seed Hopper-medium ≥ 40 normalized | Only then report CQL benchmark results |
| "Benchmark results are statistically reliable" | Upgrade from 5 to 20 seeds after CQL is fixed | All 4 algorithms × 3 datasets × 20 seeds |
| "BC size effect is robust" | Additional BC seeds under wide coverage; check if bimodality persists | Replicate Δ=+0.074 with reduced std |

---

## 5. Keep / Rebuild / Postpone Matrix

| Component | Decision | Rationale |
|-----------|----------|-----------|
| EnvA_v2 main experiment results | **KEEP** | Complete, 3 algorithms × 4 conditions × 20 seeds. Primary evidence. |
| Quality sweep results | **KEEP** | Complete, 3 algorithms × 5 levels × 20 seeds. Secondary evidence. |
| Mechanism analysis | **KEEP** | Corroborates claim 2.2. SA coverage metric validated. |
| Final tables, figures, report | **KEEP** | Frozen outputs. Do not modify. |
| Frozen datasets (13 .npz) | **KEEP** | Reproducibility anchor. Do not modify. |
| Behavior pool checkpoints | **KEEP** | Required to regenerate datasets from scratch. |
| EnvB/C (current, single-path) | **KEEP results + REBUILD environments** | Current results retained as boundary condition evidence. Environments must be redesigned as multi-path before they can serve as genuine validation. |
| Hopper benchmark (5 seeds) | **KEEP as pilot + POSTPONE upgrade** | Retain current results as appendix. Upgrade to 20 seeds only after CQL debug is confirmed. Do not upgrade with broken CQL. |
| CQL D4RL configuration | **REBUILD (targeted debug)** | Identify D4RL-appropriate `cql_alpha`. 3-seed diagnostic before committing to full rerun. |
| EnvA (15×15 FourRooms) | **KEEP in codebase, do not use** | Retain as historical reference. Document as pre-pivot baseline. |
| EnvB/C full 4-condition expansion | **POSTPONE indefinitely** | Not in scope. Validation-only role is maintained even after rebuild. |
| Walker2d / HalfCheetah benchmarks | **POSTPONE** | Conditional on Hopper being fully complete and stable at 20 seeds. |
| Additional discrete algorithms | **POSTPONE** | BC/CQL/IQL scope is frozen. No new algorithms for discrete main line. |
| Crossed quality × coverage experiment | **POSTPONE** | Would be a new research question. Out of scope for this project. |
