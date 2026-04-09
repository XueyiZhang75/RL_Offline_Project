# PRESENTATION_OUTLINE.md
# Course Presentation Outline — 12–14 Slides

> Topic: Does State-Action Coverage Matter More Than Dataset Size in Offline RL?
> Format: ~20 minutes + Q&A
> Audience: course audience familiar with RL basics

---

## Slide 1 — Title / Motivation

**Title**: *Does State-Action Coverage Matter More Than Dataset Size in Offline RL?*

Bullet points:
- Offline RL uses a fixed dataset — the quality of that dataset determines the policy ceiling
- Two natural levers: *how much* data (size) vs *which states/actions* are covered (coverage)
- Practitioners often equate "better data" with "more data" — is this correct?

*No figure. Opening slide.*

---

## Slide 2 — The Core Question

**Title**: Research Question and Experimental Design

Bullet points:
- **Primary question**: When we can't collect more data, does broader SA coverage beat more data with narrower coverage?
- **Setup**: 2×2 factorial — coverage (wide ~21% vs narrow ~6%) × size (50k vs 200k transitions)
- **Primary contrast**: small-wide vs large-narrow — 4× less data, 3.5× more coverage
- Environment: 30×30 discrete four-corridor gridworld (EnvA_v2), 3 algorithms × 20 seeds each

*No figure. Conceptual slide.*

---

## Slide 3 — What Is State-Action Coverage?

**Title**: SA Coverage: The Key Metric

Bullet points:
- SA coverage = fraction of (state, action) pairs visited in the dataset out of all reachable pairs
- Wide dataset: BFS-guided controllers across all 4 corridors → covers ~21% of 2,680 SA pairs
- Narrow dataset: BFS-guided controller restricted to one corridor → covers ~6% of SA pairs
- Same environment, same quality — only *which trajectories* differ

*Show Figure 4 (fig4_envbc_validation.png) or fig1_main_coverage_vs_size.png illustrating SA coverage distinction. OR draw a simple diagram of wide vs narrow corridor coverage.*

---

## Slide 4 — Main Result: Coverage Wins

**Title**: Small-Wide Outperforms Large-Narrow Across All Three Algorithms

Bullet points:
- BC: +0.057 gap (small-wide 0.327 vs large-narrow 0.270)
- CQL: +0.127 gap (0.397 vs 0.270)
- IQL: +0.127 gap (0.397 vs 0.270)
- All 20 seeds per algorithm confirm the direction — no exceptions

*Show Figure 2: fig2_core_smallwide_vs_largenarrow.png — bar chart of SW vs LN means with CIs.*

---

## Slide 5 — Confirming Size Is Not the Driver

**Title**: Size Adds Nothing When Coverage Is Fixed Narrow

Bullet points:
- **Small-narrow vs Large-narrow**: Δ = exactly 0.000 for all algorithms, all seeds
- Quadrupling data from the same narrow distribution: zero performance gain
- The narrow dataset has already saturated its coverage ceiling
- Also: under wide coverage, CQL and IQL are insensitive to 4× more data (Δ < 0.005)

*Show Figure 1: fig1_main_coverage_vs_size.png — 4-condition bar chart showing size null under narrow.*

---

## Slide 6 — Statistical Confidence

**Title**: Statistical Closure

Bullet points:
- Coverage effect (Claim 1.1): **formally closed** — BC MWU p = 1.13e-07; CQL/IQL t p = 7.47e-23
- Size = 0 under narrow (Claim 1.2): **exact equality** — no inference needed; std = 0.000
- Large-narrow policies are fully degenerate: all 20 seeds converge to the identical policy
- BC wide-size effect: directional (p = 0.09, not significant at α = 0.05); CQL null confirmed

*Show or describe the hypothesis test table. No separate figure needed — this is a summary slide.*

---

## Slide 7 — Mechanism: Not Distribution Shift

**Title**: Why Does Coverage Matter? The Mechanism

Bullet points:
- Standard offline RL theory: coverage prevents out-of-distribution (OOD) evaluation
- **Finding**: OOD rate = 0.000 in *all* conditions — even narrow data covers evaluation states
- Real mechanism: SA coverage determines the *set of learnable strategies*
- Narrow data constrains the Q-function to one behavioral family; wide data exposes value-based methods to multiple strategies → they learn which is better

*Show Figure 5: fig5_mechanism_summary.png — SA coverage vs return scatter.*

---

## Slide 8 — Boundary Conditions: When Does Coverage NOT Matter?

**Title**: The Original EnvB/C: Delineating When the Effect Fails

Bullet points:
- Original EnvB (15×15 double-bottleneck): wide ≈ narrow ≈ 99.3% SA coverage → gap = 0 by necessity
- Original EnvC (15×15 key-door): 10pp coverage gap exists, but both regimes are *above* the support threshold → gap = 0
- Conclusion: effect requires (a) meaningful coverage gap AND (b) gap below effective support threshold
- **This told us what to fix** → inspired multi-path environment rebuild

*Show Figure 4: fig4_envbc_validation.png if available, or describe results in bullets.*

---

## Slide 9 — Cross-Environment Replication: EnvB_v2

**Title**: EnvB_v2 Rebuilt — Three-Corridor Validation (Confirmed: 3/3 Algorithms)

Bullet points:
- Redesigned as a 20×20 three-corridor gridworld (270 states, 1080 SA pairs)
- Wide covers all 3 corridors (24.1% SA); narrow-A covers only corridor A (8.5% SA)
- **Results**: BC +0.020, IQL +0.026, CQL +0.025 — all three algorithms confirm direction
- Mechanism visible: narrow policies 100% locked to corridor A; wide discovers shorter corridor B in 9–11/20 seeds

*Optional: show a simple 2-column table of the three-algorithm results.*

---

## Slide 10 — Cross-Environment Replication: EnvC_v2

**Title**: EnvC_v2 Rebuilt — Key-Door Staged Multi-Route (Confirmed: 2/3 Algorithms)

Bullet points:
- Key-door environment with extended state ((row,col), has_key), four combined route families
- 269 extended states; wide 32.8% SA, narrow-LU 8.7% SA; gap = 0.24, ratio = 3.76×
- **Results**: BC +0.020, IQL +0.024 — both CIs non-overlapping (10/20, 12/20 seeds find shorter path); CQL +0.002 (negligible, CI overlap, 1/20 seeds — mixed evidence)
- Same mechanism: narrow → 100% LU path (return 0.66); wide → 10–12/20 seeds find shorter RU/RD (return 0.70)

*Summary table of the two validated environments together.*

---

## Slide 11 — Quality Sweep: Auxiliary Evidence

**Title**: Quality Sweep — Random Is a Universal Failure Floor

Bullet points:
- Random data: complete failure for all algorithms regardless of coverage (~38% SA, success = 0%)
- Above random floor: BC/IQL show small variation (Δ < 0.010, suboptimal to expert)
- **Important caveat**: coverage also varies across quality bins → quality and coverage effects are confounded
- CQL is more sensitive to low-quality data (high variance at suboptimal level)
- This is auxiliary evidence; the quality × coverage interaction is not cleanly separable

*Show Figure 3: fig3_quality_modulation.png — quality sweep bar chart.*

---

## Slide 12 — Benchmark Appendix

**Title**: Hopper D4RL Benchmark — Implementation Validation (Appendix)

Bullet points:
- BC, IQL, TD3+BC results match published baselines (BC 20–47, IQL 29–31, TD3+BC 22–63)
- **CQL anomaly**: results inconsistent across splits (medium: 3.17, medium-replay: 26.65, medium-expert: 1.32 vs. published ~58/46/91)
- Inconsistency pattern suggests `cql_alpha` misconfiguration, not a fundamental bug
- CQL D4RL results excluded from all comparative claims; does not affect discrete conclusions

*Show Figure 6: fig6_benchmark_validation.png — benchmark bar chart.*

---

## Slide 13 — Summary Table

**Title**: Complete Evidence Picture

Bullet points / table:

| Finding | Evidence | Strength |
|---------|----------|----------|
| Coverage > size in primary environment | EnvA_v2 3 algos × 20 seeds | **Strong / Closed** |
| Size = 0 under narrow coverage | Exact equality, all seeds | **Strong / Closed** |
| BC shows modest wide-size effect | Directional trend (p = 0.09) | Moderate |
| Coverage effect replicates in multi-path envs | EnvB_v2 (3/3) + EnvC_v2 (2/3) | Moderate-Strong |
| Original single-path envs show zero effect | Structural explanation confirmed | Moderate (boundary cond.) |
| Random data is universal failure floor | Quality sweep, all seeds | Strong |

*Summary slide — no figure needed.*

---

## Slide 14 — Limitations and Future Work

**Title**: What This Study Does and Does Not Show

**Limitations**:
- All results are in small deterministic discrete gridworlds (≤ 670 states); scalability unknown
- OOD rate = 0 → cannot test the distribution-shift mechanism directly at this scale
- BC/IQL Claim 1.3 (wide-size effect) is directional only, not formally confirmed
- EnvC_v2 CQL shows mixed evidence (gap=+0.002, CI overlapping, 1/20 seeds) — conservative Q-penalty suppresses staged-route discovery; EnvC_v2 is "2/3 strong + CQL mixed," not 3/3 confirmed
- D4RL CQL configuration issue unresolved; continuous coverage claims not supported

**Future directions**:
- Scale to larger discrete environments (100×100+) to test OOD mechanism separately
- Crossed coverage × quality design to cleanly isolate quality effects
- Continuous control: test coverage-vs-size in D4RL with properly configured CQL
- Coverage-aware dataset acquisition: can we design collection policies that maximize SA coverage?

*Final slide. No figure — leave time for Q&A.*

---

## Figure Assignment Summary (Canonical)

All figures in `figures/final/mainline/` (M-series) and `figures/final/auxiliary/` (A-series).
Backward-compatible aliases also exist in `figures/final/`.

| Slide | Canonical figure | File |
|-------|-----------------|------|
| 3 (SA coverage concept) | M1 — Factorial overview | `mainline/fig01_envA_factorial_overview.png` |
| 4 (Main result) | M2 — Primary contrast | `mainline/fig02_envA_primary_contrast_sw_vs_ln.png` |
| 5 (Size null) | M3 — Size effect decomposition | `mainline/fig03_envA_size_effect_decomposition.png` |
| 6 (Statistics) | M4 — Statistical closure matrix | `mainline/fig04_statistical_closure_matrix.png` |
| 7 (Mechanism) | M5 — Behavioral diversity mechanism | `mainline/fig05_mechanism_behavioral_diversity.png` |
| 8 (Boundary) | A1 — Original EnvB/C boundary | `auxiliary/fig08_original_envbc_boundary_conditions.png` |
| 9 (EnvB_v2) | M6 + M7 — Rebuilt validation gaps + routes | `mainline/fig06_rebuilt_validation_gap_summary.png`, `fig07_rebuilt_route_family_convergence.png` |
| 10 (EnvC_v2) | M6 + M7 (same figures, EnvC_v2 rows) | same as slide 9 |
| 11 (Quality) | A2 — Quality sweep + coverage caveat | `auxiliary/fig09_quality_sweep_with_coverage_caveat.png` |
| 12 (Benchmark) | A5 — Hopper appendix | `auxiliary/fig12_hopper_benchmark_appendix.png` |
| 13 (Summary) | M6 — Rebuilt validation gap summary | `mainline/fig06_rebuilt_validation_gap_summary.png` |
| 14 (Limitations) | No figure | — |
