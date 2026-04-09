# FINAL_SUBMISSION_BRIEF.md
# Project Submission Brief — Offline RL Coverage Study

> Date: 2026-04-08
> For: course submission / defense preparation
> Reference: reports/final_project_results.md (full details)

---

## Abstract

This study investigates whether state-action (SA) coverage is a more decisive determinant
of Offline RL policy performance than dataset size in discrete multi-path environments.
Using a controlled 2×2 factorial design (coverage × size) on a 30×30 four-corridor
gridworld, we evaluate BC, CQL, and IQL across 20 training seeds per condition.
A small dataset with broad SA coverage (50k transitions, ~21% SA pairs) consistently
outperforms a larger dataset with narrow coverage (200k transitions, ~6% SA pairs) by
+0.057 to +0.127 across all three algorithms. Crucially, when coverage is held fixed at
the narrow level, quadrupling dataset size yields zero performance improvement for all
algorithms. The effect replicates in two rebuilt multi-path validation environments:
a three-corridor gridworld (EnvB_v2, 3/3 algorithms confirmed) and a key-door staged
environment (EnvC_v2, 2/3 algorithms confirmed). The mechanism operates through
behavioral diversity rather than distribution-shift prevention: narrow datasets lock
agents to a single behavioral strategy, while wide datasets expose value-based methods
to multiple strategies, enabling discovery of higher-return trajectories.

---

## Five Strongest Claims

**Claim 1 — Coverage dominates dataset size** *(formally closed, p < 1e-7)*

In the primary environment (EnvA_v2), a 4× smaller dataset with broad SA coverage
outperforms a 4× larger dataset with narrow coverage for all three algorithms (BC +0.057,
CQL/IQL +0.127). This holds across 20 seeds with no exceptions.

**Claim 2 — Size is completely ineffective when coverage is narrow** *(formally closed, exact equality)*

Increasing dataset size from 50k to 200k under fixed narrow SA coverage (~6%) yields
exactly Δ = 0.000 for all algorithms, all seeds. The narrow dataset has already saturated
its coverage ceiling; more data from the same narrow distribution cannot raise the ceiling.

**Claim 3 — The mechanism is behavioral diversity, not distribution shift** *(closed by mechanism analysis)*

OOD action rate = 0 in all conditions. Coverage does not prevent out-of-distribution
evaluation — the environment is too small for that. Instead, SA coverage determines which
behavioral strategies are represented in training data. Narrow datasets lock policies to one
strategy; wide datasets enable discovery of better ones through value learning.

**Claim 4 — The effect replicates across structurally distinct multi-path environments**
*(EnvB_v2 3/3 confirmed; EnvC_v2 2/3 strong confirmed; CQL mixed in EnvC_v2)*

The coverage effect appears in both a three-corridor environment (EnvB_v2: BC/IQL/CQL
all positive, gaps +0.020–+0.026) and a key-door staged environment (EnvC_v2: BC/IQL
positive, +0.020/+0.024, CIs non-overlapping). EnvC_v2 CQL shows a positive but negligible
gap (+0.002, CI overlapping) — consistent with CQL's conservative penalty suppressing
path-diversity in multi-phase environments. Effect sizes are smaller than in the primary
environment, consistent with simpler two-outcome path structures.

**Claim 5 — The BC size effect under wide coverage is real but algorithm-specific**
*(directional, not formally closed)*

Under wide coverage, BC shows a +0.074 improvement with 4× more data. CQL and IQL show
< 0.005 change. This confirms that the coverage finding is not "data always helps" — it
is specifically about what the data *covers*, not how much there is.

---

## Three Required Caveats

**Caveat 1 — Results are specific to small discrete gridworlds**

All primary experiments use a 30×30 discrete deterministic environment with ~670 reachable
states. OOD rate = 0 in all conditions, which means the mechanism observed here (behavioral
diversity, not distribution shift) may differ in larger or stochastic environments. Whether
"SA coverage determines the performance ceiling" holds at scale is an open question.

**Caveat 2 — EnvC_v2 CQL shows mixed evidence (positive but negligible gap)**

EnvC_v2 CQL formal 20-seed: gap=+0.002 (negligible, CI overlapping, 1/20 seeds). EnvC_v2
is 2/3 strong confirmed (BC + IQL), not 3/3. CQL's conservative Q-penalty suppresses
path-diversity in the staged key-door structure. Describe EnvC_v2 as "2/3 confirmed with
CQL mixed" — never as "three-algorithm confirmed" in any comparison to EnvB_v2.

**Caveat 3 — Quality sweep has a coverage confound**

The quality sweep varies data quality but does not control SA coverage, which also varies
across bins. BC/IQL show small sensitivity above the random floor, but this cannot be
attributed solely to quality. Do not cite the quality sweep as clean evidence for
"quality effects are negligible" — the quality–coverage confound prevents that conclusion.

---

## Five Expected Defense Questions and Standard Answers

**Q1: Why not just collect more data? Couldn't you get both high coverage and large datasets?**

A: In real Offline RL deployments, data collection is expensive and coverage is actively
constrained — logs come from previously deployed policies that may cover only narrow
behavioral regions. The study's contribution is showing that *within a fixed data budget
and behavioral profile*, coverage determines the ceiling while size doesn't. The 4-condition
design specifically tests size at fixed coverage levels, demonstrating the zero size effect
under narrow coverage even when doubling to 4× size.

**Q2: The validation environments (EnvB_v2, EnvC_v2) show much smaller gaps than EnvA_v2.
Is this consistent?**

A: Yes. The effect size reflects the *available return differential* between the best
and worst paths in each environment. In EnvA_v2, wide data provides access to paths with
up to +0.13 higher return; in EnvB_v2 and EnvC_v2, the path-length differential is smaller
(e.g., 33-step vs 21-step corridor, or 34-step vs 27-step key-door route), producing
return differences of ~0.12 and ~0.07 respectively. The coverage effect is always present
when data covers these better paths — but its magnitude is bounded by the environment's
path reward structure, not by the mechanism's strength.

**Q3: Your OOD rate is always zero. Doesn't that undermine the standard offline RL
justification for coverage?**

A: This is our most interesting mechanistic finding. The standard argument ("coverage
prevents out-of-distribution evaluation") does not apply at this environment scale — even
the narrow dataset covers all states encountered during greedy evaluation. The mechanism
is different: coverage determines the *set of learnable strategies*. Value-based methods
(CQL, IQL) can only optimize toward strategies that are represented in the data. A narrow
dataset forecloses entire behavioral families, capping the achievable Q-value. We call
this "behavioral diversity coverage" as opposed to "support coverage." Both may matter
at larger scales — we cannot disentangle them without larger environments.

**Q4: Why not include more algorithms (TD3+BC, DT, etc.) in the discrete main line?**

A: The study's scope was deliberately limited to three algorithms (BC, CQL, IQL) chosen
to span the imitation-learning, conservative-Q-learning, and implicit-Q-learning paradigms.
Adding more algorithms would increase training cost substantially without changing the
primary question. The three chosen algorithms already show a consistent effect (all gaps
positive), which validates the conclusion without requiring exhaustive coverage of the
algorithm space. TD3+BC was used only in the continuous benchmark (Hopper), where it
validates the implementation but not the coverage claims.

**Q5: The CQL benchmark results are very poor. Does this affect your credibility?**

A: The CQL D4RL results are acknowledged as an anomaly with a known likely cause:
D4RL requires different hyperparameters (larger `cql_alpha`, longer training, larger batch)
than the discrete gridworld configuration. The inconsistency pattern (medium-replay closer
to expected; medium and medium-expert near zero) is more consistent with misconfiguration
than a fundamental bug. Crucially, CQL works correctly in the discrete environments — its
results on EnvA_v2, EnvB_v2, and EnvC_v2 are internally consistent and match BC/IQL in
direction. The D4RL section is explicitly labeled as a 5-seed pilot appendix with a known
limitation; it does not affect any primary conclusion.
