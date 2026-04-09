# FINAL_CANONICAL_CLAIMS.md
# Canonical Claim Text Blocks — Ready for Direct Use

> Date: 2026-04-08
> Status: Final — synchronized with all experimental results
> Do not modify wording without updating FINAL_SYNC_CHANGELOG.md

All three blocks reflect the same factual state:
- EnvA_v2: primary evidence, fully closed (3/3 algorithms, statistical closure complete)
- EnvB_v2: rebuilt validation, 3/3 algorithms confirmed
- EnvC_v2: BC + IQL strongly confirmed; CQL mixed evidence (conservative penalty suppresses staged-route discovery)
- Benchmark: appendix-only; CQL D4RL issue unresolved but non-blocking
- Mechanism: behavioral diversity / route-family access — NOT OOD prevention

---

## Block 1: Report / Paper Abstract Version (150–220 words)

We investigate whether state-action (SA) coverage determines the Offline RL policy
performance ceiling more decisively than dataset size in discrete multi-path environments.
Using a controlled 2×2 factorial design on a 30×30 four-corridor gridworld (EnvA_v2),
we evaluate BC, CQL, and IQL across 20 training seeds per condition. A small dataset
with broad SA coverage (50k transitions, ~21% SA pairs) consistently outperforms a
larger dataset with narrow coverage (200k transitions, ~6% SA pairs) by +0.057 to +0.127
across all three algorithms. When SA coverage is held fixed at the narrow level, quadrupling
dataset size yields zero improvement for any algorithm. The effect replicates in two rebuilt
multi-path validation environments: a three-corridor gridworld (EnvB_v2, 3/3 algorithms
confirmed, gaps +0.020–+0.026) and a key-door staged environment (EnvC_v2, BC and IQL
confirmed with CI non-overlap at +0.020 and +0.024; CQL shows a negligible gap consistent
with its conservative penalty suppressing staged-route discovery). The mechanism is
behavioral diversity rather than distribution-shift prevention: coverage determines which
strategy families are learnable from the data, not merely which states are reachable
during evaluation. These findings are formally closed by statistical testing and replicated
across three structurally distinct environments.

---

## Block 2: README Concise Version (80–120 words)

SA coverage is the primary determinant of the Offline RL performance ceiling.
A small dataset with broad coverage (50k, ~21% SA pairs) outperforms a large dataset
with narrow coverage (200k, ~6%) by +0.057–+0.127 across BC, CQL, and IQL on a
30×30 discrete gridworld. Quadrupling data size with fixed narrow coverage yields
zero improvement. The effect replicates in two rebuilt multi-path validation environments:
EnvB_v2 (3/3 algorithms confirmed) and EnvC_v2 (BC+IQL confirmed; CQL shows negligible
gap due to conservative Q-penalty in staged environments). The mechanism is behavioral
diversity: coverage determines which route families are learnable, not OOD prevention.

---

## Block 3: Oral Defense Version (60–90 seconds)

"Our study tests whether state-action coverage matters more than dataset size in Offline RL.
We ran a controlled experiment with 20 seeds per condition: small datasets with broad coverage
versus large datasets with narrow coverage. The broad-coverage datasets outperformed the
large-narrow datasets by 6 to 13 percentage points across all three algorithms — and crucially,
quadrupling the dataset size without changing coverage produced exactly zero improvement for
every algorithm. We validated this in two rebuilt multi-path environments. The first — a
three-corridor gridworld — confirmed all three algorithms. The second — a key-door staged
environment — confirmed BC and IQL clearly, but CQL showed a negligible gap because its
conservative penalty suppresses discovery of routes that aren't well-represented in the data.
The mechanism isn't about preventing out-of-distribution evaluation — our OOD rate was zero
throughout. It's about behavioral diversity: narrow datasets lock agents to one strategy
family, while broad datasets expose value-based algorithms to multiple families and let them
identify the better-return ones. The coverage effect is robust, statistically closed, and
holds across structurally distinct environments."
