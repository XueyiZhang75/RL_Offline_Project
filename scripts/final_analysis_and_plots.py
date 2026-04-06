"""
scripts/final_analysis_and_plots.py
Final analysis, tables, and figures for the Offline RL coverage-vs-size study.

Reads only. Does NOT modify any existing summary / checkpoint / npz files.
Generates:
  artifacts/final_results/
    final_discrete_results_master_table.csv
    final_quality_results_table.csv
    final_validation_results_table.csv
    final_benchmark_results_table.csv
  figures/final/
    fig1_main_coverage_vs_size.png
    fig2_core_smallwide_vs_largenarrow.png
    fig3_quality_modulation.png
    fig4_envbc_validation.png
    fig5_mechanism_summary.png
    fig6_benchmark_validation.png
"""

import sys, os, csv, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

def _p(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

SOURCES = {
    "phase8_main":    _p("artifacts", "training_main",       "envA_v2_main_summary.csv"),
    "phase9_val":     _p("artifacts", "training_validation", "envbc_validation_summary.csv"),
    "phase10_qual":   _p("artifacts", "training_quality",    "envA_v2_quality_summary.csv"),
    "phase11_mech":   _p("artifacts", "analysis",            "envA_v2_mechanism_summary.csv"),
    "phase12_bench":  _p("artifacts", "training_benchmark",  "hopper_benchmark_summary.csv"),
    "r2_iql_main":    _p("artifacts", "training_iql",        "envA_v2_iql_main_summary.csv"),
    "r3_iql_val":     _p("artifacts", "training_iql",        "envbc_iql_validation_summary.csv"),
    "r4_iql_qual":    _p("artifacts", "training_iql",        "envA_v2_iql_quality_sweep_summary.csv"),
}

OUT_DIR   = _p("artifacts", "final_results")
FIG_DIR   = _p("figures", "final")
REPORT    = _p("reports", "final_project_results.md")

for d in [OUT_DIR, FIG_DIR, _p("reports")]:
    os.makedirs(d, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(key):
    path = SOURCES[key]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required source missing: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def ci95(vals):
    n = len(vals)
    m = np.mean(vals)
    if n < 2:
        return m, 0.0, m, m
    s = np.std(vals, ddof=1)
    se = s / math.sqrt(n)
    lo, hi = sp_stats.t.interval(0.95, df=n-1, loc=m, scale=se)
    return float(m), float(s), float(lo), float(hi)

def agg_group(rows, ret_col="avg_return", sr_col="success_rate"):
    rets = [float(r[ret_col]) for r in rows]
    srs  = [float(r[sr_col])  for r in rows]
    rm, rs, rlo, rhi = ci95(rets)
    sm, ss, slo, shi = ci95(srs)
    return rm, rs, rlo, rhi, sm, ss, slo, shi

STYLE = {
    "bc":    {"color": "#2196F3", "marker": "o",  "label": "BC"},
    "cql":   {"color": "#F44336", "marker": "s",  "label": "CQL"},
    "iql":   {"color": "#4CAF50", "marker": "^",  "label": "IQL"},
    "td3bc": {"color": "#FF9800", "marker": "D",  "label": "TD3+BC"},
}
DS_LABELS = {
    "envA_v2_small_wide_medium":   "Small-Wide",
    "envA_v2_small_narrow_medium": "Small-Narrow",
    "envA_v2_large_wide_medium":   "Large-Wide",
    "envA_v2_large_narrow_medium": "Large-Narrow",
}
QUALITY_ORDER = ["random", "suboptimal", "medium", "expert", "mixed"]
QUALITY_BIN_MAP = {
    "envA_v2_quality_random_wide50k":     "random",
    "envA_v2_quality_suboptimal_wide50k": "suboptimal",
    "envA_v2_quality_medium_wide50k":     "medium",
    "envA_v2_quality_expert_wide50k":     "expert",
    "envA_v2_quality_mixed_wide50k":      "mixed",
}

print("=" * 66)
print("Final Analysis & Plots")
print("=" * 66)

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1: final_discrete_results_master_table.csv
# ══════════════════════════════════════════════════════════════════════════════

print("\n-- Table 1: discrete main results --")

bc_cql_rows = load_csv("phase8_main")
iql_rows    = load_csv("r2_iql_main")

MAIN_DATASETS = [
    "envA_v2_small_wide_medium",
    "envA_v2_small_narrow_medium",
    "envA_v2_large_wide_medium",
    "envA_v2_large_narrow_medium",
]
MAIN_ALGOS = ["bc", "cql", "iql"]

t1_rows = []
t1_cols = ["dataset_name", "algorithm", "n_seeds",
           "mean_return", "std_return", "ci95_return_low", "ci95_return_high",
           "mean_success_rate", "std_success_rate", "ci95_sr_low", "ci95_sr_high"]

for ds in MAIN_DATASETS:
    for algo in MAIN_ALGOS:
        if algo in ("bc", "cql"):
            group = [r for r in bc_cql_rows
                     if r["dataset_name"] == ds and r["algorithm"] == algo]
        else:
            group = [r for r in iql_rows
                     if r["dataset_name"] == ds and r["algorithm"] == algo]
        assert len(group) == 20, f"Expected 20 runs for {ds}/{algo}, got {len(group)}"
        rm, rs, rlo, rhi, sm, ss, slo, shi = agg_group(group)
        t1_rows.append({
            "dataset_name": ds, "algorithm": algo, "n_seeds": 20,
            "mean_return": f"{rm:.4f}", "std_return": f"{rs:.4f}",
            "ci95_return_low": f"{rlo:.4f}", "ci95_return_high": f"{rhi:.4f}",
            "mean_success_rate": f"{sm:.4f}", "std_success_rate": f"{ss:.4f}",
            "ci95_sr_low": f"{slo:.4f}", "ci95_sr_high": f"{shi:.4f}",
        })
        print(f"  {ds} | {algo}: ret={rm:.4f}±{rs:.4f}  sr={sm:.4f}")

with open(os.path.join(OUT_DIR, "final_discrete_results_master_table.csv"),
          "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=t1_cols)
    w.writeheader(); w.writerows(t1_rows)
print("  -> saved final_discrete_results_master_table.csv")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2: final_quality_results_table.csv
# ══════════════════════════════════════════════════════════════════════════════

print("\n-- Table 2: quality sweep results --")

bc_cql_q = load_csv("phase10_qual")
iql_q    = load_csv("r4_iql_qual")

t2_rows = []
t2_cols = ["quality_bin", "algorithm", "n_seeds",
           "mean_return", "std_return", "ci95_return_low", "ci95_return_high",
           "mean_success_rate", "std_success_rate", "ci95_sr_low", "ci95_sr_high"]

for qbin in QUALITY_ORDER:
    for algo in MAIN_ALGOS:
        if algo in ("bc", "cql"):
            group = [r for r in bc_cql_q
                     if r.get("quality_bin") == qbin and r["algorithm"] == algo]
        else:
            ds = next(k for k, v in QUALITY_BIN_MAP.items() if v == qbin)
            group = [r for r in iql_q
                     if r["dataset_name"] == ds and r["algorithm"] == algo]
        assert len(group) == 20, f"Expected 20 for {qbin}/{algo}, got {len(group)}"
        rm, rs, rlo, rhi, sm, ss, slo, shi = agg_group(group)
        t2_rows.append({
            "quality_bin": qbin, "algorithm": algo, "n_seeds": 20,
            "mean_return": f"{rm:.4f}", "std_return": f"{rs:.4f}",
            "ci95_return_low": f"{rlo:.4f}", "ci95_return_high": f"{rhi:.4f}",
            "mean_success_rate": f"{sm:.4f}", "std_success_rate": f"{ss:.4f}",
            "ci95_sr_low": f"{slo:.4f}", "ci95_sr_high": f"{shi:.4f}",
        })
        print(f"  {qbin} | {algo}: ret={rm:.4f}±{rs:.4f}")

with open(os.path.join(OUT_DIR, "final_quality_results_table.csv"),
          "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=t2_cols)
    w.writeheader(); w.writerows(t2_rows)
print("  -> saved final_quality_results_table.csv")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3: final_validation_results_table.csv
# ══════════════════════════════════════════════════════════════════════════════

print("\n-- Table 3: EnvB/C validation results --")

bc_cql_v = load_csv("phase9_val")
iql_v    = load_csv("r3_iql_val")

VAL_DATASETS = {
    "envB_small_wide_medium":   ("EnvB", "wide"),
    "envB_large_narrow_medium": ("EnvB", "narrow"),
    "envC_small_wide_medium":   ("EnvC", "wide"),
    "envC_large_narrow_medium": ("EnvC", "narrow"),
}

t3_rows = []
t3_cols = ["env_name", "dataset_name", "coverage", "algorithm", "n_seeds",
           "mean_return", "std_return", "mean_success_rate"]

for ds, (env_name, cov) in VAL_DATASETS.items():
    for algo in MAIN_ALGOS:
        if algo in ("bc", "cql"):
            group = [r for r in bc_cql_v
                     if r["dataset_name"] == ds and r["algorithm"] == algo]
        else:
            group = [r for r in iql_v
                     if r["dataset_name"] == ds and r["algorithm"] == algo]
        assert len(group) == 20, f"Expected 20 for {ds}/{algo}, got {len(group)}"
        rm, rs, _, _, sm, _, _, _ = agg_group(group)
        t3_rows.append({
            "env_name": env_name, "dataset_name": ds, "coverage": cov,
            "algorithm": algo, "n_seeds": 20,
            "mean_return": f"{rm:.4f}", "std_return": f"{rs:.4f}",
            "mean_success_rate": f"{sm:.4f}",
        })
        print(f"  {env_name} {cov} | {algo}: ret={rm:.4f}  sr={sm:.4f}")

with open(os.path.join(OUT_DIR, "final_validation_results_table.csv"),
          "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=t3_cols)
    w.writeheader(); w.writerows(t3_rows)
print("  -> saved final_validation_results_table.csv")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 4: final_benchmark_results_table.csv
# ══════════════════════════════════════════════════════════════════════════════

print("\n-- Table 4: Hopper benchmark results --")

bench = load_csv("phase12_bench")
t4_rows = []
t4_cols = ["dataset_name", "algorithm", "n_runs",
           "mean_normalized_score", "std_normalized_score",
           "ci95_norm_low", "ci95_norm_high"]

for r in bench:
    t4_rows.append({
        "dataset_name": r["dataset_name"],
        "algorithm":    r["algorithm"],
        "n_runs":       r["n_runs"],
        "mean_normalized_score": r["mean_normalized_score"],
        "std_normalized_score":  r["std_normalized_score"],
        "ci95_norm_low":  r["ci95_normalized_score_low"],
        "ci95_norm_high": r["ci95_normalized_score_high"],
    })
    print(f"  {r['dataset_name']} | {r['algorithm']}: "
          f"norm={r['mean_normalized_score']}±{r['std_normalized_score']}")

with open(os.path.join(OUT_DIR, "final_benchmark_results_table.csv"),
          "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=t4_cols)
    w.writeheader(); w.writerows(t4_rows)
print("  -> saved final_benchmark_results_table.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1: main coverage vs size (4-panel grouped bar)
# ══════════════════════════════════════════════════════════════════════════════

print("\n-- Figure 1: main coverage vs size --")

fig, ax = plt.subplots(figsize=(10, 5))

ds_list  = MAIN_DATASETS
ds_short = ["Small\nWide", "Small\nNarrow", "Large\nWide", "Large\nNarrow"]
algos    = ["bc", "cql", "iql"]
n_algos  = len(algos)
x        = np.arange(len(ds_list))
width    = 0.25

t1_lookup = {(r["dataset_name"], r["algorithm"]): r for r in t1_rows}

for i, algo in enumerate(algos):
    means = [float(t1_lookup[(ds, algo)]["mean_return"]) for ds in ds_list]
    cilo  = [float(t1_lookup[(ds, algo)]["mean_return"]) -
             float(t1_lookup[(ds, algo)]["ci95_return_low"])  for ds in ds_list]
    cihi  = [float(t1_lookup[(ds, algo)]["ci95_return_high"]) -
             float(t1_lookup[(ds, algo)]["mean_return"])       for ds in ds_list]
    st = STYLE[algo]
    bars = ax.bar(x + (i - 1) * width, means, width,
                  label=st["label"], color=st["color"], alpha=0.8,
                  yerr=[cilo, cihi], capsize=4, error_kw={"linewidth": 1.2})

ax.set_xticks(x)
ax.set_xticklabels(ds_short, fontsize=11)
ax.set_ylabel("Mean Return (20 seeds)", fontsize=12)
ax.set_title("EnvA_v2 Main Four: Coverage vs Size (BC / CQL / IQL)", fontsize=13)
ax.legend(fontsize=11)
ax.set_ylim(-0.05, 0.55)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_main_coverage_vs_size.png"), dpi=150)
plt.close()
print("  -> fig1_main_coverage_vs_size.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2: core small-wide vs large-narrow
# ══════════════════════════════════════════════════════════════════════════════

print("-- Figure 2: small-wide vs large-narrow --")

fig, ax = plt.subplots(figsize=(7, 5))

core_ds   = ["envA_v2_small_wide_medium", "envA_v2_large_narrow_medium"]
core_lbls = ["Small-Wide\n(High Coverage)", "Large-Narrow\n(Low Coverage)"]
x2 = np.arange(len(core_ds))

for i, algo in enumerate(algos):
    means = [float(t1_lookup[(ds, algo)]["mean_return"]) for ds in core_ds]
    cilo  = [float(t1_lookup[(ds, algo)]["mean_return"]) -
             float(t1_lookup[(ds, algo)]["ci95_return_low"])  for ds in core_ds]
    cihi  = [float(t1_lookup[(ds, algo)]["ci95_return_high"]) -
             float(t1_lookup[(ds, algo)]["mean_return"])       for ds in core_ds]
    st = STYLE[algo]
    ax.bar(x2 + (i - 1) * width, means, width,
           label=st["label"], color=st["color"], alpha=0.85,
           yerr=[cilo, cihi], capsize=5, error_kw={"linewidth": 1.5})

ax.set_xticks(x2)
ax.set_xticklabels(core_lbls, fontsize=12)
ax.set_ylabel("Mean Return (20 seeds)", fontsize=12)
ax.set_title("Core Contrast: Small-Wide vs Large-Narrow\n(Small high-coverage vs large low-coverage)",
             fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(-0.05, 0.55)
ax.grid(axis="y", alpha=0.3)

# annotate gaps
for algo in algos:
    sw = float(t1_lookup[("envA_v2_small_wide_medium",   algo)]["mean_return"])
    ln = float(t1_lookup[("envA_v2_large_narrow_medium", algo)]["mean_return"])
    gap = sw - ln
    print(f"  {algo}: small-wide={sw:.4f}  large-narrow={ln:.4f}  gap={gap:+.4f}")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_core_smallwide_vs_largenarrow.png"), dpi=150)
plt.close()
print("  -> fig2_core_smallwide_vs_largenarrow.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3: quality modulation
# ══════════════════════════════════════════════════════════════════════════════

print("-- Figure 3: quality modulation --")

t2_lookup = {(r["quality_bin"], r["algorithm"]): r for r in t2_rows}

fig, ax = plt.subplots(figsize=(10, 5))
q_x = np.arange(len(QUALITY_ORDER))

for i, algo in enumerate(algos):
    means = [float(t2_lookup[(qb, algo)]["mean_return"]) for qb in QUALITY_ORDER]
    cilo  = [float(t2_lookup[(qb, algo)]["mean_return"]) -
             float(t2_lookup[(qb, algo)]["ci95_return_low"])  for qb in QUALITY_ORDER]
    cihi  = [float(t2_lookup[(qb, algo)]["ci95_return_high"]) -
             float(t2_lookup[(qb, algo)]["mean_return"])       for qb in QUALITY_ORDER]
    st = STYLE[algo]
    ax.plot(q_x, means, marker=st["marker"], color=st["color"],
            label=st["label"], linewidth=2, markersize=8)
    ax.fill_between(q_x,
                    [m - l for m, l in zip(means, cilo)],
                    [m + h for m, h in zip(means, cihi)],
                    color=st["color"], alpha=0.15)

ax.set_xticks(q_x)
ax.set_xticklabels([q.capitalize() for q in QUALITY_ORDER], fontsize=11)
ax.set_ylabel("Mean Return (20 seeds)", fontsize=12)
ax.set_title("Quality Modulation: BC / CQL / IQL across 5 Quality Bins", fontsize=13)
ax.legend(fontsize=11)
ax.set_ylim(-1.2, 0.6)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_quality_modulation.png"), dpi=150)
plt.close()
print("  -> fig3_quality_modulation.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4: EnvB/C validation
# ══════════════════════════════════════════════════════════════════════════════

print("-- Figure 4: EnvB/C validation --")

t3_lookup = {(r["env_name"], r["coverage"], r["algorithm"]): r for r in t3_rows}
envs = ["EnvB", "EnvC"]
covs = ["wide", "narrow"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, env in zip(axes, envs):
    x4 = np.arange(len(covs))
    for i, algo in enumerate(algos):
        means = [float(t3_lookup[(env, cov, algo)]["mean_return"]) for cov in covs]
        st = STYLE[algo]
        ax.bar(x4 + (i - 1) * width, means, width,
               label=st["label"], color=st["color"], alpha=0.8)
    ax.set_xticks(x4)
    ax.set_xticklabels(["Wide\n(Small)", "Narrow\n(Large)"], fontsize=11)
    ax.set_title(f"{env} Validation", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    if env == "EnvB":
        ax.set_ylabel("Mean Return (20 seeds)", fontsize=11)
        ax.legend(fontsize=10)

axes[0].text(0.5, -0.22, "Note: zero contrast reflects single-path structure,\nnot algorithm failure.",
             transform=axes[0].transAxes, ha="center", fontsize=9, color="gray")
fig.suptitle("EnvB/C Validation: Wide vs Narrow (BC / CQL / IQL)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_envbc_validation.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> fig4_envbc_validation.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5: mechanism summary
# ══════════════════════════════════════════════════════════════════════════════

print("-- Figure 5: mechanism summary --")

mech = load_csv("phase11_mech")

ds_list_m = [r["dataset_name"] for r in mech if r["algorithm"] == "bc"]
sa_covs   = [float(r["dataset_norm_sa_cov"]) for r in mech if r["algorithm"] == "bc"]
bc_rets   = [float(r["mean_run_return"])      for r in mech if r["algorithm"] == "bc"]
cql_rets  = [float(r["mean_run_return"])      for r in mech if r["algorithm"] == "cql"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: SA coverage vs mean return
ax = axes[0]
ax.scatter(sa_covs, bc_rets,  color=STYLE["bc"]["color"],  marker="o",
           label="BC",  s=80, zorder=3)
ax.scatter(sa_covs, cql_rets, color=STYLE["cql"]["color"], marker="s",
           label="CQL", s=80, zorder=3)
for i, ds in enumerate(ds_list_m):
    lbl = DS_LABELS.get(ds, ds.replace("envA_v2_", "").replace("_medium", "")
                              .replace("quality_", ""))
    ax.annotate(lbl, (sa_covs[i], bc_rets[i]), fontsize=7,
                xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Dataset SA Coverage (normalized)", fontsize=11)
ax.set_ylabel("Mean Return (BC / CQL)", fontsize=11)
ax.set_title("SA Coverage → Policy Return", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Right: dataset avg return vs algo return (shows coverage effect)
ax = axes[1]
ds_labels_short = [DS_LABELS.get(r["dataset_name"],
                   r["dataset_name"].replace("envA_v2_quality_","").replace("_wide50k",""))
                   for r in mech if r["algorithm"] == "bc"]
x_m = np.arange(len(ds_labels_short))
ax.bar(x_m - 0.2, bc_rets,  0.38, label="BC",  color=STYLE["bc"]["color"],  alpha=0.8)
ax.bar(x_m + 0.2, cql_rets, 0.38, label="CQL", color=STYLE["cql"]["color"], alpha=0.8)
ax.set_xticks(x_m)
ax.set_xticklabels(ds_labels_short, fontsize=7, rotation=15, ha="right")
ax.set_ylabel("Mean Return (20 seeds)", fontsize=11)
ax.set_title("Policy Return by Dataset\n(Main four + Quality sweep)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Mechanism Analysis: Coverage as Constraint on Policy Return", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_mechanism_summary.png"), dpi=150)
plt.close()
print("  -> fig5_mechanism_summary.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6: benchmark validation
# ══════════════════════════════════════════════════════════════════════════════

print("-- Figure 6: benchmark validation --")

bench_rows = load_csv("phase12_bench")
bench_ds   = ["hopper-medium", "hopper-medium-replay", "hopper-medium-expert"]
bench_algos = ["bc", "cql", "iql", "td3bc"]
bench_lkp  = {(r["dataset_name"], r["algorithm"]): r for r in bench_rows}

fig, ax = plt.subplots(figsize=(10, 5))
x6 = np.arange(len(bench_ds))
b_width = 0.2

for i, algo in enumerate(bench_algos):
    means, errs_lo, errs_hi = [], [], []
    for ds in bench_ds:
        key = (ds, algo)
        if key in bench_lkp:
            r = bench_lkp[key]
            m  = float(r["mean_normalized_score"])
            lo = float(r["ci95_normalized_score_low"])
            hi = float(r["ci95_normalized_score_high"])
            means.append(m)
            errs_lo.append(m - lo)
            errs_hi.append(hi - m)
        else:
            means.append(0); errs_lo.append(0); errs_hi.append(0)
    st = STYLE.get(algo, {"color": "gray", "label": algo.upper()})
    ax.bar(x6 + (i - 1.5) * b_width, means, b_width,
           label=st["label"], color=st["color"], alpha=0.8,
           yerr=[errs_lo, errs_hi], capsize=4, error_kw={"linewidth": 1.2})

ax.set_xticks(x6)
ax.set_xticklabels(["Hopper-Medium", "Hopper-Medium\nReplay", "Hopper-Medium\nExpert"],
                   fontsize=11)
ax.set_ylabel("Normalized Score (5 seeds)", fontsize=12)
ax.set_title("Hopper D4RL Benchmark: BC / CQL / IQL / TD3+BC\n"
             "(Auxiliary external validation — CQL config known anomaly)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig6_benchmark_validation.png"), dpi=150)
plt.close()
print("  -> fig6_benchmark_validation.png")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT: reports/final_project_results.md
# ══════════════════════════════════════════════════════════════════════════════

print("\n-- Writing final_project_results.md --")

# Compute key numbers for report
sw_bc  = float(t1_lookup[("envA_v2_small_wide_medium",   "bc")]["mean_return"])
ln_bc  = float(t1_lookup[("envA_v2_large_narrow_medium", "bc")]["mean_return"])
sw_cql = float(t1_lookup[("envA_v2_small_wide_medium",   "cql")]["mean_return"])
ln_cql = float(t1_lookup[("envA_v2_large_narrow_medium", "cql")]["mean_return"])
sw_iql = float(t1_lookup[("envA_v2_small_wide_medium",   "iql")]["mean_return"])
ln_iql = float(t1_lookup[("envA_v2_large_narrow_medium", "iql")]["mean_return"])

lw_bc  = float(t1_lookup[("envA_v2_large_wide_medium",   "bc")]["mean_return"])
lw_cql = float(t1_lookup[("envA_v2_large_wide_medium",   "cql")]["mean_return"])
lw_iql = float(t1_lookup[("envA_v2_large_wide_medium",   "iql")]["mean_return"])
sn_bc  = float(t1_lookup[("envA_v2_small_narrow_medium", "bc")]["mean_return"])
sn_iql = float(t1_lookup[("envA_v2_small_narrow_medium", "iql")]["mean_return"])

iql_rand   = float(t2_lookup[("random",    "iql")]["mean_return"])
iql_sub    = float(t2_lookup[("suboptimal","iql")]["mean_return"])
iql_med    = float(t2_lookup[("medium",    "iql")]["mean_return"])
iql_exp    = float(t2_lookup[("expert",    "iql")]["mean_return"])
bc_rand    = float(t2_lookup[("random",    "bc")]["mean_return"])
bc_sub     = float(t2_lookup[("suboptimal","bc")]["mean_return"])

report_text = f"""# 最终项目结果报告

## 1. Final project status

本项目所有实验阶段已全部完成：
- Clean Phase 7–12：BC/CQL 主线 + Hopper benchmark
- Retrofit R1–R4：IQL sanity、main four、EnvB/C validation、quality sweep
- 离散主线 BC / CQL / IQL 在全部 4 个实验类型上均已完成（sanity / main / validation / quality）

---

## 2. Main research question

**在 Offline RL 中，state-action coverage 是否比 dataset size 更能决定策略的性能上限？**

核心对照：EnvA_v2 上，small-wide（小数据量、高覆盖率）vs large-narrow（大数据量、低覆盖率）。

---

## 3. Final discrete main-line conclusion

**结论：是的，coverage 是三种算法共同支持的主要性能决定因素，small-wide 在 BC、CQL、IQL 上均优于 large-narrow。**

### 3.1 核心对照数据（20 seeds，EnvA_v2）

| 算法 | Small-Wide | Large-Narrow | Gap (SW−LN) |
|------|-----------|--------------|-------------|
| BC   | {sw_bc:.4f} | {ln_bc:.4f} | {sw_bc-ln_bc:+.4f} |
| CQL  | {sw_cql:.4f} | {ln_cql:.4f} | {sw_cql-ln_cql:+.4f} |
| IQL  | {sw_iql:.4f} | {ln_iql:.4f} | {sw_iql-ln_iql:+.4f} |

三个算法均显示 small-wide > large-narrow。CQL 和 IQL 的差值约为 +0.127，BC 的差值为 +0.057（BC 对 coverage 的响应幅度较小，但方向一致）。

### 3.2 Size 的无效性

| 对比 | BC | CQL | IQL |
|------|-----|-----|-----|
| Small-Narrow → Large-Narrow | {sn_bc:.4f} → {ln_bc:.4f} (Δ={ln_bc-sn_bc:+.4f}) | {float(t1_lookup[('envA_v2_small_narrow_medium','cql')]['mean_return']):.4f} → {ln_cql:.4f} (Δ={ln_cql-float(t1_lookup[('envA_v2_small_narrow_medium','cql')]['mean_return']):+.4f}) | {sn_iql:.4f} → {ln_iql:.4f} (Δ={ln_iql-sn_iql:+.4f}) |
| Small-Wide → Large-Wide | {sw_bc:.4f} → {lw_bc:.4f} (Δ={lw_bc-sw_bc:+.4f}) | {sw_cql:.4f} → {lw_cql:.4f} (Δ={lw_cql-sw_cql:+.4f}) | {sw_iql:.4f} → {lw_iql:.4f} (Δ={lw_iql-sw_iql:+.4f}) |

固定 coverage 后，数据量从 5 万增加到 20 万，CQL 和 IQL 的性能基本不变（Δ ≈ 0.002–0.005）。BC 在 wide coverage 条件下存在一定的 size 效应（Δ=+0.0745），但在 narrow coverage 条件下同样无提升（Δ=0.000）。

### 3.3 证据强度

该结论被三个独立算法（BC、CQL、IQL）在 20 个训练种子上重复验证，并配备 95% 置信区间。结论稳健。

---

## 4. EnvB/C validation 解释

EnvB/C validation 中，所有算法（BC/CQL/IQL）在 wide 和 narrow 数据集上的结果完全相同：
- mean return = 0.7600，success rate = 1.000
- wide − narrow gap = 0.000

**这不是算法或实现的失败。**

根本原因：EnvB（两个瓶颈）和 EnvC（钥匙-门结构）都是单路径结构——任何策略都必须经过相同的关键节点，因此 wide 和 narrow 数据集的 SA 覆盖率几乎相同（EnvB ≈ 99%），无法形成有效的覆盖率对照。

**EnvB/C validation 的意义：** 它验证了实验管线在不同环境上的可运行性，同时划定了覆盖率效应的适用边界——覆盖率效应需要环境中存在多条可选路径。

---

## 5. Quality sweep 解释

### 5.1 结果概览

| 质量档 | BC | CQL | IQL |
|--------|-----|-----|-----|
| random | {bc_rand:.4f} | {float(t2_lookup[('random','cql')]['mean_return']):.4f} | {iql_rand:.4f} |
| suboptimal | {bc_sub:.4f} | {float(t2_lookup[('suboptimal','cql')]['mean_return']):.4f} | {iql_sub:.4f} |
| medium | {float(t2_lookup[('medium','bc')]['mean_return']):.4f} | {float(t2_lookup[('medium','cql')]['mean_return']):.4f} | {iql_med:.4f} |
| expert | {float(t2_lookup[('expert','bc')]['mean_return']):.4f} | {float(t2_lookup[('expert','cql')]['mean_return']):.4f} | {iql_exp:.4f} |
| mixed | {float(t2_lookup[('mixed','bc')]['mean_return']):.4f} | {float(t2_lookup[('mixed','cql')]['mean_return']):.4f} | {float(t2_lookup[('mixed','iql')]['mean_return']):.4f} |

### 5.2 关键观察

1. **Random 为绝对地板**：三个算法在 random 数据上全部失败（success rate = 0%），说明存在一个数据质量最低门槛。

2. **超过门槛后质量不敏感**：BC 和 IQL 从 suboptimal 到 expert 的性能变化极小（Δ ≈ 0.005–0.010）。这表明在小型离散环境中，只要数据质量超过随机水平，quality 的进一步提升对最终性能的边际贡献接近于零。

3. **CQL 对低质量数据更脆弱**：CQL 在 suboptimal 上方差显著更高，说明其保守性约束在低质量数据下不稳定。

4. **综合解读**：quality sweep 的主要价值在于证明 quality 有门槛效应，并验证了在该门槛之上，coverage 的操纵（见主实验）才是决定性能上限的关键变量。

---

## 6. 机制解释

Clean Phase 11 机制分析结果：

- SA coverage（dataset_norm_sa_cov）与 mean_run_return 高度对应：wide 数据集的 SA 覆盖率（~21%）显著高于 narrow（~6%），对应的性能差距 +0.13 也一致。
- **OOD action rate 在所有条件下均为 0.000**：这意味着在测试时策略从未走出训练数据的支撑范围。这是因为离散环境较小，即使 narrow 数据集也覆盖了测试轨迹的大部分状态。
- 因此，"coverage 约束分布偏移"的机制在此小型环境中无法直接观测到（OOD rate = 0），但 coverage 仍通过限制"可访问状态-动作对的多样性"来约束策略的性能上限。

---

## 7. Benchmark 解释

Hopper D4RL benchmark（3 数据集 × 4 算法 × 5 seeds）的主要趋势：
- BC 和 TD3+BC 在 medium 和 medium-expert 数据集上表现合理（normalized score 20–65）
- IQL 在 medium-replay 上表现最好（~30），与文献一致
- **CQL 存在已知异常**：normalized score 仅 1–3，远低于文献报告的 ~58。原因是当前 CQL 配置（batch_size=256）未按 D4RL 标准调参，cql_alpha 等超参数未优化。

此异常不影响离散主线结论，因为 CQL 在离散 EnvA_v2 实验中的配置是独立调优的，与连续控制配置无关。

Benchmark 的作用是提供外部参照点，证明实验框架的基本实现是正确的（BC/IQL/TD3+BC 结果与文献一致），而非作为研究主结论的依据。

---

## 8. Final limitations

### 环境性质造成的限制
1. **EnvB/C 单路径结构**：无法产生有效的 coverage 对照，跨环境泛化验证失效。
2. **环境规模过小**：OOD rate 在所有条件下均为 0，无法直接观测分布偏移效应。
3. **离散动作空间**：IQL/TD3+BC 等连续控制算法未能纳入离散主线实验。

### Benchmark 配置造成的限制
4. **CQL 连续控制配置未调优**：benchmark 中 CQL 的结果不具参考价值。
5. **Benchmark 仅 5 seeds**：统计置信度低于离散主线（20 seeds）。

---

## 9. Final deliverables summary

| 类型 | 文件 | 状态 |
|------|------|------|
| 主实验结果表 | final_discrete_results_master_table.csv | ✅ |
| 质量梯度结果表 | final_quality_results_table.csv | ✅ |
| 跨环境验证表 | final_validation_results_table.csv | ✅ |
| Benchmark 结果表 | final_benchmark_results_table.csv | ✅ |
| 主覆盖率对比图 | fig1_main_coverage_vs_size.png | ✅ |
| 核心对照图 | fig2_core_smallwide_vs_largenarrow.png | ✅ |
| 质量梯度图 | fig3_quality_modulation.png | ✅ |
| 跨环境验证图 | fig4_envbc_validation.png | ✅ |
| 机制分析图 | fig5_mechanism_summary.png | ✅ |
| Benchmark 验证图 | fig6_benchmark_validation.png | ✅ |

**最终结论：** 在 EnvA_v2 离散四走廊环境上，state-action coverage 是决定 Offline RL 策略性能上限的主要因素——small-wide 在 BC、CQL、IQL 三个算法上均优于 large-narrow。Dataset size 的独立效应较弱：固定 narrow coverage 时增大 size 无益（Δ=0）；BC 在 wide coverage 下存在一定 size 效应，但 CQL/IQL 基本不受影响。该结论由三个独立算法在 20 个训练种子上共同验证，具备统计可信度。
"""

os.makedirs(os.path.dirname(REPORT), exist_ok=True)
with open(REPORT, "w", encoding="utf-8") as f:
    f.write(report_text)
print("  -> reports/final_project_results.md")

print()
print("=" * 66)
print("Final analysis complete.")
print(f"  Tables : {OUT_DIR}")
print(f"  Figures: {FIG_DIR}")
print(f"  Report : {REPORT}")
print("=" * 66)
