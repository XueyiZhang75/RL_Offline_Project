"""
scripts/generate_final_figure_suite.py
Final canonical figure suite — 12 publication-quality figures.

Outputs:
  figures/final/mainline/  → M1–M7  (primary evidence)
  figures/final/auxiliary/ → A1–A5  (auxiliary evidence)
  figures/final/           → compat aliases for old fig1–fig6 names

Hardcoded facts from authoritative CSVs (2026-04-08 freeze):
  EnvA_v2: SW BC=0.3265/CQL=0.3970/IQL=0.3970; LN all=0.2700;
            LW BC=0.4010/CQL=0.3990/IQL=0.4020; SN all=0.2700
  EnvB_v2: SW BC=0.6900/IQL=0.6960/CQL=0.6950; LN all=0.6700
  EnvC_v2: SW BC=0.6800/IQL=0.6840/CQL=0.6620; LN all=0.6600
  CQL EnvC_v2 gap=+0.002 → MIXED; NOT 3/3 confirmed.
"""

import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import csv, math, shutil

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
FIGS_MAIN = os.path.join(PROJECT_ROOT, "figures", "final", "mainline")
FIGS_AUX  = os.path.join(PROJECT_ROOT, "figures", "final", "auxiliary")
FIGS_COMPAT = os.path.join(PROJECT_ROOT, "figures", "final")
os.makedirs(FIGS_MAIN,   exist_ok=True)
os.makedirs(FIGS_AUX,    exist_ok=True)
os.makedirs(FIGS_COMPAT, exist_ok=True)

def fmain(name): return os.path.join(FIGS_MAIN, name)
def faux(name):  return os.path.join(FIGS_AUX,  name)
def fcompat(name): return os.path.join(FIGS_COMPAT, name)
def artdir(*parts): return os.path.join(PROJECT_ROOT, "artifacts", *parts)

# ── Style constants ────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   1.5,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
})

# Fixed algorithm colors
C_BC    = "#2196F3"   # blue
C_CQL   = "#E53935"   # red
C_IQL   = "#43A047"   # green
C_TD3BC = "#FB8C00"   # orange
ALGO_COLORS = {"bc": C_BC, "cql": C_CQL, "iql": C_IQL, "td3bc": C_TD3BC}
ALGO_LABELS = {"bc": "BC", "cql": "CQL", "iql": "IQL", "td3bc": "TD3+BC"}

# Fixed condition colors
C_SW = "#1565C0"   # dark blue = small-wide
C_LN = "#B71C1C"   # dark red  = large-narrow
C_SN = "#EF9A9A"   # light red = small-narrow
C_LW = "#90CAF9"   # light blue= large-wide

# Coverage-encoding colors
COV_HIGH = "#1565C0"
COV_LOW  = "#B71C1C"

# Quality bin colors
QUAL_COLORS = {
    "random":     "#7B1FA2",
    "suboptimal": "#F57F17",
    "medium":     "#1565C0",
    "expert":     "#2E7D32",
    "mixed":      "#4E342E",
}

# Status label positions and styles
def add_status_label(ax, text, loc="lower right", fontsize=7.5, alpha=0.6):
    import matplotlib.patheffects as pe
    props = dict(boxstyle="round,pad=0.2", facecolor="white",
                 edgecolor="gray", alpha=alpha)
    x_map = {"lower right": 0.98, "lower left": 0.02, "upper right": 0.98, "upper left": 0.02}
    y_map = {"lower right": 0.02, "lower left": 0.02, "upper right": 0.97, "upper left": 0.97}
    ha_map = {"lower right": "right", "lower left": "left",
              "upper right": "right", "upper left": "left"}
    va_map = {"lower right": "bottom", "lower left": "bottom",
              "upper right": "top", "upper left": "top"}
    ax.text(x_map[loc], y_map[loc], text, transform=ax.transAxes,
            fontsize=fontsize, ha=ha_map[loc], va=va_map[loc],
            bbox=props, style="italic", color="#444")

def ci95(mean, std, n):
    if n < 2 or std == 0: return mean, mean
    se = std / math.sqrt(n)
    return mean - 1.96*se, mean + 1.96*se

# ── Frozen numerical facts ─────────────────────────────────────────────────────

# EnvA_v2 main (20 seeds each)
ENVA = {
    "small_wide":   {"bc":(0.3265,0.3124), "cql":(0.3970,0.0098), "iql":(0.3970,0.0098)},
    "small_narrow": {"bc":(0.2700,0.0000), "cql":(0.2700,0.0000), "iql":(0.2700,0.0000)},
    "large_wide":   {"bc":(0.4010,0.3226), "cql":(0.3990,0.0097), "iql":(0.4020,0.0098)},
    "large_narrow": {"bc":(0.2700,0.0000), "cql":(0.2700,0.0000), "iql":(0.2700,0.0000)},
}
ENVA_COVERAGE = {
    "small_wide":  (50_000,  0.2052),
    "small_narrow":(50_000,  0.0601),
    "large_wide":  (200_000, 0.2123),
    "large_narrow":(200_000, 0.0601),
}

# EnvB_v2 (20 seeds each)
ENVB_SW = {"bc":(0.6900,0.0519), "iql":(0.6960,0.0562), "cql":(0.6950,0.0569)}
ENVB_LN = {"bc":(0.6700,0.0000), "iql":(0.6700,0.0000), "cql":(0.6700,0.0000)}

# EnvC_v2 (20 seeds each)
ENVC_SW = {"bc":(0.6800,0.0205), "iql":(0.6840,0.0201), "cql":(0.6620,0.0089)}
ENVC_LN = {"bc":(0.6600,0.0000), "iql":(0.6600,0.0000), "cql":(0.6600,0.0000)}

# Quality sweep
QUALITY = {
    "random":     {"bc":(-1.0000,0.0), "cql":(-1.0000,0.0),    "iql":(-1.0000,0.0)},
    "suboptimal": {"bc":(0.3930,0.2), "cql":(0.0505,0.622),  "iql":(0.4020,0.0098)},
    "medium":     {"bc":(0.4000,0.2), "cql":(0.3315,0.185),  "iql":(0.4020,0.0098)},
    "expert":     {"bc":(0.3960,0.2), "cql":(0.3890,0.030),  "iql":(0.4000,0.0098)},
    "mixed":      {"bc":(0.3930,0.2), "cql":(0.3930,0.030),  "iql":(0.3930,0.0098)},
}
QUALITY_SA = {
    "random":     0.382,
    "suboptimal": 0.208,
    "medium":     0.208,
    "expert":     0.152,
    "mixed":      0.425,
}

# Statistical closure
STAT_FACTS = {
    "sw_vs_ln": {
        "bc":  {"delta":0.0565, "test":"MWU",    "p":1.13e-7,  "verdict":"CLOSED"},
        "cql": {"delta":0.1270, "test":"Welch-t","p":7.47e-23, "verdict":"CLOSED"},
        "iql": {"delta":0.1270, "test":"Welch-t","p":7.47e-23, "verdict":"CLOSED"},
    },
    "sn_vs_ln": {
        "bc":  {"delta":0.0000, "test":"Exact=", "p":None,      "verdict":"CLOSED (=0)"},
        "cql": {"delta":0.0000, "test":"Exact=", "p":None,      "verdict":"CLOSED (=0)"},
        "iql": {"delta":0.0000, "test":"Exact=", "p":None,      "verdict":"CLOSED (=0)"},
    },
    "sw_vs_lw": {
        "bc":  {"delta":0.0745, "test":"MWU",    "p":9.32e-2,  "verdict":"DIRECTIONAL"},
        "cql": {"delta":0.0020, "test":"Welch-t","p":5.31e-1,  "verdict":"NULL"},
        "iql": {"delta":0.0050, "test":"Welch-t","p":1.19e-1,  "verdict":"DIRECTIONAL"},
    },
    "wide_vs_narrow": {
        "bc":  {"delta":0.0937, "test":"MWU",    "p":1.84e-15, "verdict":"CLOSED"},
        "cql": {"delta":0.1280, "test":"Welch-t","p":3.38e-45, "verdict":"CLOSED"},
        "iql": {"delta":0.1295, "test":"Welch-t","p":4.52e-45, "verdict":"CLOSED"},
    },
}

# Mechanism summary (from envA_v2_mechanism_summary.csv — 20-seed aggregates)
MECH_SA_COV = {
    "small_wide":   0.2052,
    "small_narrow": 0.0601,
    "large_wide":   0.2123,
    "large_narrow": 0.0601,
}
MECH_RETURNS = {  # mean returns
    "small_wide":   {"bc":0.3265,"cql":0.3970,"iql":0.3970},
    "small_narrow": {"bc":0.2700,"cql":0.2700,"iql":0.2700},
    "large_wide":   {"bc":0.4010,"cql":0.3990,"iql":0.4020},
    "large_narrow": {"bc":0.2700,"cql":0.2700,"iql":0.2700},
}

# Hopper benchmark
HOPPER = {
    "hopper-medium":        {"bc":42.51,"cql":3.17, "iql":30.64,"td3bc":44.46},
    "hopper-medium-replay": {"bc":19.79,"cql":26.65,"iql":30.68,"td3bc":22.03},
    "hopper-medium-expert": {"bc":46.98,"cql":1.32, "iql":28.16,"td3bc":63.11},
}

# Route families
ROUTEFAM_B2 = {
    "bc":  {"wide":{"A":12,"B":4,"C":4}, "narrow":{"A":20,"B":0,"C":0}},
    "iql": {"wide":{"A":11,"B":5,"C":4}, "narrow":{"A":20,"B":0,"C":0}},
    "cql": {"wide":{"A":10,"B":5,"C":5}, "narrow":{"A":20,"B":0,"C":0}},
}
ROUTEFAM_C2 = {
    "bc":  {"wide":{"LU":5,"LD":5,"RU":6,"RD":4}, "narrow":{"LU":20,"LD":0,"RU":0,"RD":0}},
    "iql": {"wide":{"LU":5,"LD":3,"RU":6,"RD":6}, "narrow":{"LU":20,"LD":0,"RU":0,"RD":0}},
    "cql": {"wide":{"LU":14,"LD":5,"RU":0,"RD":1},"narrow":{"LU":20,"LD":0,"RU":0,"RD":0}},
}

print("=" * 70)
print("Generating final canonical figure suite (12 figures)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# M1 — EnvA_v2 Factorial Overview
# ═══════════════════════════════════════════════════════════════════════════

def make_M1():
    algos = ["bc","cql","iql"]
    conds = ["small_wide","small_narrow","large_wide","large_narrow"]
    cond_labels = ["Small-Wide\n(50k, 21%)", "Small-Narrow\n(50k, 6%)",
                   "Large-Wide\n(200k, 21%)", "Large-Narrow\n(200k, 6%)"]
    cond_colors = [C_SW, C_SN, C_LW, C_LN]

    N_SEEDS = 20
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    fig.suptitle("EnvA_v2 — 2×2 Factorial Design: Mean Return per Dataset Condition",
                 fontsize=12, fontweight="bold", y=1.02)

    bar_w = 0.18
    x = np.arange(len(conds))

    for ax, algo in zip(axes, algos):
        for i, (cond, col) in enumerate(zip(conds, cond_colors)):
            mean, std = ENVA[cond][algo]
            lo, hi = ci95(mean, std, N_SEEDS)
            err = [[mean - lo], [hi - mean]]
            ax.bar(i, mean, width=0.62, color=col, alpha=0.85,
                   edgecolor="white", linewidth=0.5,
                   yerr=err, error_kw=dict(ecolor="#333", capsize=4, linewidth=1.4))
            ax.text(i, max(hi, mean) + 0.015, f"{mean:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(ALGO_LABELS[algo], fontsize=11, fontweight="bold",
                     color=ALGO_COLORS[algo])
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(cond_labels, fontsize=8)
        ax.set_ylim(-0.05, 0.60)
        ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")
        if algo == "bc":
            ax.set_ylabel("Mean Return (50 eval eps, 20 seeds)", fontsize=9)
        add_status_label(ax, "Primary Evidence")

    # SA coverage legend
    legend_patches = [
        mpatches.Patch(facecolor=C_SW, label="Small-Wide (50k, ~21% SA)"),
        mpatches.Patch(facecolor=C_SN, label="Small-Narrow (50k, ~6% SA)"),
        mpatches.Patch(facecolor=C_LW, label="Large-Wide (200k, ~21% SA)"),
        mpatches.Patch(facecolor=C_LN, label="Large-Narrow (200k, ~6% SA)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08), fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(fmain("fig01_envA_factorial_overview.png"))
    plt.close()
    print("  M1 saved")

make_M1()

# ═══════════════════════════════════════════════════════════════════════════
# M2 — Primary contrast SW vs LN
# ═══════════════════════════════════════════════════════════════════════════

def make_M2():
    algos = ["bc","cql","iql"]
    N = 20
    fig, ax = plt.subplots(figsize=(7, 4.5))

    sw_vals = [(ENVA["small_wide"][a][0], ENVA["small_wide"][a][1]) for a in algos]
    ln_vals = [(ENVA["large_narrow"][a][0], ENVA["large_narrow"][a][1]) for a in algos]

    x = np.arange(len(algos))
    w = 0.32

    for i, algo in enumerate(algos):
        sw_m, sw_s = sw_vals[i]; ln_m, ln_s = ln_vals[i]
        sw_lo, sw_hi = ci95(sw_m, sw_s, N);  ln_lo, ln_hi = ci95(ln_m, ln_s, N)
        ax.bar(i - w/2, sw_m, w, color=C_SW, alpha=0.9, label="Small-Wide" if i==0 else "",
               yerr=[[sw_m-sw_lo],[sw_hi-sw_m]], error_kw=dict(ecolor="#333",capsize=5,lw=1.5))
        ax.bar(i + w/2, ln_m, w, color=C_LN, alpha=0.9, label="Large-Narrow" if i==0 else "",
               yerr=[[ln_m-ln_lo],[ln_hi-ln_m]], error_kw=dict(ecolor="#333",capsize=5,lw=1.5))
        gap = sw_m - ln_m
        y_top = max(sw_hi, ln_hi) + 0.02
        ax.annotate("", xy=(i + w/2, sw_m), xytext=(i - w/2, ln_m),
                    arrowprops=dict(arrowstyle="-|>", color=ALGO_COLORS[algo], lw=1.8))
        ax.text(i, y_top, f"Δ=+{gap:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=ALGO_COLORS[algo])
        ax.text(i - w/2, sw_m + 0.01, f"{sw_m:.3f}", ha="center", va="bottom",
                fontsize=8.5, color="#1A237E")
        ax.text(i + w/2, ln_m + 0.01, f"{ln_m:.3f}", ha="center", va="bottom",
                fontsize=8.5, color="#B71C1C")

    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in algos], fontsize=11)
    ax.set_ylim(0.0, 0.62)
    ax.set_ylabel("Mean Return (50 eval eps, 20 seeds)", fontsize=10)
    ax.set_title("Primary Contrast: Small-Wide vs Large-Narrow\n"
                 "( 4× less data, 3.5× more SA coverage → higher returns )",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.99, 0.97,
            "Small-Wide: 50k transitions, ~21% SA\nLarge-Narrow: 200k transitions, ~6% SA",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.9))
    add_status_label(ax, "Primary Evidence")
    plt.tight_layout()
    fig.savefig(fmain("fig02_envA_primary_contrast_sw_vs_ln.png"))
    plt.close()
    print("  M2 saved")

make_M2()

# ═══════════════════════════════════════════════════════════════════════════
# M3 — Size effect decomposition
# ═══════════════════════════════════════════════════════════════════════════

def make_M3():
    algos = ["bc","cql","iql"]
    N = 20
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("EnvA_v2 — Size Effect Decomposition: Is More Data Better?",
                 fontsize=12, fontweight="bold", y=1.02)

    # Panel A: SN → LN (fixed narrow coverage)
    gaps_narrow = [ENVA["large_narrow"][a][0] - ENVA["small_narrow"][a][0] for a in algos]
    colors_narrow = [ALGO_COLORS[a] for a in algos]
    bars = ax1.bar([ALGO_LABELS[a] for a in algos], gaps_narrow, color=colors_narrow,
                   alpha=0.85, width=0.5, edgecolor="white")
    ax1.axhline(0, color="#444", linewidth=1.2, linestyle="--")
    for bar, g in zip(bars, gaps_narrow):
        ax1.text(bar.get_x()+bar.get_width()/2, 0.002, f"Δ={g:+.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold",
                 color="#B71C1C")
    ax1.set_title("Panel A: Fixed Narrow Coverage\n50k → 200k (both ~6% SA)",
                  fontsize=10, fontweight="bold")
    ax1.set_ylabel("Return improvement (LN − SN)", fontsize=9)
    ax1.set_ylim(-0.05, 0.15)
    ax1.text(0.5, 0.80, "EXACT ZERO\nfor all algorithms",
             transform=ax1.transAxes, ha="center", va="center",
             fontsize=11, color="#B71C1C", fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="#FFEBEE", alpha=0.9))
    add_status_label(ax1, "Primary Evidence")

    # Panel B: SW → LW (fixed wide coverage)
    gaps_wide = [ENVA["large_wide"][a][0] - ENVA["small_wide"][a][0] for a in algos]
    bars2 = ax2.bar([ALGO_LABELS[a] for a in algos], gaps_wide, color=colors_narrow,
                    alpha=0.85, width=0.5, edgecolor="white")
    ax2.axhline(0, color="#444", linewidth=1.2, linestyle="--")
    for bar, g, algo in zip(bars2, gaps_wide, algos):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 (g + 0.004) if g >= 0 else (g - 0.008),
                 f"Δ={g:+.3f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=ALGO_COLORS[algo])
    ax2.set_title("Panel B: Fixed Wide Coverage\n50k → 200k (both ~21% SA)",
                  fontsize=10, fontweight="bold")
    ax2.set_ylabel("Return improvement (LW − SW)", fontsize=9)
    ax2.set_ylim(-0.02, 0.18)
    note = ("BC: modest size effect\n(bimodal convergence)\n"
            "CQL/IQL: negligible (< 0.01)")
    ax2.text(0.97, 0.96, note, transform=ax2.transAxes, ha="right", va="top",
             fontsize=8.5, bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.9))
    add_status_label(ax2, "Primary Evidence")

    plt.tight_layout()
    fig.savefig(fmain("fig03_envA_size_effect_decomposition.png"))
    plt.close()
    print("  M3 saved")

make_M3()

# ═══════════════════════════════════════════════════════════════════════════
# M4 — Statistical closure matrix
# ═══════════════════════════════════════════════════════════════════════════

def make_M4():
    comparisons = ["sw_vs_ln", "sn_vs_ln", "sw_vs_lw", "wide_vs_narrow"]
    comp_labels  = ["SW vs LN\n(primary contrast)",
                    "SN vs LN\n(size under narrow)",
                    "SW vs LW\n(size under wide)",
                    "Wide vs Narrow\n(pooled)"]
    algos = ["bc","cql","iql"]

    verdict_colors = {
        "CLOSED":      "#2E7D32",
        "CLOSED (=0)": "#2E7D32",
        "DIRECTIONAL": "#F57F17",
        "NULL":        "#757575",
    }

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(-0.5, len(comparisons) - 0.5)
    ax.set_ylim(-0.5, len(algos) - 0.5)

    for ci, comp in enumerate(comparisons):
        for ai, algo in enumerate(algos):
            f = STAT_FACTS[comp][algo]
            delta = f["delta"]; p = f["p"]
            test  = f["test"];  verdict = f["verdict"]
            vcol  = verdict_colors.get(verdict, "#999")
            # Cell background
            ax.add_patch(plt.Rectangle((ci-0.48, ai-0.48), 0.96, 0.96,
                         facecolor=vcol, alpha=0.18, zorder=0))
            ax.add_patch(plt.Rectangle((ci-0.48, ai-0.48), 0.96, 0.96,
                         fill=False, edgecolor=vcol, linewidth=1.2))
            # Text
            p_str = f"p={p:.2e}" if p is not None else "exact = 0"
            cell_text = f"Δ={delta:+.3f}\n{test}\n{p_str}\n{verdict}"
            ax.text(ci, ai, cell_text, ha="center", va="center",
                    fontsize=8.2, fontweight="bold" if "CLOSED" in verdict else "normal",
                    color=vcol if "CLOSED" in verdict else "#333",
                    linespacing=1.4)

    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels(comp_labels, fontsize=9)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels([ALGO_LABELS[a] for a in algos], fontsize=10)
    for ai, algo in enumerate(algos):
        ax.get_yticklabels()[ai].set_color(ALGO_COLORS[algo])
        ax.get_yticklabels()[ai].set_fontweight("bold")

    ax.set_title("EnvA_v2 — Statistical Closure Matrix (4 Comparisons × 3 Algorithms)",
                 fontsize=11, fontweight="bold")
    ax.grid(False)

    # Legend
    leg_patches = [
        mpatches.Patch(facecolor="#2E7D32", alpha=0.4, label="CLOSED (formally supported)"),
        mpatches.Patch(facecolor="#F57F17", alpha=0.4, label="DIRECTIONAL (trend, p>0.05)"),
        mpatches.Patch(facecolor="#757575", alpha=0.4, label="NULL (no effect confirmed)"),
    ]
    ax.legend(handles=leg_patches, loc="upper right", fontsize=8.5,
              bbox_to_anchor=(1.0, -0.12), ncol=3)
    add_status_label(ax, "Primary Evidence")
    plt.tight_layout()
    fig.savefig(fmain("fig04_statistical_closure_matrix.png"))
    plt.close()
    print("  M4 saved")

make_M4()

# ═══════════════════════════════════════════════════════════════════════════
# M5 — Mechanism (3-panel)
# ═══════════════════════════════════════════════════════════════════════════

def make_M5():
    fig = plt.figure(figsize=(13, 4.0))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    fig.suptitle("EnvA_v2 — Coverage Effect Mechanism Analysis",
                 fontsize=12, fontweight="bold", y=1.02)

    # Panel A: SA coverage vs return scatter
    for cond, sa_cov in MECH_SA_COV.items():
        for algo in ["bc","cql","iql"]:
            ret = MECH_RETURNS[cond][algo]
            marker = {"small_wide":"o","small_narrow":"^",
                      "large_wide":"s","large_narrow":"D"}[cond]
            ax1.scatter(sa_cov, ret, color=ALGO_COLORS[algo],
                        marker=marker, s=90, alpha=0.85, zorder=3,
                        edgecolors="white", linewidths=0.8)

    ax1.set_xlabel("Dataset SA Coverage", fontsize=9.5)
    ax1.set_ylabel("Mean Policy Return", fontsize=9.5)
    ax1.set_title("Panel A: SA Coverage\nvs Policy Return", fontsize=10)
    ax1.set_xlim(-0.01, 0.28); ax1.set_ylim(0.20, 0.45)

    # Create proxy artists for legend
    cond_markers = [
        plt.Line2D([0],[0],marker="o",linestyle="",color="#555",ms=7,label="Small-Wide"),
        plt.Line2D([0],[0],marker="^",linestyle="",color="#555",ms=7,label="Small-Narrow"),
        plt.Line2D([0],[0],marker="s",linestyle="",color="#555",ms=7,label="Large-Wide"),
        plt.Line2D([0],[0],marker="D",linestyle="",color="#555",ms=7,label="Large-Narrow"),
    ]
    algo_patches = [mpatches.Patch(facecolor=ALGO_COLORS[a],label=ALGO_LABELS[a])
                    for a in ["bc","cql","iql"]]
    ax1.legend(handles=cond_markers + algo_patches, fontsize=7.5,
               ncol=2, loc="upper left")
    add_status_label(ax1, "Primary Evidence")

    # Panel B: OOD rate = 0 summary
    conditions = ["Small-Wide", "Small-Narrow", "Large-Wide", "Large-Narrow"]
    ood_rates   = [0.0, 0.0, 0.0, 0.0]
    bar_colors  = [C_SW, C_SN, C_LW, C_LN]
    ax2.bar(conditions, ood_rates, color=bar_colors, alpha=0.85, width=0.55,
            edgecolor="white")
    ax2.set_ylim(0, 0.10)
    ax2.set_ylabel("Mean OOD Action Rate (eval)", fontsize=9.5)
    ax2.set_title("Panel B: OOD Action Rate\nAll Conditions = 0.000", fontsize=10)
    ax2.tick_params(axis="x", labelsize=8)
    ax2.text(0.5, 0.62,
             "OOD = 0 in ALL conditions\nMechanism is NOT\ndistribution-shift prevention",
             transform=ax2.transAxes, ha="center", va="center",
             fontsize=9.5, fontweight="bold", color="#1A237E",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8EAF6", alpha=0.9))
    add_status_label(ax2, "Primary Evidence")

    # Panel C: Strategy-family access summary
    ax3.axis("off")
    summary_text = (
        "Panel C: Coverage → Strategy Access\n\n"
        "Narrow dataset (~6% SA):\n"
        "  → Single behavioral family\n"
        "  → Policy locked to one strategy\n"
        "  → Performance ceiling = that strategy\n\n"
        "Wide dataset (~21% SA):\n"
        "  → Multiple behavioral families\n"
        "  → Value-based algos can compare\n"
        "  → Policy finds better-return strategy\n\n"
        "NOT about avoiding OOD states\n"
        "(OOD rate = 0 everywhere)\n"
        "→ Behavioral diversity is the mechanism"
    )
    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
             fontsize=9.2, va="top", ha="left", linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8E1", alpha=0.95,
                       edgecolor="#F9A825"))
    ax3.set_title("Panel C: Mechanism Summary", fontsize=10)

    plt.tight_layout()
    fig.savefig(fmain("fig05_mechanism_behavioral_diversity.png"))
    plt.close()
    print("  M5 saved")

make_M5()

# ═══════════════════════════════════════════════════════════════════════════
# M6 — Rebuilt validation gap summary
# ═══════════════════════════════════════════════════════════════════════════

def make_M6():
    algos = ["bc","iql","cql"]
    N = 20
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    fig.suptitle("Rebuilt Validation Environments — Coverage Effect Gaps (SW − LN)",
                 fontsize=12, fontweight="bold", y=1.02)

    def plot_env_gaps(ax, sw_data, ln_data, env_name, env_status):
        x = np.arange(len(algos)); w = 0.52
        for i, algo in enumerate(algos):
            sw_m, sw_s = sw_data[algo]; ln_m, ln_s = ln_data[algo]
            gap = sw_m - ln_m
            sw_lo, sw_hi = ci95(sw_m, sw_s, N); ln_lo, ln_hi = ci95(ln_m, ln_s, N)
            # Check CI overlap
            ci_overlap = sw_lo < ln_hi
            bar_alpha = 0.9 if not ci_overlap else 0.55
            ax.bar(i, gap, w, color=ALGO_COLORS[algo], alpha=bar_alpha,
                   edgecolor="white", linewidth=0.8)
            # Annotate
            lbl = f"+{gap:.3f}"
            style = "bold" if not ci_overlap else "normal"
            color = ALGO_COLORS[algo]
            ax.text(i, gap + 0.001, lbl, ha="center", va="bottom",
                    fontsize=10.5, fontweight=style, color=color)
            # CI overlap marker
            if ci_overlap:
                ax.text(i, gap/2, "CI\noverlap", ha="center", va="center",
                        fontsize=7, color="#555", style="italic")

        ax.axhline(0, color="#555", linewidth=1.2, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_LABELS[a] for a in algos], fontsize=11)
        ax.set_ylabel("Return gap (SW mean − LN mean)", fontsize=9.5)
        ax.set_title(f"{env_name}\n{env_status}", fontsize=10, fontweight="bold")
        ax.set_ylim(-0.003, 0.045)
        add_status_label(ax, "Rebuilt Validation")

    plot_env_gaps(ax1, ENVB_SW, ENVB_LN,
                  "EnvB_v2 (Three-Corridor, 270 states)",
                  "✓ 3/3 CONFIRMED — BC, IQL, CQL all positive")
    plot_env_gaps(ax2, ENVC_SW, ENVC_LN,
                  "EnvC_v2 (Key-Door Staged, 269 ext. states)",
                  "⚠  BC + IQL confirmed; CQL gap=+0.002 MIXED")

    # Add CQL annotation for EnvC
    ax2.annotate("CQL: gap=+0.002\n(CI overlapping;\n1/20 seeds;\nMIXED evidence)",
                 xy=(2, 0.002), xytext=(2.25, 0.018),
                 fontsize=8, color=C_CQL,
                 arrowprops=dict(arrowstyle="->", color=C_CQL, lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEBEE", alpha=0.9))

    plt.tight_layout()
    fig.savefig(fmain("fig06_rebuilt_validation_gap_summary.png"))
    plt.close()
    print("  M6 saved")

make_M6()

# ═══════════════════════════════════════════════════════════════════════════
# M7 — Route-family convergence grid
# ═══════════════════════════════════════════════════════════════════════════

def make_M7():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("Route-Family Convergence: Wide vs Narrow Datasets (20 seeds each)",
                 fontsize=12, fontweight="bold", y=1.02)

    envs  = [("EnvB_v2", ROUTEFAM_B2, ["A","B","C"], ["#E53935","#1565C0","#2E7D32"]),
             ("EnvC_v2", ROUTEFAM_C2, ["LU","LD","RU","RD"], ["#1565C0","#90CAF9","#2E7D32","#A5D6A7"])]
    algos = [("bc", "BC"), ("iql", "IQL"), ("cql", "CQL")]

    env_status_notes = {
        "EnvB_v2": "3/3 confirmed",
        "EnvC_v2": "BC+IQL confirmed; CQL mixed",
    }
    cql_note = {
        "EnvB_v2": "",
        "EnvC_v2": "CQL: conservative penalty\nsuppresses short-path discovery",
    }

    for row, (env_name, rfam_data, families, fam_colors) in enumerate(envs):
        for col, (algo, algo_label) in enumerate(algos):
            ax = axes[row][col]
            wide_d  = rfam_data[algo]["wide"]
            narrow_d = rfam_data[algo]["narrow"]

            w_counts = [wide_d.get(f, 0) for f in families]
            n_counts = [narrow_d.get(f, 0) for f in families]
            w_total  = sum(w_counts); n_total = sum(n_counts)
            w_fracs  = [c/w_total for c in w_counts]
            n_fracs  = [c/n_total for c in n_counts]

            x = np.array([0, 1])
            bottoms = [0.0, 0.0]
            for fi, (fam, col_f) in enumerate(zip(families, fam_colors)):
                wf = w_fracs[fi]; nf = n_fracs[fi]
                ax.bar(0, wf, 0.6, bottom=bottoms[0], color=col_f, alpha=0.85,
                       edgecolor="white", linewidth=0.5)
                ax.bar(1, nf, 0.6, bottom=bottoms[1], color=col_f, alpha=0.85,
                       edgecolor="white", linewidth=0.5)
                # Label if significant
                if wf > 0.08:
                    ax.text(0, bottoms[0]+wf/2, f"{fam}\n{w_counts[fi]}",
                            ha="center", va="center", fontsize=7.5, fontweight="bold",
                            color="white" if wf > 0.15 else "#333")
                if nf > 0.08:
                    ax.text(1, bottoms[1]+nf/2, f"{fam}\n{n_counts[fi]}",
                            ha="center", va="center", fontsize=7.5, fontweight="bold",
                            color="white" if nf > 0.15 else "#333")
                bottoms[0] += wf; bottoms[1] += nf

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Wide\n(50k)", "Narrow\n(200k)"], fontsize=9)
            ax.set_ylim(0, 1.12)
            ax.set_ylabel("Fraction of seeds", fontsize=8.5)

            title = f"{env_name} — {algo_label}"
            if algo == "cql" and env_name == "EnvC_v2":
                title += "\n⚠ MIXED"
                ax.set_facecolor("#FFF8E1")
            ax.set_title(title, fontsize=9, fontweight="bold",
                         color=ALGO_COLORS[algo])

            # CQL annotation
            if algo == "cql" and env_name == "EnvC_v2":
                ax.text(0.5, 0.07, "CQL wide: 14/20 LU\n(barely different from narrow)",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=7.5, color=C_CQL, style="italic",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEBEE", alpha=0.85))

    # Env status labels on the side
    for row, (env_name, _, _, _) in enumerate(envs):
        axes[row][2].text(1.04, 0.5, env_status_notes[env_name],
                          transform=axes[row][2].transAxes,
                          fontsize=8.5, ha="left", va="center", rotation=90,
                          color="#333",
                          bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.8))

    plt.tight_layout()
    fig.savefig(fmain("fig07_rebuilt_route_family_convergence.png"))
    plt.close()
    print("  M7 saved")

make_M7()

# ═══════════════════════════════════════════════════════════════════════════
# A1 — Original EnvB/C boundary conditions
# ═══════════════════════════════════════════════════════════════════════════

def make_A1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.suptitle("Original Single-Path Environments — Boundary Conditions (Zero Coverage Effect)",
                 fontsize=11, fontweight="bold", y=1.02)

    algos = ["bc","cql","iql"]

    # Panel A: EnvB — structural saturation
    envb_means = {"wide": 0.760, "narrow": 0.760}  # From final_validation_results_table.csv
    x = [0, 1]; labels = ["EnvB Wide", "EnvB Narrow"]
    colors_ab = [C_SW, C_LN]
    ax1.bar(x, [envb_means["wide"], envb_means["narrow"]], 0.55,
            color=colors_ab, alpha=0.85, edgecolor="white")
    ax1.text(0, envb_means["wide"]+0.01, f"{envb_means['wide']:.3f}", ha="center",
             fontsize=10, fontweight="bold", color="#1A237E")
    ax1.text(1, envb_means["narrow"]+0.01, f"{envb_means['narrow']:.3f}", ha="center",
             fontsize=10, fontweight="bold", color="#B71C1C")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0.5, 0.90); ax1.set_ylabel("Mean Return", fontsize=9.5)
    ax1.set_title("Panel A: EnvB (Double-Bottleneck)\nGap = 0.000 (structural saturation)",
                  fontsize=10, fontweight="bold")
    ax1.text(0.5, 0.20,
             "SA coverage:\nWide ≈ 99.3%\nNarrow ≈ 99.3%\n→ No contrast possible",
             transform=ax1.transAxes, ha="center", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="#FFEBEE", alpha=0.9))
    add_status_label(ax1, "Boundary Condition")

    # Panel B: EnvC — above threshold
    x2 = [0, 1]; labels2 = ["EnvC Wide", "EnvC Narrow"]
    envbc_cov = [0.984, 0.885]
    envc_means = {"wide": 0.760, "narrow": 0.760}
    ax2b = ax2.twinx()
    ax2.bar(x2, [envc_means["wide"], envc_means["narrow"]], 0.55,
            color=[C_SW, C_LN], alpha=0.85, edgecolor="white")
    ax2b.plot(x2, envbc_cov, "k--o", linewidth=1.5, markersize=7, label="SA Coverage")
    ax2b.set_ylabel("Norm. SA Coverage", fontsize=9)
    ax2b.set_ylim(0.8, 1.02)
    ax2b.legend(loc="lower right", fontsize=8.5)
    for xi, (lbl, r) in enumerate(zip(labels2, [envc_means["wide"], envc_means["narrow"]])):
        ax2.text(xi, r+0.01, f"{r:.3f}", ha="center", fontsize=10, fontweight="bold",
                 color="#1A237E" if xi==0 else "#B71C1C")
    ax2.set_xticks(x2); ax2.set_xticklabels(labels2, fontsize=10)
    ax2.set_ylim(0.5, 0.90); ax2.set_ylabel("Mean Return", fontsize=9.5)
    ax2.set_title("Panel B: EnvC (Key-Door)\n10pp coverage gap, yet Gap = 0.000",
                  fontsize=10, fontweight="bold")
    ax2.text(0.5, 0.20,
             "Coverage gap exists (~10pp)\nbut both regimes are ABOVE\neffective support threshold",
             transform=ax2.transAxes, ha="center", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="#FFF8E1", alpha=0.9))
    add_status_label(ax2, "Boundary Condition")

    plt.tight_layout()
    fig.savefig(faux("fig08_original_envbc_boundary_conditions.png"))
    plt.close()
    print("  A1 saved")

make_A1()

# ═══════════════════════════════════════════════════════════════════════════
# A2 — Quality sweep with coverage caveat
# ═══════════════════════════════════════════════════════════════════════════

def make_A2():
    bins = ["random","suboptimal","medium","expert","mixed"]
    bin_labels = ["Random\n(38% SA)", "Suboptimal\n(21% SA)", "Medium\n(21% SA)",
                  "Expert\n(15% SA)", "Mixed\n(43% SA)"]
    algos = ["bc","cql","iql"]
    N = 20

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), gridspec_kw={"hspace":0.05})
    fig.suptitle("Quality Sweep — Performance Across Data Quality Levels\n"
                 "(⚠ Coverage also varies across bins — not a clean quality-only isolation)",
                 fontsize=11, fontweight="bold", y=1.03)

    x = np.arange(len(bins)); w = 0.25
    for ai, algo in enumerate(algos):
        means = [QUALITY[b][algo][0] for b in bins]
        stds  = [QUALITY[b][algo][1] for b in bins]
        yerr_lo = []; yerr_hi = []
        for m, s in zip(means, stds):
            lo, hi = ci95(m, s, N)
            yerr_lo.append(m - lo); yerr_hi.append(hi - m)
        offset = (ai - 1) * w
        bars = ax1.bar(x + offset, means, w, color=ALGO_COLORS[algo],
                       label=ALGO_LABELS[algo], alpha=0.85, edgecolor="white",
                       yerr=[yerr_lo, yerr_hi],
                       error_kw=dict(ecolor="#333", capsize=3, lw=1.2))

    ax1.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax1.axhspan(-1.05, -0.7, alpha=0.06, color="#7B1FA2")
    ax1.set_xticks(x); ax1.set_xticklabels([""]*len(bins))
    ax1.set_ylim(-1.10, 0.55)
    ax1.set_ylabel("Mean Return", fontsize=9.5)
    ax1.legend(loc="upper right", fontsize=9.5)
    ax1.text(0, -0.85, "RANDOM =\nFAILURE FLOOR",
             ha="center", va="center", fontsize=8.5, fontweight="bold", color="#7B1FA2",
             bbox=dict(boxstyle="round", facecolor="#F3E5F5", alpha=0.9))
    add_status_label(ax1, "Auxiliary Evidence")

    # SA coverage overlay
    sa_covs = [QUALITY_SA[b] for b in bins]
    ax2.bar(x, sa_covs, 0.55, color=[QUAL_COLORS[b] for b in bins], alpha=0.80, edgecolor="white")
    ax2.set_xticks(x); ax2.set_xticklabels(bin_labels, fontsize=9)
    ax2.set_ylabel("SA Coverage\n(confound)", fontsize=9.5)
    ax2.set_ylim(0, 0.55)
    for xi, s in enumerate(sa_covs):
        ax2.text(xi, s + 0.01, f"{s:.2f}", ha="center", fontsize=8.5)
    ax2.text(0.99, 0.93,
             "⚠ Coverage not controlled\nacross quality bins",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8.5,
             color="#B71C1C",
             bbox=dict(boxstyle="round", facecolor="#FFEBEE", alpha=0.9))
    add_status_label(ax2, "Auxiliary Evidence")

    plt.tight_layout()
    fig.savefig(faux("fig09_quality_sweep_with_coverage_caveat.png"))
    plt.close()
    print("  A2 saved")

make_A2()

# ═══════════════════════════════════════════════════════════════════════════
# A3 — Dataset audit overview
# ═══════════════════════════════════════════════════════════════════════════

def make_A3():
    datasets = [
        # (label, transitions, sa_cov, track, env)
        ("envA_v2\nsmall-wide",   50_000,  0.2052, "main",       "EnvA_v2"),
        ("envA_v2\nsmall-narrow", 50_000,  0.0601, "main",       "EnvA_v2"),
        ("envA_v2\nlarge-wide",   200_000, 0.2123, "main",       "EnvA_v2"),
        ("envA_v2\nlarge-narrow", 200_000, 0.0601, "main",       "EnvA_v2"),
        ("quality\nrandom",       50_000,  0.3817, "quality",    "EnvA_v2"),
        ("quality\nsuboptimal",   50_000,  0.2078, "quality",    "EnvA_v2"),
        ("quality\nmedium",       50_000,  0.2078, "quality",    "EnvA_v2"),
        ("quality\nexpert",       50_000,  0.1515, "quality",    "EnvA_v2"),
        ("quality\nmixed",        50_000,  0.4246, "quality",    "EnvA_v2"),
        ("EnvB_v2\nwide",         50_000,  0.2407, "validation", "EnvB_v2"),
        ("EnvB_v2\nnarrow-A",     200_000, 0.0852, "validation", "EnvB_v2"),
        ("EnvC_v2\nwide",         50_000,  0.3281, "validation", "EnvC_v2"),
        ("EnvC_v2\nnarrow-LU",    200_000, 0.0874, "validation", "EnvC_v2"),
    ]
    track_colors = {"main": "#1565C0", "quality": "#F57F17", "validation": "#2E7D32"}
    env_markers  = {"EnvA_v2": "o", "EnvB_v2": "s", "EnvC_v2": "D"}
    env_sizes    = {"EnvA_v2": 100, "EnvB_v2": 110, "EnvC_v2": 110}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    handles = []
    for label, trans, sa, track, env in datasets:
        sc = ax.scatter(sa, trans/1000, s=env_sizes[env],
                        color=track_colors[track], marker=env_markers[env],
                        alpha=0.85, edgecolors="white", linewidths=0.8, zorder=3)

    ax.set_xlabel("Normalized SA Coverage", fontsize=10)
    ax.set_ylabel("Dataset Size (thousands of transitions)", fontsize=10)
    ax.set_title("Dataset Audit Overview — All Datasets by Coverage and Size",
                 fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlim(-0.01, 0.50); ax.set_ylim(20, 800)

    # Annotate key points
    for label, trans, sa, track, env in datasets:
        ax.annotate(label, (sa, trans/1000), fontsize=7,
                    xytext=(4, 3), textcoords="offset points",
                    color=track_colors[track], alpha=0.85)

    # Track legend
    track_handles = [mpatches.Patch(facecolor=c, label=t.capitalize())
                     for t, c in track_colors.items()]
    env_handles = [plt.Line2D([0],[0], marker=m, linestyle="", color="#555",
                              ms=8, label=e) for e, m in env_markers.items()]
    ax.legend(handles=track_handles+env_handles, loc="upper right",
              fontsize=8.5, ncol=2)
    add_status_label(ax, "Auxiliary / Audit")
    plt.tight_layout()
    fig.savefig(faux("fig10_dataset_audit_overview.png"))
    plt.close()
    print("  A3 saved")

make_A3()

# ═══════════════════════════════════════════════════════════════════════════
# A4 — Visitation heatmaps (load existing pilot PNGs + compose)
# ═══════════════════════════════════════════════════════════════════════════

def make_A4():
    from PIL import Image
    pilot_dir = artdir("pilot_envbc_v2")
    available = os.listdir(pilot_dir)

    hm_files = {
        "envB_v2 Wide vs Narrow": "envB_v2_route_heatmap.png",
        "envC_v2 v2 Pre-door": "envC_v2_v2_route_heatmap_pre_v2.png",
        "envC_v2 v2 Post-door": "envC_v2_v2_route_heatmap_post_v2.png",
    }

    # Check what's available
    found = {k: os.path.join(pilot_dir, v) for k, v in hm_files.items()
             if v in available}

    if not found:
        # Fallback: create a placeholder figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Visitation heatmaps:\nSee artifacts/pilot_envbc_v2/ for existing PNG files.\n"
                "EnvB_v2: envB_v2_route_heatmap.png\n"
                "EnvC_v2: envC_v2_v2_route_heatmap_pre_v2.png / post_v2.png",
                transform=ax.transAxes, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.95))
        ax.axis("off")
        ax.set_title("Visitation Heatmaps (see pilot_envbc_v2/ directory)",
                     fontsize=11, fontweight="bold")
        add_status_label(ax, "Auxiliary / Appendix")
    else:
        n = len(found)
        fig, axes = plt.subplots(1, n, figsize=(7*n, 5))
        if n == 1: axes = [axes]
        for ax, (title, path) in zip(axes, found.items()):
            try:
                img = Image.open(path)
                ax.imshow(np.array(img))
                ax.set_title(title, fontsize=10, fontweight="bold")
            except Exception:
                ax.text(0.5, 0.5, f"Load error:\n{os.path.basename(path)}",
                        transform=ax.transAxes, ha="center", va="center")
            ax.axis("off")
        fig.suptitle("Visitation Frequency Heatmaps — Wide vs Narrow Datasets",
                     fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(faux("fig11_visitation_heatmaps.png"))
    plt.close()
    print("  A4 saved")

try:
    make_A4()
except ImportError:
    # PIL not available
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(0.5, 0.5, "Visitation heatmaps:\nSee artifacts/pilot_envbc_v2/ for existing PNG files.\n"
            "EnvB_v2: envB_v2_route_heatmap.png\n"
            "EnvC_v2 (pre/post-door): envC_v2_v2_route_heatmap_*_v2.png",
            transform=ax.transAxes, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.95))
    ax.axis("off")
    ax.set_title("Visitation Heatmaps (see pilot_envbc_v2/ directory)", fontsize=11)
    add_status_label(ax, "Auxiliary / Appendix")
    plt.tight_layout()
    fig.savefig(faux("fig11_visitation_heatmaps.png"))
    plt.close()
    print("  A4 saved (placeholder — PIL not available)")

# ═══════════════════════════════════════════════════════════════════════════
# A5 — Hopper benchmark appendix
# ═══════════════════════════════════════════════════════════════════════════

def make_A5():
    splits = ["hopper-medium", "hopper-medium-replay", "hopper-medium-expert"]
    split_labels = ["Hopper\nMedium", "Hopper\nMedium-Replay", "Hopper\nMedium-Expert"]
    algos_b = ["bc","cql","iql","td3bc"]
    algo_labels_b = {"bc":"BC","cql":"CQL","iql":"IQL","td3bc":"TD3+BC"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    fig.suptitle("Hopper D4RL Benchmark — 5-Seed Pilot (Appendix Only)\n"
                 "⚠ CQL anomaly: hyperparameter misconfiguration — excluded from all claims",
                 fontsize=11, fontweight="bold", y=1.03)

    published = {
        "hopper-medium":        {"bc":29,"cql":58,"iql":66,"td3bc":59},
        "hopper-medium-replay": {"bc":18,"cql":46,"iql":47,"td3bc":41},
        "hopper-medium-expert": {"bc":54,"cql":91,"iql":91,"td3bc":98},
    }

    for ax, split, slabel in zip(axes, splits, split_labels):
        x = np.arange(len(algos_b)); w = 0.38
        our_vals = [HOPPER[split][a] for a in algos_b]
        pub_vals  = [published[split][a] for a in algos_b]
        bars1 = ax.bar(x - w/2, our_vals, w, label="Our results (5-seed)",
                       color=[ALGO_COLORS[a] for a in algos_b], alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + w/2, pub_vals, w, label="Published baselines",
                       color=[ALGO_COLORS[a] for a in algos_b], alpha=0.35,
                       edgecolor=[ALGO_COLORS[a] for a in algos_b], linewidth=1.5,
                       linestyle="dashed", hatch="//")

        for bar, val, algo in zip(bars1, our_vals, algos_b):
            is_cql_anomaly = (algo == "cql" and val < 10)
            color = "#B71C1C" if is_cql_anomaly else "#333"
            ax.text(bar.get_x()+bar.get_width()/2, val+0.5, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8,
                    fontweight="bold" if is_cql_anomaly else "normal",
                    color=color)
            if is_cql_anomaly:
                ax.annotate("CQL\nanomaly", xy=(bar.get_x()+bar.get_width()/2, val),
                            xytext=(bar.get_x()+bar.get_width()/2+0.3, val+8),
                            fontsize=7.5, color="#B71C1C", fontweight="bold",
                            arrowprops=dict(arrowstyle="->",color="#B71C1C",lw=1.0))

        ax.set_title(slabel, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([algo_labels_b[a] for a in algos_b], fontsize=9)
        ax.set_ylabel("Normalized Score", fontsize=9)
        if split == splits[0]:
            ax.legend(fontsize=8, loc="upper left")
        add_status_label(ax, "Appendix")

    plt.tight_layout()
    fig.savefig(faux("fig12_hopper_benchmark_appendix.png"))
    plt.close()
    print("  A5 saved")

make_A5()

# ═══════════════════════════════════════════════════════════════════════════
# Task B — Compatibility aliases (old fig1–fig6 names)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Task B: Creating compatibility aliases ──")
compat_map = {
    "fig1_main_coverage_vs_size.png":          fmain("fig01_envA_factorial_overview.png"),
    "fig2_core_smallwide_vs_largenarrow.png":  fmain("fig02_envA_primary_contrast_sw_vs_ln.png"),
    "fig3_quality_modulation.png":             faux("fig09_quality_sweep_with_coverage_caveat.png"),
    "fig4_envbc_validation.png":               faux("fig08_original_envbc_boundary_conditions.png"),
    "fig5_mechanism_summary.png":              fmain("fig05_mechanism_behavioral_diversity.png"),
    "fig6_benchmark_validation.png":           faux("fig12_hopper_benchmark_appendix.png"),
}
for alias, source in compat_map.items():
    dest = fcompat(alias)
    if os.path.exists(source):
        shutil.copy2(source, dest)
        print(f"  {alias} → {os.path.basename(source)}")
    else:
        print(f"  WARNING: source not found for {alias}")

print()
print("=" * 70)
print("Figure suite complete.")
print(f"  Mainline (M1–M7): {FIGS_MAIN}")
print(f"  Auxiliary (A1–A5): {FIGS_AUX}")
print(f"  Compat aliases:   {FIGS_COMPAT}")
print("=" * 70)
