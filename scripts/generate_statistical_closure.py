"""
scripts/generate_statistical_closure.py
Statistical closure for the Offline RL coverage-vs-size study.

Reads existing experiment summary CSVs (no new training).
Computes 4 comparisons × 3 algorithms = 12 rows of hypothesis-test statistics.
Writes artifacts/final_results/final_hypothesis_test_summary.csv.

Comparisons
-----------
C1  sw_vs_ln   small-wide vs large-narrow   [PRIMARY — tests H1a]
C2  sw_vs_lw   small-wide vs large-wide     [size effect, wide coverage]
C3  sn_vs_ln   small-narrow vs large-narrow [size effect, narrow coverage]
C4  wide_vs_narrow  pooled wide vs pooled narrow [collapsed coverage main effect]

Edge cases handled per STATISTICAL_CLOSURE_PLAN.md §4:
  - C3: both groups constant and identical → exact equality, no test
  - BC C1: bimodal small-wide → MWU primary, Welch t secondary/unreliable
  - CQL/IQL C1: one group constant → note inflated Cohen's d

Verification
------------
Prints run summary and checks against plan pre-computed values (§8).
Exits with error if any deviation exceeds tolerance.
"""

import sys, os, csv, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from scipy import stats as sp_stats

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

def _p(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

SRC_MAIN = _p("artifacts", "training_main",  "envA_v2_main_summary.csv")
SRC_IQL  = _p("artifacts", "training_iql",   "envA_v2_iql_main_summary.csv")
OUT_CSV  = _p("artifacts", "final_results",  "final_hypothesis_test_summary.csv")

OUTPUT_COLUMNS = [
    "comparison", "algorithm",
    "n_group_A", "mean_A", "std_A", "ci95_low_A", "ci95_high_A",
    "n_group_B", "mean_B", "std_B", "ci95_low_B", "ci95_high_B",
    "delta",
    "cohens_d", "effect_label",
    "t_stat", "t_pvalue", "t_notes",
    "mwu_stat", "mwu_pvalue", "mwu_notes",
    "primary_test", "primary_pvalue",
    "notes",
]

# ── Data loading ──────────────────────────────────────────────────────────────

def load_returns():
    """Load all avg_return values into a dict keyed by (dataset_name, algorithm)."""
    data = {}
    for path in [SRC_MAIN, SRC_IQL]:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["dataset_name"], row["algorithm"])
                data.setdefault(key, []).append(float(row["avg_return"]))
    return data

# ── Statistical helpers ───────────────────────────────────────────────────────

def ci95(values):
    """95% CI via t-distribution (ddof=1)."""
    n = len(values)
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1))
    if s == 0 or n < 2:
        return m, m
    se = s / math.sqrt(n)
    t_crit = sp_stats.t.ppf(0.975, df=n - 1)
    return m - t_crit * se, m + t_crit * se


def pooled_cohens_d(a, b):
    """
    Pooled Cohen's d = (mean_a - mean_b) / pooled_SD.
    Returns (d, notes_str).
    """
    na, nb = len(a), len(b)
    sa = float(np.std(a, ddof=1))
    sb = float(np.std(b, ddof=1))
    if sa == 0 and sb == 0:
        if np.mean(a) == np.mean(b):
            return 0.0, "zero variance both groups; groups are identical"
        else:
            return float("inf"), "degenerate: pooled SD=0 but means differ"
    pooled = math.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    if pooled == 0:
        return float("inf"), "degenerate: pooled SD=0"
    d = (float(np.mean(a)) - float(np.mean(b))) / pooled
    note = ""
    if abs(d) > 10:
        note = "d magnitude reflects near-zero SD, not practical effect scale"
    if sa == 0 or sb == 0:
        note = ("one group constant (zero variance); " + note).strip("; ")
    return d, note


def effect_label(d):
    """Map |d| to label per plan §6."""
    if d is None or not math.isfinite(d):
        return "undefined"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def welch_t(a, b):
    """
    Welch two-sample t-test.
    Returns (t_stat, p_value, notes) or (nan, nan, reason) if inapplicable.
    """
    sa = float(np.std(a, ddof=1))
    sb = float(np.std(b, ddof=1))
    if sa == 0 and sb == 0:
        return 0.0, 1.0, "t=0 p=1: both groups constant and identical"
    try:
        t, p = sp_stats.ttest_ind(a, b, equal_var=False)
        return float(t), float(p), ""
    except Exception as e:
        return float("nan"), float("nan"), f"error: {e}"


def mwu(a, b):
    """
    Mann-Whitney U two-sided test.
    Returns (stat, p_value, notes) or (nan, nan, reason) if inapplicable.
    """
    combined = np.concatenate([a, b])
    if len(np.unique(combined)) <= 1:
        return float("nan"), float("nan"), "inapplicable: identical values in both groups"
    try:
        stat, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(stat), float(p), ""
    except Exception as e:
        return float("nan"), float("nan"), f"error: {e}"

# ── BC bimodality detection ───────────────────────────────────────────────────

def is_bimodal_bc(values):
    """
    Detect BC's known bimodal pattern in small-wide:
    1 seed at -1.0, remainder at 0.39-0.41.
    Criterion: more than one distinct cluster with a gap > 0.5.
    """
    sorted_v = sorted(values)
    for i in range(len(sorted_v) - 1):
        if sorted_v[i + 1] - sorted_v[i] > 0.5:
            return True
    return False

# ── Primary test selection ────────────────────────────────────────────────────

def select_primary(algo, a, b, t_p, mwu_p, c3_exact):
    """
    Returns (primary_test_str, primary_pvalue_str) per plan §5.
    """
    if c3_exact:
        return "exact equality (all seeds identical)", "N/A"
    # BC bimodal check: only when group A (small-wide) is bimodal
    if algo == "bc" and is_bimodal_bc(list(a)):
        p_str = f"{mwu_p:.4e}" if math.isfinite(mwu_p) else "nan"
        return "Mann-Whitney U (primary; t-test unreliable due to bimodal BC distribution)", p_str
    p_str = f"{t_p:.4e}" if math.isfinite(t_p) else "nan"
    return "Welch t-test", p_str

# ── Group statistics bundle ───────────────────────────────────────────────────

def group_stats(values):
    n = len(values)
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1))
    lo, hi = ci95(values)
    return n, m, s, lo, hi

# ── Single comparison × algorithm row ────────────────────────────────────────

def compute_row(comparison_label, algo, group_a_vals, group_b_vals):
    a = np.array(group_a_vals, dtype=float)
    b = np.array(group_b_vals, dtype=float)

    nA, mA, sA, loA, hiA = group_stats(a)
    nB, mB, sB, loB, hiB = group_stats(b)
    delta = mA - mB

    # C3 exact equality detection: both groups constant AND identical mean.
    # Use a tolerance of 1e-10 for std (handles floating-point noise ~1e-16
    # that arises when all values are numerically identical).
    c3_exact = (sA < 1e-10 and sB < 1e-10 and abs(mA - mB) < 1e-10)

    # Cohen's d
    d, d_note = pooled_cohens_d(a, b)
    el = effect_label(d) if math.isfinite(d) else "undefined"

    # Welch t
    t_stat, t_p, t_note = welch_t(a, b)

    # Mann-Whitney U
    mwu_stat, mwu_p, mwu_note = mwu(a, b)

    # Primary test
    primary_test, primary_p = select_primary(algo, a, b, t_p, mwu_p, c3_exact)

    # Assemble notes
    notes_parts = []
    if d_note:
        notes_parts.append(d_note)
    if algo == "bc" and is_bimodal_bc(list(a)):
        notes_parts.append(
            "BC distribution is bimodal (1 failure seed at -1.0, "
            "remainder 0.39-0.41 in small-wide); Welch t-test unreliable; "
            "MWU is primary test"
        )
    if not d_note and (sA == 0 or sB == 0) and not c3_exact:
        notes_parts.append("one group has zero variance")
    if abs(d) > 10:
        notes_parts.append(
            f"Cohen's d={d:.2f} is inflated by near-zero pooled SD; "
            "magnitude does not reflect practical effect scale"
        )

    def fmt(v, decimals=4):
        """Format a float with fixed decimal places; return 'nan' for non-finite."""
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "nan"
        return f"{v:.{decimals}f}"

    def fmt_p(v):
        """
        Format a p-value as scientific notation (always).
        Handles the case where scipy returns 0.0 for extremely small p-values
        by using the smallest representable positive float instead.
        Returns 'nan' for non-finite, 'N/A' passthrough handled by caller.
        """
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "nan"
        if v == 0.0:
            # scipy returns exact 0 when p underflows; indicate as < machine epsilon
            return "<2.22e-16"
        return f"{v:.4e}"

    def fmt_primary_p(v):
        """
        Format primary_pvalue: passes through 'N/A' strings unchanged,
        otherwise uses fmt_p for numeric values.
        """
        if isinstance(v, str):
            return v          # preserves "N/A" from exact equality rows
        return fmt_p(v)

    # Recompute primary_p as a raw value (not string) so fmt_primary_p can handle it
    # select_primary already returns a formatted string — re-derive the numeric value
    if c3_exact:
        primary_p_final = "N/A"
    elif algo == "bc" and is_bimodal_bc(list(a)):
        primary_p_final = fmt_p(mwu_p)
    else:
        primary_p_final = fmt_p(t_p)

    return {
        "comparison":    comparison_label,
        "algorithm":     algo,
        "n_group_A":     nA,
        "mean_A":        fmt(mA),
        "std_A":         fmt(sA),
        "ci95_low_A":    fmt(loA),
        "ci95_high_A":   fmt(hiA),
        "n_group_B":     nB,
        "mean_B":        fmt(mB),
        "std_B":         fmt(sB),
        "ci95_low_B":    fmt(loB),
        "ci95_high_B":   fmt(hiB),
        "delta":         fmt(delta),
        "cohens_d":      fmt(d, 4) if math.isfinite(d) else ("inf" if d == float("inf") else "nan"),
        "effect_label":  el,
        "t_stat":        fmt(t_stat),
        "t_pvalue":      fmt_p(t_p),
        "t_notes":       t_note if t_note else "Welch two-sample t-test",
        "mwu_stat":      fmt(mwu_stat, 1),
        "mwu_pvalue":    fmt_p(mwu_p),
        "mwu_notes":     mwu_note if mwu_note else "Mann-Whitney U two-sided",
        "primary_test":  primary_test,
        "primary_pvalue": primary_p_final,
        "notes":         "; ".join(notes_parts) if notes_parts else "",
    }

# ── Verification against plan §8 ─────────────────────────────────────────────

def verify(rows):
    """
    Check computed rows against plan pre-computed values.
    Print PASS/FAIL per check. Return True if all pass.
    """
    print()
    print("── Verification against STATISTICAL_CLOSURE_PLAN.md §8 ──")
    ok = True

    def find_row(cmp, algo):
        for r in rows:
            if r["comparison"] == cmp and r["algorithm"] == algo:
                return r
        return None

    def check(label, got, expected, tol=0.05, mode="abs"):
        nonlocal ok
        try:
            g = float(got)
            e = float(expected)
        except (ValueError, TypeError):
            print(f"  [SKIP] {label}: cannot parse got={got!r} expected={expected!r}")
            return
        if mode == "abs":
            diff = abs(g - e)
            passed = diff <= tol
        else:  # relative
            passed = (abs(g) < 1e-10 and abs(e) < 1e-10) or abs(g - e) / (abs(e) + 1e-10) <= tol
        status = "PASS" if passed else "FAIL"
        if not passed:
            ok = False
        print(f"  [{status}] {label}: got={g:.4f} expected≈{e:.4f} (tol={tol})")

    # C1 BC
    r = find_row("sw_vs_ln", "bc")
    if r:
        check("C1 BC: Cohen's d ≈ 0.256", r["cohens_d"], 0.256, tol=0.05)
        check("C1 BC: Welch t_p ≈ 0.43", r["t_pvalue"], 0.43, tol=0.05)
        check("C1 BC: delta = +0.057", r["delta"], 0.057, tol=0.005)
        mwu_p = float(r["mwu_pvalue"]) if r["mwu_pvalue"] != "nan" else float("nan")
        print(f"  [{'PASS' if mwu_p < 0.0001 else 'FAIL'}] C1 BC: MWU p < 0.0001 (got={mwu_p:.6f})")
        if mwu_p >= 0.0001:
            ok = False
        primary_ok = "Mann-Whitney" in r["primary_test"]
        print(f"  [{'PASS' if primary_ok else 'FAIL'}] C1 BC: primary_test = MWU (got={r['primary_test']!r})")
        if not primary_ok:
            ok = False
    else:
        print("  [FAIL] C1 BC row not found")
        ok = False

    # C1 CQL
    r = find_row("sw_vs_ln", "cql")
    if r:
        check("C1 CQL: Cohen's d ≈ 18.35", r["cohens_d"], 18.35, tol=0.5)
        t_p = float(r["t_pvalue"]) if r["t_pvalue"] != "nan" else float("nan")
        print(f"  [{'PASS' if t_p < 1e-4 else 'FAIL'}] C1 CQL: t_p < 1e-4 (got={t_p:.6e})")
        if t_p >= 1e-4:
            ok = False

    # C1 IQL
    r = find_row("sw_vs_ln", "iql")
    if r:
        check("C1 IQL: Cohen's d ≈ 18.35", r["cohens_d"], 18.35, tol=0.5)
        t_p = float(r["t_pvalue"]) if r["t_pvalue"] != "nan" else float("nan")
        print(f"  [{'PASS' if t_p < 1e-4 else 'FAIL'}] C1 IQL: t_p < 1e-4 (got={t_p:.6e})")
        if t_p >= 1e-4:
            ok = False

    # C3 all algos: exact equality
    for algo in ["bc", "cql", "iql"]:
        r = find_row("sn_vs_ln", algo)
        if r:
            delta_ok = abs(float(r["delta"])) < 1e-9
            exact_ok = "exact equality" in r["primary_test"]
            print(f"  [{'PASS' if delta_ok else 'FAIL'}] C3 {algo}: delta = 0 (got={r['delta']})")
            print(f"  [{'PASS' if exact_ok else 'FAIL'}] C3 {algo}: primary_test = exact equality (got={r['primary_test']!r})")
            if not delta_ok or not exact_ok:
                ok = False
        else:
            print(f"  [FAIL] C3 {algo} row not found")
            ok = False

    print()
    if ok:
        print("  All pre-computed verifications PASSED.")
    else:
        print("  WARNING: One or more verifications FAILED — review output before use.")
    return ok

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("generate_statistical_closure.py")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n── Input files ──")
    for path in [SRC_MAIN, SRC_IQL]:
        assert os.path.isfile(path), f"Missing: {path}"
        print(f"  {path}")

    data = load_returns()

    # Dataset keys
    SW = "envA_v2_small_wide_medium"
    SN = "envA_v2_small_narrow_medium"
    LW = "envA_v2_large_wide_medium"
    LN = "envA_v2_large_narrow_medium"

    # ── Print sample counts ───────────────────────────────────────────────────
    print("\n── Sample counts per cell ──")
    for ds in [SW, SN, LW, LN]:
        for algo in ["bc", "cql", "iql"]:
            vals = data.get((ds, algo), [])
            print(f"  {ds} x {algo}: n={len(vals)}")
            if len(vals) != 20:
                print(f"    WARNING: expected 20, got {len(vals)}")

    # ── Compute rows ──────────────────────────────────────────────────────────
    rows = []
    ALGOS = ["bc", "cql", "iql"]

    # C1: sw vs ln (PRIMARY)
    for algo in ALGOS:
        rows.append(compute_row("sw_vs_ln", algo,
                                data[(SW, algo)], data[(LN, algo)]))

    # C2: sw vs lw
    for algo in ALGOS:
        rows.append(compute_row("sw_vs_lw", algo,
                                data[(SW, algo)], data[(LW, algo)]))

    # C3: sn vs ln
    for algo in ALGOS:
        rows.append(compute_row("sn_vs_ln", algo,
                                data[(SN, algo)], data[(LN, algo)]))

    # C4: wide-pooled vs narrow-pooled
    for algo in ALGOS:
        wide_vals  = list(data[(SW, algo)]) + list(data[(LW, algo)])
        narrow_vals = list(data[(SN, algo)]) + list(data[(LN, algo)])
        rows.append(compute_row("wide_vs_narrow", algo, wide_vals, narrow_vals))

    # ── Print run summary ─────────────────────────────────────────────────────
    print("\n── Run summary (12 rows) ──")
    print(f"  {'comparison':<20} {'algo':<5} {'mean_A':>8} {'mean_B':>8} "
          f"{'delta':>8} {'d':>8} {'primary_p':>12}  primary_test")
    print("  " + "-" * 85)
    for r in rows:
        print(f"  {r['comparison']:<20} {r['algorithm']:<5} "
              f"{r['mean_A']:>8} {r['mean_B']:>8} "
              f"{r['delta']:>8} {r['cohens_d']:>8} "
              f"{r['primary_pvalue']:>12}  {r['primary_test'][:50]}")

    # ── Verification ──────────────────────────────────────────────────────────
    all_ok = verify(rows)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n── Output ──")
    print(f"  {OUT_CSV}")
    print(f"  Rows written: {len(rows)}")
    print()
    if all_ok:
        print("Statistical closure script completed. All verifications passed.")
    else:
        print("Statistical closure script completed with verification warnings.")
        print("Review FAILED checks above before treating output as final.")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
