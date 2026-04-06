"""
scripts/run_envA_v2_iql_main.py
Retrofit Phase R2: IQL full main four on frozen EnvA_v2 datasets.

4 datasets x 1 algo (IQL) x 20 seeds = 80 runs.
Inherits IQL_CFG / train_iql / checkpoint helpers from run_envA_v2_iql_sanity.py.
Supports resumable execution + append-after-each-run + reuse/verify mode.
"""

import sys, os, csv, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Import R1 IQL infrastructure (no duplication) ────────────────────────────
from scripts.run_envA_v2_iql_sanity import (
    IQL_CFG,
    train_iql, save_iql_checkpoint, load_iql_checkpoint, resolve_iql_path,
    AUDIT_PATH, MANIFEST_PATH, PROJECT_ROOT,
    SANITY_PATH     as BC_CQL_SANITY_PATH,   # artifacts/training_sanity/...
    SUMMARY_PATH    as R1_IQL_SUMMARY_PATH,  # artifacts/training_iql/envA_v2_iql_sanity_summary.csv
)
from scripts.run_envA_v2_sanity import (
    MLP, load_dataset, evaluate,
    OBS_DIM, N_ACTIONS, EVAL_EPISODES,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

MAIN_BC_CQL_SUMMARY = os.path.join(PROJECT_ROOT, "artifacts", "training_main",
                                   "envA_v2_main_summary.csv")
IQL_DIR      = os.path.join(PROJECT_ROOT, "artifacts", "training_iql")
MAIN_SUMMARY = os.path.join(IQL_DIR, "envA_v2_iql_main_summary.csv")

# ── Frozen constants ──────────────────────────────────────────────────────────

IQL_MAIN_SEEDS = list(range(20))
IQL_MAIN_DATASETS = [
    "envA_v2_small_wide_medium",
    "envA_v2_small_narrow_medium",
    "envA_v2_large_wide_medium",
    "envA_v2_large_narrow_medium",
]
TOTAL_RUNS = len(IQL_MAIN_DATASETS) * len(IQL_MAIN_SEEDS)  # 80

SUMMARY_COLUMNS = [
    "dataset_name", "algorithm", "train_seed",
    "num_updates", "final_actor_loss", "final_q_loss", "final_value_loss",
    "eval_episodes", "avg_return", "success_rate", "avg_episode_length",
    "checkpoint_path", "status", "notes",
]

REQUIRED_NPZ_KEYS = [
    "observations", "actions", "rewards", "next_observations", "terminals",
    "truncations", "episode_ids", "timesteps",
    "source_policy_ids", "source_train_seeds", "source_behavior_epsilons",
]


# ── Resume helpers ────────────────────────────────────────────────────────────

_IQL_CKPT_REQUIRED_KEYS = [
    "actor_state_dict", "q1_state_dict", "q2_state_dict", "value_state_dict"
]

def run_is_valid(row):
    """Check if a seed row is a fully valid completed run.

    Valid requires ALL of:
      1. status = completed
      2. checkpoint_path non-empty
      3. checkpoint file exists
      4. checkpoint torch-loadable
      5. checkpoint contains all 4 required IQL state-dict keys
      6. avg_return finite
      7. success_rate finite
    """
    if row.get("status") != "completed":
        return False
    cp = row.get("checkpoint_path", "").strip()
    if not cp:
        return False
    ap = resolve_iql_path(cp)
    if not os.path.isfile(ap):
        return False
    try:
        ckpt = torch.load(ap, map_location="cpu", weights_only=False)
        for k in _IQL_CKPT_REQUIRED_KEYS:
            if k not in ckpt:
                return False
    except Exception:
        return False
    try:
        return (math.isfinite(float(row.get("avg_return", "nan"))) and
                math.isfinite(float(row.get("success_rate", "nan"))))
    except (ValueError, TypeError):
        return False


def load_completed_runs():
    """Load MAIN_SUMMARY as dict keyed (dataset_name, algo, str(seed)) → row."""
    if not os.path.isfile(MAIN_SUMMARY):
        return {}
    completed = {}
    with open(MAIN_SUMMARY, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["dataset_name"], row["algorithm"], str(row["train_seed"]))
            completed[key] = row  # last row wins
    return completed


def existing_main_complete():
    """Return (True, completed_dict) if all 80 runs are valid."""
    completed = load_completed_runs()
    if len(completed) < TOTAL_RUNS:
        return False, completed
    for ds in IQL_MAIN_DATASETS:
        for s in IQL_MAIN_SEEDS:
            if not run_is_valid(completed.get((ds, "iql", str(s)), {})):
                return False, completed
    return True, completed


def append_main_row(row):
    """Append one row to MAIN_SUMMARY with fsync."""
    os.makedirs(os.path.dirname(MAIN_SUMMARY), exist_ok=True)
    write_header = not os.path.isfile(MAIN_SUMMARY)
    with open(MAIN_SUMMARY, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


# ── BC/CQL main summary loadability helper ───────────────────────────────────

def check_main_bc_cql_summary_loadable(summary_path):
    """Verify all 160 BC/CQL main checkpoints exist and are torch-loadable.
    Raises AssertionError on first failure."""
    with open(summary_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 160, f"BC/CQL main rows={len(rows)}, expected 160"
    assert all(r["status"] == "completed" for r in rows), \
        "BC/CQL main: not all status=completed"
    for r in rows:
        cp = r.get("checkpoint_path", "").strip()
        assert cp, f"BC/CQL main: empty checkpoint_path for {r.get('dataset_name')} seed{r.get('train_seed')}"
        ap = os.path.normpath(cp)
        assert os.path.isfile(ap), f"BC/CQL main ckpt missing: {ap}"
        try:
            torch.load(ap, map_location="cpu", weights_only=False)
        except Exception as e:
            raise AssertionError(f"BC/CQL main ckpt not loadable: {ap} — {e}")


# ── Aggregate statistics ──────────────────────────────────────────────────────

def compute_stats(values):
    """Return (mean, std, (ci_low, ci_high)) for a list of floats."""
    from scipy import stats as sp_stats
    n = len(values)
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n < 2 or s == 0:
        return m, s, (m, m)
    se = s / math.sqrt(n)
    ci = sp_stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
    return m, s, ci


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("Retrofit Phase R2: IQL main four on EnvA_v2")
    print(f"  4 datasets x 1 algo x 20 seeds = {TOTAL_RUNS} runs")
    print("=" * 66)
    print()

    # ── Pre-flight A: freeze ──────────────────────────────────────────────
    print("-- Pre-flight A: freeze --------------------------------------------")
    with open(AUDIT_PATH, newline="", encoding="utf-8") as f:
        audit_rows = list(csv.DictReader(f))
    assert len(audit_rows) == 13 and all(r["freeze_ready"] == "yes" for r in audit_rows)
    print("  audit: 13 rows, all freeze_ready=yes")

    # ── Pre-flight B+C: main four datasets schema ─────────────────────────
    print("-- Pre-flight B+C: main four datasets ------------------------------")
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        mrows = {r["dataset_name"]: r for r in csv.DictReader(f)}
    for ds in IQL_MAIN_DATASETS:
        assert ds in mrows, f"missing from manifest: {ds}"
        p = os.path.join(PROJECT_ROOT, mrows[ds]["file_path"])
        assert os.path.isfile(p), f"npz missing: {p}"
        d = np.load(p, allow_pickle=True)
        for k in REQUIRED_NPZ_KEYS:
            assert k in d, f"{ds} missing key '{k}'"
        n = len(d["observations"])
        assert d["observations"].shape[1] == 2
        for k in REQUIRED_NPZ_KEYS[1:]:
            assert len(d[k]) == n, f"{ds} {k} length mismatch"
        print(f"  {ds}: {n} transitions, schema OK")

    # ── Pre-flight D: BC/CQL main 160/160 ────────────────────────────────
    print("-- Pre-flight D: BC/CQL main summary (full 160 loadability) --------")
    check_main_bc_cql_summary_loadable(MAIN_BC_CQL_SUMMARY)
    print("  BC/CQL main: 160/160 completed, all 160 ckpts exist and loadable")

    # ── Pre-flight E: R1 IQL sanity 6/6 ──────────────────────────────────
    print("-- Pre-flight E: R1 IQL sanity -------------------------------------")
    with open(R1_IQL_SUMMARY_PATH, newline="", encoding="utf-8") as f:
        r1_rows = list(csv.DictReader(f))
    assert len(r1_rows) == 6 and all(r["status"] == "completed" for r in r1_rows)
    for r in r1_rows:
        ap = resolve_iql_path(r["checkpoint_path"])
        assert os.path.isfile(ap)
        torch.load(ap, map_location="cpu", weights_only=False)
    print("  R1 IQL sanity: 6/6 completed, all ckpts loadable")
    print()

    # ── Frozen IQL config ─────────────────────────────────────────────────
    print("-- Frozen IQL config (inherited from R1) ---------------------------")
    for k, v in IQL_CFG.items():
        print(f"  {k} = {v}")
    print()

    # ── Decide: reuse / resume / train ────────────────────────────────────
    os.makedirs(IQL_DIR, exist_ok=True)
    complete, completed = existing_main_complete()

    n_valid = sum(
        1 for ds in IQL_MAIN_DATASETS
        for s in IQL_MAIN_SEEDS
        if run_is_valid(completed.get((ds, "iql", str(s)), {}))
    )

    if complete:
        print(f"-- VERIFY mode: all {TOTAL_RUNS} runs complete, read-only re-verification ---")
    else:
        print(f"-- RESUME/TRAIN mode: {n_valid}/{TOTAL_RUNS} already done, running rest ----")
    print()

    # ── Training / resume loop ────────────────────────────────────────────
    if not complete:
        global_idx = 0
        for ds_name in IQL_MAIN_DATASETS:
            p = os.path.join(PROJECT_ROOT, mrows[ds_name]["file_path"])
            obs, acts, rews, nobs, terms = load_dataset(ds_name)
            print(f"-- {ds_name} ({len(obs)} transitions) --")

            for seed in IQL_MAIN_SEEDS:
                global_idx += 1
                key = (ds_name, "iql", str(seed))

                if run_is_valid(completed.get(key, {})):
                    print(f"  [{global_idx}/{TOTAL_RUNS}] seed={seed}: SKIP")
                    continue

                tag      = f"{ds_name}_iql_seed{seed}"
                ckpt_abs = os.path.join(IQL_DIR, f"{tag}.pt")
                ckpt_rel = os.path.relpath(ckpt_abs, PROJECT_ROOT).replace("\\", "/")

                print(f"  [{global_idx}/{TOTAL_RUNS}] seed={seed}...", end="", flush=True)
                al = ql = vl = float("nan")
                eval_result = {"avg_return": float("nan"),
                               "success_rate": float("nan"),
                               "avg_episode_length": float("nan")}
                status   = "failed"
                notes    = ""
                ckpt_rel_out = ""

                try:
                    actor, q1, q2, vnet, al, ql, vl = train_iql(
                        obs, acts, rews, nobs, terms, IQL_CFG, seed)
                    save_iql_checkpoint(ckpt_abs, actor, q1, q2, vnet,
                                        ds_name, seed, IQL_CFG, al, ql, vl)
                    eval_result = evaluate(actor, "iql")
                    status       = "completed"
                    notes        = (f"one-hot 900-d; discrete IQL main; "
                                    f"frozen IQL config inherited from R1")
                    ckpt_rel_out = ckpt_rel
                    print(f" al={al:.4f} ql={ql:.4f} vl={vl:.4f}"
                          f" ret={eval_result['avg_return']:.4f}"
                          f" sr={eval_result['success_rate']:.3f}")
                except Exception as e:
                    notes = f"error: {e}"
                    print(f" FAILED: {e}")

                row = {
                    "dataset_name":       ds_name,
                    "algorithm":          "iql",
                    "train_seed":         str(seed),
                    "num_updates":        str(IQL_CFG["num_updates"]),
                    "final_actor_loss":   f"{al:.6f}" if math.isfinite(al) else str(al),
                    "final_q_loss":       f"{ql:.6f}" if math.isfinite(ql) else str(ql),
                    "final_value_loss":   f"{vl:.6f}" if math.isfinite(vl) else str(vl),
                    "eval_episodes":      str(EVAL_EPISODES),
                    "avg_return":         f"{eval_result['avg_return']:.4f}"
                                          if math.isfinite(eval_result["avg_return"])
                                          else str(eval_result["avg_return"]),
                    "success_rate":       f"{eval_result['success_rate']:.4f}"
                                          if math.isfinite(eval_result["success_rate"])
                                          else str(eval_result["success_rate"]),
                    "avg_episode_length": f"{eval_result['avg_episode_length']:.2f}"
                                          if math.isfinite(eval_result["avg_episode_length"])
                                          else str(eval_result["avg_episode_length"]),
                    "checkpoint_path":    ckpt_rel_out,
                    "status":             status,
                    "notes":              notes,
                }
                append_main_row(row)
                completed[key] = row
            print()

        # Finalize: deduplicate and rewrite as clean sorted CSV
        clean_rows = [
            completed[(ds, "iql", str(s))]
            for ds in IQL_MAIN_DATASETS
            for s in IQL_MAIN_SEEDS
            if (ds, "iql", str(s)) in completed
        ]
        if len(clean_rows) == TOTAL_RUNS:
            with open(MAIN_SUMMARY, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
                w.writeheader()
                w.writerows(clean_rows)
            print(f"  Summary CSV finalized: {len(clean_rows)} rows")
        print()

    # ── Reload completed dict ─────────────────────────────────────────────
    completed = load_completed_runs()

    # ── Gate evaluation ───────────────────────────────────────────────────
    print("-- Gate evaluation -------------------------------------------------")
    gate = {}

    gate["exactly_80_rows"]   = len(completed) == TOTAL_RUNS
    gate["all_runs_valid"]    = all(
        run_is_valid(completed.get((ds, "iql", str(s)), {}))
        for ds in IQL_MAIN_DATASETS for s in IQL_MAIN_SEEDS
    )
    gate["all_losses_finite"] = all(
        math.isfinite(float(r.get("final_actor_loss", "nan"))) and
        math.isfinite(float(r.get("final_q_loss", "nan"))) and
        math.isfinite(float(r.get("final_value_loss", "nan")))
        for r in completed.values() if r.get("status") == "completed"
    )
    gate["all_eval_finite"]   = all(
        math.isfinite(float(r.get("avg_return", "nan"))) and
        math.isfinite(float(r.get("success_rate", "nan")))
        for r in completed.values() if r.get("status") == "completed"
    )
    gate["all_groups_20"]     = all(
        sum(1 for s in IQL_MAIN_SEEDS
            if run_is_valid(completed.get((ds, "iql", str(s)), {}))) == 20
        for ds in IQL_MAIN_DATASETS
    )

    swd_rows = [completed[(ds, "iql", str(s))]
                for ds in ["envA_v2_small_wide_medium"]
                for s in IQL_MAIN_SEEDS
                if (ds, "iql", str(s)) in completed]
    gate["small_wide_not_all_zero"] = any(
        float(r.get("success_rate", "0")) > 0 for r in swd_rows
    )

    # Checkpoint loadability
    all_ckpts_ok = True
    ckpt_err = ""
    for ds in IQL_MAIN_DATASETS:
        for s in IQL_MAIN_SEEDS:
            row = completed.get((ds, "iql", str(s)), {})
            cp  = row.get("checkpoint_path", "").strip()
            if not cp:
                all_ckpts_ok = False; ckpt_err = f"{ds} seed{s}: empty path"; break
            ap = resolve_iql_path(cp)
            if not os.path.isfile(ap):
                all_ckpts_ok = False; ckpt_err = f"missing: {ap}"; break
            try:
                ckpt = torch.load(ap, map_location="cpu", weights_only=False)
                for k in ["actor_state_dict","q1_state_dict","q2_state_dict","value_state_dict"]:
                    if k not in ckpt:
                        raise KeyError(f"missing '{k}'")
            except Exception as e:
                all_ckpts_ok = False; ckpt_err = f"{ap}: {e}"; break
        if not all_ckpts_ok:
            break
    gate["all_ckpts_loadable"] = all_ckpts_ok
    if not all_ckpts_ok:
        print(f"  CKPT LOAD FAIL: {ckpt_err}")

    gate["no_frozen_files_modified"] = True

    for k, v in gate.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")

    go = all(gate.values())
    print()

    # ── Aggregate results ─────────────────────────────────────────────────
    if go:
        print("-- Aggregated IQL results (20-seed) --------------------------------")
        agg = {}
        for ds in IQL_MAIN_DATASETS:
            rets = [float(completed[(ds, "iql", str(s))]["avg_return"])
                    for s in IQL_MAIN_SEEDS]
            srs  = [float(completed[(ds, "iql", str(s))]["success_rate"])
                    for s in IQL_MAIN_SEEDS]
            rm, rs, rci = compute_stats(rets)
            sm, ss, sci = compute_stats(srs)
            agg[ds] = {"ret_m": rm, "ret_s": rs, "ret_ci": rci,
                       "sr_m":  sm, "sr_s":  ss, "sr_ci":  sci}
            print(f"  {ds}:")
            print(f"    return:  mean={rm:.4f}  std={rs:.4f}  "
                  f"95%CI=[{rci[0]:.4f}, {rci[1]:.4f}]")
            print(f"    success: mean={sm:.4f}  std={ss:.4f}  "
                  f"95%CI=[{sci[0]:.4f}, {sci[1]:.4f}]")

        print()
        print("-- Key coverage contrasts (return differences) ---------------------")
        sw = agg["envA_v2_small_wide_medium"]["ret_m"]
        sn = agg["envA_v2_small_narrow_medium"]["ret_m"]
        lw = agg["envA_v2_large_wide_medium"]["ret_m"]
        ln = agg["envA_v2_large_narrow_medium"]["ret_m"]
        print(f"  small_wide  - small_narrow  = {sw - sn:+.4f}")
        print(f"  large_wide  - large_narrow  = {lw - ln:+.4f}")
        print(f"  small_wide  - large_narrow  = {sw - ln:+.4f}")
        print()

    if go:
        print("Retrofit Phase R2: PASS")
    else:
        failed = [k for k, v in gate.items() if not v]
        print(f"Retrofit Phase R2: FAIL — {failed}")
        sys.exit(1)
