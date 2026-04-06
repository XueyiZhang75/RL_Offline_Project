"""
scripts/run_envA_v2_iql_quality_sweep.py
Retrofit Phase R4: IQL quality sweep on frozen EnvA_v2 quality datasets.

5 datasets x 1 algo (IQL) x 20 seeds = 100 runs.
Inherits:
  - IQL_CFG / train_iql / save_iql_checkpoint / load_iql_checkpoint / resolve_iql_path
    from run_envA_v2_iql_sanity.py (R1)
  - run_is_valid / check_iql_main_summary_valid
    from run_envA_v2_iql_main.py (R2)
  - load_dataset / evaluate / encode_obs / encode_single / OBS_DIM / N_ACTIONS / EVAL_EPISODES
    from run_envA_v2_sanity.py (Clean Phase 7)
  - frozen-file snapshot helpers
    from run_envbc_iql_validation.py (R3 patch)
Supports resumable + append-after-each-run + reuse/verify mode.
"""

import sys, os, csv, math, hashlib
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Import R1 IQL infrastructure ─────────────────────────────────────────────
from scripts.run_envA_v2_iql_sanity import (
    IQL_CFG,
    train_iql, save_iql_checkpoint, load_iql_checkpoint, resolve_iql_path,
    AUDIT_PATH, MANIFEST_PATH, PROJECT_ROOT,
)

# ── Import R2/R3 prerequisite helpers ────────────────────────────────────────
from scripts.run_envA_v2_iql_main import (
    check_main_bc_cql_summary_loadable,
    _IQL_CKPT_REQUIRED_KEYS,
)
from scripts.run_envbc_iql_validation import (
    check_iql_main_summary_valid,
)

# ── Import Clean Phase 7 EnvA_v2 infrastructure ──────────────────────────────
from scripts.run_envA_v2_sanity import (
    MLP, load_dataset, evaluate,
    OBS_DIM, N_ACTIONS, EVAL_EPISODES,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

IQL_DIR          = os.path.join(PROJECT_ROOT, "artifacts", "training_iql")
BC_CQL_QUAL_SUM  = os.path.join(PROJECT_ROOT, "artifacts", "training_quality",
                                 "envA_v2_quality_summary.csv")
IQL_MAIN_SUM     = os.path.join(PROJECT_ROOT, "artifacts", "training_iql",
                                 "envA_v2_iql_main_summary.csv")
SUMMARY_PATH     = os.path.join(IQL_DIR, "envA_v2_iql_quality_sweep_summary.csv")

# ── Frozen constants ──────────────────────────────────────────────────────────

IQL_QUALITY_SEEDS = list(range(20))
IQL_QUALITY_DATASETS = [
    "envA_v2_quality_random_wide50k",
    "envA_v2_quality_suboptimal_wide50k",
    "envA_v2_quality_medium_wide50k",
    "envA_v2_quality_expert_wide50k",
    "envA_v2_quality_mixed_wide50k",
]
QUALITY_BIN_MAP = {
    "envA_v2_quality_random_wide50k":     "random",
    "envA_v2_quality_suboptimal_wide50k": "suboptimal",
    "envA_v2_quality_medium_wide50k":     "medium",
    "envA_v2_quality_expert_wide50k":     "expert",
    "envA_v2_quality_mixed_wide50k":      "mixed",
}
TOTAL_RUNS = len(IQL_QUALITY_DATASETS) * len(IQL_QUALITY_SEEDS)  # 100

REQUIRED_NPZ_KEYS = [
    "observations", "actions", "rewards", "next_observations", "terminals",
    "truncations", "episode_ids", "timesteps",
    "source_policy_ids", "source_train_seeds", "source_behavior_epsilons",
]

SUMMARY_COLUMNS = [
    "dataset_name", "algorithm", "train_seed",
    "num_updates", "final_actor_loss", "final_q_loss", "final_value_loss",
    "eval_episodes", "avg_return", "success_rate", "avg_episode_length",
    "checkpoint_path", "status", "notes",
]


# ── BC/CQL quality summary loadability helper ─────────────────────────────────

def check_bc_cql_quality_summary_loadable(summary_path):
    """Verify all 200 BC/CQL quality checkpoints exist and are torch-loadable."""
    with open(summary_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 200, f"BC/CQL quality rows={len(rows)}, expected 200"
    assert all(r["status"] == "completed" for r in rows), \
        "BC/CQL quality: not all status=completed"
    for r in rows:
        cp = r.get("checkpoint_path", "").strip()
        assert cp, "BC/CQL quality: empty checkpoint_path"
        ap = resolve_iql_path(cp)
        assert os.path.isfile(ap), f"BC/CQL quality ckpt missing: {ap}"
        try:
            torch.load(ap, map_location="cpu", weights_only=False)
        except Exception as e:
            raise AssertionError(f"BC/CQL quality ckpt not loadable: {ap} — {e}")


# ── Run validity ──────────────────────────────────────────────────────────────

def run_is_valid(row):
    """Strict run validity: status + path + existence + loadability + keys + finite metrics."""
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


# ── Resume helpers ────────────────────────────────────────────────────────────

def load_completed_runs():
    """Load SUMMARY_PATH → dict keyed (dataset_name, 'iql', str(seed)) → row."""
    if not os.path.isfile(SUMMARY_PATH):
        return {}
    completed = {}
    with open(SUMMARY_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["dataset_name"], row["algorithm"], str(row["train_seed"]))
            completed[key] = row
    return completed


def existing_quality_complete():
    """Return (True, completed_dict) if all 100 runs are strictly valid."""
    completed = load_completed_runs()
    if len(completed) < TOTAL_RUNS:
        return False, completed
    for ds in IQL_QUALITY_DATASETS:
        for s in IQL_QUALITY_SEEDS:
            if not run_is_valid(completed.get((ds, "iql", str(s)), {})):
                return False, completed
    return True, completed


def append_quality_row(row):
    """Append one row to SUMMARY_PATH with fsync."""
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    write_header = not os.path.isfile(SUMMARY_PATH)
    with open(SUMMARY_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


# ── Aggregate stats ───────────────────────────────────────────────────────────

def compute_stats(values):
    from scipy import stats as sp_stats
    n = len(values)
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n < 2 or s == 0:
        return m, s, (m, m)
    se = s / math.sqrt(n)
    ci = sp_stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
    return m, s, ci


# ── Frozen-file snapshot helpers (inherited from R3 patch) ────────────────────

def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_frozen_file_snapshots(paths):
    snap = {}
    for p in paths:
        if os.path.isfile(p):
            snap[p] = {"exists": True, "size": os.path.getsize(p), "sha256": _file_sha256(p)}
        else:
            snap[p] = {"exists": False, "size": None, "sha256": None}
    return snap


def frozen_snapshots_equal(before, after):
    diffs = []
    for p in before:
        b, a = before[p], after.get(p, {"exists": False})
        if not b["exists"]:
            diffs.append(f"{p}: was missing before snapshot")
            continue
        if not a.get("exists", False):
            diffs.append(f"{p}: existed before but is now missing")
            continue
        if b["size"] != a["size"]:
            diffs.append(f"{p}: size changed {b['size']} → {a['size']}")
        elif b["sha256"] != a["sha256"]:
            diffs.append(f"{p}: content changed (sha256 mismatch)")
    return (len(diffs) == 0), diffs


FROZEN_FILES = [AUDIT_PATH, MANIFEST_PATH, BC_CQL_QUAL_SUM, IQL_MAIN_SUM]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("Retrofit Phase R4: IQL quality sweep on EnvA_v2")
    print(f"  5 datasets x 1 algo x 20 seeds = {TOTAL_RUNS} runs")
    print("=" * 66)
    print()

    # ── Snapshot frozen files before any logic ────────────────────────────
    _frozen_before = capture_frozen_file_snapshots(FROZEN_FILES)

    # ── Pre-flight A: freeze audit ────────────────────────────────────────
    print("-- Pre-flight A: freeze --------------------------------------------")
    with open(AUDIT_PATH, newline="", encoding="utf-8") as f:
        audit_rows = list(csv.DictReader(f))
    assert len(audit_rows) == 13 and all(r["freeze_ready"] == "yes" for r in audit_rows)
    print("  audit: 13 rows, all freeze_ready=yes")

    # ── Pre-flight B: quality datasets schema ─────────────────────────────
    print("-- Pre-flight B: quality datasets schema ---------------------------")
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        mrows = {r["dataset_name"]: r for r in csv.DictReader(f)}
    for ds in IQL_QUALITY_DATASETS:
        assert ds in mrows, f"missing from manifest: {ds}"
        p = os.path.join(PROJECT_ROOT, mrows[ds]["file_path"])
        assert os.path.isfile(p), f"npz missing: {p}"
        d = np.load(p, allow_pickle=True)
        for k in REQUIRED_NPZ_KEYS:
            assert k in d, f"{ds} missing key '{k}'"
        n = len(d["observations"])
        assert d["observations"].shape[1] == 2, \
            f"{ds} obs shape {d['observations'].shape}, expected (N,2)"
        for k in REQUIRED_NPZ_KEYS[1:]:
            assert len(d[k]) == n, f"{ds} {k} length mismatch"
        print(f"  {ds}: {n} transitions, schema OK")

    # ── Pre-flight C: BC/CQL quality 200/200 loadability ─────────────────
    print("-- Pre-flight C: BC/CQL quality summary (full 200 loadability) -----")
    check_bc_cql_quality_summary_loadable(BC_CQL_QUAL_SUM)
    print("  BC/CQL quality: 200/200 completed, all ckpts exist and loadable")

    # ── Pre-flight D: R2 IQL main 80/80 loadability ───────────────────────
    print("-- Pre-flight D: R2 IQL main summary (full 80 loadability) ---------")
    check_iql_main_summary_valid(IQL_MAIN_SUM)
    print("  IQL main: 80/80 valid, all ckpts loadable with required keys")
    print()

    # ── Frozen IQL config ─────────────────────────────────────────────────
    print("-- Frozen IQL config (inherited from R1) ---------------------------")
    for k, v in IQL_CFG.items():
        print(f"  {k} = {v}")
    print()

    # ── Decide: reuse / resume / train ────────────────────────────────────
    os.makedirs(IQL_DIR, exist_ok=True)
    complete, completed = existing_quality_complete()

    n_valid = sum(
        1 for ds in IQL_QUALITY_DATASETS
        for s in IQL_QUALITY_SEEDS
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
        for ds_name in IQL_QUALITY_DATASETS:
            obs, acts, rews, nobs, terms = load_dataset(ds_name)
            qbin = QUALITY_BIN_MAP[ds_name]
            print(f"-- {ds_name} [{qbin}] ({len(obs)} transitions) --")

            for seed in IQL_QUALITY_SEEDS:
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
                    eval_result  = evaluate(actor, "iql")
                    status       = "completed"
                    notes        = (f"one-hot {OBS_DIM}-d; discrete IQL quality sweep; "
                                    f"quality_bin={qbin}; frozen IQL config from R1")
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
                append_quality_row(row)
                completed[key] = row
            print()

        # Finalize: clean sorted rewrite
        clean_rows = [
            completed[(ds, "iql", str(s))]
            for ds in IQL_QUALITY_DATASETS
            for s in IQL_QUALITY_SEEDS
            if (ds, "iql", str(s)) in completed
        ]
        if len(clean_rows) == TOTAL_RUNS:
            with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
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

    gate["exactly_100_rows"]  = len(completed) == TOTAL_RUNS
    gate["all_runs_valid"]    = all(
        run_is_valid(completed.get((ds, "iql", str(s)), {}))
        for ds in IQL_QUALITY_DATASETS for s in IQL_QUALITY_SEEDS
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
        sum(1 for s in IQL_QUALITY_SEEDS
            if run_is_valid(completed.get((ds, "iql", str(s)), {}))) == 20
        for ds in IQL_QUALITY_DATASETS
    )

    # Checkpoint loadability gate
    all_ckpts_ok = True
    ckpt_err = ""
    for ds in IQL_QUALITY_DATASETS:
        for s in IQL_QUALITY_SEEDS:
            row = completed.get((ds, "iql", str(s)), {})
            cp  = row.get("checkpoint_path", "").strip()
            if not cp:
                all_ckpts_ok = False; ckpt_err = f"{ds} seed{s}: empty path"; break
            ap = resolve_iql_path(cp)
            if not os.path.isfile(ap):
                all_ckpts_ok = False; ckpt_err = f"missing: {ap}"; break
            try:
                ckpt = torch.load(ap, map_location="cpu", weights_only=False)
                for k in _IQL_CKPT_REQUIRED_KEYS:
                    if k not in ckpt:
                        raise KeyError(f"missing '{k}'")
            except Exception as e:
                all_ckpts_ok = False; ckpt_err = f"{ap}: {e}"; break
        if not all_ckpts_ok:
            break
    gate["all_ckpts_loadable"] = all_ckpts_ok
    if not all_ckpts_ok:
        print(f"  CKPT LOAD FAIL: {ckpt_err}")

    # Frozen-file gate (real sha256 comparison)
    _frozen_after = capture_frozen_file_snapshots(FROZEN_FILES)
    _frozen_ok, _frozen_diffs = frozen_snapshots_equal(_frozen_before, _frozen_after)
    gate["no_frozen_files_modified"] = _frozen_ok
    if not _frozen_ok:
        for diff in _frozen_diffs:
            print(f"  FROZEN FILE MODIFIED: {diff}")

    for k, v in gate.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")

    go = all(gate.values())
    print()

    # ── Aggregate results ─────────────────────────────────────────────────
    if go:
        print("-- Aggregated IQL quality results (20-seed) -------------------------")
        agg = {}
        for ds in IQL_QUALITY_DATASETS:
            qbin = QUALITY_BIN_MAP[ds]
            rets = [float(completed[(ds, "iql", str(s))]["avg_return"])
                    for s in IQL_QUALITY_SEEDS]
            srs  = [float(completed[(ds, "iql", str(s))]["success_rate"])
                    for s in IQL_QUALITY_SEEDS]
            rm, rs, rci = compute_stats(rets)
            sm, ss, sci = compute_stats(srs)
            agg[ds] = {"ret_m": rm, "sr_m": sm}
            print(f"  [{qbin}] {ds}:")
            print(f"    return:  mean={rm:.4f}  std={rs:.4f}  "
                  f"95%CI=[{rci[0]:.4f}, {rci[1]:.4f}]")
            print(f"    success: mean={sm:.4f}  std={ss:.4f}  "
                  f"95%CI=[{sci[0]:.4f}, {sci[1]:.4f}]")

        print()
        print("-- Quality contrasts (for record only) ------------------------------")
        expert_r    = agg["envA_v2_quality_expert_wide50k"]["ret_m"]
        medium_r    = agg["envA_v2_quality_medium_wide50k"]["ret_m"]
        subopt_r    = agg["envA_v2_quality_suboptimal_wide50k"]["ret_m"]
        mixed_r     = agg["envA_v2_quality_mixed_wide50k"]["ret_m"]
        random_r    = agg["envA_v2_quality_random_wide50k"]["ret_m"]
        print(f"  expert  - medium     = {expert_r - medium_r:+.4f}")
        print(f"  medium  - suboptimal = {medium_r - subopt_r:+.4f}")
        print(f"  mixed vs suboptimal  = {mixed_r - subopt_r:+.4f}")
        print(f"  mixed vs medium      = {mixed_r - medium_r:+.4f}")
        print(f"  mixed vs expert      = {mixed_r - expert_r:+.4f}")
        print(f"  random               = {random_r:.4f}  (expected near floor)")
        print()

    if go:
        print("Retrofit Phase R4: PASS")
    else:
        failed = [k for k, v in gate.items() if not v]
        print(f"Retrofit Phase R4: FAIL — {failed}")
        sys.exit(1)
