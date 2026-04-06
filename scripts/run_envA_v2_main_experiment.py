"""
scripts/run_envA_v2_main_experiment.py
Clean Phase 8: full main experiment — 4 datasets x 2 algos x 20 seeds = 160 runs.

Directly reuses the frozen training framework from run_envA_v2_sanity.py.
Does NOT modify any frozen file.

Supports read-only re-verification mode: if a complete summary already exists
(160 rows, all completed, all checkpoint_paths non-empty), skips training and
only re-runs pre-flight + gate checks + aggregation.
"""

import sys, os, csv, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Reuse frozen framework from sanity ────────────────────────────────────────
from scripts.run_envA_v2_sanity import (
    BC_CFG, CQL_CFG, OBS_DIM, N_ACTIONS, EVAL_EPISODES,
    MLP, encode_obs, encode_single, load_dataset,
    train_bc, train_cql, evaluate,
    SUMMARY_COLUMNS, AUDIT_PATH, MANIFEST_PATH,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SANITY_SUMMARY = os.path.join(PROJECT_ROOT, "artifacts", "training_sanity",
                              "envA_v2_sanity_summary.csv")
MAIN_DIR     = os.path.join(PROJECT_ROOT, "artifacts", "training_main")
MAIN_SUMMARY = os.path.join(MAIN_DIR, "envA_v2_main_summary.csv")

# ── Frozen constants ──────────────────────────────────────────────────────────

MAIN_SEEDS = list(range(20))
MAIN_DATASETS = [
    "envA_v2_small_wide_medium",
    "envA_v2_small_narrow_medium",
    "envA_v2_large_wide_medium",
    "envA_v2_large_narrow_medium",
]
ALGORITHMS = ["bc", "cql"]

REQUIRED_KEYS = {
    "observations", "actions", "rewards", "next_observations",
    "terminals", "truncations", "episode_ids", "timesteps",
    "source_policy_ids", "source_train_seeds", "source_behavior_epsilons",
}

# ── Path normalization helper ─────────────────────────────────────────────────

def resolve_ckpt_path(rel_path):
    """Normalize a checkpoint_path from summary CSV to an absolute path."""
    normalized = rel_path.replace("\\", "/")
    return os.path.normpath(os.path.join(PROJECT_ROOT, normalized))


# ── Helpers for pre-flight and gate ───────────────────────────────────────────

def check_sanity_checkpoints(san_rows):
    """Return (ok, details) for sanity checkpoint existence check."""
    missing = []
    for sr in san_rows:
        cp = sr.get("checkpoint_path", "").strip()
        if not cp:
            missing.append(f"seed{sr['train_seed']}_{sr['algorithm']}: empty path")
            continue
        abs_path = resolve_ckpt_path(cp)
        if not os.path.isfile(abs_path):
            missing.append(f"{cp}: not found at {abs_path}")
    return len(missing) == 0, missing


def check_main_checkpoints_exist(summary_rows):
    """Return (ok, missing_list) for 160 checkpoint existence."""
    missing = []
    for r in summary_rows:
        cp = r.get("checkpoint_path", "").strip()
        if not cp:
            missing.append(f"{r['dataset_name']}_{r['algorithm']}_seed{r['train_seed']}: empty")
            continue
        if not os.path.isfile(resolve_ckpt_path(cp)):
            missing.append(cp)
    return len(missing) == 0, missing


def check_main_checkpoints_loadable(summary_rows):
    """Return (ok, failed_list) for 160 checkpoint loadability."""
    failed = []
    for r in summary_rows:
        cp = r.get("checkpoint_path", "").strip()
        if not cp:
            failed.append(f"{r['dataset_name']}_{r['algorithm']}_seed{r['train_seed']}: empty")
            continue
        abs_path = resolve_ckpt_path(cp)
        try:
            torch.load(abs_path, weights_only=False)
        except Exception as e:
            failed.append(f"{cp}: {e}")
    return len(failed) == 0, failed


def existing_summary_is_complete():
    """Check if MAIN_SUMMARY already has 160 completed rows with non-empty checkpoint_paths."""
    if not os.path.isfile(MAIN_SUMMARY):
        return False, []
    with open(MAIN_SUMMARY, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 160:
        return False, rows
    if not all(r["status"] == "completed" for r in rows):
        return False, rows
    if not all(r.get("checkpoint_path", "").strip() for r in rows):
        return False, rows
    return True, rows


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("Clean Phase 8: EnvA_v2 full main experiment")
    print(f"  4 datasets x 2 algos x 20 seeds = 160 runs")
    print("=" * 66)
    print()

    # ── Pre-flight A: freeze ──────────────────────────────────────────────
    print("-- Pre-flight A: freeze --------------------------------------------")
    with open(AUDIT_PATH, newline="", encoding="utf-8") as f:
        audit_rows = list(csv.DictReader(f))
    assert len(audit_rows) == 13, f"Audit rows {len(audit_rows)} != 13"
    for ar in audit_rows:
        assert ar["freeze_ready"] == "yes", f"{ar['dataset_name']} not frozen"
    print("  freeze_ready = yes for all 13: OK")

    # ── Pre-flight B: sanity (with checkpoint existence) ──────────────────
    print("-- Pre-flight B: sanity --------------------------------------------")
    with open(SANITY_SUMMARY, newline="", encoding="utf-8") as f:
        san_rows = list(csv.DictReader(f))
    assert len(san_rows) == 12, f"Sanity rows {len(san_rows)} != 12"
    for sr in san_rows:
        assert sr["status"] == "completed", (
            f"Sanity {sr['dataset_name']}_{sr['algorithm']}_seed{sr['train_seed']} not completed")
    print("  12/12 sanity runs completed: OK")

    san_ckpt_ok, san_ckpt_missing = check_sanity_checkpoints(san_rows)
    if not san_ckpt_ok:
        print(f"  FAIL: sanity checkpoints missing: {san_ckpt_missing}")
        sys.exit(1)
    print("  12/12 sanity checkpoints exist: OK")

    # ── Pre-flight C: manifest & datasets ─────────────────────────────────
    print("-- Pre-flight C: main datasets -------------------------------------")
    for dn in MAIN_DATASETS:
        obs, acts, rews, nobs, terms = load_dataset(dn)
        print(f"  {dn}: {len(obs)} transitions, loaded OK")
    print()

    # ── Print frozen configs ──────────────────────────────────────────────
    print("-- Frozen configs (inherited from sanity) ---------------------------")
    print(f"  BC_CFG:  {BC_CFG}")
    print(f"  CQL_CFG: {CQL_CFG}")
    print(f"  MAIN_SEEDS: {MAIN_SEEDS[0]}..{MAIN_SEEDS[-1]} ({len(MAIN_SEEDS)} seeds)")
    print()

    # ── Decide: reuse existing or train ───────────────────────────────────
    complete, existing_rows = existing_summary_is_complete()

    if complete:
        print("-- Reuse mode: existing summary is complete (160 rows) -------------")
        print("  Skipping training. Running read-only re-verification.")
        summary_rows = existing_rows
    else:
        print("-- Training mode: running 160 training runs -------------------------")
        os.makedirs(MAIN_DIR, exist_ok=True)
        summary_rows = []
        run_count = 0
        total_runs = len(MAIN_DATASETS) * len(ALGORITHMS) * len(MAIN_SEEDS)

        for dn in MAIN_DATASETS:
            obs, acts, rews, nobs, terms = load_dataset(dn)
            print(f"-- {dn} ({len(obs)} transitions) --")

            for algo in ALGORITHMS:
                for seed in MAIN_SEEDS:
                    run_count += 1
                    tag = f"{dn}_{algo}_seed{seed}"
                    print(f"  [{run_count}/{total_runs}] {tag}...", end="", flush=True)

                    try:
                        if algo == "bc":
                            model, final_loss = train_bc(obs, acts, BC_CFG, seed)
                            num_updates = BC_CFG["num_updates"]
                        else:
                            model, final_loss = train_cql(
                                obs, acts, rews, nobs, terms, CQL_CFG, seed)
                            num_updates = CQL_CFG["num_updates"]

                        ckpt_name = f"{tag}.pt"
                        ckpt_path = os.path.join(MAIN_DIR, ckpt_name)
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "algorithm": algo,
                            "dataset_name": dn,
                            "train_seed": seed,
                            "num_updates": num_updates,
                            "final_train_loss": final_loss,
                            "obs_dim": OBS_DIM,
                            "n_actions": N_ACTIONS,
                            "config": BC_CFG if algo == "bc" else CQL_CFG,
                        }, ckpt_path)

                        ev = evaluate(model, algo)
                        status = "completed"
                        notes = ("one-hot 900-d; full main experiment; "
                                 f"frozen {'BC_CFG' if algo == 'bc' else 'CQL_CFG'} "
                                 "inherited from sanity")
                        print(f" loss={final_loss:.4f}  SR={ev['success_rate']:.3f}  "
                              f"ret={ev['avg_return']:.3f}")

                    except Exception as e:
                        final_loss = float("nan")
                        ev = {"avg_return": float("nan"), "success_rate": float("nan"),
                              "avg_episode_length": float("nan")}
                        ckpt_path = ""
                        num_updates = 0
                        status = "failed"
                        notes = f"error: {e}"
                        print(f" FAILED: {e}")

                    summary_rows.append({
                        "dataset_name": dn,
                        "algorithm": algo,
                        "train_seed": seed,
                        "num_updates": num_updates,
                        "final_train_loss": f"{final_loss:.6f}" if math.isfinite(final_loss) else str(final_loss),
                        "eval_episodes": EVAL_EPISODES,
                        "avg_return": f"{ev['avg_return']:.4f}" if math.isfinite(ev['avg_return']) else str(ev['avg_return']),
                        "success_rate": f"{ev['success_rate']:.4f}" if math.isfinite(ev['success_rate']) else str(ev['success_rate']),
                        "avg_episode_length": f"{ev['avg_episode_length']:.2f}" if math.isfinite(ev['avg_episode_length']) else str(ev['avg_episode_length']),
                        "checkpoint_path": os.path.relpath(ckpt_path, PROJECT_ROOT) if ckpt_path else "",
                        "status": status,
                        "notes": notes,
                    })

        with open(MAIN_SUMMARY, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            w.writerows(summary_rows)
        print()
        print(f"  Main summary: {MAIN_SUMMARY} ({len(summary_rows)} rows)")

    # ── Gate evaluation ───────────────────────────────────────────────────
    print()
    print("-- Gate evaluation -------------------------------------------------")
    gate = {}

    all_completed = all(r["status"] == "completed" for r in summary_rows)
    gate["all_160_completed"] = all_completed
    print(f"  [{'OK' if all_completed else 'FAIL'}] 160/160 completed")

    # Checkpoint existence
    ckpt_exist_ok, ckpt_exist_missing = check_main_checkpoints_exist(summary_rows)
    gate["all_160_checkpoints_exist"] = ckpt_exist_ok
    print(f"  [{'OK' if ckpt_exist_ok else 'FAIL'}] 160/160 checkpoints exist")
    if not ckpt_exist_ok:
        for m in ckpt_exist_missing[:5]:
            print(f"    missing: {m}")

    # Checkpoint loadability
    ckpt_load_ok, ckpt_load_failed = check_main_checkpoints_loadable(summary_rows)
    gate["all_160_checkpoints_loadable"] = ckpt_load_ok
    print(f"  [{'OK' if ckpt_load_ok else 'FAIL'}] 160/160 checkpoints loadable")
    if not ckpt_load_ok:
        for f in ckpt_load_failed[:5]:
            print(f"    load fail: {f}")

    all_loss_finite = all(math.isfinite(float(r["final_train_loss"])) for r in summary_rows)
    gate["all_losses_finite"] = all_loss_finite
    print(f"  [{'OK' if all_loss_finite else 'FAIL'}] all losses finite")

    all_eval_finite = all(
        math.isfinite(float(r["avg_return"])) and math.isfinite(float(r["success_rate"]))
        for r in summary_rows
    )
    gate["all_eval_finite"] = all_eval_finite
    print(f"  [{'OK' if all_eval_finite else 'FAIL'}] all eval finite")

    correct_160 = len(summary_rows) == 160
    gate["exactly_160_rows"] = correct_160
    print(f"  [{'OK' if correct_160 else 'FAIL'}] summary has 160 rows")

    all_20 = True
    for dn in MAIN_DATASETS:
        for algo in ALGORITHMS:
            cnt = sum(1 for r in summary_rows
                      if r["dataset_name"] == dn and r["algorithm"] == algo
                      and r["status"] == "completed")
            if cnt != 20:
                all_20 = False
                print(f"  FAIL: {dn} x {algo} has {cnt} completed, expected 20")
    gate["all_groups_20"] = all_20
    if all_20:
        print("  [OK] each dataset x algo group has 20 completed runs")

    print()
    go = all(gate.values())
    for k, v in gate.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    print()

    # ── Aggregated results (print only) ───────────────────────────────────
    if go:
        print("-- Aggregated results (20-seed) ------------------------------------")
        from scipy import stats as sp_stats

        for dn in MAIN_DATASETS:
            for algo in ALGORITHMS:
                rets = [float(r["avg_return"]) for r in summary_rows
                        if r["dataset_name"] == dn and r["algorithm"] == algo]
                srs = [float(r["success_rate"]) for r in summary_rows
                       if r["dataset_name"] == dn and r["algorithm"] == algo]

                ret_mean = np.mean(rets)
                ret_std  = np.std(rets, ddof=1)
                ret_se   = ret_std / math.sqrt(len(rets)) if ret_std > 0 else 0
                if ret_se > 0:
                    ret_ci = sp_stats.t.interval(0.95, df=len(rets)-1,
                                                 loc=ret_mean, scale=ret_se)
                else:
                    ret_ci = (ret_mean, ret_mean)

                sr_mean = np.mean(srs)
                sr_std  = np.std(srs, ddof=1)
                sr_se   = sr_std / math.sqrt(len(srs)) if sr_std > 0 else 0
                if sr_se > 0:
                    sr_ci = sp_stats.t.interval(0.95, df=len(srs)-1,
                                                loc=sr_mean, scale=sr_se)
                else:
                    sr_ci = (sr_mean, sr_mean)

                print(f"  {dn} x {algo}:")
                print(f"    return: mean={ret_mean:.4f}  std={ret_std:.4f}  "
                      f"95%CI=[{ret_ci[0]:.4f}, {ret_ci[1]:.4f}]")
                print(f"    SR:     mean={sr_mean:.4f}  std={sr_std:.4f}  "
                      f"95%CI=[{sr_ci[0]:.4f}, {sr_ci[1]:.4f}]")
        print()

    if go:
        print("Clean Phase 8: PASS")
    else:
        failed = [k for k, v in gate.items() if not v]
        print("Clean Phase 8: FAIL")
        print(f"  Failed: {failed}")
        sys.exit(1)
