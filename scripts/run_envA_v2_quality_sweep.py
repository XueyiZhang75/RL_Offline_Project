"""
scripts/run_envA_v2_quality_sweep.py
Clean Phase 10: quality sweep — 5 datasets x 2 algos x 20 seeds = 200 runs.

Reuses frozen framework from run_envA_v2_sanity.py.
Supports reuse mode: skips training if complete summary already exists.
"""

import sys, os, csv, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_envA_v2_sanity import (
    BC_CFG, CQL_CFG, OBS_DIM, N_ACTIONS, EVAL_EPISODES,
    MLP, encode_obs, encode_single, load_dataset,
    train_bc, train_cql, evaluate,
    AUDIT_PATH, MANIFEST_PATH,
)
from scripts.run_envA_v2_main_experiment import (
    MAIN_SUMMARY as ENVA_MAIN_SUMMARY,
    resolve_ckpt_path,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
VAL_SUMMARY  = os.path.join(PROJECT_ROOT, "artifacts", "training_validation",
                            "envbc_validation_summary.csv")
QUALITY_DIR  = os.path.join(PROJECT_ROOT, "artifacts", "training_quality")
QUALITY_SUMMARY = os.path.join(QUALITY_DIR, "envA_v2_quality_summary.csv")

# ── Frozen constants ──────────────────────────────────────────────────────────

QUALITY_SEEDS = list(range(20))
QUALITY_DATASETS = [
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
ALGORITHMS = ["bc", "cql"]

SUMMARY_COLUMNS = [
    "dataset_name", "quality_bin", "algorithm", "train_seed",
    "num_updates", "final_train_loss",
    "eval_episodes", "avg_return", "success_rate", "avg_episode_length",
    "checkpoint_path", "status", "notes",
]

REQUIRED_KEYS = {
    "observations", "actions", "rewards", "next_observations",
    "terminals", "truncations", "episode_ids", "timesteps",
    "source_policy_ids", "source_train_seeds", "source_behavior_epsilons",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_qpath(rel_path):
    return os.path.normpath(os.path.join(PROJECT_ROOT, rel_path.replace("\\", "/")))


def check_summary_loadable(summary_path, expected_rows):
    with open(summary_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == expected_rows, f"{len(rows)} != {expected_rows}"
    assert all(r["status"] == "completed" for r in rows)
    for r in rows:
        cp = r.get("checkpoint_path", "").strip()
        assert cp, f"Empty checkpoint_path"
        assert os.path.isfile(resolve_qpath(cp)), f"Missing: {cp}"
        torch.load(resolve_qpath(cp), weights_only=False)
    return rows


def existing_quality_summary_complete():
    if not os.path.isfile(QUALITY_SUMMARY):
        return False, []
    with open(QUALITY_SUMMARY, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 200:
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
    print("Clean Phase 10: EnvA_v2 quality sweep")
    print(f"  5 datasets x 2 algos x 20 seeds = 200 runs")
    print("=" * 66)
    print()

    # ── Pre-flight A: freeze ──────────────────────────────────────────────
    print("-- Pre-flight A: freeze --------------------------------------------")
    with open(AUDIT_PATH, newline="", encoding="utf-8") as f:
        audit_rows = list(csv.DictReader(f))
    assert len(audit_rows) == 13
    for ar in audit_rows:
        assert ar["freeze_ready"] == "yes"
    print("  freeze_ready = yes for all 13: OK")

    # ── Pre-flight B: main experiment ─────────────────────────────────────
    print("-- Pre-flight B: main experiment ------------------------------------")
    check_summary_loadable(ENVA_MAIN_SUMMARY, 160)
    print("  160/160 main runs completed, checkpoints exist & loadable: OK")

    # ── Pre-flight C: validation ──────────────────────────────────────────
    print("-- Pre-flight C: EnvB/C validation ----------------------------------")
    check_summary_loadable(VAL_SUMMARY, 160)
    print("  160/160 validation runs completed, checkpoints exist & loadable: OK")

    # ── Pre-flight D: quality datasets ────────────────────────────────────
    print("-- Pre-flight D: quality datasets -----------------------------------")
    for dn in QUALITY_DATASETS:
        d_obs, d_acts, d_rews, d_nobs, d_terms = load_dataset(dn)
        print(f"  {dn}: {len(d_obs)} trans, loaded OK")

    # ── Pre-flight E: dataset schema ──────────────────────────────────────
    print("-- Pre-flight E: dataset schema -------------------------------------")
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        man_rows = list(csv.DictReader(f))
    for dn in QUALITY_DATASETS:
        mr = next((r for r in man_rows if r["dataset_name"] == dn), None)
        assert mr is not None, f"{dn} not in manifest"
        fpath = os.path.join(PROJECT_ROOT, mr["file_path"])
        d = np.load(fpath, allow_pickle=True)
        missing = REQUIRED_KEYS - set(d.files)
        assert not missing, f"{dn} missing keys: {missing}"
        n = len(d["observations"])
        for k in REQUIRED_KEYS:
            assert len(d[k]) == n, f"{dn} length mismatch on {k}"
        assert d["observations"].shape == (n, 2), (
            f"{dn} obs shape {d['observations'].shape} != (N,2)")
        print(f"  {dn}: schema OK")
    print()

    # ── Print frozen configs ──────────────────────────────────────────────
    print("-- Frozen configs (inherited from sanity) ---------------------------")
    print(f"  BC_CFG:  {BC_CFG}")
    print(f"  CQL_CFG: {CQL_CFG}")
    print(f"  QUALITY_SEEDS: {QUALITY_SEEDS[0]}..{QUALITY_SEEDS[-1]} ({len(QUALITY_SEEDS)} seeds)")
    print()

    # ── Decide: reuse or train ────────────────────────────────────────────
    complete, existing_rows = existing_quality_summary_complete()

    if complete:
        print("-- Reuse mode: existing summary is complete (200 rows) -------------")
        print("  Skipping training. Running read-only re-verification.")
        summary_rows = existing_rows
    else:
        print("-- Training mode: running 200 quality runs --------------------------")
        os.makedirs(QUALITY_DIR, exist_ok=True)
        summary_rows = []
        run_count = 0
        total_runs = len(QUALITY_DATASETS) * len(ALGORITHMS) * len(QUALITY_SEEDS)

        for dn in QUALITY_DATASETS:
            obs, acts, rews, nobs, terms = load_dataset(dn)
            qbin = QUALITY_BIN_MAP[dn]
            print(f"-- {dn} [{qbin}] ({len(obs)} trans) --")

            for algo in ALGORITHMS:
                for seed in QUALITY_SEEDS:
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
                        ckpt_path = os.path.join(QUALITY_DIR, ckpt_name)
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "algorithm": algo,
                            "dataset_name": dn,
                            "quality_bin": qbin,
                            "train_seed": seed,
                            "num_updates": num_updates,
                            "final_train_loss": final_loss,
                            "obs_dim": OBS_DIM,
                            "n_actions": N_ACTIONS,
                            "config": BC_CFG if algo == "bc" else CQL_CFG,
                        }, ckpt_path)

                        ev = evaluate(model, algo)
                        status = "completed"
                        notes = (f"one-hot 900-d; quality sweep; "
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
                        "quality_bin": qbin,
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

        with open(QUALITY_SUMMARY, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            w.writerows(summary_rows)
        print()
        print(f"  Quality summary: {QUALITY_SUMMARY} ({len(summary_rows)} rows)")

    # ── Gate evaluation ───────────────────────────────────────────────────
    print()
    print("-- Gate evaluation -------------------------------------------------")
    gate = {}

    all_completed = all(r["status"] == "completed" for r in summary_rows)
    gate["all_200_completed"] = all_completed
    print(f"  [{'OK' if all_completed else 'FAIL'}] 200/200 completed")

    all_exist = all(
        r.get("checkpoint_path", "").strip() and
        os.path.isfile(resolve_qpath(r["checkpoint_path"]))
        for r in summary_rows
    )
    gate["all_200_ckpts_exist"] = all_exist
    print(f"  [{'OK' if all_exist else 'FAIL'}] 200/200 checkpoints exist")

    all_loadable = True
    for r in summary_rows:
        cp = r.get("checkpoint_path", "").strip()
        if not cp:
            all_loadable = False; break
        try:
            torch.load(resolve_qpath(cp), weights_only=False)
        except:
            all_loadable = False; break
    gate["all_200_ckpts_loadable"] = all_loadable
    print(f"  [{'OK' if all_loadable else 'FAIL'}] 200/200 checkpoints loadable")

    all_loss_finite = all(math.isfinite(float(r["final_train_loss"])) for r in summary_rows)
    gate["all_losses_finite"] = all_loss_finite
    print(f"  [{'OK' if all_loss_finite else 'FAIL'}] all losses finite")

    all_eval_finite = all(
        math.isfinite(float(r["avg_return"])) and math.isfinite(float(r["success_rate"]))
        for r in summary_rows
    )
    gate["all_eval_finite"] = all_eval_finite
    print(f"  [{'OK' if all_eval_finite else 'FAIL'}] all eval finite")

    gate["exactly_200_rows"] = len(summary_rows) == 200
    print(f"  [{'OK' if gate['exactly_200_rows'] else 'FAIL'}] summary has 200 rows")

    all_20 = True
    for dn in QUALITY_DATASETS:
        for algo in ALGORITHMS:
            cnt = sum(1 for r in summary_rows
                      if r["dataset_name"] == dn and r["algorithm"] == algo
                      and r["status"] == "completed")
            if cnt != 20:
                all_20 = False
    gate["all_groups_20"] = all_20
    print(f"  [{'OK' if all_20 else 'FAIL'}] each group has 20 runs")

    print()
    go = all(gate.values())
    for k, v in gate.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    print()

    # ── Aggregated results ────────────────────────────────────────────────
    if go:
        print("-- Aggregated results (20-seed) ------------------------------------")
        from scipy import stats as sp_stats

        agg = {}
        for dn in QUALITY_DATASETS:
            qbin = QUALITY_BIN_MAP[dn]
            for algo in ALGORITHMS:
                rets = [float(r["avg_return"]) for r in summary_rows
                        if r["dataset_name"] == dn and r["algorithm"] == algo]
                srs = [float(r["success_rate"]) for r in summary_rows
                       if r["dataset_name"] == dn and r["algorithm"] == algo]

                ret_mean = np.mean(rets)
                ret_std = np.std(rets, ddof=1)
                ret_se = ret_std / math.sqrt(len(rets)) if ret_std > 0 else 0
                ret_ci = (sp_stats.t.interval(0.95, df=len(rets)-1, loc=ret_mean, scale=ret_se)
                          if ret_se > 0 else (ret_mean, ret_mean))
                sr_mean = np.mean(srs)
                sr_std = np.std(srs, ddof=1)
                sr_se = sr_std / math.sqrt(len(srs)) if sr_std > 0 else 0
                sr_ci = (sp_stats.t.interval(0.95, df=len(srs)-1, loc=sr_mean, scale=sr_se)
                         if sr_se > 0 else (sr_mean, sr_mean))

                agg[(qbin, algo)] = {"ret_mean": ret_mean, "sr_mean": sr_mean}
                print(f"  {qbin} x {algo}:")
                print(f"    return: mean={ret_mean:.4f}  std={ret_std:.4f}  "
                      f"95%CI=[{ret_ci[0]:.4f}, {ret_ci[1]:.4f}]")
                print(f"    SR:     mean={sr_mean:.4f}  std={sr_std:.4f}  "
                      f"95%CI=[{sr_ci[0]:.4f}, {sr_ci[1]:.4f}]")

        print()
        print("-- Quality ordering ------------------------------------------------")
        order = ["random", "suboptimal", "medium", "expert", "mixed"]
        for algo in ALGORITHMS:
            rets_str = " -> ".join(f"{q}={agg[(q,algo)]['ret_mean']:.4f}" for q in order)
            srs_str  = " -> ".join(f"{q}={agg[(q,algo)]['sr_mean']:.4f}" for q in order)
            print(f"  {algo.upper()} return: {rets_str}")
            print(f"  {algo.upper()} SR:     {srs_str}")

        print()
        print("-- CQL - BC mean return gap per quality dataset ---------------------")
        for qbin in order:
            diff = agg[(qbin, "cql")]["ret_mean"] - agg[(qbin, "bc")]["ret_mean"]
            print(f"  {qbin}: CQL - BC = {diff:+.4f}")
        print()

    if go:
        print("Clean Phase 10: PASS")
    else:
        failed = [k for k, v in gate.items() if not v]
        print("Clean Phase 10: FAIL")
        print(f"  Failed: {failed}")
        sys.exit(1)
