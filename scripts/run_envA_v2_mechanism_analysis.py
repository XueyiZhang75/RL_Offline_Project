"""
scripts/run_envA_v2_mechanism_analysis.py
Clean Phase 11: read-only mechanism analysis on frozen EnvA_v2 data & checkpoints.

Computes per-run (360 runs) mechanism metrics:
  - OOD-action rate
  - Support overlap rate
  - Support distance proxy
  - Q-value overestimation proxy (CQL only)

Outputs:
  - artifacts/analysis/envA_v2_mechanism_seed_metrics.csv  (360 rows)
  - artifacts/analysis/envA_v2_mechanism_summary.csv       (18 rows)
"""

import sys, os, csv, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_envA_v2_sanity import (
    MLP, encode_single, OBS_DIM, N_ACTIONS, EVAL_EPISODES,
    AUDIT_PATH, MANIFEST_PATH,
)
from envs.gridworld_envs import EnvA_v2

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MAIN_SUMMARY    = os.path.join(PROJECT_ROOT, "artifacts", "training_main",
                               "envA_v2_main_summary.csv")
QUALITY_SUMMARY = os.path.join(PROJECT_ROOT, "artifacts", "training_quality",
                               "envA_v2_quality_summary.csv")
ANALYSIS_DIR    = os.path.join(PROJECT_ROOT, "artifacts", "analysis")
SEED_CSV        = os.path.join(ANALYSIS_DIR, "envA_v2_mechanism_seed_metrics.csv")
SUMMARY_CSV     = os.path.join(ANALYSIS_DIR, "envA_v2_mechanism_summary.csv")

# ── Constants ─────────────────────────────────────────────────────────────────

ENVA_V2_DATASETS = [
    "envA_v2_small_wide_medium", "envA_v2_small_narrow_medium",
    "envA_v2_large_wide_medium", "envA_v2_large_narrow_medium",
    "envA_v2_quality_random_wide50k", "envA_v2_quality_suboptimal_wide50k",
    "envA_v2_quality_medium_wide50k", "envA_v2_quality_expert_wide50k",
    "envA_v2_quality_mixed_wide50k",
]

SEED_COLUMNS = [
    "dataset_name", "quality_bin", "algorithm", "train_seed",
    "source_summary_type",
    "dataset_norm_state_cov", "dataset_norm_sa_cov", "dataset_state_entropy",
    "dataset_avg_return", "dataset_success_rate",
    "run_avg_return", "run_success_rate", "run_avg_episode_length",
    "ood_action_rate_step", "support_overlap_rate_unique_sa",
    "eval_sa_coverage_wrt_dataset_support", "support_distance_proxy",
    "q_over_rate", "q_over_mean_excess",
    "checkpoint_path", "status", "notes",
]

SUMMARY_COLUMNS = [
    "dataset_name", "quality_bin", "algorithm", "n_runs",
    "dataset_norm_state_cov", "dataset_norm_sa_cov", "dataset_state_entropy",
    "dataset_avg_return", "dataset_success_rate",
    "mean_run_return", "std_run_return",
    "ci95_run_return_low", "ci95_run_return_high",
    "mean_run_success_rate", "std_run_success_rate",
    "ci95_run_success_rate_low", "ci95_run_success_rate_high",
    "mean_ood_action_rate_step", "mean_support_overlap_rate_unique_sa",
    "mean_eval_sa_coverage_wrt_dataset_support", "mean_support_distance_proxy",
    "mean_q_over_rate", "mean_q_over_mean_excess",
    "notes",
]

MECH_EVAL_EPISODES = 100

# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_path(rel):
    return os.path.normpath(os.path.join(PROJECT_ROOT, rel.replace("\\", "/")))


def build_dataset_support(dataset_name):
    """Load .npz, return (support_sa set, dataset_metrics dict)."""
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        man_rows = list(csv.DictReader(f))
    mr = next((r for r in man_rows if r["dataset_name"] == dataset_name), None)
    assert mr is not None
    fpath = resolve_path(mr["file_path"])
    d = np.load(fpath, allow_pickle=True)
    obs = d["observations"]  # (N,2)
    acts = d["actions"]
    n = len(obs)
    support_sa = set()
    for i in range(n):
        support_sa.add((int(obs[i, 0]), int(obs[i, 1]), int(acts[i])))
    # Dataset metrics from audit
    with open(AUDIT_PATH, newline="", encoding="utf-8") as f:
        audit_rows = list(csv.DictReader(f))
    ar = next((r for r in audit_rows if r["dataset_name"] == dataset_name), None)
    metrics = {
        "norm_state_cov":   float(ar["normalized_state_coverage"]),
        "norm_sa_cov":      float(ar["normalized_state_action_coverage"]),
        "state_entropy":    float(ar["state_visitation_entropy"]),
        "avg_return":       float(ar["avg_return"]),
        "success_rate":     float(ar["success_rate"]),
        "return_p90":       float(ar["return_p90"]),
    }
    return support_sa, metrics


def mechanism_eval(model, support_sa, algo, dataset_name, dataset_metrics):
    """Run greedy eval on EnvA_v2, compute mechanism metrics. Returns dict."""
    env = EnvA_v2()
    total_steps = 0
    ood_steps = 0
    eval_unique_sa = set()
    returns, successes, lengths = [], [], []

    model.eval()
    with torch.no_grad():
        for _ in range(MECH_EVAL_EPISODES):
            obs, _ = env.reset()
            done, ep_ret, ep_len, success = False, 0.0, 0, False
            while not done:
                r, c = obs
                s_t = encode_single(r, c).unsqueeze(0)
                out = model(s_t)
                action = int(out.argmax(dim=1).item())
                eval_unique_sa.add((r, c, action))
                if (r, c, action) not in support_sa:
                    ood_steps += 1
                total_steps += 1
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_ret += reward
                ep_len += 1
                done = terminated or truncated
                if terminated:
                    success = True
            returns.append(ep_ret)
            successes.append(float(success))
            lengths.append(ep_len)

    # OOD action rate
    ood_rate = ood_steps / total_steps if total_steps > 0 else 0.0

    # Support overlap rate
    in_support = sum(1 for sa in eval_unique_sa if sa in support_sa)
    overlap_rate = in_support / len(eval_unique_sa) if eval_unique_sa else 0.0

    # Eval coverage wrt dataset support
    eval_cov = in_support / len(support_sa) if support_sa else 0.0

    # Support distance proxy
    dist_proxy = 1.0 - overlap_rate

    # Q-value overestimation proxy (CQL only)
    q_over_rate = ""
    q_over_mean_excess = ""
    if algo == "cql":
        return_p90 = dataset_metrics["return_p90"]
        # Get unique states from dataset support
        unique_states = list(set((s[0], s[1]) for s in support_sa))
        rng = np.random.RandomState(42)
        if len(unique_states) > 5000:
            idx = rng.choice(len(unique_states), 5000, replace=False)
            unique_states = [unique_states[i] for i in idx]
        max_qs = []
        for (r, c) in unique_states:
            s_t = encode_single(r, c).unsqueeze(0)
            q_vals = model(s_t)
            max_qs.append(q_vals.max().item())
        max_qs = np.array(max_qs)
        threshold = return_p90 + 0.05
        q_over_rate = f"{(max_qs > threshold).mean():.6f}"
        excess = np.maximum(0, max_qs - return_p90)
        q_over_mean_excess = f"{excess.mean():.6f}"

    return {
        "avg_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "avg_episode_length": float(np.mean(lengths)),
        "ood_action_rate_step": ood_rate,
        "support_overlap_rate_unique_sa": overlap_rate,
        "eval_sa_coverage_wrt_dataset_support": eval_cov,
        "support_distance_proxy": dist_proxy,
        "q_over_rate": q_over_rate,
        "q_over_mean_excess": q_over_mean_excess,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("Clean Phase 11: EnvA_v2 mechanism analysis (read-only)")
    print("=" * 66)
    print()

    # ── Pre-flights ───────────────────────────────────────────────────────
    print("-- Pre-flight A: freeze --------------------------------------------")
    with open(AUDIT_PATH, newline="", encoding="utf-8") as f:
        audit_rows = list(csv.DictReader(f))
    assert len(audit_rows) == 13
    for ar in audit_rows:
        assert ar["freeze_ready"] == "yes"
    print("  freeze_ready = yes: OK")

    print("-- Pre-flight B: main summary --------------------------------------")
    with open(MAIN_SUMMARY, newline="", encoding="utf-8") as f:
        main_rows = list(csv.DictReader(f))
    assert len(main_rows) == 160
    assert all(r["status"] == "completed" for r in main_rows)
    for r in main_rows:
        assert os.path.isfile(resolve_path(r["checkpoint_path"]))
    print("  160/160 main: OK")

    print("-- Pre-flight C: quality summary ------------------------------------")
    with open(QUALITY_SUMMARY, newline="", encoding="utf-8") as f:
        qual_rows = list(csv.DictReader(f))
    assert len(qual_rows) == 200
    assert all(r["status"] == "completed" for r in qual_rows)
    for r in qual_rows:
        assert os.path.isfile(resolve_path(r["checkpoint_path"]))
    print("  200/200 quality: OK")

    print("-- Pre-flight D: 9 EnvA_v2 datasets --------------------------------")
    for dn in ENVA_V2_DATASETS:
        support_sa, _ = build_dataset_support(dn)
        print(f"  {dn}: {len(support_sa)} SA pairs in support")
    print()

    # ── Build all runs list ───────────────────────────────────────────────
    all_runs = []
    for r in main_rows:
        if r["dataset_name"].startswith("envA_v2_"):
            all_runs.append({**r, "source": "main",
                             "quality_bin": "medium"})
    for r in qual_rows:
        if r["dataset_name"].startswith("envA_v2_"):
            all_runs.append({**r, "source": "quality"})
    print(f"  Total runs to analyze: {len(all_runs)}")
    assert len(all_runs) == 360
    print()

    # ── Cache dataset supports ────────────────────────────────────────────
    ds_cache = {}
    for dn in ENVA_V2_DATASETS:
        ds_cache[dn] = build_dataset_support(dn)

    # ── Analyze each run ──────────────────────────────────────────────────
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    seed_rows = []
    count = 0

    for run in all_runs:
        count += 1
        dn = run["dataset_name"]
        algo = run["algorithm"]
        seed = int(run["train_seed"])
        cp = run["checkpoint_path"]
        source = run["source"]
        qbin = run.get("quality_bin", "medium")

        support_sa, ds_metrics = ds_cache[dn]

        print(f"  [{count}/360] {dn}_{algo}_s{seed}...", end="", flush=True)

        # Load checkpoint
        ckpt = torch.load(resolve_path(cp), weights_only=False)
        hidden = ckpt["config"]["hidden_dims"]
        model = MLP(OBS_DIM, N_ACTIONS, hidden)
        model.load_state_dict(ckpt["model_state_dict"])

        # Mechanism eval
        mech = mechanism_eval(model, support_sa, algo, dn, ds_metrics)

        print(f" OOD={mech['ood_action_rate_step']:.3f}  "
              f"overlap={mech['support_overlap_rate_unique_sa']:.3f}  "
              f"ret={mech['avg_return']:.3f}")

        seed_rows.append({
            "dataset_name": dn,
            "quality_bin": qbin,
            "algorithm": algo,
            "train_seed": seed,
            "source_summary_type": source,
            "dataset_norm_state_cov": f"{ds_metrics['norm_state_cov']:.6f}",
            "dataset_norm_sa_cov": f"{ds_metrics['norm_sa_cov']:.6f}",
            "dataset_state_entropy": f"{ds_metrics['state_entropy']:.4f}",
            "dataset_avg_return": f"{ds_metrics['avg_return']:.4f}",
            "dataset_success_rate": f"{ds_metrics['success_rate']:.4f}",
            "run_avg_return": f"{mech['avg_return']:.4f}",
            "run_success_rate": f"{mech['success_rate']:.4f}",
            "run_avg_episode_length": f"{mech['avg_episode_length']:.2f}",
            "ood_action_rate_step": f"{mech['ood_action_rate_step']:.6f}",
            "support_overlap_rate_unique_sa": f"{mech['support_overlap_rate_unique_sa']:.6f}",
            "eval_sa_coverage_wrt_dataset_support": f"{mech['eval_sa_coverage_wrt_dataset_support']:.6f}",
            "support_distance_proxy": f"{mech['support_distance_proxy']:.6f}",
            "q_over_rate": mech["q_over_rate"],
            "q_over_mean_excess": mech["q_over_mean_excess"],
            "checkpoint_path": cp,
            "status": "completed",
            "notes": "mechanism analysis; eval_episodes=100; greedy policy",
        })

    # ── Write seed-level CSV ──────────────────────────────────────────────
    with open(SEED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SEED_COLUMNS)
        w.writeheader()
        w.writerows(seed_rows)
    print()
    print(f"  Seed metrics: {SEED_CSV} ({len(seed_rows)} rows)")

    # ── Compute summary ──────────────────────────────────────────────────
    from scipy import stats as sp_stats
    summary_rows = []

    for dn in ENVA_V2_DATASETS:
        _, ds_metrics = ds_cache[dn]
        for algo in ["bc", "cql"]:
            runs = [r for r in seed_rows
                    if r["dataset_name"] == dn and r["algorithm"] == algo]
            n = len(runs)
            rets = [float(r["run_avg_return"]) for r in runs]
            srs = [float(r["run_success_rate"]) for r in runs]
            oods = [float(r["ood_action_rate_step"]) for r in runs]
            overlaps = [float(r["support_overlap_rate_unique_sa"]) for r in runs]
            covs = [float(r["eval_sa_coverage_wrt_dataset_support"]) for r in runs]
            dists = [float(r["support_distance_proxy"]) for r in runs]

            ret_m, ret_s = np.mean(rets), np.std(rets, ddof=1)
            ret_se = ret_s / math.sqrt(n) if ret_s > 0 else 0
            ret_ci = (sp_stats.t.interval(0.95, df=n-1, loc=ret_m, scale=ret_se)
                      if ret_se > 0 else (ret_m, ret_m))
            sr_m, sr_s = np.mean(srs), np.std(srs, ddof=1)
            sr_se = sr_s / math.sqrt(n) if sr_s > 0 else 0
            sr_ci = (sp_stats.t.interval(0.95, df=n-1, loc=sr_m, scale=sr_se)
                     if sr_se > 0 else (sr_m, sr_m))

            q_rates = [r["q_over_rate"] for r in runs if r["q_over_rate"] not in ("", "NA")]
            q_excs = [r["q_over_mean_excess"] for r in runs if r["q_over_mean_excess"] not in ("", "NA")]

            qbin = runs[0]["quality_bin"] if runs else ""

            summary_rows.append({
                "dataset_name": dn,
                "quality_bin": qbin,
                "algorithm": algo,
                "n_runs": n,
                "dataset_norm_state_cov": f"{ds_metrics['norm_state_cov']:.6f}",
                "dataset_norm_sa_cov": f"{ds_metrics['norm_sa_cov']:.6f}",
                "dataset_state_entropy": f"{ds_metrics['state_entropy']:.4f}",
                "dataset_avg_return": f"{ds_metrics['avg_return']:.4f}",
                "dataset_success_rate": f"{ds_metrics['success_rate']:.4f}",
                "mean_run_return": f"{ret_m:.4f}",
                "std_run_return": f"{ret_s:.4f}",
                "ci95_run_return_low": f"{ret_ci[0]:.4f}",
                "ci95_run_return_high": f"{ret_ci[1]:.4f}",
                "mean_run_success_rate": f"{sr_m:.4f}",
                "std_run_success_rate": f"{sr_s:.4f}",
                "ci95_run_success_rate_low": f"{sr_ci[0]:.4f}",
                "ci95_run_success_rate_high": f"{sr_ci[1]:.4f}",
                "mean_ood_action_rate_step": f"{np.mean(oods):.6f}",
                "mean_support_overlap_rate_unique_sa": f"{np.mean(overlaps):.6f}",
                "mean_eval_sa_coverage_wrt_dataset_support": f"{np.mean(covs):.6f}",
                "mean_support_distance_proxy": f"{np.mean(dists):.6f}",
                "mean_q_over_rate": f"{np.mean([float(x) for x in q_rates]):.6f}" if q_rates else "",
                "mean_q_over_mean_excess": f"{np.mean([float(x) for x in q_excs]):.6f}" if q_excs else "",
                "notes": "mechanism summary; 20-seed aggregate",
            })

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  Summary: {SUMMARY_CSV} ({len(summary_rows)} rows)")

    # ── Gate ──────────────────────────────────────────────────────────────
    print()
    print("-- Gate evaluation -------------------------------------------------")
    gate = {}
    gate["360_rows"] = len(seed_rows) == 360
    gate["18_summary"] = len(summary_rows) == 18

    oods_all = [float(r["ood_action_rate_step"]) for r in seed_rows]
    overlaps_all = [float(r["support_overlap_rate_unique_sa"]) for r in seed_rows]
    covs_all = [float(r["eval_sa_coverage_wrt_dataset_support"]) for r in seed_rows]
    dists_all = [float(r["support_distance_proxy"]) for r in seed_rows]

    gate["ood_in_01"] = all(0 <= v <= 1 for v in oods_all)
    gate["overlap_in_01"] = all(0 <= v <= 1 for v in overlaps_all)
    gate["cov_in_01"] = all(0 <= v <= 1 for v in covs_all)
    gate["dist_in_01"] = all(0 <= v <= 1 for v in dists_all)

    bc_q = [r for r in seed_rows if r["algorithm"] == "bc"]
    gate["bc_q_empty"] = all(r["q_over_rate"] in ("", "NA") for r in bc_q)
    cql_q = [r for r in seed_rows if r["algorithm"] == "cql"]
    gate["cql_q_finite"] = all(
        r["q_over_rate"] not in ("", "NA") and math.isfinite(float(r["q_over_rate"]))
        for r in cql_q)

    go = all(gate.values())
    for k, v in gate.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    print()

    # ── Aggregated views (print only) ─────────────────────────────────────
    if go:
        print("-- A. Main four mechanism view --------------------------------------")
        main_four = ["envA_v2_small_wide_medium", "envA_v2_small_narrow_medium",
                      "envA_v2_large_wide_medium", "envA_v2_large_narrow_medium"]
        for sr in summary_rows:
            if sr["dataset_name"] in main_four:
                print(f"  {sr['dataset_name']} x {sr['algorithm']}:  "
                      f"sa_cov={sr['dataset_norm_sa_cov']}  "
                      f"ret={sr['mean_run_return']}  "
                      f"OOD={sr['mean_ood_action_rate_step']}  "
                      f"overlap={sr['mean_support_overlap_rate_unique_sa']}  "
                      f"dist={sr['mean_support_distance_proxy']}")
        print()

        print("-- B. Coverage-performance ordering --------------------------------")
        sorted_sum = sorted(summary_rows,
                            key=lambda r: float(r["dataset_norm_sa_cov"]))
        for sr in sorted_sum:
            print(f"  {sr['dataset_name']:45s} {sr['algorithm']:4s}  "
                  f"sa_cov={sr['dataset_norm_sa_cov']}  "
                  f"ret={sr['mean_run_return']}  SR={sr['mean_run_success_rate']}")
        print()

        print("-- C. CQL overestimation view ---------------------------------------")
        for sr in summary_rows:
            if sr["algorithm"] == "cql":
                print(f"  {sr['dataset_name']:45s}  "
                      f"q_over_rate={sr['mean_q_over_rate']:>10s}  "
                      f"q_excess={sr['mean_q_over_mean_excess']:>10s}  "
                      f"ret={sr['mean_run_return']}")
        print()

        print("-- D. BC vs CQL mechanism gap ---------------------------------------")
        for dn in ENVA_V2_DATASETS:
            bc_sr = next(r for r in summary_rows if r["dataset_name"] == dn and r["algorithm"] == "bc")
            cql_sr = next(r for r in summary_rows if r["dataset_name"] == dn and r["algorithm"] == "cql")
            ret_gap = float(cql_sr["mean_run_return"]) - float(bc_sr["mean_run_return"])
            ood_gap = float(cql_sr["mean_ood_action_rate_step"]) - float(bc_sr["mean_ood_action_rate_step"])
            dist_gap = float(cql_sr["mean_support_distance_proxy"]) - float(bc_sr["mean_support_distance_proxy"])
            print(f"  {dn:45s}  ret_gap={ret_gap:+.4f}  ood_gap={ood_gap:+.4f}  dist_gap={dist_gap:+.4f}")
        print()

    if go:
        print("Clean Phase 11: PASS")
    else:
        failed = [k for k, v in gate.items() if not v]
        print("Clean Phase 11: FAIL")
        print(f"  Failed: {failed}")
        sys.exit(1)
