"""
scripts/generate_envA_v2_final_datasets.py
Clean Phase 5: generate 9 formal EnvA_v2 datasets + local pre-audit + manifest update.

Inherits frozen semantics from verify_envA_v2_proxy_gate.py.
Reads actual artifacts from active behavior_policy_catalog.csv.
Verifies pilot passed before generating.

9 datasets:
  4 main experiment (medium, coverage x size)
  5 quality sweep (wide, 50k)

Hard gate: local pre-audit must pass before manifest update.
"""

import sys, os, csv, collections
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.verify_envA_v2_proxy_gate import (
    FAMILIES, SEED_FAMILY_MAP, TOUR_WAYPOINTS,
    ACTION_DELTAS, OPPOSITE_ACTION, G,
    get_table, get_delay_action,
)
from envs.gridworld_envs import EnvA_v2, HORIZON, N_ACTIONS

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CATALOG_PATH = os.path.join(PROJECT_ROOT, "artifacts", "behavior_pool",
                            "behavior_policy_catalog.csv")
PILOT_SUMMARY = os.path.join(PROJECT_ROOT, "artifacts", "pilot",
                             "envA_v2_pilot_summary.csv")
DATASET_DIR  = os.path.join(PROJECT_ROOT, "artifacts", "final_datasets")
MANIFEST_PATH = os.path.join(DATASET_DIR, "final_dataset_manifest.csv")

GOAL = EnvA_v2.goal_pos
START = EnvA_v2.start_pos
_env_tmp = EnvA_v2()
TOTAL_STATES = len(_env_tmp.get_reachable_states())
TOTAL_SA     = len(_env_tmp.get_reachable_state_action_pairs())
WALLS        = _env_tmp._walls
del _env_tmp

# ── Frozen generation seeds ───────────────────────────────────────────────────

GEN_SEEDS = {
    "envA_v2_small_wide_medium":          100,
    "envA_v2_small_narrow_medium":        101,
    "envA_v2_large_wide_medium":          102,
    "envA_v2_large_narrow_medium":        103,
    "envA_v2_quality_random_wide50k":     110,
    "envA_v2_quality_suboptimal_wide50k": 111,
    "envA_v2_quality_medium_wide50k":     112,
    "envA_v2_quality_expert_wide50k":     113,
    "envA_v2_quality_mixed_wide50k":      114,
}

# ── Frozen source compositions ────────────────────────────────────────────────

NARROW_MEDIUM_IDS = ["envA_v2_seed0_medium_controller"]

WIDE_MEDIUM_IDS = [
    "envA_v2_seed0_medium_controller",
    "envA_v2_seed2_medium_controller",
    "envA_v2_seed3_medium_controller",
    "envA_v2_seed4_medium_controller",
    "envA_v2_seed5_medium_controller",
    "envA_v2_seed6_medium_controller",
]

WIDE_SUBOPTIMAL_IDS = [
    "envA_v2_seed0_suboptimal_controller",
    "envA_v2_seed2_suboptimal_controller",
    "envA_v2_seed3_suboptimal_controller",
    "envA_v2_seed4_suboptimal_controller",
    "envA_v2_seed5_suboptimal_controller",
    "envA_v2_seed6_suboptimal_controller",
]

WIDE_EXPERT_IDS = [
    "envA_v2_seed0_expert_controller",
    "envA_v2_seed2_expert_controller",
    "envA_v2_seed3_expert_controller",
    "envA_v2_seed4_expert_controller",
    "envA_v2_seed5_expert_controller",
    "envA_v2_seed6_expert_controller",
]

# Mixed tier cycle: random / medium / medium / expert (4-episode period)
MIXED_TIER_CYCLE = ["random", "medium", "medium", "expert"]

# ── Dataset specs ─────────────────────────────────────────────────────────────

DATASET_SPECS = [
    {"name": "envA_v2_small_wide_medium",    "target": 50_000,  "mode": "wide",   "bin": "medium"},
    {"name": "envA_v2_small_narrow_medium",  "target": 50_000,  "mode": "narrow", "bin": "medium"},
    {"name": "envA_v2_large_wide_medium",    "target": 200_000, "mode": "wide",   "bin": "medium"},
    {"name": "envA_v2_large_narrow_medium",  "target": 200_000, "mode": "narrow", "bin": "medium"},
    {"name": "envA_v2_quality_random_wide50k",     "target": 50_000, "mode": "wide", "bin": "random"},
    {"name": "envA_v2_quality_suboptimal_wide50k",  "target": 50_000, "mode": "wide", "bin": "suboptimal"},
    {"name": "envA_v2_quality_medium_wide50k",      "target": 50_000, "mode": "wide", "bin": "medium"},
    {"name": "envA_v2_quality_expert_wide50k",      "target": 50_000, "mode": "wide", "bin": "expert"},
    {"name": "envA_v2_quality_mixed_wide50k",       "target": 50_000, "mode": "wide", "bin": "mixed"},
]

MANIFEST_COLUMNS = [
    "dataset_name", "env_name", "target_transitions", "actual_transitions",
    "size_mode", "coverage_mode", "quality_bin",
    "num_source_policies", "source_policy_ids", "generation_seed",
    "file_path",
    "normalized_state_coverage", "normalized_state_action_coverage",
    "avg_return", "success_rate", "avg_episode_length",
    "status", "notes",
]

# ── Catalog loading ───────────────────────────────────────────────────────────

def load_catalog_selected(env_name="EnvA_v2"):
    with open(CATALOG_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = {}
    for r in rows:
        if r["env_name"] == env_name and r["selected"] == "yes":
            result[r["policy_id"]] = r
    return result


def load_artifact(catalog_row):
    rel_path = catalog_row["checkpoint_path"]
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    return torch.load(abs_path, weights_only=False)


def art_to_ctrl(art):
    return {
        "waypoints": [tuple(w) for w in art["waypoints"]],
        "delay_prob": art["delay_prob"],
        "policy_id": f"envA_v2_seed{art['controller_seed']}_{art['quality_bin']}_controller",
        "train_seed": art["controller_seed"],
    }


# ── Rollout engine ────────────────────────────────────────────────────────────

def generate_dataset(controller_schedule, target_trans, gen_seed):
    """
    controller_schedule: callable(episode_idx) -> ctrl_dict or "random"
    Returns dict with all 11 required arrays + metrics.
    """
    env = EnvA_v2()
    rng = np.random.default_rng(gen_seed)

    obs_list, act_list, rew_list, nobs_list = [], [], [], []
    term_list, trunc_list = [], []
    ep_id_list, ts_list = [], []
    pid_list, seed_list, eps_list = [], [], []

    total = 0
    ep_idx = 0
    returns, successes, lengths = [], [], []

    while total < target_trans:
        ctrl = controller_schedule(ep_idx)
        is_random = (ctrl == "random")

        obs_raw, _ = env.reset()
        pos = obs_raw
        wi = 0
        ep_ret, ep_len = 0.0, 0
        done = False
        success = False

        if is_random:
            pid_str = "envA_v2_random"
            t_seed = -1
            b_eps = 1.0
        else:
            pid_str = ctrl["policy_id"]
            t_seed = ctrl["train_seed"]
            b_eps = 0.0
            wps = ctrl["waypoints"]
            dp = ctrl["delay_prob"]

        while not done:
            if is_random:
                action = int(rng.integers(0, N_ACTIONS))
            else:
                while wi < len(wps) and pos == wps[wi]:
                    wi += 1
                target = wps[wi] if wi < len(wps) else GOAL
                tbl = get_table(target)
                bfs_a = tbl.get(pos, 0)
                if rng.random() < dp:
                    action = get_delay_action(pos, bfs_a)
                else:
                    action = bfs_a

            obs_list.append(pos)
            act_list.append(action)
            pid_list.append(pid_str)
            seed_list.append(t_seed)
            eps_list.append(b_eps)
            ep_id_list.append(ep_idx)
            ts_list.append(ep_len)

            obs_raw, r, terminated, truncated, _ = env.step(action)
            npos = obs_raw
            rew_list.append(r)
            nobs_list.append(npos)
            term_list.append(terminated)
            trunc_list.append(truncated)

            pos = npos
            ep_ret += r
            ep_len += 1
            total += 1
            done = terminated or truncated
            if terminated:
                success = True

        returns.append(ep_ret)
        successes.append(float(success))
        lengths.append(ep_len)
        ep_idx += 1

    obs_arr  = np.array(obs_list,  dtype=np.int32)
    nobs_arr = np.array(nobs_list, dtype=np.int32)
    act_arr  = np.array(act_list,  dtype=np.int32)
    rew_arr  = np.array(rew_list,  dtype=np.float32)
    term_arr = np.array(term_list, dtype=bool)
    truc_arr = np.array(trunc_list, dtype=bool)
    epid_arr = np.array(ep_id_list, dtype=np.int32)
    ts_arr   = np.array(ts_list,   dtype=np.int32)
    seed_arr = np.array(seed_list, dtype=np.int32)
    beps_arr = np.array(eps_list,  dtype=np.float32)
    pid_arr  = np.array(pid_list,  dtype=object)

    unique_states = set(map(tuple, obs_arr.tolist()))
    unique_sa = set()
    for i in range(len(obs_arr)):
        unique_sa.add((tuple(obs_arr[i]), int(act_arr[i])))

    return {
        "observations": obs_arr,
        "actions": act_arr,
        "rewards": rew_arr,
        "next_observations": nobs_arr,
        "terminals": term_arr,
        "truncations": truc_arr,
        "episode_ids": epid_arr,
        "timesteps": ts_arr,
        "source_policy_ids": pid_arr,
        "source_train_seeds": seed_arr,
        "source_behavior_epsilons": beps_arr,
        # metrics
        "actual_transitions": total,
        "num_episodes": ep_idx,
        "unique_state_count": len(unique_states),
        "unique_sa_count": len(unique_sa),
        "norm_state_cov": len(unique_states) / TOTAL_STATES,
        "norm_sa_cov": len(unique_sa) / TOTAL_SA,
        "success_rate": float(np.mean(successes)),
        "avg_return": float(np.mean(returns)),
        "avg_episode_length": float(np.mean(lengths)),
        "return_std": float(np.std(returns)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("Clean Phase 5: Generate 9 EnvA_v2 formal datasets")
    print("=" * 66)
    print()

    # ── Pre-flight A: behavior pool ───────────────────────────────────────
    print("-- Pre-flight A: behavior pool ------------------------------------")
    cat = load_catalog_selected("EnvA_v2")
    for qbin in ["suboptimal", "medium", "expert"]:
        cnt = sum(1 for pid, r in cat.items() if r["quality_bin"] == qbin)
        seeds = sorted(int(r["train_seed"]) for pid, r in cat.items()
                       if r["quality_bin"] == qbin)
        print(f"  {qbin}: {cnt} selected, seeds={seeds}")
        assert cnt == 8, f"{qbin} count {cnt} != 8"
        assert seeds == list(range(8)), f"{qbin} seeds != 0..7"
    print("  behavior pool pre-flight: OK")
    print()

    # ── Pre-flight B: pilot ───────────────────────────────────────────────
    print("-- Pre-flight B: pilot summary ------------------------------------")
    with open(PILOT_SUMMARY, newline="", encoding="utf-8") as f:
        pilot_rows = list(csv.DictReader(f))
    agg = [r for r in pilot_rows if r["record_type"] == "aggregate"]
    assert len(agg) == 1, "Expected 1 aggregate row in pilot summary"
    assert agg[0]["all_3_pass"] == "yes", "Pilot all_3_pass != yes"
    assert float(agg[0]["min_gap"]) >= 0.10, "Pilot min_gap < 0.10"
    print(f"  all_3_pass={agg[0]['all_3_pass']}  min_gap={agg[0]['min_gap']}")
    print("  pilot pre-flight: OK")
    print()

    # ── Load all needed artifacts ─────────────────────────────────────────
    def load_ctrl_list(pid_list):
        return [art_to_ctrl(load_artifact(cat[pid])) for pid in pid_list]

    narrow_med = load_ctrl_list(NARROW_MEDIUM_IDS)
    wide_med   = load_ctrl_list(WIDE_MEDIUM_IDS)
    wide_sub   = load_ctrl_list(WIDE_SUBOPTIMAL_IDS)
    wide_exp   = load_ctrl_list(WIDE_EXPERT_IDS)

    # ── Build scheduler factories ─────────────────────────────────────────
    def make_rr_scheduler(ctrl_list):
        def sched(ep_idx):
            return ctrl_list[ep_idx % len(ctrl_list)]
        return sched

    def make_narrow_scheduler():
        return make_rr_scheduler(narrow_med)

    def make_random_scheduler():
        def sched(ep_idx):
            return "random"
        return sched

    def make_mixed_scheduler():
        med_idx = [0]
        exp_idx = [0]
        def sched(ep_idx):
            tier = MIXED_TIER_CYCLE[ep_idx % len(MIXED_TIER_CYCLE)]
            if tier == "random":
                return "random"
            elif tier == "medium":
                ctrl = wide_med[med_idx[0] % len(wide_med)]
                med_idx[0] += 1
                return ctrl
            elif tier == "expert":
                ctrl = wide_exp[exp_idx[0] % len(wide_exp)]
                exp_idx[0] += 1
                return ctrl
        return sched

    schedulers = {
        "envA_v2_small_wide_medium":    make_rr_scheduler(wide_med),
        "envA_v2_small_narrow_medium":  make_narrow_scheduler(),
        "envA_v2_large_wide_medium":    make_rr_scheduler(wide_med),
        "envA_v2_large_narrow_medium":  make_rr_scheduler(narrow_med),
        "envA_v2_quality_random_wide50k":     make_random_scheduler(),
        "envA_v2_quality_suboptimal_wide50k":  make_rr_scheduler(wide_sub),
        "envA_v2_quality_medium_wide50k":      make_rr_scheduler(wide_med),
        "envA_v2_quality_expert_wide50k":      make_rr_scheduler(wide_exp),
        "envA_v2_quality_mixed_wide50k":       make_mixed_scheduler(),
    }

    source_id_strs = {
        "envA_v2_small_wide_medium":    ";".join(WIDE_MEDIUM_IDS),
        "envA_v2_small_narrow_medium":  ";".join(NARROW_MEDIUM_IDS),
        "envA_v2_large_wide_medium":    ";".join(WIDE_MEDIUM_IDS),
        "envA_v2_large_narrow_medium":  ";".join(NARROW_MEDIUM_IDS),
        "envA_v2_quality_random_wide50k":     "envA_v2_random",
        "envA_v2_quality_suboptimal_wide50k":  ";".join(WIDE_SUBOPTIMAL_IDS),
        "envA_v2_quality_medium_wide50k":      ";".join(WIDE_MEDIUM_IDS),
        "envA_v2_quality_expert_wide50k":      ";".join(WIDE_EXPERT_IDS),
        "envA_v2_quality_mixed_wide50k":       "envA_v2_random;" + ";".join(WIDE_MEDIUM_IDS) + ";" + ";".join(WIDE_EXPERT_IDS),
    }

    # ── Generate datasets ─────────────────────────────────────────────────
    print("-- Generating datasets ---------------------------------------------")
    results = {}
    for spec in DATASET_SPECS:
        name = spec["name"]
        target = spec["target"]
        gseed = GEN_SEEDS[name]
        sched = schedulers[name]
        print(f"  {name} (target={target}, seed={gseed})...", end="", flush=True)
        data = generate_dataset(sched, target, gseed)
        # Save .npz
        fpath = os.path.join(DATASET_DIR, f"{name}.npz")
        np.savez(fpath,
                 observations=data["observations"],
                 actions=data["actions"],
                 rewards=data["rewards"],
                 next_observations=data["next_observations"],
                 terminals=data["terminals"],
                 truncations=data["truncations"],
                 episode_ids=data["episode_ids"],
                 timesteps=data["timesteps"],
                 source_policy_ids=data["source_policy_ids"],
                 source_train_seeds=data["source_train_seeds"],
                 source_behavior_epsilons=data["source_behavior_epsilons"])
        results[name] = data
        print(f" {data['actual_transitions']} trans, "
              f"SA_cov={data['norm_sa_cov']:.4f}, "
              f"SR={data['success_rate']:.3f}")

    # ── Local pre-audit ───────────────────────────────────────────────────
    print()
    print("-- Local pre-audit -------------------------------------------------")

    # A. Integrity
    all_ok = True
    for spec in DATASET_SPECS:
        name = spec["name"]
        fpath = os.path.join(DATASET_DIR, f"{name}.npz")
        d = np.load(fpath, allow_pickle=True)
        keys = set(d.files)
        required = {"observations", "actions", "rewards", "next_observations",
                     "terminals", "truncations", "episode_ids", "timesteps",
                     "source_policy_ids", "source_train_seeds", "source_behavior_epsilons"}
        if not required.issubset(keys):
            print(f"  FAIL integrity: {name} missing keys {required - keys}")
            all_ok = False
        n = len(d["observations"])
        for k in ["actions", "rewards", "next_observations", "terminals",
                   "truncations", "episode_ids", "timesteps"]:
            if len(d[k]) != n:
                print(f"  FAIL integrity: {name} length mismatch on {k}")
                all_ok = False
        if results[name]["actual_transitions"] < spec["target"]:
            print(f"  FAIL integrity: {name} actual < target")
            all_ok = False

    if all_ok:
        print("  integrity: OK (all 9 datasets)")
    else:
        print("  integrity: FAIL")
        print("Clean Phase 5: FAIL")
        sys.exit(1)

    # B. Main four local gates
    sw = results["envA_v2_small_wide_medium"]["norm_sa_cov"]
    sn = results["envA_v2_small_narrow_medium"]["norm_sa_cov"]
    lw = results["envA_v2_large_wide_medium"]["norm_sa_cov"]
    ln = results["envA_v2_large_narrow_medium"]["norm_sa_cov"]

    main_gates = {
        "sw > sn":        sw > sn,
        "lw > ln":        lw > ln,
        "sw-sn >= 0.01":  (sw - sn) >= 0.01,
        "lw-ln >= 0.005": (lw - ln) >= 0.005,
    }
    print(f"  sw={sw:.4f}  sn={sn:.4f}  diff={sw-sn:.4f}")
    print(f"  lw={lw:.4f}  ln={ln:.4f}  diff={lw-ln:.4f}")
    for k, v in main_gates.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")

    if not all(main_gates.values()):
        print("Clean Phase 5: FAIL (main four gates)")
        sys.exit(1)

    # C. Quality sweep local gates
    q_sr = {}
    q_ret = {}
    q_ret_std = {}
    for qname in ["random", "suboptimal", "medium", "expert", "mixed"]:
        dname = f"envA_v2_quality_{qname}_wide50k"
        q_sr[qname]  = results[dname]["success_rate"]
        q_ret[qname] = results[dname]["avg_return"]
        q_ret_std[qname] = results[dname]["return_std"]

    print(f"\n  quality SR:  random={q_sr['random']:.3f}  sub={q_sr['suboptimal']:.3f}  "
          f"med={q_sr['medium']:.3f}  exp={q_sr['expert']:.3f}  mixed={q_sr['mixed']:.3f}")
    print(f"  quality ret: random={q_ret['random']:.3f}  sub={q_ret['suboptimal']:.3f}  "
          f"med={q_ret['medium']:.3f}  exp={q_ret['expert']:.3f}  mixed={q_ret['mixed']:.3f}")
    print(f"  return std:  med={q_ret_std['medium']:.4f}  mixed={q_ret_std['mixed']:.4f}")

    q_gates = {
        "SR monotone":    q_sr["random"] < q_sr["suboptimal"] < q_sr["medium"] < q_sr["expert"],
        "ret monotone":   q_ret["random"] < q_ret["suboptimal"] < q_ret["medium"] < q_ret["expert"],
        "mixed SR range": q_sr["suboptimal"] < q_sr["mixed"] < q_sr["expert"],
        "mixed std > med std": q_ret_std["mixed"] > q_ret_std["medium"],
    }
    for k, v in q_gates.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")

    if not all(q_gates.values()):
        print("Clean Phase 5: FAIL (quality gates)")
        sys.exit(1)

    print("\n  local pre-audit: ALL PASS")

    # ── Update manifest ───────────────────────────────────────────────────
    print()
    print("-- Updating manifest -----------------------------------------------")
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        existing = [r for r in reader if r["env_name"] != "EnvA_v2"]

    new_rows = []
    for spec in DATASET_SPECS:
        name = spec["name"]
        data = results[name]
        target = spec["target"]
        size_mode = "small" if target == 50_000 else "large"
        cov_mode = spec["mode"]
        qbin = spec["bin"]
        n_src = len(source_id_strs[name].split(";"))
        notes = (f"Clean Phase 5; deterministic round-robin; "
                 f"source={cov_mode}_{qbin}; gen_seed={GEN_SEEDS[name]}")
        if qbin == "random":
            notes = f"Clean Phase 5; uniform random policy; gen_seed={GEN_SEEDS[name]}"
        elif qbin == "mixed":
            notes = (f"Clean Phase 5; mixed tier cycle=random/medium/medium/expert; "
                     f"gen_seed={GEN_SEEDS[name]}")

        new_rows.append({
            "dataset_name": name,
            "env_name": "EnvA_v2",
            "target_transitions": str(target),
            "actual_transitions": str(data["actual_transitions"]),
            "size_mode": size_mode,
            "coverage_mode": cov_mode if qbin not in ("random","suboptimal","medium","expert","mixed") else "wide" if cov_mode == "wide" else cov_mode,
            "quality_bin": qbin,
            "num_source_policies": str(n_src),
            "source_policy_ids": source_id_strs[name],
            "generation_seed": str(GEN_SEEDS[name]),
            "file_path": f"artifacts/final_datasets/{name}.npz",
            "normalized_state_coverage": f"{data['norm_state_cov']:.6f}",
            "normalized_state_action_coverage": f"{data['norm_sa_cov']:.6f}",
            "avg_return": f"{data['avg_return']:.4f}",
            "success_rate": f"{data['success_rate']:.4f}",
            "avg_episode_length": f"{data['avg_episode_length']:.2f}",
            "status": "generated",
            "notes": notes,
        })

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(existing)
        w.writerows(new_rows)

    n_bc = len(existing)
    print(f"  Preserved {n_bc} EnvB/EnvC rows")
    print(f"  Appended {len(new_rows)} EnvA_v2 rows")
    print(f"  Total manifest rows: {n_bc + len(new_rows)}")
    print()
    print("Clean Phase 5: PASS")
