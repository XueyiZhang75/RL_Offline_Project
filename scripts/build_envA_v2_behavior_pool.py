"""
scripts/build_envA_v2_behavior_pool.py
Clean Phase 3: build controller-based behavior pool for EnvA_v2.

Inherits frozen semantics from verify_envA_v2_proxy_gate.py.
Two-stage delay_prob sweep: coarse (13 points) + optional bracket refinement.
Outputs: .pt artifacts, eval CSV, and (on PASS) appends to active catalog.
"""

import sys, os, csv, collections
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Inherit frozen semantics (source of truth: Phase 2) ──────────────────────
from scripts.verify_envA_v2_proxy_gate import (
    FAMILIES, SEED_FAMILY_MAP, TOUR_WAYPOINTS,
    ACTION_DELTAS, OPPOSITE_ACTION, G,
    get_table, get_delay_action,
)
from envs.gridworld_envs import EnvA_v2, HORIZON

# ── Constants (frozen for this phase) ─────────────────────────────────────────

TRAIN_SEEDS = list(range(8))

COARSE_DELAY_PROBS = [
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
]
REFINE_STEP = 0.01
_COARSE_SET = frozenset(COARSE_DELAY_PROBS)

EVAL_EPISODES = 100

BIN_BOUNDS = {
    "suboptimal": (0.20, 0.50),
    "medium":     (0.60, 0.85),
    "expert":     (0.95, 1.01),
}
BIN_CENTERS = {
    "suboptimal": 0.35,
    "medium":     0.725,
    "expert":     0.975,
}

POLICY_SEMANTICS_VERSION = "envA_v2_controller_r1"

ARTIFACT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "artifacts", "behavior_pool"))
CATALOG_PATH = os.path.join(ARTIFACT_DIR, "behavior_policy_catalog.csv")
EVAL_CSV_PATH = os.path.join(ARTIFACT_DIR, "envA_v2_controller_eval.csv")

CATALOG_COLUMNS = [
    "env_name", "train_seed", "policy_id", "policy_type",
    "checkpoint_step", "checkpoint_path",
    "behavior_epsilon",
    "eval_episodes", "avg_return", "success_rate", "avg_episode_length",
    "quality_bin", "selected", "notes",
]

EVAL_CSV_COLUMNS = [
    "env_name", "controller_seed", "route_family", "delay_prob",
    "success_rate", "avg_return", "avg_episode_length",
    "quality_bin", "distance_to_bin_center", "selected_for_bin",
]

START = EnvA_v2.start_pos
GOAL  = EnvA_v2.goal_pos

# ── Controller evaluation ─────────────────────────────────────────────────────

def evaluate_controller(family, delay_prob, n_episodes=EVAL_EPISODES):
    env = EnvA_v2()
    wps = TOUR_WAYPOINTS[family]
    walls = env._walls
    returns, successes, lengths = [], [], []
    for ep in range(n_episodes):
        rng = np.random.default_rng(8888 + ep)
        obs, _ = env.reset()
        pos, wi = obs, 0
        ep_ret, ep_len, done, success = 0.0, 0, False, False
        while not done:
            while wi < len(wps) and pos == wps[wi]:
                wi += 1
            target = wps[wi] if wi < len(wps) else GOAL
            tbl = get_table(target)
            bfs_a = tbl.get(pos, 0)
            if rng.random() < delay_prob:
                action = get_delay_action(pos, bfs_a)
            else:
                action = bfs_a
            obs, r, terminated, truncated, _ = env.step(action)
            pos = obs
            ep_ret += r
            ep_len += 1
            done = terminated or truncated
            if terminated:
                success = True
        returns.append(ep_ret)
        successes.append(float(success))
        lengths.append(ep_len)
    return {
        "avg_return":         float(np.mean(returns)),
        "success_rate":       float(np.mean(successes)),
        "avg_episode_length": float(np.mean(lengths)),
    }


def assign_quality_bin(sr):
    for name, (lo, hi) in BIN_BOUNDS.items():
        if lo <= sr < hi:
            return name
    if sr < 0.20:
        return "unassigned_low"
    if sr < 0.60:
        return "unassigned_mid"
    if sr < 0.95:
        return "unassigned_high"
    return "expert"


# ── Refinement logic ──────────────────────────────────────────────────────────

def find_bracket(results, target_bin):
    """Find adjacent coarse pair that straddles the target bin boundary."""
    lo_bound, hi_bound = BIN_BOUNDS[target_bin]
    sr_map = {round(r["delay_prob"], 10): r["sr"] for r in results}
    brackets = []
    for i in range(len(COARSE_DELAY_PROBS) - 1):
        p_lo = COARSE_DELAY_PROBS[i]
        p_hi = COARSE_DELAY_PROBS[i + 1]
        sr_lo = sr_map.get(p_lo)
        sr_hi = sr_map.get(p_hi)
        if sr_lo is None or sr_hi is None:
            continue
        if target_bin == "medium":
            if sr_lo >= hi_bound and sr_hi < lo_bound:
                brackets.append((p_lo, p_hi))
        elif target_bin == "suboptimal":
            if sr_lo >= 0.50 and sr_hi < 0.20:
                brackets.append((p_lo, p_hi))
    if not brackets:
        return None
    brackets.sort(key=lambda x: (round(x[1] - x[0], 9), x[0]))
    return brackets[0]


def generate_refined(p_lo, p_hi):
    candidates = []
    k = 1
    while True:
        p = round(p_lo + k * REFINE_STEP, 2)
        if p >= p_hi:
            break
        if p not in _COARSE_SET:
            candidates.append(p)
        k += 1
    return candidates


# ── Selection logic ───────────────────────────────────────────────────────────

def select_for_bin(candidates, bin_name):
    lo, hi = BIN_BOUNDS[bin_name]
    center = BIN_CENTERS[bin_name]
    eligible = [c for c in candidates if lo <= c["sr"] < hi]
    if not eligible:
        return None
    eligible.sort(key=lambda c: (abs(c["sr"] - center), c["delay_prob"], c["order"]))
    return eligible[0]


# ── Artifact construction ─────────────────────────────────────────────────────

def make_artifact(seed, family, delay_prob, quality_bin, eval_result):
    return {
        "controller_type":          "waypoint_bfs",
        "env_name":                 "EnvA_v2",
        "route_family":             family,
        "controller_seed":          seed,
        "delay_prob":               delay_prob,
        "start":                    START,
        "goal":                     GOAL,
        "waypoints":                list(TOUR_WAYPOINTS[family]),
        "policy_semantics_version": POLICY_SEMANTICS_VERSION,
        "quality_bin":              quality_bin,
        "eval_episodes":            EVAL_EPISODES,
        "success_rate":             eval_result["success_rate"],
        "avg_return":               eval_result["avg_return"],
        "avg_episode_length":       eval_result["avg_episode_length"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("Clean Phase 3: EnvA_v2 behavior pool construction")
    print("=" * 66)
    print()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    all_eval_rows = []
    selected_artifacts = {}  # (seed, bin) -> artifact dict

    # ── Stage 1: coarse sweep ─────────────────────────────────────────────
    print("-- Stage 1: coarse sweep -------------------------------------------")
    seed_candidates = {s: [] for s in TRAIN_SEEDS}

    for seed in TRAIN_SEEDS:
        family = SEED_FAMILY_MAP[seed]
        print(f"  seed {seed} (family {family}):", end="", flush=True)
        for idx, dp in enumerate(COARSE_DELAY_PROBS):
            ev = evaluate_controller(family, dp)
            sr = ev["success_rate"]
            qb = assign_quality_bin(sr)
            dist = abs(sr - BIN_CENTERS.get(qb, 0.5)) if qb in BIN_CENTERS else 999
            entry = {"delay_prob": dp, "sr": sr, "eval": ev, "bin": qb,
                     "dist": dist, "order": idx, "stage": "coarse"}
            seed_candidates[seed].append(entry)
            all_eval_rows.append({
                "env_name": "EnvA_v2", "controller_seed": seed,
                "route_family": family, "delay_prob": f"{dp:.2f}",
                "success_rate": f"{sr:.4f}",
                "avg_return": f"{ev['avg_return']:.4f}",
                "avg_episode_length": f"{ev['avg_episode_length']:.2f}",
                "quality_bin": qb,
                "distance_to_bin_center": f"{dist:.4f}",
                "selected_for_bin": "",
            })
            print(f" {dp:.2f}={sr:.2f}", end="", flush=True)
        print()

    # ── Stage 2: refinement (only for missing bins) ───────────────────────
    print()
    print("-- Stage 2: refinement (if needed) ---------------------------------")
    for seed in TRAIN_SEEDS:
        family = SEED_FAMILY_MAP[seed]
        for target_bin in ["suboptimal", "medium"]:
            sel = select_for_bin(seed_candidates[seed], target_bin)
            if sel is not None:
                continue
            coarse_results = [{"delay_prob": c["delay_prob"], "sr": c["sr"]}
                              for c in seed_candidates[seed] if c["stage"] == "coarse"]
            bracket = find_bracket(coarse_results, target_bin)
            if bracket is None:
                print(f"  seed {seed} {target_bin}: no bracket found, skipping")
                continue
            p_lo, p_hi = bracket
            refined_dps = generate_refined(p_lo, p_hi)
            if not refined_dps:
                print(f"  seed {seed} {target_bin}: empty refinement in ({p_lo},{p_hi})")
                continue
            print(f"  seed {seed} {target_bin}: refining ({p_lo},{p_hi}) -> {refined_dps}")
            base_order = len(COARSE_DELAY_PROBS)
            for ri, dp in enumerate(refined_dps):
                ev = evaluate_controller(family, dp)
                sr = ev["success_rate"]
                qb = assign_quality_bin(sr)
                dist = abs(sr - BIN_CENTERS.get(qb, 0.5)) if qb in BIN_CENTERS else 999
                entry = {"delay_prob": dp, "sr": sr, "eval": ev, "bin": qb,
                         "dist": dist, "order": base_order + ri, "stage": "refined"}
                seed_candidates[seed].append(entry)
                all_eval_rows.append({
                    "env_name": "EnvA_v2", "controller_seed": seed,
                    "route_family": family, "delay_prob": f"{dp:.2f}",
                    "success_rate": f"{sr:.4f}",
                    "avg_return": f"{ev['avg_return']:.4f}",
                    "avg_episode_length": f"{ev['avg_episode_length']:.2f}",
                    "quality_bin": qb,
                    "distance_to_bin_center": f"{dist:.4f}",
                    "selected_for_bin": "",
                })

    # ── Selection ─────────────────────────────────────────────────────────
    print()
    print("-- Selection -------------------------------------------------------")
    for seed in TRAIN_SEEDS:
        family = SEED_FAMILY_MAP[seed]
        for target_bin in ["suboptimal", "medium", "expert"]:
            sel = select_for_bin(seed_candidates[seed], target_bin)
            if sel is None:
                print(f"  seed {seed} ({family}) {target_bin}: NO CANDIDATE")
                continue
            ev = sel["eval"]
            art = make_artifact(seed, family, sel["delay_prob"], target_bin, ev)
            selected_artifacts[(seed, target_bin)] = art
            # Mark in eval rows
            for row in all_eval_rows:
                if (int(row["controller_seed"]) == seed
                        and row["delay_prob"] == f"{sel['delay_prob']:.2f}"
                        and row["selected_for_bin"] == ""):
                    row["selected_for_bin"] = target_bin
                    break
            print(f"  seed {seed} ({family}) {target_bin}: "
                  f"dp={sel['delay_prob']:.2f} SR={sel['sr']:.3f}")

    # ── Write eval CSV ────────────────────────────────────────────────────
    with open(EVAL_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EVAL_CSV_COLUMNS)
        w.writeheader()
        w.writerows(all_eval_rows)
    print(f"\n  Eval CSV: {EVAL_CSV_PATH} ({len(all_eval_rows)} rows)")

    # ── Save .pt artifacts ────────────────────────────────────────────────
    print()
    print("-- Saving .pt artifacts --------------------------------------------")
    for (seed, qbin), art in sorted(selected_artifacts.items()):
        fname = f"envA_v2_seed{seed}_{qbin}_controller.pt"
        fpath = os.path.join(ARTIFACT_DIR, fname)
        torch.save(art, fpath)
        print(f"  {fname}")

    # ── Gate evaluation ───────────────────────────────────────────────────
    print()
    print("-- Gate evaluation -------------------------------------------------")
    bin_counts = {"suboptimal": [], "medium": [], "expert": []}
    for (seed, qbin), art in selected_artifacts.items():
        bin_counts[qbin].append(seed)

    gate = {}
    for qbin in ["suboptimal", "medium", "expert"]:
        seeds = bin_counts[qbin]
        cnt = len(seeds)
        uniq = len(set(seeds))
        gate[f"{qbin}_count_ge_6"] = cnt >= 6
        gate[f"{qbin}_seeds_ge_6"] = uniq >= 6
        print(f"  {qbin}: {cnt} selected from {uniq} seeds"
              f"  [{'OK' if cnt >= 6 and uniq >= 6 else 'FAIL'}]")

    # Check artifacts loadable
    all_loadable = True
    for (seed, qbin), art in selected_artifacts.items():
        fname = f"envA_v2_seed{seed}_{qbin}_controller.pt"
        fpath = os.path.join(ARTIFACT_DIR, fname)
        try:
            loaded = torch.load(fpath, weights_only=False)
            assert loaded["controller_type"] == "waypoint_bfs"
            assert loaded["env_name"] == "EnvA_v2"
        except Exception as e:
            print(f"  LOAD FAIL: {fname}: {e}")
            all_loadable = False
    gate["all_artifacts_loadable"] = all_loadable
    print(f"  artifacts loadable: {'OK' if all_loadable else 'FAIL'}")

    go = all(gate.values())
    print()
    for k, v in gate.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    print()

    if not go:
        failed = [k for k, v in gate.items() if not v]
        print("Clean Phase 3: FAIL")
        print(f"  Failed gates: {failed}")
        print("  Active catalog NOT updated.")
        sys.exit(1)

    # ── Update active catalog (only on PASS) ──────────────────────────────
    print("-- Updating active catalog -----------------------------------------")
    with open(CATALOG_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
        fieldnames = reader.fieldnames

    new_rows = []
    for (seed, qbin), art in sorted(selected_artifacts.items()):
        family = SEED_FAMILY_MAP[seed]
        fname = f"envA_v2_seed{seed}_{qbin}_controller.pt"
        fpath = os.path.join("artifacts", "behavior_pool", fname)
        notes = (f"controller-based; route_family={family}; "
                 f"delay_prob={art['delay_prob']:.2f}; "
                 f"frozen semantics inherited from verify_envA_v2_proxy_gate.py; "
                 f"behavior_epsilon=0.0 compatibility only")
        row = {
            "env_name":           "EnvA_v2",
            "train_seed":         str(seed),
            "policy_id":          f"envA_v2_seed{seed}_{qbin}_controller",
            "policy_type":        "scripted_controller",
            "checkpoint_step":    "0",
            "checkpoint_path":    fpath,
            "behavior_epsilon":   "0.0",
            "eval_episodes":      str(art["eval_episodes"]),
            "avg_return":         f"{art['avg_return']:.4f}",
            "success_rate":       f"{art['success_rate']:.4f}",
            "avg_episode_length": f"{art['avg_episode_length']:.2f}",
            "quality_bin":        qbin,
            "selected":           "yes",
            "notes":              notes,
        }
        new_rows.append(row)

    with open(CATALOG_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(existing_rows)
        w.writerows(new_rows)

    envb_c_count = sum(1 for r in existing_rows if r["env_name"] in ("EnvB", "EnvC"))
    print(f"  Preserved {envb_c_count} EnvB/EnvC rows")
    print(f"  Appended {len(new_rows)} EnvA_v2 rows")
    print(f"  Total catalog rows: {envb_c_count + len(new_rows)}")
    print()
    print("Clean Phase 3: PASS")
