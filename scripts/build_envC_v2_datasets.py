"""
scripts/build_envC_v2_datasets.py
EnvC_v2 dataset generation — scripted BFS-guided behavior pool.

Frozen dataset config (v2 redesign pilot, 2026-04-08):
  small-wide:       50k transitions,  families LU+LD+RU+RD, delay=0.05,
                    delay_type=uniform_random,  seed=600
  large-narrow-LU: 200k transitions, family LU only,        delay=0.05,
                    delay_type=opposite,         seed=601

Controller design:
  Two-phase BFS:
    - has_key=0 phase: BFS to the key cell (left half only, door blocked)
    - has_key=1 phase: BFS to door, then post-door waypoint, then goal

  delay_type='uniform_random': any of 4 actions (wide dataset, boosts SA coverage)
  delay_type='opposite':       opposite of BFS optimal (narrow dataset, minimal exploration)

Outputs (never overwrites final_datasets or any other directory):
  artifacts/envC_v2_datasets/envC_v2_small_wide.npz
  artifacts/envC_v2_datasets/envC_v2_large_narrow_LU.npz
  artifacts/envC_v2_datasets/envC_v2_dataset_manifest.csv

Each .npz file contains:
  observations      : (N, 3) int32 — (row, col, has_key)
  actions           : (N,)   int64
  rewards           : (N,)   float32
  next_observations : (N, 3) int32 — (row, col, has_key)
  terminals         : (N,)   float32  (1.0 = episode ended)
  norm_sa_coverage  : scalar float64
  norm_state_coverage : scalar float64
  route_family_coverage : scalar float64
"""

import sys, os, csv
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from collections import deque

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from envs.gridworld_envs import (
    EnvC_v2, FAMILY_WAYPOINTS_C2, CORRIDOR_C2,
    _WALLS_C2, _REACHABLE_C2, _DOOR_C2, _KEY_CELLS_C2, _GRID_C2,
    N_ACTIONS,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "envC_v2_datasets")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Frozen dataset config ──────────────────────────────────────────────────────

WIDE_CONFIG = {
    "name":        "envC_v2_small_wide",
    "families":    ["LU", "LD", "RU", "RD"],
    "delay_prob":  0.05,
    "delay_type":  "uniform_random",
    "n_trans":     50_000,
    "seed":        600,
}
NARROW_CONFIG = {
    "name":        "envC_v2_large_narrow_LU",
    "families":    ["LU"],
    "delay_prob":  0.05,
    "delay_type":  "opposite",
    "n_trans":     200_000,
    "seed":        601,
}

# ── Internal constants ────────────────────────────────────────────────────────

_DELTAS  = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
_OPP     = {0: 1, 1: 0, 2: 3, 3: 2}
_G       = _GRID_C2
_START   = EnvC_v2.start_pos
_GOAL    = EnvC_v2.goal_pos
_HORIZON = EnvC_v2._horizon

# ── BFS reachability for no-key and with-key sub-graphs ───────────────────────

def _build_nokey_reachable():
    """Positions reachable with has_key=0 (door treated as wall)."""
    no_key_walls = _WALLS_C2 | {_DOOR_C2}
    vis = {_START}; q = deque([_START])
    while q:
        r, c = q.popleft()
        for dr, dc in _DELTAS.values():
            p = (r+dr, c+dc)
            if p not in vis and p not in no_key_walls and 0 <= p[0] < _G and 0 <= p[1] < _G:
                vis.add(p); q.append(p)
    return frozenset(vis)


def _build_withkey_reachable():
    """Positions reachable with has_key=1 (door passable)."""
    vis = {_START}; q = deque([_START])
    while q:
        r, c = q.popleft()
        for dr, dc in _DELTAS.values():
            p = (r+dr, c+dc)
            if p not in vis and p not in _WALLS_C2 and 0 <= p[0] < _G and 0 <= p[1] < _G:
                vis.add(p); q.append(p)
    return frozenset(vis)


_REACH_NK = _build_nokey_reachable()
_REACH_WK = _build_withkey_reachable()

# ── BFS action tables ──────────────────────────────────────────────────────────

_BFS_NK_CACHE = {}
_BFS_WK_CACHE = {}


def _build_bfs_table(target, reachable):
    """Reverse BFS from target through reachable positions.
    Returns dict: pos → best_action_to_reach_target."""
    dist = {target: 0}; q = deque([target])
    while q:
        cur = q.popleft()
        for a, (dr, dc) in _DELTAS.items():
            nbr = (cur[0]-dr, cur[1]-dc)
            if nbr in reachable and nbr not in dist:
                dist[nbr] = dist[cur]+1; q.append(nbr)
    tbl = {}
    for pos in reachable:
        if pos == target: tbl[pos] = 0; continue
        best_a, best_d = None, float("inf")
        for a, (dr, dc) in _DELTAS.items():
            nbr = (pos[0]+dr, pos[1]+dc)
            if nbr in reachable and dist.get(nbr, float("inf")) < best_d:
                best_d, best_a = dist[nbr], a
        if best_a is not None: tbl[pos] = best_a
    return tbl


def _get_nokey_bfs(target):
    if target not in _BFS_NK_CACHE:
        _BFS_NK_CACHE[target] = _build_bfs_table(target, _REACH_NK)
    return _BFS_NK_CACHE[target]


def _get_withkey_bfs(target):
    if target not in _BFS_WK_CACHE:
        _BFS_WK_CACHE[target] = _build_bfs_table(target, _REACH_WK)
    return _BFS_WK_CACHE[target]


# Pre-warm BFS tables for all waypoints
for _fam_wps in FAMILY_WAYPOINTS_C2.values():
    for _wp in _fam_wps:
        if _wp in _REACH_NK: _get_nokey_bfs(_wp)
        if _wp in _REACH_WK: _get_withkey_bfs(_wp)

# ── Controller ─────────────────────────────────────────────────────────────────

def _controller_step(pos, has_key, family, wp_idx, delay_prob, delay_type, rng):
    """Return (action, new_wp_idx) for one scripted controller step.

    Two-phase controller:
      Phase 0 (has_key=0): BFS from left half toward the key cell
      Phase 1 (has_key=1): BFS through the full grid toward door/waypoints/goal
    """
    wps = FAMILY_WAYPOINTS_C2[family]
    while wp_idx < len(wps) and pos == wps[wp_idx]:
        wp_idx += 1
    if wp_idx >= len(wps):
        return 0, wp_idx
    target = wps[wp_idx]
    tbl = _get_nokey_bfs(target) if has_key == 0 else _get_withkey_bfs(target)
    bfs_a = tbl.get(pos, 0)
    if rng.random() < delay_prob:
        if delay_type == "opposite":
            action = _OPP[bfs_a]
        else:  # uniform_random
            action = int(rng.integers(0, N_ACTIONS))
    else:
        action = bfs_a
    return action, wp_idx


def _env_step_c2(pos, has_key, action):
    """One deterministic step.  Returns (new_pos, new_has_key)."""
    dr, dc = _DELTAS[action]
    nxt = (pos[0]+dr, pos[1]+dc)
    if nxt in _WALLS_C2:
        return pos, has_key
    if nxt == _DOOR_C2 and has_key == 0:
        return pos, has_key
    new_hk = 1 if (nxt in _KEY_CELLS_C2 and has_key == 0) else has_key
    return nxt, new_hk

# ── Exclusive SA precomputation ────────────────────────────────────────────────

_ALL_EXTENDED_SA = frozenset(
    (state, a) for state in _REACHABLE_C2 for a in range(N_ACTIONS)
)


def _compute_exclusive_sa():
    """Return dict mapping stage label → frozenset of (extended_state, action)."""
    excl = {}
    for stage, spec in CORRIDOR_C2.items():
        hk   = spec["has_key"]
        rlo  = spec["row_lo"]; rhi  = spec["row_hi"]
        clo  = spec["col_lo"]; chi  = spec["col_hi"]
        sa_set = set()
        for (pos, h), a in _ALL_EXTENDED_SA:
            r, c = pos
            if h == hk and rlo <= r <= rhi and clo <= c <= chi:
                sa_set.add(((pos, h), a))
        excl[stage] = frozenset(sa_set)
    return excl


_EXCLUSIVE_SA_C2 = _compute_exclusive_sa()

# ── Dataset generation ─────────────────────────────────────────────────────────

def generate_dataset(families, delay_prob, delay_type, n_transitions, seed):
    """Generate offline transitions using scripted BFS controllers.

    Returns
    -------
    transitions : list of (state, action, reward, next_state, terminal)
                  where state = ((row, col), has_key)
    visited_sa  : frozenset of (state, action) pairs visited
    """
    rng = np.random.default_rng(seed)
    transitions = []
    visited_sa  = set()
    n_fam = len(families)
    fam_cycle = 0

    while len(transitions) < n_transitions:
        family = families[fam_cycle % n_fam]; fam_cycle += 1
        pos = _START; has_key = 0; wp_idx = 0; step = 0; done = False

        while not done and len(transitions) < n_transitions:
            action, wp_idx = _controller_step(
                pos, has_key, family, wp_idx, delay_prob, delay_type, rng)
            npos, nhk = _env_step_c2(pos, has_key, action)

            reward    = 1.0 if npos == _GOAL else -0.01
            terminal  = (npos == _GOAL)
            truncated = (not terminal) and (step + 1 >= _HORIZON)
            done_flag = terminal or truncated

            state     = (pos, has_key)
            nxt_state = (npos, nhk)
            transitions.append((state, action, reward, nxt_state, done_flag))
            visited_sa.add((state, action))

            pos = npos; has_key = nhk; step += 1; done = done_flag

    return transitions[:n_transitions], frozenset(visited_sa)


def compute_coverage(visited_sa):
    """Return coverage metrics dict for extended state space."""
    norm_sa = len(visited_sa & _ALL_EXTENDED_SA) / len(_ALL_EXTENDED_SA)
    vis_states = frozenset(s for s, a in visited_sa)
    norm_state = len(vis_states) / len(_REACHABLE_C2)

    # Route-family coverage: a family is "covered" if its key cell AND mid-waypoint are visited
    fams_covered = set()
    for fam, wps in FAMILY_WAYPOINTS_C2.items():
        key_cell = wps[0]; mid_wp = wps[2]
        key_vis = (key_cell, 0) in vis_states or (key_cell, 1) in vis_states
        mid_vis = (mid_wp, 1) in vis_states
        if key_vis and mid_vis:
            fams_covered.add(fam)
    rfc = len(fams_covered) / 4.0

    # Per-phase (has_key) coverage
    sa_hk0   = frozenset(sa for sa in visited_sa if sa[0][1] == 0)
    sa_hk1   = frozenset(sa for sa in visited_sa if sa[0][1] == 1)
    all_hk0  = frozenset(sa for sa in _ALL_EXTENDED_SA if sa[0][1] == 0)
    all_hk1  = frozenset(sa for sa in _ALL_EXTENDED_SA if sa[0][1] == 1)
    cov_hk0  = len(sa_hk0 & all_hk0) / max(len(all_hk0), 1)
    cov_hk1  = len(sa_hk1 & all_hk1) / max(len(all_hk1), 1)

    return {
        "norm_sa_coverage":    norm_sa,
        "norm_state_coverage": norm_state,
        "route_family_coverage": rfc,
        "families_covered":    sorted(fams_covered),
        "cov_hk0":             cov_hk0,
        "cov_hk1":             cov_hk1,
    }


def save_dataset(transitions, name, cov):
    """Save transitions to .npz in OUT_DIR.  Observations stored as (N,3) int32."""
    n = len(transitions)
    obs  = np.array([(t[0][0][0], t[0][0][1], t[0][1]) for t in transitions], dtype=np.int32)
    acts = np.array([t[1] for t in transitions], dtype=np.int64)
    rews = np.array([t[2] for t in transitions], dtype=np.float32)
    nobs = np.array([(t[3][0][0], t[3][0][1], t[3][1]) for t in transitions], dtype=np.int32)
    term = np.array([float(t[4]) for t in transitions], dtype=np.float32)
    path = os.path.join(OUT_DIR, f"{name}.npz")
    np.savez_compressed(
        path,
        observations       = obs,
        actions            = acts,
        rewards            = rews,
        next_observations  = nobs,
        terminals          = term,
        norm_sa_coverage        = np.array([cov["norm_sa_coverage"]]),
        norm_state_coverage     = np.array([cov["norm_state_coverage"]]),
        route_family_coverage   = np.array([cov["route_family_coverage"]]),
        n_transitions           = np.array([n]),
    )
    return path


def load_dataset(name):
    """Load .npz.  Returns (obs, acts, rews, nobs, terms) as numpy arrays.

    obs / nobs : (N, 3) int32 — (row, col, has_key)
    acts       : (N,)   int64
    rews       : (N,)   float32
    terms      : (N,)   float32
    """
    path = os.path.join(OUT_DIR, f"{name}.npz")
    d = np.load(path)
    return (d["observations"], d["actions"].astype(np.int64),
            d["rewards"].astype(np.float32), d["next_observations"],
            d["terminals"].astype(np.float32))

# ── Manifest ───────────────────────────────────────────────────────────────────

def write_manifest(entries):
    path = os.path.join(OUT_DIR, "envC_v2_dataset_manifest.csv")
    fieldnames = ["dataset_name", "coverage_mode", "target_transitions",
                  "actual_transitions", "norm_state_coverage", "norm_sa_coverage",
                  "route_family_coverage", "families_covered",
                  "delay_prob", "delay_type", "source_families", "status"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(entries)
    return path

# ── Public entry point ─────────────────────────────────────────────────────────

def build_all(verbose=True):
    """Build both canonical datasets and write manifest.
    Returns (wide_cov, narrow_cov) dicts."""
    entries = []; results = {}

    for cfg, mode in [(WIDE_CONFIG, "wide"), (NARROW_CONFIG, "narrow")]:
        if verbose:
            print(f"  Generating {cfg['name']} ({cfg['n_trans']//1000}k, "
                  f"delay={cfg['delay_prob']}, type={cfg['delay_type']}, "
                  f"fam={cfg['families']})...")
        trans, vis_sa = generate_dataset(
            cfg["families"], cfg["delay_prob"], cfg["delay_type"],
            cfg["n_trans"], cfg["seed"])
        cov  = compute_coverage(vis_sa)
        path = save_dataset(trans, cfg["name"], cov)
        results[mode] = cov
        entries.append({
            "dataset_name":          cfg["name"],
            "coverage_mode":         mode,
            "target_transitions":    cfg["n_trans"],
            "actual_transitions":    len(trans),
            "norm_state_coverage":   f"{cov['norm_state_coverage']:.4f}",
            "norm_sa_coverage":      f"{cov['norm_sa_coverage']:.4f}",
            "route_family_coverage": f"{cov['route_family_coverage']:.4f}",
            "families_covered":      ",".join(cov["families_covered"]),
            "delay_prob":            cfg["delay_prob"],
            "delay_type":            cfg["delay_type"],
            "source_families":       ",".join(cfg["families"]),
            "status":                "formal_prep",
        })
        if verbose:
            print(f"    norm_sa={cov['norm_sa_coverage']:.4f}  "
                  f"rfc={cov['route_family_coverage']:.3f}  "
                  f"fams={cov['families_covered']}  path={path}")

    man = write_manifest(entries)
    gap = results["wide"]["norm_sa_coverage"] - results["narrow"]["norm_sa_coverage"]
    if verbose:
        print(f"\n  Coverage gap: {gap:.4f}  "
              f"({'submission-worthy' if gap >= 0.15 else 'below 0.15'})")
        print(f"  Manifest: {man}")
    return results["wide"], results["narrow"]


if __name__ == "__main__":
    print("Building EnvC_v2 datasets (formal prep)...")
    w, n = build_all(verbose=True)
    print("\nDone.")
