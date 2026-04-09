"""
scripts/build_envB_v2_datasets.py
EnvB_v2 dataset generation — scripted BFS-guided behavior pool.

Frozen dataset config (v4 fairness audit):
  small-wide:   50k transitions,  families A+B+C, delay=0.25, seed=425
  large-narrow: 200k transitions, family A only,  delay=0.10, seed=411
  (seed=425 = 400+int(0.25*100); seed=411 = 401+int(0.10*100) — matches v4 audit run)

Outputs (never overwrites final_datasets):
  artifacts/envB_v2_datasets/envB_v2_small_wide.npz
  artifacts/envB_v2_datasets/envB_v2_large_narrow_A.npz

Usage:
  python scripts/build_envB_v2_datasets.py
  # or import and call build_all()

Each .npz file contains:
  observations      : (N, 2) int32 — (row, col) positions
  actions           : (N,)   int64
  rewards           : (N,)   float32
  next_observations : (N, 2) int32
  terminals         : (N,)   float32  (1.0 = episode ended)
  norm_sa_coverage  : scalar float64  (stored in metadata array)
  route_family_coverage : scalar float64
"""

import sys, os, csv
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from collections import deque

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from envs.gridworld_envs import (
    EnvB_v2, DOORWAYS_B2, CORRIDOR_COLS_B2, FAMILY_WAYPOINTS_B2,
    _WALLS_B2, _REACHABLE_B2, N_ACTIONS,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "envB_v2_datasets")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Frozen dataset config ──────────────────────────────────────────────────────

WIDE_DELAY   = 0.25
NARROW_DELAY = 0.10
NARROW_FAM   = "A"

WIDE_CONFIG = {
    "name":       "envB_v2_small_wide",
    "families":   ["A", "B", "C"],
    "delay_prob": WIDE_DELAY,
    "n_trans":    50_000,
    "seed":       425,   # = 400 + int(0.25*100); matches v4 audit run (seed=425 → gap=0.1556)
}
NARROW_CONFIG = {
    "name":       "envB_v2_large_narrow_A",
    "families":   [NARROW_FAM],
    "delay_prob": NARROW_DELAY,
    "n_trans":    200_000,
    "seed":       411,   # = 401 + int(0.10*100); matches v4 audit run (seed=411 → gap=0.1556)
}

# ── BFS path tables ─────────────────────────────────────────────────────────────

_DELTAS  = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
_OPP     = {0: 1, 1: 0, 2: 3, 3: 2}
_G       = EnvB_v2.grid_size
_START   = EnvB_v2.start_pos
_GOAL    = EnvB_v2.goal_pos
_HORIZON = EnvB_v2._horizon

_BFS_CACHE = {}


def _get_bfs_table(target):
    """Return a dict mapping each reachable pos -> best action toward target."""
    if target in _BFS_CACHE:
        return _BFS_CACHE[target]
    dist = {target: 0}
    q = deque([target])
    while q:
        cur = q.popleft()
        for a, (dr, dc) in _DELTAS.items():
            nbr = (cur[0] - dr, cur[1] - dc)
            if nbr in _REACHABLE_B2 and nbr not in dist:
                dist[nbr] = dist[cur] + 1
                q.append(nbr)
    tbl = {}
    for pos in _REACHABLE_B2:
        if pos == target:
            tbl[pos] = 0
            continue
        best_a, best_d = None, float("inf")
        for a, (dr, dc) in _DELTAS.items():
            nbr = (pos[0] + dr, pos[1] + dc)
            if nbr in _REACHABLE_B2 and dist.get(nbr, float("inf")) < best_d:
                best_d, best_a = dist[nbr], a
        if best_a is not None:
            tbl[pos] = best_a
    _BFS_CACHE[target] = tbl
    return tbl


# Pre-warm BFS tables for all waypoints
for _fam_wps in FAMILY_WAYPOINTS_B2.values():
    for _wp in _fam_wps:
        _get_bfs_table(_wp)


def _controller_step(pos, family, wp_idx, delay_prob, rng):
    """Return (action, new_wp_idx) for one scripted controller step."""
    wps = FAMILY_WAYPOINTS_B2[family]
    while wp_idx < len(wps) and pos == wps[wp_idx]:
        wp_idx += 1
    if wp_idx >= len(wps):
        return 0, wp_idx
    tbl = _get_bfs_table(wps[wp_idx])
    bfs_a = tbl.get(pos, 0)
    action = _OPP[bfs_a] if rng.random() < delay_prob else bfs_a
    return action, wp_idx


def _env_step(pos, action):
    dr, dc = _DELTAS[action]
    nxt = (pos[0] + dr, pos[1] + dc)
    return nxt if (nxt not in _WALLS_B2 and nxt in _REACHABLE_B2) else pos

# ── Exclusive SA pair precomputation ──────────────────────────────────────────

def _compute_exclusive_sa():
    """Return dict {family: frozenset of (pos, action)} for corridor-exclusive SA."""
    all_sa = frozenset((s, a) for s in _REACHABLE_B2 for a in range(N_ACTIONS))
    excl = {}
    for fam, (lo, hi) in CORRIDOR_COLS_B2.items():
        fam_sa = set()
        for pos in _REACHABLE_B2:
            r, c = pos
            if 4 <= r <= 15 and lo <= c <= hi:
                for a in range(N_ACTIONS):
                    if (pos, a) in all_sa:
                        fam_sa.add((pos, a))
        excl[fam] = frozenset(fam_sa)
    return excl


_EXCLUSIVE_SA = _compute_exclusive_sa()
_ALL_SA = frozenset((s, a) for s in _REACHABLE_B2 for a in range(N_ACTIONS))

# ── Dataset generation ─────────────────────────────────────────────────────────

def generate_dataset(families, delay_prob, n_transitions, seed):
    """Generate offline transitions using scripted BFS controllers.

    Returns
    -------
    transitions : list of (pos, action, reward, next_pos, terminal)
    visited_sa  : frozenset of (pos, action) pairs visited
    """
    rng = np.random.default_rng(seed)
    transitions = []
    visited_sa = set()
    n_fam = len(families)
    fam_cycle = 0

    while len(transitions) < n_transitions:
        family = families[fam_cycle % n_fam]
        fam_cycle += 1
        pos = _START
        wp_idx = 0
        step = 0
        done = False

        while not done and len(transitions) < n_transitions:
            action, wp_idx = _controller_step(pos, family, wp_idx, delay_prob, rng)
            npos = _env_step(pos, action)
            reward = 1.0 if npos == _GOAL else -0.01
            terminal  = (npos == _GOAL)
            truncated = (not terminal) and (step + 1 >= _HORIZON)
            done_flag = terminal or truncated

            transitions.append((pos, action, reward, npos, done_flag))
            visited_sa.add((pos, action))
            pos = npos
            step += 1
            done = done_flag

    transitions = transitions[:n_transitions]
    return transitions, frozenset(visited_sa)


def compute_coverage(visited_sa):
    """Return coverage metrics dict."""
    norm_sa = len(visited_sa & _ALL_SA) / len(_ALL_SA)
    vis_states = frozenset(s for s, a in visited_sa)
    norm_state = len(vis_states) / len(_REACHABLE_B2)
    fams = set()
    for fam, excl in _EXCLUSIVE_SA.items():
        if visited_sa & excl:
            fams.add(fam)
    rfc = len(fams) / 3.0
    return {
        "norm_sa_coverage":    norm_sa,
        "norm_state_coverage": norm_state,
        "route_family_coverage": rfc,
        "families_covered":    sorted(fams),
    }


def save_dataset(transitions, name, cov):
    """Save transitions to .npz in OUT_DIR and return path."""
    n = len(transitions)
    obs  = np.array([t[0] for t in transitions], dtype=np.int32)
    acts = np.array([t[1] for t in transitions], dtype=np.int64)
    rews = np.array([t[2] for t in transitions], dtype=np.float32)
    nobs = np.array([t[3] for t in transitions], dtype=np.int32)
    term = np.array([float(t[4]) for t in transitions], dtype=np.float32)
    path = os.path.join(OUT_DIR, f"{name}.npz")
    np.savez_compressed(
        path,
        observations      = obs,
        actions           = acts,
        rewards           = rews,
        next_observations = nobs,
        terminals         = term,
        norm_sa_coverage       = np.array([cov["norm_sa_coverage"]]),
        norm_state_coverage    = np.array([cov["norm_state_coverage"]]),
        route_family_coverage  = np.array([cov["route_family_coverage"]]),
        n_transitions          = np.array([n]),
    )
    return path


def load_dataset(name):
    """Load .npz and return (obs, acts, rews, nobs, terms) as numpy arrays.

    obs  / nobs : (N, 2) int32 — (row, col) positions (NOT yet one-hot encoded)
    acts        : (N,)   int64
    rews        : (N,)   float32
    terms       : (N,)   float32
    """
    path = os.path.join(OUT_DIR, f"{name}.npz")
    d = np.load(path)
    return (d["observations"], d["actions"].astype(np.int64),
            d["rewards"].astype(np.float32), d["next_observations"],
            d["terminals"].astype(np.float32))

# ── Manifest ───────────────────────────────────────────────────────────────────

def write_manifest(entries):
    path = os.path.join(OUT_DIR, "envB_v2_dataset_manifest.csv")
    fieldnames = ["dataset_name", "coverage_mode", "target_transitions",
                  "actual_transitions", "norm_state_coverage", "norm_sa_coverage",
                  "route_family_coverage", "families_covered",
                  "delay_prob", "source_families", "status"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(entries)
    return path

# ── Public entry point ─────────────────────────────────────────────────────────

def build_all(verbose=True):
    """Build both canonical datasets and write manifest. Returns (wide_cov, narrow_cov)."""
    entries = []
    results = {}

    for cfg, mode in [(WIDE_CONFIG, "wide"), (NARROW_CONFIG, "narrow")]:
        if verbose:
            print(f"  Generating {cfg['name']} ({cfg['n_trans']//1000}k, "
                  f"delay={cfg['delay_prob']}, fam={cfg['families']})...")
        trans, vis_sa = generate_dataset(
            cfg["families"], cfg["delay_prob"], cfg["n_trans"], cfg["seed"])
        cov = compute_coverage(vis_sa)
        path = save_dataset(trans, cfg["name"], cov)
        results[mode] = cov
        entries.append({
            "dataset_name":        cfg["name"],
            "coverage_mode":       mode,
            "target_transitions":  cfg["n_trans"],
            "actual_transitions":  len(trans),
            "norm_state_coverage": f"{cov['norm_state_coverage']:.4f}",
            "norm_sa_coverage":    f"{cov['norm_sa_coverage']:.4f}",
            "route_family_coverage": f"{cov['route_family_coverage']:.4f}",
            "families_covered":    ",".join(cov["families_covered"]),
            "delay_prob":          cfg["delay_prob"],
            "source_families":     ",".join(cfg["families"]),
            "status":              "formal_prep",
        })
        if verbose:
            print(f"    norm_sa={cov['norm_sa_coverage']:.4f}  "
                  f"rfc={cov['route_family_coverage']:.3f}  "
                  f"families={cov['families_covered']}")
            print(f"    saved -> {path}")

    man = write_manifest(entries)
    gap = results["wide"]["norm_sa_coverage"] - results["narrow"]["norm_sa_coverage"]
    if verbose:
        print(f"\n  Coverage gap: {gap:.4f}  "
              f"({'submission-worthy' if gap >= 0.15 else 'below submission target'})")
        print(f"  Manifest: {man}")
    return results["wide"], results["narrow"]


if __name__ == "__main__":
    print("Building EnvB_v2 datasets (formal prep)...")
    w, n = build_all(verbose=True)
    print("\nDone.")
