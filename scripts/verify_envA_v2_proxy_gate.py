"""
scripts/verify_envA_v2_proxy_gate.py
EnvA_v2 corridor structure and scripted controller definitions.

Defines the four-corridor route families for the 30×30 EnvA_v2 gridworld
and the scripted controller primitives used to build the behavior policy
pool.  All behavior pool construction and dataset generation scripts import
from this module as the single source of truth for corridor semantics.

Corridor layout (EnvA_v2, 30×30 grid)
---------------------------------------
  Branch doorways : (3,3)  (3,10) (3,18) (3,25)
  Merge  doorways : (26,3) (26,10)(26,18)(26,25)
  Family A : cols 1–6   (seeds 0–1)
  Family B : cols 8–13  (seeds 2–3)
  Family C : cols 15–20 (seeds 4–5)
  Family D : cols 22–28 (seeds 6–7)

Exports
-------
FAMILIES        -- {family: [seed_indices]}
SEED_FAMILY_MAP -- {seed_index: family}
TOUR_WAYPOINTS  -- {family: [waypoint_positions]}  (branch_door → merge_door)
ACTION_DELTAS   -- {action: (dr, dc)}
OPPOSITE_ACTION -- {action: opposite_action}
G               -- adjacency graph {pos: [(neighbor_pos, action), ...]}
get_table(target)                    -- BFS table {pos: action} → target
get_delay_action(pos, bfs_action)    -- return a non-optimal action
"""

import sys, os
from collections import deque

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from envs.gridworld_envs import EnvA_v2, N_ACTIONS

# ── Corridor family definitions ───────────────────────────────────────────────

FAMILIES = {
    "A": [0, 1],   # corridor cols 1–6,   branch (3,3),  merge (26,3)
    "B": [2, 3],   # corridor cols 8–13,  branch (3,10), merge (26,10)
    "C": [4, 5],   # corridor cols 15–20, branch (3,18), merge (26,18)
    "D": [6, 7],   # corridor cols 22–28, branch (3,25), merge (26,25)
}

SEED_FAMILY_MAP = {}
for _fam, _seeds in FAMILIES.items():
    for _s in _seeds:
        SEED_FAMILY_MAP[_s] = _fam

# ── Tour waypoints (branch doorway → merge doorway per family) ────────────────
# The scripted controller follows these intermediate targets in sequence,
# then heads to the final goal (28,14).

TOUR_WAYPOINTS = {
    "A": [(3,  3), (26,  3)],
    "B": [(3, 10), (26, 10)],
    "C": [(3, 18), (26, 18)],
    "D": [(3, 25), (26, 25)],
}

# ── Action primitives ─────────────────────────────────────────────────────────

ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up/down/left/right
OPPOSITE_ACTION = {0: 1, 1: 0, 2: 3, 3: 2}

# ── Grid adjacency graph ──────────────────────────────────────────────────────
# G maps each reachable position to a list of (neighbor_pos, action) pairs.

_env_tmp = EnvA_v2()
_REACHABLE = _env_tmp.get_reachable_states()
_WALLS     = _env_tmp._walls
del _env_tmp

G = {}
for _pos in _REACHABLE:
    _neighbors = []
    for _a, (_dr, _dc) in ACTION_DELTAS.items():
        _nr, _nc = _pos[0] + _dr, _pos[1] + _dc
        _npos = (_nr, _nc)
        if _npos not in _WALLS and _npos in _REACHABLE:
            _neighbors.append((_npos, _a))
    G[_pos] = _neighbors

# ── BFS path tables ───────────────────────────────────────────────────────────

def get_table(target):
    """Compute a BFS action table from every reachable position to `target`.

    Returns
    -------
    dict {pos: action}
        For each reachable position, the first action on the shortest path
        to `target`.  Positions from which `target` is unreachable are
        omitted (returns action 0 as fallback via .get(pos, 0)).
    """
    # Backward BFS from target
    dist = {target: 0}
    queue = deque([target])
    parent_action = {}   # pos -> action taken FROM pos to reach target

    while queue:
        cur = queue.popleft()
        for nbr, act in G.get(cur, []):
            if nbr not in dist:
                dist[nbr] = dist[cur] + 1
                queue.append(nbr)

    # For each position, find the action that leads to the neighbor
    # closest (by BFS distance) to target.
    table = {}
    for pos in _REACHABLE:
        if pos == target:
            table[pos] = 0   # already at target, any action works
            continue
        best_action = None
        best_dist   = float("inf")
        for nbr, act in G.get(pos, []):
            d = dist.get(nbr, float("inf"))
            if d < best_dist:
                best_dist   = d
                best_action = act
        if best_action is not None:
            table[pos] = best_action
    return table


# ── Delay action ──────────────────────────────────────────────────────────────

def get_delay_action(pos, bfs_action):
    """Return any action other than `bfs_action` to simulate controller delay.

    Prefers the opposite direction first; falls back to the next available
    non-optimal action.  Always returns a valid integer action in [0, 3].
    """
    opposite = OPPOSITE_ACTION[bfs_action]
    if opposite != bfs_action:
        return opposite
    # Fallback: first action that differs from bfs_action
    for a in range(N_ACTIONS):
        if a != bfs_action:
            return a
    return bfs_action   # degenerate case: all actions same (should not occur)
