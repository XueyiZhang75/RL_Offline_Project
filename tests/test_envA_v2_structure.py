"""
tests/test_envA_v2_structure.py
Clean Phase 1 structural validation for EnvA_v2 (Four-Corridor grid).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from collections import deque
from envs.gridworld_envs import (
    EnvA, EnvB, EnvC, EnvA_v2,
    N_ACTIONS, HORIZON, GOAL_REWARD, STEP_PENALTY,
)

_BRANCH_DOORWAYS = [(3, 3), (3, 10), (3, 18), (3, 25)]
_MERGE_DOORWAYS  = [(26, 3), (26, 10), (26, 18), (26, 25)]
_ALL_DOORWAYS    = _BRANCH_DOORWAYS + _MERGE_DOORWAYS

# Corridor column ranges (inclusive)
_CORRIDORS = {
    "A": (1, 6),
    "B": (8, 13),
    "C": (15, 20),
    "D": (22, 28),
}


def _bfs_path(start, goal, walls, grid_size=30):
    if start == goal:
        return [start]
    parent = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (r+dr, c+dc)
            if nb in parent or nb in walls or not (0<=nb[0]<grid_size and 0<=nb[1]<grid_size):
                continue
            parent[nb] = (r, c)
            if nb == goal:
                path, cur = [], goal
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return list(reversed(path))
            q.append(nb)
    return None


# ── 1. Basic interface ────────────────────────────────────────────────────────

class TestBasicInterface:

    def test_instantiates(self):
        assert EnvA_v2() is not None

    def test_grid_size_30(self):
        assert EnvA_v2.grid_size == 30

    def test_start_pos(self):
        assert EnvA_v2.start_pos == (1, 14)

    def test_goal_pos(self):
        assert EnvA_v2.goal_pos == (28, 14)

    def test_reset_returns_start(self):
        env = EnvA_v2()
        obs, _ = env.reset()
        assert obs == (1, 14)

    def test_transition_deterministic(self):
        env = EnvA_v2()
        for a in range(N_ACTIONS):
            assert env.transition((1, 14), a) == env.transition((1, 14), a)

    def test_action_space_4(self):
        assert N_ACTIONS == 4

    def test_horizon_100(self):
        assert HORIZON == 100


# ── 2. Goal / reward / truncation ─────────────────────────────────────────────

class TestRewardAndTermination:

    def test_goal_terminated(self):
        env = EnvA_v2()
        env.reset()
        env._pos = (28, 13)
        obs, reward, terminated, truncated, _ = env.step(3)  # right to (28,14)
        assert obs == (28, 14)
        assert terminated is True
        assert truncated is False
        assert abs(reward - (GOAL_REWARD + STEP_PENALTY)) < 1e-9

    def test_step_penalty(self):
        env = EnvA_v2()
        env.reset()
        _, reward, term, trunc, _ = env.step(1)  # down
        assert not term
        assert abs(reward - STEP_PENALTY) < 1e-9

    def test_horizon_truncated(self):
        env = EnvA_v2()
        env.reset()
        for _ in range(HORIZON - 1):
            _, _, t, tr, _ = env.step(0)  # up into wall, stay
            assert not t and not tr
        _, _, t, tr, _ = env.step(0)
        assert tr is True and t is False


# ── 3. Walls / doorways / corridors ───────────────────────────────────────────

class TestWallsAndDoorways:

    def test_branch_doorways_intact(self):
        env = EnvA_v2()
        for dw in _BRANCH_DOORWAYS:
            assert dw not in env._walls, f"Branch doorway {dw} is blocked"

    def test_merge_doorways_intact(self):
        env = EnvA_v2()
        for dw in _MERGE_DOORWAYS:
            assert dw not in env._walls, f"Merge doorway {dw} is blocked"

    def test_divider_col7_is_wall(self):
        env = EnvA_v2()
        for r in range(4, 26):
            assert (r, 7) in env._walls, f"Divider (r={r}, 7) missing"

    def test_divider_col14_is_wall(self):
        env = EnvA_v2()
        for r in range(4, 26):
            assert (r, 14) in env._walls, f"Divider (r={r}, 14) missing"

    def test_divider_col21_is_wall(self):
        env = EnvA_v2()
        for r in range(4, 26):
            assert (r, 21) in env._walls, f"Divider (r={r}, 21) missing"

    def test_corridors_internally_open(self):
        """Each corridor interior (rows 4-25, corridor cols) has no extra walls."""
        env = EnvA_v2()
        for name, (c_lo, c_hi) in _CORRIDORS.items():
            for r in range(4, 26):
                for c in range(c_lo, c_hi + 1):
                    assert (r, c) not in env._walls, (
                        f"Corridor {name} cell ({r},{c}) is unexpectedly a wall"
                    )

    def test_branch_wall_blocks_non_doorway(self):
        env = EnvA_v2()
        assert (3, 5) in env._walls  # not a doorway
        assert (3, 14) in env._walls  # not a doorway (col 14 is divider at row 3)


# ── 4. Reachable states / SA pairs ───────────────────────────────────────────

class TestReachableCounts:

    def test_reachable_nonempty(self):
        rs = EnvA_v2().get_reachable_states()
        assert isinstance(rs, frozenset)
        assert len(rs) > 0

    def test_sa_pairs_ge_2500(self):
        n_sa = len(EnvA_v2().get_reachable_state_action_pairs())
        assert n_sa >= 2500, f"SA pairs {n_sa} < 2500"

    def test_start_and_goal_reachable(self):
        rs = EnvA_v2().get_reachable_states()
        assert (1, 14) in rs
        assert (28, 14) in rs

    def test_all_doorways_reachable(self):
        rs = EnvA_v2().get_reachable_states()
        for dw in _ALL_DOORWAYS:
            assert dw in rs, f"Doorway {dw} not reachable"

    def test_reachable_reproducible(self):
        assert EnvA_v2().get_reachable_states() == EnvA_v2().get_reachable_states()

    def test_wall_cells_not_reachable(self):
        env = EnvA_v2()
        rs = env.get_reachable_states()
        for w in env._walls:
            assert w not in rs


# ── 5. Route family reachability ──────────────────────────────────────────────

class TestRouteFamilies:

    def test_goal_reachable_via_corridor_A(self):
        env = EnvA_v2()
        p = _bfs_path((1, 14), (28, 14), env._walls, 30)
        assert p is not None, "Goal not reachable at all"

    def test_each_corridor_reachable_from_start(self):
        """Can reach at least one cell inside each corridor from start."""
        env = EnvA_v2()
        targets = {"A": (10, 3), "B": (10, 10), "C": (10, 18), "D": (10, 25)}
        for name, tgt in targets.items():
            p = _bfs_path((1, 14), tgt, env._walls, 30)
            assert p is not None, f"Corridor {name} cell {tgt} not reachable"

    def test_each_corridor_connects_to_goal(self):
        """Can reach goal from inside each corridor."""
        env = EnvA_v2()
        sources = {"A": (20, 3), "B": (20, 10), "C": (20, 18), "D": (20, 25)}
        for name, src in sources.items():
            p = _bfs_path(src, (28, 14), env._walls, 30)
            assert p is not None, f"Cannot reach goal from corridor {name} cell {src}"

    def test_corridors_are_isolated(self):
        """No direct path between corridor interiors without going through stems."""
        env = EnvA_v2()
        # Try going from corridor A interior to corridor B interior
        # without passing through rows 1-2 or 27-28 (stems)
        a_cell = (10, 3)
        b_cell = (10, 10)
        # BFS within rows 3-26 only (corridor area + walls)
        restricted_walls = set(env._walls)
        for c in range(1, 29):
            for r in [1, 2, 27, 28]:
                restricted_walls.add((r, c))
        p = _bfs_path(a_cell, b_cell, frozenset(restricted_walls), 30)
        assert p is None, "Corridors A and B are not isolated"


# ── 6. Existing environments not broken ──────────────────────────────────────

class TestExistingEnvsIntact:

    def test_envA_still_works(self):
        env = EnvA()
        obs, _ = env.reset()
        assert obs == (1, 1)
        assert len(env.get_reachable_states()) == 148

    def test_envB_still_works(self):
        env = EnvB()
        obs, _ = env.reset()
        assert obs == (1, 1)
        assert len(env.get_reachable_states()) > 0

    def test_envC_still_works(self):
        env = EnvC()
        obs, _ = env.reset()
        assert obs == ((1, 1), 0)
        assert len(env.get_reachable_states()) > 0
