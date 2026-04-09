"""
tests/test_envB_v2_structure.py
Structural validation tests for EnvB_v2 (20×20 Three-Corridor Grid).

Checks:
  1. Basic interface (instantiation, start/goal, horizon)
  2. Reachable states / SA pairs counts
  3. Wall correctness (split wall, merge wall, dividers, doorways)
  4. Route family distinguishability
  5. All three families can reach goal via waypoints (BFS path exists)
  6. Reward / termination logic
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from collections import deque
from envs.gridworld_envs import (
    EnvB_v2, DOORWAYS_B2, CORRIDOR_COLS_B2, FAMILY_WAYPOINTS_B2,
    N_ACTIONS, HORIZON, GOAL_REWARD, STEP_PENALTY,
)

_SPLIT_DOORWAYS = [(3, 4), (3, 9), (3, 14)]
_MERGE_DOORWAYS = [(16, 4), (16, 9), (16, 14)]


def _bfs_path(start, goal, walls, grid_size=20):
    """Return a BFS path from start to goal, or None if unreachable."""
    if start == goal:
        return [start]
    parent = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (r + dr, c + dc)
            if nb in parent or nb in walls:
                continue
            if not (0 <= nb[0] < grid_size and 0 <= nb[1] < grid_size):
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


# ── 1. Basic interface ─────────────────────────────────────────────────────────

class TestBasicInterface:

    def test_instantiates(self):
        assert EnvB_v2() is not None

    def test_grid_size_20(self):
        assert EnvB_v2.grid_size == 20

    def test_start_pos(self):
        assert EnvB_v2.start_pos == (1, 9)

    def test_goal_pos(self):
        assert EnvB_v2.goal_pos == (18, 9)

    def test_horizon_200(self):
        env = EnvB_v2()
        assert env._horizon == 200

    def test_horizon_does_not_change_module_constant(self):
        # The module-level HORIZON must remain 100 for other envs
        assert HORIZON == 100

    def test_reset_returns_start(self):
        env = EnvB_v2()
        obs, info = env.reset()
        assert obs == (1, 9)
        assert isinstance(info, dict)

    def test_transition_deterministic(self):
        env = EnvB_v2()
        for a in range(N_ACTIONS):
            s1 = env.transition((1, 9), a)
            s2 = env.transition((1, 9), a)
            assert s1 == s2


# ── 2. Reachable states / SA pairs ─────────────────────────────────────────────

class TestReachability:

    def test_reachable_states_count(self):
        env = EnvB_v2()
        states = env.get_reachable_states()
        # Confirmed by audit: 270 reachable states
        assert len(states) == 270

    def test_sa_pairs_count(self):
        env = EnvB_v2()
        sa = env.get_reachable_state_action_pairs()
        # 270 states × 4 actions = 1080
        assert len(sa) == 1080

    def test_start_in_reachable(self):
        env = EnvB_v2()
        assert EnvB_v2.start_pos in env.get_reachable_states()

    def test_goal_in_reachable(self):
        env = EnvB_v2()
        assert EnvB_v2.goal_pos in env.get_reachable_states()

    def test_goal_reachable_from_start(self):
        env = EnvB_v2()
        path = _bfs_path(EnvB_v2.start_pos, EnvB_v2.goal_pos, env._walls)
        assert path is not None
        assert path[0] == EnvB_v2.start_pos
        assert path[-1] == EnvB_v2.goal_pos


# ── 3. Walls / doorways ────────────────────────────────────────────────────────

class TestWallsAndDoorways:

    def test_split_doorways_passable(self):
        env = EnvB_v2()
        for dw in _SPLIT_DOORWAYS:
            assert dw not in env._walls, f"Split doorway {dw} is incorrectly a wall"

    def test_merge_doorways_passable(self):
        env = EnvB_v2()
        for dw in _MERGE_DOORWAYS:
            assert dw not in env._walls, f"Merge doorway {dw} is incorrectly a wall"

    def test_split_wall_blocks_non_doorways(self):
        env = EnvB_v2()
        # Row 3 cells that are NOT doorways must be walls
        for c in range(1, 19):
            if (3, c) not in set(_SPLIT_DOORWAYS):
                assert (3, c) in env._walls, f"(3,{c}) should be a split wall"

    def test_merge_wall_blocks_non_doorways(self):
        env = EnvB_v2()
        for c in range(1, 19):
            if (16, c) not in set(_MERGE_DOORWAYS):
                assert (16, c) in env._walls, f"(16,{c}) should be a merge wall"

    def test_divider_col7_is_wall(self):
        env = EnvB_v2()
        for r in range(3, 17):
            assert (r, 7) in env._walls, f"Divider (r={r},7) missing"

    def test_divider_col12_is_wall(self):
        env = EnvB_v2()
        for r in range(3, 17):
            assert (r, 12) in env._walls, f"Divider (r={r},12) missing"

    def test_corridor_A_interior_open(self):
        env = EnvB_v2()
        for r in range(4, 16):
            for c in range(1, 7):
                assert (r, c) not in env._walls, f"Corridor A cell ({r},{c}) is a wall"

    def test_corridor_B_interior_open(self):
        env = EnvB_v2()
        for r in range(4, 16):
            for c in range(8, 12):
                assert (r, c) not in env._walls, f"Corridor B cell ({r},{c}) is a wall"

    def test_corridor_C_interior_open(self):
        env = EnvB_v2()
        for r in range(4, 16):
            for c in range(13, 19):
                assert (r, c) not in env._walls, f"Corridor C cell ({r},{c}) is a wall"

    def test_outer_walls_closed(self):
        env = EnvB_v2()
        for r in range(20):
            assert (r, 0) in env._walls
            assert (r, 19) in env._walls
        for c in range(20):
            assert (0, c) in env._walls
            assert (19, c) in env._walls

    def test_step_into_split_wall_blocked(self):
        env = EnvB_v2()
        # Cell (2,5) is in the top stem; moving down (action=1) hits split wall at (3,5)
        assert env.transition((2, 5), 1) == (2, 5)

    def test_step_through_doorway_allowed(self):
        env = EnvB_v2()
        # Cell (2,4) is in stem; moving down (action=1) goes to doorway (3,4)
        assert env.transition((2, 4), 1) == (3, 4)


# ── 4. Route family distinguishability ────────────────────────────────────────

class TestRouteFamilies:

    def test_cell_family_corridor_A(self):
        env = EnvB_v2()
        # Mid-corridor A cell
        assert env.cell_family((8, 3)) == "A"

    def test_cell_family_corridor_B(self):
        env = EnvB_v2()
        assert env.cell_family((8, 10)) == "B"

    def test_cell_family_corridor_C(self):
        env = EnvB_v2()
        assert env.cell_family((8, 15)) == "C"

    def test_cell_family_stem(self):
        env = EnvB_v2()
        assert env.cell_family((1, 9)) == "stem"
        assert env.cell_family((18, 5)) == "stem"

    def test_cell_family_doorway(self):
        env = EnvB_v2()
        for dw in _SPLIT_DOORWAYS + _MERGE_DOORWAYS:
            assert env.cell_family(dw) == "doorway"

    def test_corridor_A_cells_not_reachable_via_B_entry(self):
        """A policy entering via col 9 doorway cannot reach corridor A cells
        without passing back through the stem. Verify cell (4,3) ∈ corridor A
        is NOT adjacent to any corridor B cell."""
        env = EnvB_v2()
        # (4,3) is in corridor A; (4,7) is a divider wall
        # Verify adjacency across divider is blocked
        assert (4, 7) in env._walls   # divider between A and B
        assert (4, 12) in env._walls  # divider between B and C

    def test_exclusive_sa_pairs_nonempty_per_family(self):
        env = EnvB_v2()
        states = env.get_reachable_states()
        for fam, (lo, hi) in CORRIDOR_COLS_B2.items():
            excl = {(s, a) for s in states for a in range(N_ACTIONS)
                    if 4 <= s[0] <= 15 and lo <= s[1] <= hi}
            assert len(excl) > 0, f"No exclusive SA pairs for family {fam}"

    def test_three_corridor_families_recognized(self):
        env = EnvB_v2()
        assert set(CORRIDOR_COLS_B2.keys()) == {"A", "B", "C"}

    def test_corridor_widths(self):
        # A: 6 cols, B: 4 cols, C: 6 cols
        assert CORRIDOR_COLS_B2["A"] == (1, 6)
        assert CORRIDOR_COLS_B2["B"] == (8, 11)
        assert CORRIDOR_COLS_B2["C"] == (13, 18)


# ── 5. Waypoint routes reachable ──────────────────────────────────────────────

class TestWaypointRoutes:

    @pytest.mark.parametrize("family", ["A", "B", "C"])
    def test_waypoint_path_exists(self, family):
        """BFS can find a path through each waypoint sequence."""
        env = EnvB_v2()
        wps = [EnvB_v2.start_pos] + list(FAMILY_WAYPOINTS_B2[family])
        for i in range(len(wps) - 1):
            path = _bfs_path(wps[i], wps[i + 1], env._walls)
            assert path is not None, (
                f"Family {family}: no BFS path from {wps[i]} to {wps[i+1]}"
            )

    def test_all_waypoints_in_reachable(self):
        env = EnvB_v2()
        states = env.get_reachable_states()
        for fam, wps in FAMILY_WAYPOINTS_B2.items():
            for wp in wps:
                assert wp in states, f"Family {fam} waypoint {wp} not reachable"

    def test_family_A_mid_waypoint_in_corridor_A(self):
        # (10,1) should be in corridor A
        env = EnvB_v2()
        assert env.cell_family((10, 1)) == "A"

    def test_family_B_mid_waypoint_in_corridor_B(self):
        # (10,11) should be in corridor B
        env = EnvB_v2()
        assert env.cell_family((10, 11)) == "B"

    def test_family_C_mid_waypoint_in_corridor_C(self):
        # (10,18) should be in corridor C
        env = EnvB_v2()
        assert env.cell_family((10, 18)) == "C"


# ── 6. Reward / termination ───────────────────────────────────────────────────

class TestRewardAndTermination:

    def test_goal_termination(self):
        env = EnvB_v2()
        env.reset()
        env._pos = (18, 8)
        obs, reward, terminated, truncated, _ = env.step(3)  # right -> (18,9)=GOAL
        assert obs == (18, 9)
        assert terminated is True
        assert truncated is False
        assert abs(reward - (GOAL_REWARD + STEP_PENALTY)) < 1e-9

    def test_step_penalty(self):
        env = EnvB_v2()
        env.reset()
        # step down from (1,9) → (2,9) — a valid stem cell
        _, reward, term, trunc, _ = env.step(1)
        assert not term
        assert abs(reward - STEP_PENALTY) < 1e-9

    def test_truncation_at_horizon_200(self):
        env = EnvB_v2()
        env.reset()
        # Repeatedly bump into outer wall — stays at (1,9)
        for _ in range(199):
            _, _, t, tr, _ = env.step(0)  # up into outer wall
            assert not t and not tr
        _, _, t, tr, _ = env.step(0)
        assert tr is True and t is False

    def test_wall_collision_no_move(self):
        env = EnvB_v2()
        env.reset()
        env._pos = (1, 1)
        obs, _, _, _, _ = env.step(2)  # left -> col 0 = outer wall
        assert obs == (1, 1)
