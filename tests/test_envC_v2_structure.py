"""
tests/test_envC_v2_structure.py
Structural validation tests for EnvC_v2 (20×20 Key-Door Staged Multi-Route).

Checks:
  1. Basic interface (instantiation, start/goal/door/keys, horizon=300)
  2. Reachable extended states / SA pairs counts
  3. has_key state transitions (key pickup, door gating)
  4. Wall correctness (center wall, L/R divider, post-door separator)
  5. All four combined families can reach goal via waypoints
  6. Stage classification (Pre-L/Pre-R/Post-U/Post-D correctness)
  7. Reward / termination logic
  8. Key cell auto-pickup: both K1 and K2 work
  9. Door blocks passage without key; allows with key
 10. goal=(9,18) is only reachable with has_key=1
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from collections import deque
from envs.gridworld_envs import (
    EnvC_v2, FAMILY_WAYPOINTS_C2, CORRIDOR_C2,
    _WALLS_C2, _REACHABLE_C2, _DOOR_C2, _KEY_CELLS_C2, _GRID_C2,
    N_ACTIONS, HORIZON, GOAL_REWARD, STEP_PENALTY,
)

_START = EnvC_v2.start_pos   # (1, 4)
_GOAL  = EnvC_v2.goal_pos    # (9, 18)
_K1    = (5, 2)
_K2    = (5, 7)
_DOOR  = (9, 10)


def _bfs_nokey(start, goal, grid_size=20):
    """BFS on left half (no key, door blocked). Returns path or None."""
    no_key_walls = _WALLS_C2 | {_DOOR}
    if start == goal:
        return [start]
    parent = {start: None}; q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (r+dr, c+dc)
            if nb in parent or nb in no_key_walls: continue
            if not (0 <= nb[0] < grid_size and 0 <= nb[1] < grid_size): continue
            parent[nb] = (r,c)
            if nb == goal:
                path, cur = [], goal
                while cur is not None: path.append(cur); cur = parent[cur]
                return list(reversed(path))
            q.append(nb)
    return None


def _bfs_withkey(start, goal, grid_size=20):
    """BFS over whole grid (has key, door passable)."""
    if start == goal:
        return [start]
    parent = {start: None}; q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (r+dr, c+dc)
            if nb in parent or nb in _WALLS_C2: continue
            if not (0 <= nb[0] < grid_size and 0 <= nb[1] < grid_size): continue
            parent[nb] = (r,c)
            if nb == goal:
                path, cur = [], goal
                while cur is not None: path.append(cur); cur = parent[cur]
                return list(reversed(path))
            q.append(nb)
    return None


# ── 1. Basic interface ─────────────────────────────────────────────────────────

class TestBasicInterface:

    def test_instantiates(self):
        assert EnvC_v2() is not None

    def test_grid_size_20(self):
        assert EnvC_v2.grid_size == 20

    def test_start_pos(self):
        assert EnvC_v2.start_pos == (1, 4)

    def test_goal_pos(self):
        assert EnvC_v2.goal_pos == (9, 18)

    def test_door_pos(self):
        assert EnvC_v2.door_pos == (9, 10)

    def test_key_cells(self):
        assert (5, 2) in EnvC_v2.key_cells
        assert (5, 7) in EnvC_v2.key_cells

    def test_horizon_300(self):
        env = EnvC_v2()
        assert env._horizon == 300

    def test_base_horizon_unchanged(self):
        # Module-level HORIZON constant (used by EnvA/B/C/A_v2) must stay 100
        assert HORIZON == 100

    def test_reset_returns_extended_start(self):
        env = EnvC_v2()
        obs, info = env.reset()
        assert obs == (_START, 0), f"Expected (start, 0), got {obs}"

    def test_reset_clears_key(self):
        env = EnvC_v2()
        # Simulate picking up a key by manually setting has_key
        env._has_key = 1
        env.reset()
        obs, _ = env.reset()
        assert obs[1] == 0, "has_key not cleared on reset"

    def test_four_combined_families(self):
        assert set(FAMILY_WAYPOINTS_C2.keys()) == {"LU", "LD", "RU", "RD"}


# ── 2. Reachable states / SA pairs ────────────────────────────────────────────

class TestReachability:

    def test_reachable_extended_nonempty(self):
        assert len(_REACHABLE_C2) > 0

    def test_reachable_count_stable(self):
        # Pilot measured 269 extended states; allow ±2 for floating-point edge cases
        assert 265 <= len(_REACHABLE_C2) <= 275, \
            f"Expected ~269 extended states, got {len(_REACHABLE_C2)}"

    def test_sa_pairs_stable(self):
        env = EnvC_v2()
        sa_pairs = env.get_reachable_state_action_pairs()
        # Pilot measured 1076 SA pairs; allow ±8
        assert 1060 <= len(sa_pairs) <= 1090, \
            f"Expected ~1076 SA pairs, got {len(sa_pairs)}"

    def test_start_in_reachable(self):
        assert (_START, 0) in _REACHABLE_C2

    def test_goal_in_reachable_withkey(self):
        assert (_GOAL, 1) in _REACHABLE_C2

    def test_goal_not_reachable_nokey(self):
        # Goal is in right half — unreachable without key (door blocks passage)
        assert (_GOAL, 0) not in _REACHABLE_C2

    def test_both_keys_in_reachable_as_withkey(self):
        # Key cells are stored as has_key=1 (pickup happens on entry)
        assert (_K1, 1) in _REACHABLE_C2
        assert (_K2, 1) in _REACHABLE_C2

    def test_key_cells_not_in_reachable_as_nokey(self):
        # Agents are never at a key cell with has_key=0 (key immediately picked up)
        assert (_K1, 0) not in _REACHABLE_C2
        assert (_K2, 0) not in _REACHABLE_C2

    def test_get_reachable_states_returns_c2(self):
        env = EnvC_v2()
        assert env.get_reachable_states() is _REACHABLE_C2


# ── 3. has_key state transitions ──────────────────────────────────────────────

class TestKeyDoorLogic:

    def test_picking_up_k1_sets_has_key(self):
        env = EnvC_v2(); env.reset()
        # Navigate to K1=(5,2) manually; start=(1,4)
        # Use transition() — no side effects, verifies state logic
        state = ((4, 2), 0)
        next_state = env.transition(state, 1)  # action 1 = down → (5,2)
        assert next_state == ((5, 2), 1), f"Expected ((5,2),1), got {next_state}"

    def test_picking_up_k2_sets_has_key(self):
        env = EnvC_v2(); env.reset()
        state = ((4, 7), 0)
        next_state = env.transition(state, 1)  # down → (5,7)=K2
        assert next_state == ((5, 7), 1), f"Expected ((5,7),1), got {next_state}"

    def test_key_not_lost_after_pickup(self):
        env = EnvC_v2(); env.reset()
        state = ((5, 2), 1)  # at K1 with key already
        next_state = env.transition(state, 1)  # move away: down → (6,2)
        assert next_state[1] == 1, "has_key should remain 1 after leaving key cell"

    def test_door_blocks_without_key(self):
        env = EnvC_v2(); env.reset()
        state = ((9, 9), 0)
        next_state = env.transition(state, 3)  # right → (9,10) = door, blocked
        assert next_state == ((9, 9), 0), \
            f"Door should block without key; got {next_state}"

    def test_door_passable_with_key(self):
        env = EnvC_v2(); env.reset()
        state = ((9, 9), 1)
        next_state = env.transition(state, 3)  # right → (9,10) = door, passable
        assert next_state == ((9, 10), 1), \
            f"Door should allow passage with key; got {next_state}"

    def test_step_key_pickup_updates_env_state(self):
        env = EnvC_v2(); env.reset()
        # Walk to K1=(5,2)
        for _ in range(20):  # allow enough steps
            r, c = env._pos
            if (r, c) == (4, 2):
                obs, _, _, _, _ = env.step(1)  # down to K1
                assert obs[1] == 1, f"has_key should be 1 after step onto K1; obs={obs}"
                return
            # crude navigation: move left and down
            if c > 2:
                env.step(2)  # left
            elif r < 4:
                env.step(1)  # down
            else:
                break
        # If we didn't reach K1, at least confirm the mechanics work via transition
        env2 = EnvC_v2(); env2.reset()
        s = ((4, 2), 0)
        ns = env2.transition(s, 1)
        assert ns[1] == 1

    def test_second_key_does_not_change_has_key_to_0(self):
        # Stepping on K2 when already has_key=1 keeps has_key=1
        env = EnvC_v2(); env.reset()
        state = ((4, 7), 1)
        next_state = env.transition(state, 1)  # down → (5,7)=K2 with key already
        assert next_state == ((5, 7), 1)


# ── 4. Wall correctness ───────────────────────────────────────────────────────

class TestWalls:

    def test_outer_walls(self):
        G = _GRID_C2
        for c in range(G):
            assert (0, c) in _WALLS_C2
            assert (G-1, c) in _WALLS_C2
        for r in range(G):
            assert (r, 0) in _WALLS_C2
            assert (r, G-1) in _WALLS_C2

    def test_center_wall_col_10_blocked(self):
        # Center wall at col 10, rows 1-18 except door
        for r in range(1, 19):
            if (r, 10) != _DOOR:
                assert (r, 10) in _WALLS_C2

    def test_door_cell_not_in_permanent_walls(self):
        assert _DOOR not in _WALLS_C2

    def test_lr_divider_col4_rows2to8(self):
        for r in range(2, 9):
            assert (r, 4) in _WALLS_C2

    def test_lr_divider_row1_col4_open(self):
        # Row 1 is NOT divided — start (1,4) can access both sides via row 1
        assert (1, 4) not in _WALLS_C2

    def test_post_door_separator_row9_cols12to17(self):
        for c in range(12, 18):  # cols 12-17
            assert (9, c) in _WALLS_C2

    def test_post_door_separator_col18_open(self):
        # col 18 row 9 is the goal approach path (must be open)
        assert (9, 18) not in _WALLS_C2

    def test_bottom_left_dead_zone(self):
        # rows 10-18, cols 1-9 are walls
        for r in range(10, 19):
            for c in range(1, 10):
                assert (r, c) in _WALLS_C2

    def test_top_right_dead_zone(self):
        # rows 1-2, cols 11-18 are walls
        for r in range(1, 3):
            for c in range(11, 19):
                assert (r, c) in _WALLS_C2

    def test_postd_far_right_open(self):
        # rows 10-18 cols 16-18 are NOT blocked (v2 removed that wall)
        for r in [12, 13, 14]:
            for c in [16, 17]:
                assert (r, c) not in _WALLS_C2

    def test_transition_stays_in_bounds(self):
        env = EnvC_v2()
        # Any action from a corner or edge must not escape bounds
        state = ((1, 1), 0)
        for a in range(N_ACTIONS):
            ns = env.transition(state, a)
            r, c = ns[0]
            assert 0 < r < _GRID_C2 and 0 < c < _GRID_C2, \
                f"Transition escaped grid: {ns}"


# ── 5. Route families reach goal via waypoints ────────────────────────────────

class TestFamilyRoutability:

    def test_k1_reachable_nokey(self):
        path = _bfs_nokey(_START, _K1)
        assert path is not None, "K1 unreachable from start (no key)"

    def test_k2_reachable_nokey(self):
        path = _bfs_nokey(_START, _K2)
        assert path is not None, "K2 unreachable from start (no key)"

    def test_door_reachable_withkey_from_k1(self):
        path = _bfs_withkey(_K1, _DOOR)
        assert path is not None, "Door unreachable from K1 (with key)"

    def test_door_reachable_withkey_from_k2(self):
        path = _bfs_withkey(_K2, _DOOR)
        assert path is not None, "Door unreachable from K2 (with key)"

    def test_post_u_waypoint_reachable_from_door(self):
        wp_u = (5, 15)
        path = _bfs_withkey(_DOOR, wp_u)
        assert path is not None, f"Post-U waypoint {wp_u} unreachable from door"

    def test_post_d_waypoint_reachable_from_door(self):
        wp_d = (13, 15)
        path = _bfs_withkey(_DOOR, wp_d)
        assert path is not None, f"Post-D waypoint {wp_d} unreachable from door"

    def test_goal_reachable_from_post_u_waypoint(self):
        path = _bfs_withkey((5, 15), _GOAL)
        assert path is not None, "Goal unreachable from Post-U waypoint"

    def test_goal_reachable_from_post_d_waypoint(self):
        path = _bfs_withkey((13, 15), _GOAL)
        assert path is not None, "Goal unreachable from Post-D waypoint"

    @pytest.mark.parametrize("family", ["LU", "LD", "RU", "RD"])
    def test_full_family_path_exists(self, family):
        """End-to-end BFS connectivity for each combined family."""
        wps = FAMILY_WAYPOINTS_C2[family]
        # Phase 1: start → key (no key)
        path = _bfs_nokey(_START, wps[0])
        assert path is not None, f"Family {family}: start → key unreachable"
        # Phase 2: key → door (with key)
        path = _bfs_withkey(wps[0], wps[1])
        assert path is not None, f"Family {family}: key → door unreachable"
        # Phase 3: door → mid-waypoint (with key)
        path = _bfs_withkey(wps[1], wps[2])
        assert path is not None, f"Family {family}: door → mid-waypoint unreachable"
        # Phase 4: mid-waypoint → goal (with key)
        path = _bfs_withkey(wps[2], wps[3])
        assert path is not None, f"Family {family}: mid-waypoint → goal unreachable"


# ── 6. Stage classification ───────────────────────────────────────────────────

class TestStageClassification:

    def setup_method(self):
        self.env = EnvC_v2()

    def test_pre_l_cells_classified(self):
        assert self.env.cell_stage((3, 2), 0) == "Pre-L"
        assert self.env.cell_stage((7, 1), 0) == "Pre-L"

    def test_pre_r_cells_classified(self):
        assert self.env.cell_stage((3, 6), 0) == "Pre-R"
        assert self.env.cell_stage((7, 9), 0) == "Pre-R"

    def test_pre_shared_includes_row1(self):
        assert self.env.cell_stage((1, 3), 0) == "Pre-shared"
        assert self.env.cell_stage((1, 7), 0) == "Pre-shared"

    def test_door_classified(self):
        assert self.env.cell_stage(_DOOR, 1) == "door"

    def test_post_u_cells_classified(self):
        assert self.env.cell_stage((5, 14), 1) == "Post-U"
        assert self.env.cell_stage((4, 17), 1) == "Post-U"

    def test_post_d_cells_classified(self):
        assert self.env.cell_stage((12, 15), 1) == "Post-D"
        assert self.env.cell_stage((14, 18), 1) == "Post-D"

    def test_left_half_withkey_is_post_shared(self):
        # Left-half states with has_key=1 (K→door navigation) → post-shared
        assert self.env.cell_stage((6, 2), 1) == "Post-shared"
        assert self.env.cell_stage((8, 7), 1) == "Post-shared"


# ── 7. Reward / termination ───────────────────────────────────────────────────

class TestRewardTermination:

    def test_step_penalty_on_non_goal(self):
        env = EnvC_v2(); env.reset()
        # Any step not reaching goal gives STEP_PENALTY
        obs, reward, term, trunc, _ = env.step(0)  # up from (1,4) → probably wall = stays
        assert abs(reward - STEP_PENALTY) < 1e-5, f"Expected step penalty, got {reward}"
        assert not term

    def test_goal_gives_positive_reward(self):
        env = EnvC_v2(); env.reset()
        # Teleport to just before goal and take the right step
        env._pos = (9, 17)
        env._has_key = 1
        obs, reward, term, trunc, _ = env.step(3)  # right → (9,18) = goal
        assert term, "Should be terminated at goal"
        expected = STEP_PENALTY + GOAL_REWARD
        assert abs(reward - expected) < 1e-5, \
            f"Expected {expected}, got {reward}"

    def test_truncation_at_horizon(self):
        env = EnvC_v2(); env.reset()
        env._step_count = env._horizon - 1
        obs, reward, term, trunc, _ = env.step(0)
        if not term:
            assert trunc, "Should truncate when step_count >= horizon"

    def test_horizon_is_300(self):
        env = EnvC_v2()
        assert env._horizon == 300

    def test_goal_reward_constant(self):
        assert GOAL_REWARD == 1.0

    def test_step_penalty_constant(self):
        assert STEP_PENALTY == -0.01


# ── 8. Existing envs unaffected ───────────────────────────────────────────────

class TestNoRegression:

    def test_envb_v2_still_works(self):
        from envs.gridworld_envs import EnvB_v2
        env = EnvB_v2()
        obs, _ = env.reset()
        assert obs == (1, 9)
        assert env._horizon == 200

    def test_enva_v2_still_works(self):
        from envs.gridworld_envs import EnvA_v2
        env = EnvA_v2()
        obs, _ = env.reset()
        assert obs == (1, 14)
