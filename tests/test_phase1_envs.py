"""
Phase 1 unit tests for EnvA, EnvB, EnvC.
Run with: pytest tests/test_phase1_envs.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from envs.gridworld_envs import (
    EnvA, EnvB, EnvC,
    HORIZON, GOAL_REWARD, STEP_PENALTY,
    N_ACTIONS, DOOR_POS_C, KEY_POS_C,
)


# ─── Env-A ────────────────────────────────────────────────────────────────────

class TestEnvA:

    def test_reset_returns_fixed_start(self):
        env = EnvA()
        obs, _ = env.reset(seed=0)
        assert obs == EnvA.start_pos
        obs2, _ = env.reset(seed=42)
        assert obs2 == EnvA.start_pos

    def test_deterministic_transition_same_state_same_action(self):
        env = EnvA()
        env.reset()
        obs1, _, _, _, _ = env.step(1)  # down from start

        env.reset()
        obs2, _, _, _, _ = env.step(1)
        assert obs1 == obs2

    def test_transition_helper_is_deterministic(self):
        env = EnvA()
        s = EnvA.start_pos
        for a in range(N_ACTIONS):
            ns1 = env.transition(s, a)
            ns2 = env.transition(s, a)
            assert ns1 == ns2

    def test_no_wall_penetration(self):
        env = EnvA()
        # (1,1) moving up hits outer wall row-0, must stay at (1,1)
        env.reset()
        obs, _, _, _, _ = env.step(0)  # up
        assert obs == (1, 1)

        # (1,1) moving left hits outer wall col-0, must stay at (1,1)
        env.reset()
        obs, _, _, _, _ = env.step(2)  # left
        assert obs == (1, 1)

    def test_wall_cells_not_reachable(self):
        env = EnvA()
        reachable = env.get_reachable_states()
        for w in env._walls:
            assert w not in reachable

    def test_goal_reward_and_terminated(self):
        env = EnvA()
        env.reset()
        # Manually place agent adjacent to goal and step onto it
        env._pos = (13, 12)  # one step left of goal (13,13)
        obs, reward, terminated, truncated, _ = env.step(3)  # right -> (13,13)
        assert obs == EnvA.goal_pos
        assert terminated is True
        assert truncated is False
        assert abs(reward - (GOAL_REWARD + STEP_PENALTY)) < 1e-9

    def test_step_penalty_on_non_goal(self):
        env = EnvA()
        env.reset()
        # Move into free space — not goal, just penalty
        env._pos = (1, 2)
        obs, reward, terminated, truncated, _ = env.step(1)  # down to (2,2)
        assert terminated is False
        assert abs(reward - STEP_PENALTY) < 1e-9

    def test_horizon_truncated(self):
        env = EnvA()
        env.reset()
        # Keep bumping into the same wall so we stay far from goal
        for _ in range(HORIZON - 1):
            _, _, terminated, truncated, _ = env.step(0)  # up (wall)
            assert not terminated
            assert not truncated
        _, _, terminated, truncated, _ = env.step(0)
        assert truncated is True
        assert terminated is False

    def test_reachable_states_nonempty_finite(self):
        env = EnvA()
        rs = env.get_reachable_states()
        assert len(rs) > 0
        assert len(rs) < 15 * 15  # walls excluded

    def test_reachable_sa_pairs_nonempty_finite(self):
        env = EnvA()
        pairs = env.get_reachable_state_action_pairs()
        assert len(pairs) > 0
        assert len(pairs) == len(env.get_reachable_states()) * N_ACTIONS

    def test_reachable_states_reproducible(self):
        env1, env2 = EnvA(), EnvA()
        assert env1.get_reachable_states() == env2.get_reachable_states()

    def test_doorways_are_reachable(self):
        env = EnvA()
        rs = env.get_reachable_states()
        for d in EnvA.doorway_positions:
            assert d in rs, f"Doorway {d} not in reachable states"

    def test_start_and_goal_in_reachable(self):
        env = EnvA()
        rs = env.get_reachable_states()
        assert EnvA.start_pos in rs
        assert EnvA.goal_pos in rs


# ─── Env-B ────────────────────────────────────────────────────────────────────

class TestEnvB:

    def test_reset_returns_fixed_start(self):
        env = EnvB()
        obs, _ = env.reset()
        assert obs == EnvB.start_pos

    def test_deterministic_transition(self):
        env = EnvB()
        env.reset()
        obs1, _, _, _, _ = env.step(1)
        env.reset()
        obs2, _, _, _, _ = env.step(1)
        assert obs1 == obs2

    def test_transition_helper_deterministic(self):
        env = EnvB()
        s = EnvB.start_pos
        for a in range(N_ACTIONS):
            assert env.transition(s, a) == env.transition(s, a)

    def test_bottleneck_walls_block_movement(self):
        env = EnvB()
        # Row 5, col 1 is a wall in barrier 1.
        # Try to move down from (4,1) — should be blocked by wall at (5,1)
        env.reset()
        env._pos = (4, 1)
        obs, _, _, _, _ = env.step(1)  # down -> (5,1) is wall -> stays (4,1)
        assert obs == (4, 1)

    def test_bottleneck_gap_is_passable(self):
        env = EnvB()
        # Gap at (5,5): moving down from (4,5) should reach (5,5)
        env.reset()
        env._pos = (4, 5)
        obs, _, _, _, _ = env.step(1)  # down
        assert obs == (5, 5)

    def test_goal_terminated(self):
        env = EnvB()
        env.reset()
        env._pos = (13, 12)
        obs, reward, terminated, truncated, _ = env.step(3)  # right
        assert obs == EnvB.goal_pos
        assert terminated is True
        assert truncated is False
        assert abs(reward - (GOAL_REWARD + STEP_PENALTY)) < 1e-9

    def test_horizon_truncated(self):
        env = EnvB()
        env.reset()
        for _ in range(HORIZON - 1):
            _, _, terminated, truncated, _ = env.step(0)
            assert not terminated and not truncated
        _, _, _, truncated, _ = env.step(0)
        assert truncated is True

    def test_reachable_states_nonempty_finite(self):
        env = EnvB()
        rs = env.get_reachable_states()
        assert len(rs) > 0
        assert len(rs) < 15 * 15

    def test_reachable_sa_pairs_nonempty_finite(self):
        env = EnvB()
        pairs = env.get_reachable_state_action_pairs()
        assert len(pairs) == len(env.get_reachable_states()) * N_ACTIONS

    def test_reachable_states_reproducible(self):
        env1, env2 = EnvB(), EnvB()
        assert env1.get_reachable_states() == env2.get_reachable_states()

    def test_start_goal_in_reachable(self):
        env = EnvB()
        rs = env.get_reachable_states()
        assert EnvB.start_pos in rs
        assert EnvB.goal_pos in rs

    def test_bottleneck_cells_in_reachable(self):
        from envs.gridworld_envs import BOTTLENECKS_B
        env = EnvB()
        rs = env.get_reachable_states()
        for b in BOTTLENECKS_B:
            assert b in rs, f"Bottleneck gap {b} not reachable"

    def test_wall_cells_not_reachable(self):
        env = EnvB()
        rs = env.get_reachable_states()
        for w in env._walls:
            assert w not in rs


# ─── Env-C ────────────────────────────────────────────────────────────────────

class TestEnvC:

    def test_reset_has_key_zero(self):
        env = EnvC()
        obs, _ = env.reset()
        pos, has_key = obs
        assert has_key == 0
        assert pos == EnvC.start_pos

    def test_reset_always_same_start(self):
        env = EnvC()
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=99)
        assert obs1 == obs2 == (EnvC.start_pos, 0)

    def test_door_blocked_without_key(self):
        env = EnvC()
        env.reset()
        door_r, door_c = DOOR_POS_C
        # Place agent one step above the door (row door_r-1, col door_c)
        # Moving down should be blocked because has_key=0
        env._pos = (door_r - 1, door_c)
        env._has_key = 0
        obs, _, _, _, _ = env.step(1)  # down
        pos, hk = obs
        assert pos == (door_r - 1, door_c), "Agent should not pass door without key"

    def test_door_passable_with_key(self):
        env = EnvC()
        env.reset()
        door_r, door_c = DOOR_POS_C
        env._pos = (door_r - 1, door_c)
        env._has_key = 1
        obs, _, _, _, _ = env.step(1)  # down -> through door
        pos, hk = obs
        assert pos == DOOR_POS_C, "Agent should pass door with key"

    def test_key_pickup_auto(self):
        env = EnvC()
        env.reset()
        key_r, key_c = KEY_POS_C
        # Place agent one step above the key
        env._pos = (key_r - 1, key_c)
        env._has_key = 0
        obs, _, _, _, _ = env.step(1)  # down -> key cell
        pos, hk = obs
        assert pos == KEY_POS_C
        assert hk == 1

    def test_has_key_stays_after_leaving_key_cell(self):
        env = EnvC()
        env.reset()
        env._pos = KEY_POS_C
        env._has_key = 1
        obs, _, _, _, _ = env.step(1)  # leave key cell downward
        _, hk = obs
        assert hk == 1

    def test_goal_reward_and_terminated(self):
        env = EnvC()
        env.reset()
        env._pos = (13, 12)
        env._has_key = 1
        obs, reward, terminated, truncated, _ = env.step(3)  # right -> goal
        pos, hk = obs
        assert pos == EnvC.goal_pos
        assert terminated is True
        assert truncated is False
        assert abs(reward - (GOAL_REWARD + STEP_PENALTY)) < 1e-9

    def test_horizon_truncated(self):
        env = EnvC()
        env.reset()
        for _ in range(HORIZON - 1):
            _, _, terminated, truncated, _ = env.step(0)
            assert not terminated and not truncated
        _, _, terminated, truncated, _ = env.step(0)
        assert truncated is True
        assert terminated is False

    def test_reachable_states_nonempty_finite(self):
        env = EnvC()
        rs = env.get_reachable_states()
        assert len(rs) > 0
        assert len(rs) < 15 * 15 * 2  # at most 2 copies of grid

    def test_reachable_sa_pairs_nonempty_finite(self):
        env = EnvC()
        pairs = env.get_reachable_state_action_pairs()
        assert len(pairs) == len(env.get_reachable_states()) * N_ACTIONS

    def test_reachable_states_reproducible(self):
        env1, env2 = EnvC(), EnvC()
        assert env1.get_reachable_states() == env2.get_reachable_states()

    def test_reachable_includes_both_has_key_values(self):
        env = EnvC()
        rs = env.get_reachable_states()
        has_key_0 = {s for s in rs if s[1] == 0}
        has_key_1 = {s for s in rs if s[1] == 1}
        assert len(has_key_0) > 0, "No states with has_key=0"
        assert len(has_key_1) > 0, "No states with has_key=1"

    def test_key_and_goal_in_reachable(self):
        env = EnvC()
        rs = env.get_reachable_states()
        # Key is auto-picked up on entry, so (KEY_POS_C, 0) is never a valid state.
        # Agent arrives at key cell already holding the key.
        assert (KEY_POS_C, 0) not in rs
        assert (KEY_POS_C, 1) in rs
        assert (EnvC.goal_pos, 1) in rs

    def test_goal_not_reachable_without_key_if_door_required(self):
        env = EnvC()
        rs = env.get_reachable_states()
        # Goal is on the right side; to reach it must pass door, requiring key.
        # So (goal_pos, 0) should NOT be reachable.
        assert (EnvC.goal_pos, 0) not in rs

    def test_transition_helper_deterministic(self):
        env = EnvC()
        s = (EnvC.start_pos, 0)
        for a in range(N_ACTIONS):
            assert env.transition(s, a) == env.transition(s, a)

    def test_transition_door_logic(self):
        env = EnvC()
        door_r, door_c = DOOR_POS_C
        above_door = (door_r - 1, door_c)
        # Without key: blocked
        ns = env.transition((above_door, 0), 1)
        assert ns == (above_door, 0)
        # With key: passes through
        ns = env.transition((above_door, 1), 1)
        assert ns == (DOOR_POS_C, 1)

    def test_reset_clears_has_key(self):
        env = EnvC()
        env.reset()
        env._pos = KEY_POS_C
        env._has_key = 1
        env.reset()
        _, hk = env._make_obs()
        assert hk == 0
