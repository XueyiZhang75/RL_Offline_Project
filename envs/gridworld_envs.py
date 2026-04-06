"""
Phase 1: Discrete gridworld environments for Offline RL coverage study.

Environments:
  EnvA — 15x15 FourRooms (baseline, known to saturate at 50k/200k scale)
  EnvB — 15x15 Bottleneck (verification environment)
  EnvC — 15x15 Key-Door (verification environment)

Actions: 0=up, 1=down, 2=left, 3=right
Deterministic, horizon=100, reward: +1 at goal, -0.01 per step
"""

from collections import deque

# Action deltas: 0=up, 1=down, 2=left, 3=right
_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
N_ACTIONS = 4
HORIZON = 100
GOAL_REWARD = 1.0
STEP_PENALTY = -0.01


class _GridBase:
    """Lightweight base for deterministic discrete gridworlds."""

    def __init__(self):
        self._step_count = 0

    # Subclasses must define:
    #   self._walls: set of (r, c) impassable cells
    #   self._start: (r, c)
    #   self._goal:  (r, c)
    #   self._state_from_pos(pos) -> hashable state
    #   self._pos_from_state(state) -> (r, c)
    #   self._extra_blocked(pos, next_pos) -> bool  (extra logic beyond walls)
    #   self._extra_transition(pos, next_pos) -> (next_pos, extra_reward)

    def _try_move(self, pos, action):
        dr, dc = _DELTAS[action]
        r, c = pos
        nr, nc = r + dr, c + dc
        if (nr, nc) in self._walls:
            return pos
        return (nr, nc)

    def reset(self, seed=None):
        self._step_count = 0
        self._pos = self._start
        self._on_reset()
        obs = self._make_obs()
        return obs, {}

    def _on_reset(self):
        pass

    def step(self, action):
        self._step_count += 1
        next_pos = self._try_move(self._pos, action)
        next_pos, extra_reward = self._apply_special(next_pos)
        self._pos = next_pos

        terminated = (self._pos == self._goal)
        truncated = (not terminated) and (self._step_count >= HORIZON)

        reward = STEP_PENALTY + extra_reward
        if terminated:
            reward += GOAL_REWARD

        obs = self._make_obs()
        return obs, reward, terminated, truncated, {}

    def _apply_special(self, next_pos):
        return next_pos, 0.0

    def _make_obs(self):
        return self._state_from_pos(self._pos)

    def get_reachable_states(self):
        raise NotImplementedError

    def get_reachable_state_action_pairs(self):
        states = self.get_reachable_states()
        return [(s, a) for s in states for a in range(N_ACTIONS)]

    def transition(self, state, action):
        """Deterministic transition: given state + action -> next state (no side effects)."""
        raise NotImplementedError


# ─────────────────────────────────────────────
# Env-A: 15x15 FourRooms
# ─────────────────────────────────────────────
#
# Layout (0-indexed rows 0..14, cols 0..14):
#
#   Outer walls: row 0, row 14, col 0, col 14.
#
#   Horizontal wall: row 7, cols 1..13, with doorways at col 3 and col 11.
#   Vertical wall:   col 7, rows 1..13, with doorways at row 3 and row 11.
#
#   This creates 4 rooms:
#     Top-left    (rows 1-6,  cols 1-6)
#     Top-right   (rows 1-6,  cols 8-13)
#     Bottom-left (rows 8-13, cols 1-6)
#     Bottom-right(rows 8-13, cols 8-13)
#
#   Doorways (wall cells removed):
#     Horizontal wall: (7, 3) and (7, 11)
#     Vertical wall:   (3, 7) and (11, 7)
#
#   Start: (1, 1)   — top-left room corner
#   Goal:  (13, 13) — bottom-right room corner
#
DOORWAYS_A = {(7, 3), (7, 11), (3, 7), (11, 7)}


def _build_walls_A():
    walls = set()
    for r in range(15):
        walls.add((r, 0))
        walls.add((r, 14))
    for c in range(15):
        walls.add((0, c))
        walls.add((14, c))
    # horizontal divider at row 7
    for c in range(1, 14):
        if (7, c) not in DOORWAYS_A:
            walls.add((7, c))
    # vertical divider at col 7
    for r in range(1, 14):
        if (r, 7) not in DOORWAYS_A:
            walls.add((r, 7))
    return walls


_WALLS_A = _build_walls_A()
_FREE_A = {(r, c) for r in range(15) for c in range(15) if (r, c) not in _WALLS_A}


class EnvA(_GridBase):
    """
    15x15 deterministic FourRooms.
    Start: (1,1), Goal: (13,13)
    Doorways: (7,3), (7,11), (3,7), (11,7)
    """

    start_pos = (1, 1)
    goal_pos = (13, 13)
    doorway_positions = DOORWAYS_A

    def __init__(self):
        super().__init__()
        self._walls = _WALLS_A
        self._start = self.start_pos
        self._goal = self.goal_pos

    def _state_from_pos(self, pos):
        return pos  # (row, col) tuple

    def _pos_from_state(self, state):
        return state

    def get_reachable_states(self):
        return frozenset(_FREE_A)

    def transition(self, state, action):
        pos = self._pos_from_state(state)
        next_pos = self._try_move(pos, action)
        return self._state_from_pos(next_pos)


# ─────────────────────────────────────────────
# Env-B: 15x15 Bottleneck / Obstacle
# ─────────────────────────────────────────────
#
# Layout:
#   Outer walls as usual.
#   Two horizontal obstacle walls creating a narrow corridor:
#     Wall 1: row 5, cols 1..10  (with single-cell gap at col 5)
#     Wall 2: row 9, cols 4..13  (with single-cell gap at col 9)
#   A vertical barrier at col 7, rows 6..8, blocking direct diagonal shortcut,
#   with a single-cell gap at row 7.
#
#   This forces the agent through two sequential single-cell bottlenecks.
#
#   Start: (1, 1)   — top-left region
#   Goal:  (13, 13) — bottom-right region
#
BOTTLENECKS_B = {(5, 5), (9, 9), (7, 7)}   # gap cells (NOT walls)


def _build_walls_B():
    walls = set()
    # outer walls
    for r in range(15):
        walls.add((r, 0))
        walls.add((r, 14))
    for c in range(15):
        walls.add((0, c))
        walls.add((14, c))
    # horizontal barrier row 5, cols 1..10, gap at col 5
    for c in range(1, 11):
        if (5, c) not in BOTTLENECKS_B:
            walls.add((5, c))
    # horizontal barrier row 9, cols 4..13, gap at col 9
    for c in range(4, 14):
        if (9, c) not in BOTTLENECKS_B:
            walls.add((9, c))
    # vertical barrier col 7, rows 6..8, gap at row 7
    for r in range(6, 9):
        if (r, 7) not in BOTTLENECKS_B:
            walls.add((r, 7))
    return walls


_WALLS_B = _build_walls_B()
_FREE_B = {(r, c) for r in range(15) for c in range(15) if (r, c) not in _WALLS_B}


def _bfs_reachable(start, walls, grid_size=15):
    reachable = set()
    queue = deque([start])
    reachable.add(start)
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DELTAS.values():
            nr, nc = r + dr, c + dc
            if (nr, nc) not in reachable and (nr, nc) not in walls:
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    reachable.add((nr, nc))
                    queue.append((nr, nc))
    return frozenset(reachable)


class EnvB(_GridBase):
    """
    15x15 Bottleneck/Obstacle environment.
    Start: (1,1), Goal: (13,13)
    Two sequential single-cell bottlenecks at (5,5) and (9,9),
    plus vertical barrier gap at (7,7).
    """

    start_pos = (1, 1)
    goal_pos = (13, 13)
    bottleneck_positions = BOTTLENECKS_B

    def __init__(self):
        super().__init__()
        self._walls = _WALLS_B
        self._start = self.start_pos
        self._goal = self.goal_pos
        self._reachable = _bfs_reachable(self._start, self._walls)

    def _state_from_pos(self, pos):
        return pos

    def _pos_from_state(self, state):
        return state

    def get_reachable_states(self):
        return self._reachable

    def transition(self, state, action):
        pos = self._pos_from_state(state)
        next_pos = self._try_move(pos, action)
        return self._state_from_pos(next_pos)


# ─────────────────────────────────────────────
# Env-C: 15x15 Key-Door Maze
# ─────────────────────────────────────────────
#
# Layout:
#   Outer walls as usual.
#   A vertical wall at col 7, rows 1..13, with the door at row 7.
#     The door cell (7, 7) is passable only when has_key=1.
#     Without key, (7, 7) acts as a wall.
#   Key location: (2, 3) — left half
#   Start:  (1, 1)   — left half
#   Door:   (7, 7)   — in the wall, requires key
#   Goal:   (13, 13) — right half
#
#   State: ((row, col), has_key)  where has_key in {0, 1}
#   Stepping onto the key cell when has_key=0 picks up the key (has_key->1).
#   The door cell can only be entered when has_key=1.
#
KEY_POS_C = (2, 3)
DOOR_POS_C = (7, 7)


def _build_walls_C():
    walls = set()
    # outer walls
    for r in range(15):
        walls.add((r, 0))
        walls.add((r, 14))
    for c in range(15):
        walls.add((0, c))
        walls.add((14, c))
    # vertical wall at col 7, rows 1..13; door at row 7 (conditional, not a permanent wall)
    for r in range(1, 14):
        if r != 7:
            walls.add((r, 7))
    # door cell (7,7) is NOT in permanent walls; blocked conditionally via _apply_special
    return walls


_WALLS_C = _build_walls_C()


def _bfs_reachable_C(start_state, walls):
    """BFS over extended state (pos, has_key). Door (7,7) requires has_key=1."""
    reachable = set()
    queue = deque([start_state])
    reachable.add(start_state)
    while queue:
        (r, c), hk = queue.popleft()
        for dr, dc in _DELTAS.values():
            nr, nc = r + dr, c + dc
            if (nr, nc) in walls:
                continue
            # door logic
            if (nr, nc) == DOOR_POS_C and hk == 0:
                continue
            new_hk = 1 if (nr, nc) == KEY_POS_C else hk
            ns = ((nr, nc), new_hk)
            if ns not in reachable:
                reachable.add(ns)
                queue.append(ns)
    return frozenset(reachable)


class EnvC(_GridBase):
    """
    15x15 Key-Door Maze.
    Start: (1,1), Key: (2,3), Door: (7,7), Goal: (13,13)
    Extended state: ((row, col), has_key).
    Door requires has_key=1 to pass.
    Picking up key is automatic on entering key cell.
    """

    start_pos = (1, 1)
    key_pos = KEY_POS_C
    door_pos = DOOR_POS_C
    goal_pos = (13, 13)

    def __init__(self):
        super().__init__()
        self._walls = _WALLS_C
        self._start = self.start_pos
        self._goal = self.goal_pos
        self._has_key = 0
        self._reachable = _bfs_reachable_C((self._start, 0), self._walls)

    def _on_reset(self):
        self._has_key = 0

    def _state_from_pos(self, pos):
        return (pos, self._has_key)

    def _pos_from_state(self, state):
        return state[0]

    def _try_move(self, pos, action):
        dr, dc = _DELTAS[action]
        r, c = pos
        nr, nc = r + dr, c + dc
        if (nr, nc) in self._walls:
            return pos
        # door is blocked without key
        if (nr, nc) == DOOR_POS_C and self._has_key == 0:
            return pos
        return (nr, nc)

    def _apply_special(self, next_pos):
        if next_pos == KEY_POS_C and self._has_key == 0:
            self._has_key = 1
        return next_pos, 0.0

    def _make_obs(self):
        return (self._pos, self._has_key)

    def get_reachable_states(self):
        return self._reachable

    def transition(self, state, action):
        pos, hk = state
        dr, dc = _DELTAS[action]
        r, c = pos
        nr, nc = r + dr, c + dc
        if (nr, nc) in self._walls:
            return state
        if (nr, nc) == DOOR_POS_C and hk == 0:
            return state
        new_hk = 1 if (nr, nc) == KEY_POS_C else hk
        return ((nr, nc), new_hk)


# ─────────────────────────────────────────────
# EnvA_v2: 30×30 Four-Corridor Grid
# ─────────────────────────────────────────────
#
# Design rationale (Clean Phase 1):
#   4 fully isolated corridor route families maximise exclusive SA coverage.
#   Wide (6 controllers across 4 families) vs narrow (1 family) yields
#   ~300 exclusive SA pairs of gap at 10k transitions — well above 0.10.
#
# Layout:
#   Outer walls: row 0, row 29, col 0, col 29.
#   Top stem:    rows 1-2, cols 1-28 (open, shared by all routes).
#   Branch wall: row 3, cols 1-28 minus 4 doorways.
#   4 corridors: rows 4-25 separated by dividers at cols 7, 14, 21.
#     Corridor A: cols 1-6   (22×6 = 132 cells)
#     Corridor B: cols 8-13  (22×6 = 132 cells)
#     Corridor C: cols 15-20 (22×6 = 132 cells)
#     Corridor D: cols 22-28 (22×7 = 154 cells)
#   Merge wall:  row 26, cols 1-28 minus 4 doorways.
#   Bottom stem: rows 27-28, cols 1-28 (open, shared by all routes).
#
#   Start: (1, 14)   Goal: (28, 14)
#
#   Doorways (branch): (3,3), (3,10), (3,18), (3,25)
#   Doorways (merge):  (26,3), (26,10), (26,18), (26,25)
#
_GRID_V2 = 30
_DOORWAYS_BRANCH_V2 = frozenset({(3, 3), (3, 10), (3, 18), (3, 25)})
_DOORWAYS_MERGE_V2  = frozenset({(26, 3), (26, 10), (26, 18), (26, 25)})
DOORWAYS_V2 = _DOORWAYS_BRANCH_V2 | _DOORWAYS_MERGE_V2


def _build_walls_v2():
    walls = set()
    G = _GRID_V2
    # outer walls
    for r in range(G):
        walls.add((r, 0))
        walls.add((r, G - 1))
    for c in range(G):
        walls.add((0, c))
        walls.add((G - 1, c))
    # branch wall at row 3
    for c in range(1, G - 1):
        if (3, c) not in _DOORWAYS_BRANCH_V2:
            walls.add((3, c))
    # merge wall at row 26
    for c in range(1, G - 1):
        if (26, c) not in _DOORWAYS_MERGE_V2:
            walls.add((26, c))
    # corridor dividers: cols 7, 14, 21 for rows 3-26 (inclusive)
    for r in range(3, 27):
        walls.add((r, 7))
        walls.add((r, 14))
        walls.add((r, 21))
    return frozenset(walls)


_WALLS_V2 = _build_walls_v2()
_REACHABLE_V2 = _bfs_reachable((1, 14), _WALLS_V2, grid_size=_GRID_V2)


class EnvA_v2(_GridBase):
    """
    30×30 Four-Corridor gridworld (Clean Phase 1 candidate main environment).

    4 fully isolated corridors (A-D) connected to shared top/bottom stems
    via 8 doorways.  Each corridor is a route family whose cells are
    exclusively visited only by controllers assigned to that family.

    Start: (1,14), Goal: (28,14).
    Reachable states: ~670.  SA pairs: ~2680.
    """

    start_pos = (1, 14)
    goal_pos  = (28, 14)
    grid_size = _GRID_V2
    doorway_positions = DOORWAYS_V2

    def __init__(self):
        super().__init__()
        self._walls = _WALLS_V2
        self._start = self.start_pos
        self._goal  = self.goal_pos

    def _state_from_pos(self, pos):
        return pos

    def _pos_from_state(self, state):
        return state

    def get_reachable_states(self):
        return _REACHABLE_V2

    def transition(self, state, action):
        pos = self._pos_from_state(state)
        next_pos = self._try_move(pos, action)
        return self._state_from_pos(next_pos)
