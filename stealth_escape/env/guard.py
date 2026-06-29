import numpy as np
from env.fov import is_in_fov, has_line_of_sight
from config import (
    GUARD_FOV_ANGLE, GUARD_FOV_RANGE,
    GUARD_SPEED, ALERT_DURATION
)

# Guard facing directions (angle in degrees)
DIRECTIONS = {
    "right": 0,
    "down":  90,
    "left":  180,
    "up":    270,
}

DIR_VECTORS = {
    "right": (0,  1),
    "down":  (1,  0),
    "left":  (0, -1),
    "up":    (-1, 0),
}


class GuardState:
    PATROL = "patrol"
    ALERT  = "alert"
    CHASE  = "chase"


class Guard:
    def __init__(self, start_pos, patrol_path, grid_size,
                 fov_range=GUARD_FOV_RANGE, fov_angle=GUARD_FOV_ANGLE):
        """
        Args:
            start_pos    : (row, col) starting position
            patrol_path  : list of (row, col) waypoints to cycle through
            grid_size    : int, size of the NxN grid
            fov_range    : override FOV range (curriculum difficulty)
            fov_angle    : override FOV angle (curriculum difficulty)
        """
        self.pos         = list(start_pos)
        self.patrol_path = patrol_path
        self.grid_size   = grid_size
        self.fov_range   = fov_range
        self.fov_angle   = fov_angle

        self.patrol_idx  = 0
        self.state       = GuardState.PATROL
        self.facing      = DIRECTIONS["right"]   # degrees
        self.alert_timer = 0
        self.last_seen   = None

    # ── Core update ────────────────────────────────────────────────────

    def update(self, grid, agent_pos):
        """
        Called once per environment step.
        Returns True if agent is detected this step.
        """
        detected = self._check_detection(grid, agent_pos)

        if detected:
            self.last_seen   = tuple(agent_pos)
            self.alert_timer = ALERT_DURATION

            if self.state == GuardState.PATROL:
                self.state = GuardState.ALERT

        if self.state == GuardState.PATROL:
            self._patrol_move(grid)

        elif self.state == GuardState.ALERT:
            self._alert_move(grid)
            self.alert_timer -= 1
            if self.alert_timer <= 0:
                self.state     = GuardState.PATROL
                self.last_seen = None

        return detected

    # ── Detection ──────────────────────────────────────────────────────

    def _check_detection(self, grid, agent_pos):
        """Returns True if agent is within FOV cone AND has line of sight."""
        in_fov = is_in_fov(
            guard_pos    = tuple(self.pos),
            guard_facing = self.facing,
            target_pos   = tuple(agent_pos),
            fov_angle    = self.fov_angle,
            fov_range    = self.fov_range,
        )
        if not in_fov:
            return False
        return has_line_of_sight(grid, tuple(self.pos), tuple(agent_pos))

    # ── Movement ───────────────────────────────────────────────────────

    def _patrol_move(self, grid):
        """Move toward next waypoint in patrol path."""
        if not self.patrol_path:
            return

        target = self.patrol_path[self.patrol_idx]
        moved  = self._step_toward(grid, target)

        if moved and tuple(self.pos) == tuple(target):
            self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_path)

    def _alert_move(self, grid):
        """Move toward last seen agent position."""
        if self.last_seen:
            self._step_toward(grid, self.last_seen)

    def _step_toward(self, grid, target):
        """Take one step toward target, updating facing direction."""
        dr = target[0] - self.pos[0]
        dc = target[1] - self.pos[1]

        if dr == 0 and dc == 0:
            return False

        moves = []
        if abs(dr) >= abs(dc):
            moves = [("down" if dr > 0 else "up"), ("right" if dc > 0 else "left")]
        else:
            moves = [("right" if dc > 0 else "left"), ("down" if dr > 0 else "up")]

        for direction in moves:
            vec = DIR_VECTORS[direction]
            nr  = self.pos[0] + vec[0]
            nc  = self.pos[1] + vec[1]
            if self._valid(grid, nr, nc):
                self.pos    = [nr, nc]
                self.facing = DIRECTIONS[direction]
                return True

        return False

    def _valid(self, grid, r, c):
        """Check if cell is within bounds and not a wall."""
        return (
            0 <= r < self.grid_size and
            0 <= c < self.grid_size and
            grid[r][c] != 1
        )

    # ── Observation helper ─────────────────────────────────────────────

    def get_obs_vector(self, grid_size):
        """
        Returns a flat observation vector for this guard:
        [row/N, col/N, facing/360, state_patrol, state_alert]
        """
        n = grid_size
        state_patrol = 1.0 if self.state == GuardState.PATROL else 0.0
        state_alert  = 1.0 if self.state == GuardState.ALERT  else 0.0
        return np.array([
            self.pos[0] / n,
            self.pos[1] / n,
            self.facing / 360.0,
            state_patrol,
            state_alert,
        ], dtype=np.float32)