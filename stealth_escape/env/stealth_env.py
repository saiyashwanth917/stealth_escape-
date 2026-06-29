"""
stealth_env.py  —  Stealth Escape RL Environment  (FIXED v3)

Key changes vs original:
  1. Safe spawn  : agent starts >=6 cells (Manhattan) from every guard
  2. Guards only in bottom-2/3 : never overlap agent top-left quadrant
  3. Richer obs  : relative guard vectors + per-guard FOV flag + exit direction
  4. Strong caught penalty : -15  (was -1)
  5. Anti-idle   : escalating penalty when agent stays in the same cell
  6. In-FOV step penalty : -0.8 per guard cone the agent is inside
  7. Proximity penalty   : graded penalty when agent is <=2 cells from a guard
  8. Progress shaping    : delta-distance reward (continuous pull toward exit)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.guard import Guard
from env.fov import is_in_fov
from config import (
    GRID_SIZE, NUM_GUARDS, MAX_STEPS, NUM_WALLS,
    GUARD_FOV_ANGLE, GUARD_FOV_RANGE,
    REWARD_REACH_EXIT, REWARD_STEP,
)

EMPTY = 0
WALL  = 1
AGENT = 2
GUARD = 3
EXIT  = 4

MAX_GUARDS           = 3
MIN_AGENT_GUARD_DIST = 6


class StealthEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, stage=None):
        super().__init__()
        self.render_mode = render_mode
        self._stage      = stage
        self._apply_stage(stage)

        self.action_space = spaces.Discrete(4)

        # obs = 5 base + 8 rays + 4 exit_dir + MAX_GUARDS*6
        obs_dim = 5 + 8 + 4 + MAX_GUARDS * 6
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._renderer   = None
        self.grid        = None
        self.agent_pos   = None
        self.exit_pos    = None
        self.guards      = []
        self.step_count  = 0
        self._prev_dist  = None
        self._idle_steps = 0

    # ── Stage ──────────────────────────────────────────────────────────

    def _apply_stage(self, stage):
        if stage is not None:
            self.grid_size  = stage.grid_size
            self.num_guards = stage.num_guards
            self.num_walls  = stage.num_walls
            self.fov_range  = stage.fov_range
            self.fov_angle  = stage.fov_angle
            self.max_steps  = stage.max_steps
        else:
            self.grid_size  = GRID_SIZE
            self.num_guards = NUM_GUARDS
            self.num_walls  = NUM_WALLS
            self.fov_range  = GUARD_FOV_RANGE
            self.fov_angle  = GUARD_FOV_ANGLE
            self.max_steps  = MAX_STEPS

    def set_stage(self, stage):
        self._stage = stage
        self._apply_stage(stage)
        if self.grid is not None:
            self._build_grid()
            self._prev_dist  = self._dist_to_exit()
            self.step_count  = 0
            self._idle_steps = 0
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count  = 0
        self._idle_steps = 0
        self._build_grid()
        self._prev_dist = self._dist_to_exit()
        return self._get_obs(), {}

    # ── Grid ───────────────────────────────────────────────────────────

    def _build_grid(self):
        rng = self.np_random
        n   = self.grid_size

        self.grid = np.zeros((n, n), dtype=np.int32)

        placed, attempts = 0, 0
        while placed < self.num_walls and attempts < 10000:
            attempts += 1
            r = rng.integers(1, n - 1)
            c = rng.integers(1, n - 1)
            if self.grid[r][c] == EMPTY:
                self.grid[r][c] = WALL
                placed += 1

        self.agent_pos = self._rand_empty(rng, 0, max(2, n // 3), 0, max(2, n // 3))
        self.exit_pos  = self._rand_empty(rng, 2 * n // 3, n, 2 * n // 3, n)
        self.grid[self.exit_pos[0]][self.exit_pos[1]] = EXIT

        self.guards = []
        taken = [self.agent_pos, self.exit_pos]
        patrol_regions = [
            (n // 3, 2 * n // 3, 0,      n),
            (n // 2, n,          0,      n // 2),
            (n // 2, n,          n // 2, n),
        ]
        for i in range(self.num_guards):
            mr, xr, mc, xc = patrol_regions[i % len(patrol_regions)]
            pos = self._rand_empty_safe(rng, mr, xr, mc, xc, taken, self.agent_pos)
            taken.append(pos)
            self.guards.append(Guard(
                pos, self._make_patrol(pos, n), n,
                fov_range=self.fov_range, fov_angle=self.fov_angle,
            ))

    def _make_patrol(self, center, n, radius=3):
        r, c = center
        return [
            (max(0, r-radius), max(0, c-radius)),
            (max(0, r-radius), min(n-1, c+radius)),
            (min(n-1, r+radius), min(n-1, c+radius)),
            (min(n-1, r+radius), max(0, c-radius)),
        ]

    def _rand_empty(self, rng, r0, r1, c0, c1, exc=None):
        n  = self.grid_size
        r1 = max(r0+1, min(r1, n))
        c1 = max(c0+1, min(c1, n))
        exc = exc or []
        for _ in range(3000):
            r = rng.integers(r0, r1)
            c = rng.integers(c0, c1)
            if self.grid[r][c] == EMPTY and [r,c] not in exc and (r,c) not in exc:
                return [r, c]
        for r in range(r0, r1):
            for c in range(c0, c1):
                if self.grid[r][c] == EMPTY:
                    return [r, c]
        return [r0, c0]

    def _rand_empty_safe(self, rng, r0, r1, c0, c1, exc, avoid):
        n  = self.grid_size
        r1 = max(r0+1, min(r1, n))
        c1 = max(c0+1, min(c1, n))
        for _ in range(5000):
            r = rng.integers(r0, r1)
            c = rng.integers(c0, c1)
            if self.grid[r][c] != EMPTY:
                continue
            if [r,c] in exc or (r,c) in exc:
                continue
            if abs(r - avoid[0]) + abs(c - avoid[1]) < MIN_AGENT_GUARD_DIST:
                continue
            return [r, c]
        return self._rand_empty(rng, r0, r1, c0, c1, exc)

    # ── Step ───────────────────────────────────────────────────────────

    def step(self, action):
        self.step_count += 1
        prev_pos = list(self.agent_pos)

        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = deltas[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        if self._valid_agent(nr, nc):
            self.agent_pos = [nr, nc]

        caught       = any(g.update(self.grid, self.agent_pos) for g in self.guards)
        reached_exit = (self.agent_pos[0] == self.exit_pos[0] and
                        self.agent_pos[1] == self.exit_pos[1])

        # ── Reward ────────────────────────────────────────────────────
        if caught:
            reward = -15.0

        elif reached_exit:
            reward = float(REWARD_REACH_EXIT)

        else:
            reward = REWARD_STEP   # -0.001

            # (a) Continuous progress: delta distance to exit
            curr_dist = self._dist_to_exit()
            reward   += (self._prev_dist - curr_dist) * 0.5
            self._prev_dist = curr_dist

            # (b) Survive bonus
            reward += 0.02

            # (c) Anti-idle: escalating penalty for not moving
            if self.agent_pos == prev_pos:
                self._idle_steps += 1
                reward -= 0.05 * min(self._idle_steps, 10)
            else:
                self._idle_steps = 0

            # (d) In-FOV penalty per guard cone
            fov_count = sum(
                1 for g in self.guards
                if is_in_fov(tuple(g.pos), g.facing, tuple(self.agent_pos),
                             g.fov_angle, g.fov_range)
            )
            if fov_count:
                reward -= 0.8 * fov_count

            # (e) Proximity danger — graded
            if self.guards:
                md = min(abs(g.pos[0]-self.agent_pos[0]) + abs(g.pos[1]-self.agent_pos[1])
                         for g in self.guards)
                if md <= 2:
                    reward -= (3 - md) * 0.3

        terminated = caught or reached_exit
        truncated  = self.step_count >= self.max_steps
        info = {"caught": caught, "reached_exit": reached_exit, "step": self.step_count}

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    # ── Observation ────────────────────────────────────────────────────

    def _get_obs(self):
        n      = self.grid_size
        ar, ac = self.agent_pos
        er, ec = self.exit_pos

        # 5 base features
        base = [ar/n, ac/n, er/n, ec/n, self._dist_to_exit()/(n*1.414)]

        # 8 danger rays
        rays = self._danger_rays().tolist()

        # 4 exit direction features
        ddr = er - ar
        ddc = ec - ac
        d   = max(1.0, np.sqrt(ddr**2 + ddc**2))
        exit_dir = [ddr/d, ddc/d, abs(ddr)/n, abs(ddc)/n]

        # 6 features per guard
        guard_feats = []
        for g in self.guards:
            gr, gc = g.pos
            in_fov = 1.0 if is_in_fov(
                tuple(g.pos), g.facing, tuple(self.agent_pos),
                g.fov_angle, g.fov_range
            ) else 0.0
            guard_feats += [
                (gr - ar) / n,
                (gc - ac) / n,
                g.facing / 360.0,
                1.0 if g.state != "patrol" else 0.0,
                in_fov,
                (abs(gr-ar) + abs(gc-ac)) / (2*n),
            ]
        while len(guard_feats) < MAX_GUARDS * 6:
            guard_feats += [0.0] * 6

        obs = np.array(base + rays + exit_dir + guard_feats, dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _danger_rays(self):
        RAY_RANGE = 6
        dirs = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
        ar, ac = self.agent_pos
        rays   = []
        for dr, dc in dirs:
            hit = 1.0
            for d in range(1, RAY_RANGE + 1):
                r, c = ar + dr*d, ac + dc*d
                if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                    break
                if self.grid[r][c] == WALL:
                    break
                if any(g.pos[0]==r and g.pos[1]==c for g in self.guards):
                    hit = (d-1) / RAY_RANGE
                    break
            rays.append(hit)
        return np.array(rays, dtype=np.float32)

    def _dist_to_exit(self):
        dr = self.agent_pos[0] - self.exit_pos[0]
        dc = self.agent_pos[1] - self.exit_pos[1]
        return np.sqrt(dr**2 + dc**2)

    def _valid_agent(self, r, c):
        return (0 <= r < self.grid_size and 0 <= c < self.grid_size
                and self.grid[r][c] != WALL)

    # ── Render ─────────────────────────────────────────────────────────

    def render(self):
        from env.renderer import Renderer
        if self._renderer is None:
            self._renderer = Renderer(self.grid_size)
        return self._renderer.draw(
            grid=self.grid, agent_pos=self.agent_pos,
            exit_pos=self.exit_pos, guards=self.guards, step=self.step_count,
        )

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None