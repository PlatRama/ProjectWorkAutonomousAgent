from __future__ import annotations
import math
import random
from typing import List, Optional, Tuple

import numpy as np

from config import Config


FIXED_AGENT_POS: Tuple[float, float] = (80.0, 300.0)
FIXED_GOAL_POS:  Tuple[float, float] = (520.0, 300.0)

FIXED_OBSTACLES_DATA: List[Tuple[float, float, float]] = [
    (185.0, 210.0, 28.0),
    (185.0, 390.0, 28.0),
    (295.0, 155.0, 25.0),
    (295.0, 445.0, 25.0),
    (390.0, 215.0, 30.0),
    (390.0, 385.0, 30.0),
    (478.0, 160.0, 26.0),
    (478.0, 440.0, 26.0),
]

_INV_SQRT2 = 1.0 / math.sqrt(2.0)

ACTION_DIRS: List[Tuple[float, float]] = [
    (0.0,         -1.0),       # 0 Up
    (0.0,          1.0),       # 1 Down
    (1.0,          0.0),       # 2 Right
    (-1.0,         0.0),       # 3 Left
    (_INV_SQRT2,  -_INV_SQRT2),# 4 Up-Right
    (_INV_SQRT2,   _INV_SQRT2),# 5 Down-Right
    (-_INV_SQRT2,  _INV_SQRT2),# 6 Down-Left
    (-_INV_SQRT2, -_INV_SQRT2),# 7 Up-Left
]


class CircleObstacle:
    def __init__(self, cx: float, cy: float, r: float):
        self.cx = cx
        self.cy = cy
        self.r  = r

    def ray_distance(self, ox: float, oy: float, dx: float, dy: float) -> float:
        """
        Distance from ray origin (ox, oy) in unit direction (dx, dy)
        to the *surface* of this circle.  Returns math.inf if no hit.
        """
        fx   = ox - self.cx
        fy   = oy - self.cy
        b    = 2.0 * (fx * dx + fy * dy)
        c    = fx * fx + fy * fy - self.r * self.r
        disc = b * b - 4.0 * c            # a == 1 because |D| == 1
        if disc < 0.0:
            return math.inf
        sq = math.sqrt(disc)
        t1 = (-b - sq) * 0.5
        t2 = (-b + sq) * 0.5
        eps = 1e-6
        if t1 > eps:
            return t1
        if t2 > eps:
            return t2
        return math.inf

    def contains_point(self, px: float, py: float, agent_radius: float) -> bool:
        """True when agent circle (centre px,py, radius agent_radius) overlaps this obstacle."""
        dx = px - self.cx
        dy = py - self.cy
        return dx * dx + dy * dy < (self.r + agent_radius) ** 2

    # Convenience helpers used during random spawn
    def dist_to(self, cx: float, cy: float) -> float:
        dx = cx - self.cx
        dy = cy - self.cy
        return math.sqrt(dx * dx + dy * dy)


# ── Main environment ──────────────────────────────────────────────────────────

class DQNNavEnv:
    """
    Continuous 2-D navigation environment.

    Observation vector (length = config.STATE_DIM):
        [sin θ, cos θ, dist_norm,       ← goal direction & normalised distance
         r0 … r_{N-1},                  ← LiDAR readings ∈ [0,1]
         a0 … a7]                       ← last action one-hot

    Episode terminates on:
        • Goal reached   (+REWARD_GOAL, done=True)
        • Collision      (+REWARD_COLLISION, done=True)
        • Timeout        (after MAX_STEPS steps, done=True)
    """

    def __init__(self, config: Config, render: bool = False):
        self.cfg    = config
        self.render_mode = render

        # Precompute action deltas (scaled by STEP_SIZE)
        self._deltas: List[Tuple[float, float]] = [
            (ACTION_DIRS[i][0] * config.STEP_SIZE,
             ACTION_DIRS[i][1] * config.STEP_SIZE)
            for i in range(config.N_ACTIONS)
        ]

        # Maximum possible distance in the environment (canvas diagonal)
        self._max_dist = math.sqrt(2.0) * config.CANVAS_SIZE

        # Runtime state — initialised by reset()
        self.agent_x: float       = 0.0
        self.agent_y: float       = 0.0
        self.goal_x:  float       = 0.0
        self.goal_y:  float       = 0.0
        self.obstacles: List      = []
        self.last_action: Optional[int] = None
        self.steps: int           = 0
        self._prev_dist: float    = 0.0

        # Pygame (optional)
        self._screen = None
        self._clock  = None
        if render:
            self._init_pygame()

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        spawn_mode: str = "random",
        n_obstacles: Optional[int] = None,
        obstacles: Optional[List] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Reset the environment and return the initial observation.

        Parameters
        ----------
        spawn_mode : "random" | "fixed"
        n_obstacles : override config.N_OBSTACLES for this episode
        obstacles   : pre-built obstacle objects (used by evaluate_shapes.py)
        seed        : optional RNG seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.steps       = 0
        self.last_action = None
        n_obs = n_obstacles if n_obstacles is not None else self.cfg.N_OBSTACLES

        if obstacles is not None:
            # External obstacles provided (e.g. non-circular shapes)
            self.obstacles = obstacles
            if spawn_mode == "fixed":
                self.agent_x, self.agent_y = FIXED_AGENT_POS
                self.goal_x,  self.goal_y  = FIXED_GOAL_POS
            else:
                self._spawn_agent_goal_random()
        elif spawn_mode == "fixed":
            self._spawn_fixed(n_obs)
        elif spawn_mode == "agent_random":
            self._spawn_agent_random(n_obs)
        else:
            self._spawn_random(n_obs)

        self._prev_dist = self._dist_to_goal()
        return self._compute_state()

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply action and return (next_obs, reward, done, info).

        Wall sliding is applied for diagonal actions:
          full move → try X-only → try Y-only → collision (terminal).
        """
        self.last_action = action
        prev_dist = self._dist_to_goal()

        dx, dy     = self._deltas[action]
        is_diagonal = action >= 4

        result = self._try_move(dx, dy, is_diagonal)

        if result == "collision":
            return (
                self._compute_state(),
                self.cfg.REWARD_COLLISION,
                True,
                {"reason": "collision"},
            )

        curr_dist = self._dist_to_goal()

        # Goal check
        if curr_dist <= self.cfg.GOAL_SUCCESS_RADIUS:
            return (
                self._compute_state(),
                self.cfg.REWARD_GOAL,
                True,
                {"reason": "goal"},
            )

        # Reward
        reward = self.cfg.REWARD_STEP
        if curr_dist < self.cfg.PROXIMITY_ZONE:
            delta   = prev_dist - curr_dist
            reward += self.cfg.PROXIMITY_SCALE * delta / self._max_dist

        self.steps += 1
        done = self.steps >= self.cfg.MAX_STEPS

        return (
            self._compute_state(),
            reward,
            done,
            {"reason": "timeout" if done else "step"},
        )

    # ── Movement & collision ───────────────────────────────────────────────────

    def _try_move(self, dx: float, dy: float, is_diagonal: bool) -> str:
        """
        Attempt to move by (dx, dy).  Wall-clips position; returns:
          "ok"        — move succeeded
          "slide_x"   — moved along X only (wall-slide)
          "slide_y"   — moved along Y only (wall-slide)
          "collision" — all attempts failed → terminal collision
        """
        cfg = self.cfg
        lo  = float(cfg.AGENT_RADIUS)
        hi  = float(cfg.CANVAS_SIZE - cfg.AGENT_RADIUS)

        # Full move (with wall clipping)
        nx = float(np.clip(self.agent_x + dx, lo, hi))
        ny = float(np.clip(self.agent_y + dy, lo, hi))

        if not self._obstacle_collision(nx, ny):
            self.agent_x, self.agent_y = nx, ny
            return "ok"

        if is_diagonal:
            # Slide along X
            nx1 = float(np.clip(self.agent_x + dx, lo, hi))
            ny1 = self.agent_y
            if not self._obstacle_collision(nx1, ny1):
                self.agent_x, self.agent_y = nx1, ny1
                return "slide_x"

            # Slide along Y
            nx2 = self.agent_x
            ny2 = float(np.clip(self.agent_y + dy, lo, hi))
            if not self._obstacle_collision(nx2, ny2):
                self.agent_x, self.agent_y = nx2, ny2
                return "slide_y"

        return "collision"

    def _obstacle_collision(self, px: float, py: float) -> bool:
        """True if agent at (px, py) overlaps any obstacle."""
        r = self.cfg.AGENT_RADIUS
        for obs in self.obstacles:
            if obs.contains_point(px, py, r):
                return True
        return False

    def _dist_to_goal(self) -> float:
        dx = self.agent_x - self.goal_x
        dy = self.agent_y - self.goal_y
        return math.sqrt(dx * dx + dy * dy)

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _spawn_fixed(self, n_obs: int) -> None:
        self.agent_x, self.agent_y = FIXED_AGENT_POS
        self.goal_x,  self.goal_y  = FIXED_GOAL_POS
        self.obstacles = [
            CircleObstacle(cx, cy, r)
            for cx, cy, r in FIXED_OBSTACLES_DATA[:n_obs]
        ]

    def _spawn_agent_random(self, n_obs: int) -> None:
        cfg = self.cfg
        ar  = cfg.AGENT_RADIUS

        self.goal_x, self.goal_y = FIXED_GOAL_POS
        self.obstacles = [
            CircleObstacle(cx, cy, r)
            for cx, cy, r in FIXED_OBSTACLES_DATA[:n_obs]
        ]

        margin      = ar + 5.0
        lo, hi      = margin, cfg.CANVAS_SIZE - margin
        min_ag_dist = max(cfg.CANVAS_SIZE * 0.25, 120.0)

        for _ in range(2000):
            ax = random.uniform(lo, hi)
            ay = random.uniform(lo, hi)
            dist = math.sqrt(
                (ax - self.goal_x) ** 2 + (ay - self.goal_y) ** 2
            )
            if dist >= min_ag_dist and not self._obstacle_collision(ax, ay):
                self.agent_x, self.agent_y = ax, ay
                return

        self.agent_x, self.agent_y = FIXED_AGENT_POS

    def _spawn_random(self, n_obs: int) -> None:
        cfg          = self.cfg
        ar           = cfg.AGENT_RADIUS
        margin       = ar + 5.0
        lo           = margin
        hi           = cfg.CANVAS_SIZE - margin
        goal_clear   = cfg.GOAL_SUCCESS_RADIUS + cfg.OBSTACLE_CLEARANCE
        min_ag_dist  = max(cfg.CANVAS_SIZE * 0.30, 150.0)

        # 1. Goal
        for _ in range(2000):
            gx = random.uniform(lo + cfg.GOAL_RADIUS, hi - cfg.GOAL_RADIUS)
            gy = random.uniform(lo + cfg.GOAL_RADIUS, hi - cfg.GOAL_RADIUS)
            self.goal_x, self.goal_y = gx, gy
            break  # validated against obstacles below

        # 2. Agent (far enough from goal)
        for _ in range(2000):
            ax = random.uniform(lo, hi)
            ay = random.uniform(lo, hi)
            dist = math.sqrt((ax - self.goal_x) ** 2 + (ay - self.goal_y) ** 2)
            if dist >= min_ag_dist:
                self.agent_x, self.agent_y = ax, ay
                break

        # 3. Obstacles (rejection sampling)
        self.obstacles = []
        for _ in range(n_obs):
            r = random.uniform(cfg.OBS_RADIUS_MIN, cfg.OBS_RADIUS_MAX)
            placed = False
            for _ in range(500):
                cx = random.uniform(r + margin, cfg.CANVAS_SIZE - r - margin)
                cy = random.uniform(r + margin, cfg.CANVAS_SIZE - r - margin)

                # Clearance from goal
                dg = math.sqrt((cx - self.goal_x) ** 2 + (cy - self.goal_y) ** 2)
                if dg < r + goal_clear:
                    continue

                # Clearance from agent start
                da = math.sqrt((cx - self.agent_x) ** 2 + (cy - self.agent_y) ** 2)
                if da < r + ar + 15.0:
                    continue

                # No overlap with existing obstacles
                ok = all(
                    math.sqrt((cx - o.cx) ** 2 + (cy - o.cy) ** 2) >= r + o.r + 5.0
                    for o in self.obstacles
                    if hasattr(o, "cx")      # circles only
                )
                if ok:
                    self.obstacles.append(CircleObstacle(cx, cy, r))
                    placed = True
                    break

            # If placement failed after 500 attempts, skip this obstacle
            # (avoids infinite loops with very crowded configurations)

    def _spawn_agent_goal_random(self) -> None:
        """Random agent/goal placement when obstacles are externally supplied."""
        cfg         = self.cfg
        ar          = cfg.AGENT_RADIUS
        margin      = ar + 5.0
        lo, hi      = margin, cfg.CANVAS_SIZE - margin
        min_ag_dist = max(cfg.CANVAS_SIZE * 0.30, 150.0)

        for _ in range(1000):
            gx = random.uniform(lo, hi)
            gy = random.uniform(lo, hi)
            self.goal_x, self.goal_y = gx, gy
            if not self._obstacle_collision(gx, gy):
                break

        for _ in range(1000):
            ax = random.uniform(lo, hi)
            ay = random.uniform(lo, hi)
            dist = math.sqrt((ax - self.goal_x) ** 2 + (ay - self.goal_y) ** 2)
            if not self._obstacle_collision(ax, ay) and dist >= min_ag_dist:
                self.agent_x, self.agent_y = ax, ay
                break

    # ── State computation ─────────────────────────────────────────────────────

    def _compute_state(self) -> np.ndarray:
        cfg = self.cfg
        dx  = self.goal_x - self.agent_x
        dy  = self.goal_y - self.agent_y
        dist  = math.sqrt(dx * dx + dy * dy)
        angle = math.atan2(dy, dx)

        sin_theta  = math.sin(angle)
        cos_theta  = math.cos(angle)
        dist_norm  = min(dist / self._max_dist, 1.0)

        lidar = self._cast_lidar()

        action_oh = np.zeros(cfg.N_ACTIONS, dtype=np.float32)
        if self.last_action is not None:
            action_oh[self.last_action] = 1.0

        state = np.concatenate(
            [[sin_theta, cos_theta, dist_norm], lidar, action_oh]
        ).astype(np.float32)

        return state

    # ── LiDAR ─────────────────────────────────────────────────────────────────

    def _cast_lidar(self) -> np.ndarray:
        """
        Cast N_RAYS evenly-spaced rays from agent position.
        Returns normalised distances in [0, 1] (0 = obstacle right here, 1 = max range).
        Includes optional Gaussian noise (cfg.LIDAR_NOISE_STD).
        """
        cfg      = self.cfg
        readings = np.empty(cfg.N_RAYS, dtype=np.float32)
        ox, oy   = self.agent_x, self.agent_y
        max_d    = float(cfg.RAY_MAX_DIST)

        for i in range(cfg.N_RAYS):
            angle = 2.0 * math.pi * i / cfg.N_RAYS
            dx    = math.cos(angle)
            dy    = math.sin(angle)

            t_min = min(self._ray_wall_dist(ox, oy, dx, dy), max_d)

            for obs in self.obstacles:
                t = obs.ray_distance(ox, oy, dx, dy)
                if t < t_min:
                    t_min = t

            t_norm = min(t_min, max_d) / max_d

            if cfg.LIDAR_NOISE_STD > 0.0:
                t_norm += float(np.random.normal(0.0, cfg.LIDAR_NOISE_STD))
                t_norm  = float(np.clip(t_norm, 0.0, 1.0))

            readings[i] = t_norm

        return readings

    def _ray_wall_dist(
        self, ox: float, oy: float, dx: float, dy: float
    ) -> float:
        """Distance from (ox, oy) in direction (dx, dy) to the nearest canvas wall."""
        W   = float(self.cfg.CANVAS_SIZE)
        eps = 1e-9
        t_min = math.inf

        if dx < -eps:
            t = ox / (-dx)
            if t > 1e-6:
                t_min = min(t_min, t)
        elif dx > eps:
            t = (W - ox) / dx
            if t > 1e-6:
                t_min = min(t_min, t)

        if dy < -eps:
            t = oy / (-dy)
            if t > 1e-6:
                t_min = min(t_min, t)
        elif dy > eps:
            t = (W - oy) / dy
            if t > 1e-6:
                t_min = min(t_min, t)

        return t_min

    # ── Rendering (pygame) ────────────────────────────────────────────────────

    def _init_pygame(self) -> None:
        import pygame
        pygame.init()
        self._pygame   = pygame
        self._screen   = pygame.display.set_mode(
            (self.cfg.CANVAS_SIZE, self.cfg.CANVAS_SIZE)
        )
        pygame.display.set_caption("DQN Navigation")
        self._clock    = pygame.time.Clock()
        self._font     = pygame.font.SysFont("monospace", 14)

    def render(self, fps: int = 30) -> bool:
        """
        Draw current state.  Returns False if the user closed the window.
        Only works when render=True was passed to __init__.
        """
        if self._screen is None:
            return True

        pg = self._pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False

        W   = self.cfg.CANVAS_SIZE
        scr = self._screen

        # Background
        scr.fill((30, 30, 35))

        # Obstacles
        for obs in self.obstacles:
            if hasattr(obs, "cx"):          # CircleObstacle
                pg.draw.circle(scr, (180, 80, 60),
                               (int(obs.cx), int(obs.cy)), int(obs.r))
            else:
                obs.draw(scr, pg)           # shape objects implement draw()

        # LiDAR rays (dim)
        cfg = self.cfg
        ox, oy = int(self.agent_x), int(self.agent_y)
        for i in range(cfg.N_RAYS):
            angle  = 2.0 * math.pi * i / cfg.N_RAYS
            dx     = math.cos(angle)
            dy     = math.sin(angle)
            t_min  = min(self._ray_wall_dist(self.agent_x, self.agent_y, dx, dy),
                         cfg.RAY_MAX_DIST)
            for obs in self.obstacles:
                t = obs.ray_distance(self.agent_x, self.agent_y, dx, dy)
                t_min = min(t_min, t)
            t_min = min(t_min, cfg.RAY_MAX_DIST)
            ex = int(ox + dx * t_min)
            ey = int(oy + dy * t_min)
            pg.draw.line(scr, (60, 100, 140), (ox, oy), (ex, ey), 1)
            pg.draw.circle(scr, (120, 180, 220), (ex, ey), 3)

        # Goal
        pg.draw.circle(scr, (50, 200, 80),
                       (int(self.goal_x), int(self.goal_y)),
                       cfg.GOAL_RADIUS, 2)
        pg.draw.circle(scr, (100, 240, 100),
                       (int(self.goal_x), int(self.goal_y)), 5)

        # Success zone
        pg.draw.circle(scr, (50, 200, 80),
                       (int(self.goal_x), int(self.goal_y)),
                       cfg.GOAL_SUCCESS_RADIUS, 1)

        # Agent
        pg.draw.circle(scr, (220, 200, 60),
                       (int(self.agent_x), int(self.agent_y)),
                       cfg.AGENT_RADIUS)

        # HUD
        dist_to_g = self._dist_to_goal()
        hud = self._font.render(
            f"Step {self.steps:3d}  dist {dist_to_g:5.1f}px  "
            f"obs {len(self.obstacles)}",
            True, (200, 200, 200),
        )
        scr.blit(hud, (5, 5))

        pg.display.flip()
        self._clock.tick(fps)
        return True

    def close(self) -> None:
        if self._screen is not None:
            self._pygame.quit()
            self._screen = None