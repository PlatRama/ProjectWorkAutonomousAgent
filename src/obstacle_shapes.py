from __future__ import annotations

import math
from typing import List, Tuple, Optional

import numpy as np


def _ray_segment_dist(
    ox: float, oy: float, dx: float, dy: float,
    ax: float, ay: float, bx: float, by: float,
) -> float:
    """
    Distance from ray (O + t*D) to segment AB = A + u*(B-A).
    t = distance along the ray (t > 0 required)
    u = position on the segment (valid if 0 <= u <= 1)
    """
    edx = bx - ax
    edy = by - ay
    denom = dx * edy - dy * edx
    if abs(denom) < 1e-10:
        return math.inf   # parallel

    # Vector A-O (not O-A): necessary for correct t>0 orientation
    fx = ax - ox
    fy = ay - oy
    t = (fx * edy - fy * edx) / denom   # distance along the ray (t>0 = forward)
    if t < 1e-9:
        return math.inf   # intersection behind the origin

    u = (fx * dy - fy * dx) / denom     # position on the segment [0,1]
    if u < -1e-9 or u > 1.0 + 1e-9:
        return math.inf   # outside the segment

    return t

def _point_in_convex_polygon(
    px: float, py: float, verts: List[Tuple[float, float]]
) -> bool:
    """Point-in-convex-polygon test (cross-product sign test)."""
    n = len(verts)
    sign = None
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        cross   = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        s       = 1 if cross >= 0 else -1
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


def _polygon_ray_distance(
    ox: float, oy: float, dx: float, dy: float,
    verts: List[Tuple[float, float]],
) -> float:
    t_min = math.inf
    n     = len(verts)
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        t = _ray_segment_dist(ox, oy, dx, dy, ax, ay, bx, by)
        if t < t_min:
            t_min = t
    return t_min


def _polygon_contains_agent(
    px: float, py: float, agent_r: float,
    verts: List[Tuple[float, float]],
) -> bool:
    """
    Collision of agent (circle) with convex polygon:
      1. center inside the polygon
      2. or distance from center to any side < agent_r
    """
    if _point_in_convex_polygon(px, py, verts):
        return True
    n = len(verts)
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        # Point-to-segment distance
        ex, ey = bx - ax, by - ay
        t = ((px - ax) * ex + (py - ay) * ey) / (ex * ex + ey * ey + 1e-12)
        t = max(0.0, min(1.0, t))
        cx = ax + t * ex - px
        cy = ay + t * ey - py
        if cx * cx + cy * cy < agent_r * agent_r:
            return True
    return False


# ── Axis-aligned rectangle ──────────────────────────────────────────────────

class RectObstacle:
    """Axis-aligned rectangle defined by (x, y, w, h) of the bounding box."""

    kind = "rect"

    def __init__(self, x: float, y: float, w: float, h: float):
        self.x, self.y, self.w, self.h = x, y, w, h
        # Counter-clockwise vertices
        self._verts: List[Tuple[float, float]] = [
            (x,     y),
            (x + w, y),
            (x + w, y + h),
            (x,     y + h),
        ]
        # Bounding-box centroid (for compatibility with random spawning)
        self.cx = x + w / 2
        self.cy = y + h / 2
        self.r  = math.sqrt(w * w + h * h) / 2   # radius of the circumscribed circle

    def ray_distance(self, ox, oy, dx, dy) -> float:
        return _polygon_ray_distance(ox, oy, dx, dy, self._verts)

    def contains_point(self, px, py, agent_r) -> bool:
        return _polygon_contains_agent(px, py, agent_r, self._verts)

    def draw(self, screen, pygame) -> None:
        pygame.draw.rect(screen, (180, 80, 60),
                         (int(self.x), int(self.y), int(self.w), int(self.h)))


# ── Rotated rectangle ────────────────────────────────────────────────────────

class RotatedRectObstacle:
    """Rectangle rotated by *angle* radians around its center."""

    kind = "rotated_rect"

    def __init__(self, cx: float, cy: float, w: float, h: float, angle: float):
        self.cx, self.cy = cx, cy
        self.w,  self.h  = w, h
        self.angle       = angle
        self.r           = math.sqrt(w * w + h * h) / 2

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        hw, hh = w / 2, h / 2

        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        self._verts = [
            (cx + cos_a * lx - sin_a * ly,
             cy + sin_a * lx + cos_a * ly)
            for lx, ly in corners
        ]

    def ray_distance(self, ox, oy, dx, dy) -> float:
        return _polygon_ray_distance(ox, oy, dx, dy, self._verts)

    def contains_point(self, px, py, agent_r) -> bool:
        return _polygon_contains_agent(px, py, agent_r, self._verts)

    def draw(self, screen, pygame) -> None:
        pts = [(int(x), int(y)) for x, y in self._verts]
        pygame.draw.polygon(screen, (180, 80, 60), pts)


# ── Triangle ─────────────────────────────────────────────────────────────────

class TriangleObstacle:
    """Convex triangle defined by three vertices."""

    kind = "triangle"

    def __init__(self, p1: Tuple[float, float], p2: Tuple[float, float],
                 p3: Tuple[float, float]):
        self._verts = [p1, p2, p3]
        xs = [p1[0], p2[0], p3[0]]
        ys = [p1[1], p2[1], p3[1]]
        self.cx = sum(xs) / 3
        self.cy = sum(ys) / 3
        self.r  = max(
            math.hypot(x - self.cx, y - self.cy)
            for x, y in self._verts
        )

    def ray_distance(self, ox, oy, dx, dy) -> float:
        return _polygon_ray_distance(ox, oy, dx, dy, self._verts)

    def contains_point(self, px, py, agent_r) -> bool:
        return _polygon_contains_agent(px, py, agent_r, self._verts)

    def draw(self, screen, pygame) -> None:
        pts = [(int(x), int(y)) for x, y in self._verts]
        pygame.draw.polygon(screen, (160, 100, 60), pts)


# ── L-shape ───────────────────────────────────────────────────────────────────

class LShapeObstacle:
    """
    L-shape composed of two axis-aligned rectangles (R1 horizontal, R2 vertical).
    Geometry:
              ┌────────┐
              │   R2   │
         ┌────┤        │
         │ R1 │        │
         └────┴────────┘
    """

    kind = "l_shape"

    def __init__(self, x: float, y: float,
                 long_w: float, long_h: float,
                 arm_w: float,  arm_h: float):
        """
        (x, y) top-left corner of the entire shape.
        R1: width arm_w, height long_h (left arm).
        R2: width long_w, height arm_h (top arm).
        """
        self._r1 = RectObstacle(x,          y + arm_h, arm_w,  long_h - arm_h)
        self._r2 = RectObstacle(x,          y,         long_w, arm_h)
        self.cx  = x + long_w / 2
        self.cy  = y + long_h / 2
        self.r   = math.hypot(long_w, long_h) / 2

    def ray_distance(self, ox, oy, dx, dy) -> float:
        return min(
            self._r1.ray_distance(ox, oy, dx, dy),
            self._r2.ray_distance(ox, oy, dx, dy),
        )

    def contains_point(self, px, py, agent_r) -> bool:
        return (
            self._r1.contains_point(px, py, agent_r) or
            self._r2.contains_point(px, py, agent_r)
        )

    def draw(self, screen, pygame) -> None:
        self._r1.draw(screen, pygame)
        self._r2.draw(screen, pygame)


# ── Factory ───────────────────────────────────────────────────────────────────

def make_random_obstacles(
    shape_type: str,
    n:          int,
    canvas:     int = 600,
    seed:       Optional[int] = None,
) -> list:
    if seed is not None:
        np.random.seed(seed)
        import random as rnd
        rnd.seed(seed)

    rng = np.random.default_rng(seed)
    obstacles = []
    margin = 60

    for _ in range(n):
        if shape_type == "rect":
            w  = float(rng.uniform(40, 80))
            h  = float(rng.uniform(40, 80))
            x  = float(rng.uniform(margin, canvas - margin - w))
            y  = float(rng.uniform(margin, canvas - margin - h))
            obstacles.append(RectObstacle(x, y, w, h))

        elif shape_type == "rotated_rect":
            w     = float(rng.uniform(40, 80))
            h     = float(rng.uniform(30, 60))
            angle = float(rng.uniform(0, math.pi))
            cx    = float(rng.uniform(margin, canvas - margin))
            cy    = float(rng.uniform(margin, canvas - margin))
            obstacles.append(RotatedRectObstacle(cx, cy, w, h, angle))

        elif shape_type == "triangle":
            cx   = float(rng.uniform(margin, canvas - margin))
            cy   = float(rng.uniform(margin, canvas - margin))
            size = float(rng.uniform(30, 60))
            # Randomly rotated equilateral triangle
            base_angle = float(rng.uniform(0, 2 * math.pi))
            verts = [
                (cx + size * math.cos(base_angle + i * 2 * math.pi / 3),
                 cy + size * math.sin(base_angle + i * 2 * math.pi / 3))
                for i in range(3)
            ]
            obstacles.append(TriangleObstacle(*verts))

        elif shape_type == "l_shape":
            long_w = float(rng.uniform(60, 100))
            long_h = float(rng.uniform(60, 100))
            arm_w  = float(rng.uniform(20, long_w * 0.5))
            arm_h  = float(rng.uniform(20, long_h * 0.5))
            x      = float(rng.uniform(margin, canvas - margin - long_w))
            y      = float(rng.uniform(margin, canvas - margin - long_h))
            obstacles.append(LShapeObstacle(x, y, long_w, long_h, arm_w, arm_h))

        else:
            raise ValueError(f"Unknown shape: '{shape_type}'")

    return obstacles