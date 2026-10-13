from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Point = np.ndarray


@dataclass(frozen=True)
class AABB:
    """Axis-aligned building/no-fly volume in local metric coordinates."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    name: str = "obstacle"

    def expanded(self, margin: float) -> "AABB":
        lo = tuple(v - margin for v in self.minimum)
        hi = tuple(v + margin for v in self.maximum)
        return AABB(lo, hi, self.name)

    def contains(self, point: Point, margin: float = 0.0) -> bool:
        box = self.expanded(margin)
        return bool(
            np.all(point >= np.asarray(box.minimum))
            and np.all(point <= np.asarray(box.maximum))
        )

    def intersects_segment(self, start: Point, end: Point, margin: float = 0.0) -> bool:
        """Slab-test intersection between a line segment and an expanded box."""
        box = self.expanded(margin)
        lo = np.asarray(box.minimum, dtype=float)
        hi = np.asarray(box.maximum, dtype=float)
        direction = end - start
        t_min, t_max = 0.0, 1.0
        for axis in range(3):
            if abs(direction[axis]) < 1e-12:
                if start[axis] < lo[axis] or start[axis] > hi[axis]:
                    return False
                continue
            inv = 1.0 / direction[axis]
            t1 = (lo[axis] - start[axis]) * inv
            t2 = (hi[axis] - start[axis]) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
        return True


@dataclass
class Environment:
    name: str
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    start: Point
    goal: Point
    obstacles: list[AABB]
    safety_margin: float = 1.0

    def __post_init__(self) -> None:
        self.start = np.asarray(self.start, dtype=float)
        self.goal = np.asarray(self.goal, dtype=float)
        if not self.point_is_free(self.start) or not self.point_is_free(self.goal):
            raise ValueError("Start and goal must be inside bounds and outside obstacles")

    def inside_bounds(self, point: Point) -> bool:
        return all(low <= point[i] <= high for i, (low, high) in enumerate(self.bounds))

    def point_is_free(self, point: Point) -> bool:
        return self.inside_bounds(point) and not any(
            obstacle.contains(point, self.safety_margin) for obstacle in self.obstacles
        )

    def segment_is_free(self, start: Point, end: Point) -> bool:
        if not self.inside_bounds(start) or not self.inside_bounds(end):
            return False
        return not any(
            obstacle.intersects_segment(start, end, self.safety_margin)
            for obstacle in self.obstacles
        )

    @classmethod
    def from_values(
        cls,
        name: str,
        bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        start: Iterable[float],
        goal: Iterable[float],
        obstacles: list[AABB],
        safety_margin: float = 1.0,
    ) -> "Environment":
        return cls(name, bounds, np.asarray(start, float), np.asarray(goal, float), obstacles, safety_margin)

