from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .environment import Environment, Point
from .planners import PlanResult


def path_length(path: list[Point]) -> float:
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:])))


def turning_angle_sum(path: list[Point]) -> float:
    total = 0.0
    for a, b, c in zip(path, path[1:], path[2:]):
        first, second = b - a, c - b
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator < 1e-12:
            continue
        cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
        total += math.degrees(math.acos(cosine))
    return total


def maximum_turn_angle(path: list[Point]) -> float:
    maximum = 0.0
    for a, b, c in zip(path, path[1:], path[2:]):
        first, second = b - a, c - b
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator < 1e-12:
            continue
        cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
        maximum = max(maximum, math.degrees(math.acos(cosine)))
    return maximum


def maximum_climb_angle(path: list[Point]) -> float:
    maximum = 0.0
    for start, end in zip(path, path[1:]):
        delta = end - start
        horizontal = float(np.linalg.norm(delta[:2]))
        if horizontal < 1e-12:
            continue
        maximum = max(maximum, math.degrees(math.atan2(abs(float(delta[2])), horizontal)))
    return maximum


def smooth_path(path: list[Point], environment: Environment) -> list[Point]:
    """Greedy line-of-sight shortcutting while preserving collision safety."""
    if len(path) <= 2:
        return [point.copy() for point in path]
    output = [path[0].copy()]
    anchor = 0
    while anchor < len(path) - 1:
        target = len(path) - 1
        while target > anchor + 1 and not environment.segment_is_free(path[anchor], path[target]):
            target -= 1
        output.append(path[target].copy())
        anchor = target
    return output


def validate_path(path: list[Point], environment: Environment) -> bool:
    return bool(path) and all(environment.point_is_free(p) for p in path) and all(
        environment.segment_is_free(a, b) for a, b in zip(path, path[1:])
    )


@dataclass
class Metrics:
    scenario: str
    algorithm: str
    seed: int
    success: bool
    planning_time_s: float
    path_length_m: float | None
    waypoints: int
    turning_angle_deg: float | None
    expanded_nodes: int
    collision_free: bool

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_metrics(result: PlanResult, environment: Environment, seed: int, smooth: bool = True) -> tuple[Metrics, list[Point]]:
    path = smooth_path(result.path, environment) if result.success and smooth else result.path
    valid = validate_path(path, environment) if result.success else False
    metric = Metrics(
        environment.name,
        result.algorithm,
        seed,
        result.success,
        result.planning_time,
        path_length(path) if result.success else None,
        len(path),
        turning_angle_sum(path) if result.success else None,
        result.expanded_nodes,
        valid,
    )
    return metric, path

