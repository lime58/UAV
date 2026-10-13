from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
import time

import numpy as np

from .environment import Environment, Point


@dataclass
class PlanResult:
    algorithm: str
    success: bool
    path: list[Point]
    planning_time: float
    expanded_nodes: int
    tree_edges: list[tuple[Point, Point]]
    message: str = ""


def _reconstruct_grid_path(
    came_from: dict[tuple[int, int, int], tuple[int, int, int]],
    current: tuple[int, int, int],
    origin: np.ndarray,
    resolution: float,
) -> list[Point]:
    indices = [current]
    while current in came_from:
        current = came_from[current]
        indices.append(current)
    indices.reverse()
    return [origin + resolution * np.asarray(index, dtype=float) for index in indices]


def astar_3d(environment: Environment, resolution: float = 2.0) -> PlanResult:
    started = time.perf_counter()
    origin = np.asarray([axis[0] for axis in environment.bounds], dtype=float)
    shape = tuple(int(math.floor((high - low) / resolution)) + 1 for low, high in environment.bounds)

    def point_to_index(point: Point) -> tuple[int, int, int]:
        raw = np.rint((point - origin) / resolution).astype(int)
        return tuple(int(np.clip(raw[i], 0, shape[i] - 1)) for i in range(3))

    def index_to_point(index: tuple[int, int, int]) -> Point:
        return origin + resolution * np.asarray(index, dtype=float)

    start = point_to_index(environment.start)
    goal = point_to_index(environment.goal)
    moves = [m for m in itertools.product((-1, 0, 1), repeat=3) if m != (0, 0, 0)]
    queue: list[tuple[float, int, tuple[int, int, int]]] = []
    counter = itertools.count()
    heapq.heappush(queue, (0.0, next(counter), start))
    g_score = {start: 0.0}
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    closed: set[tuple[int, int, int]] = set()

    while queue:
        _, _, current = heapq.heappop(queue)
        if current in closed:
            continue
        closed.add(current)
        if current == goal:
            path = _reconstruct_grid_path(came_from, current, origin, resolution)
            path[0] = environment.start.copy()
            path[-1] = environment.goal.copy()
            return PlanResult("A*", True, path, time.perf_counter() - started, len(closed), [])
        current_point = index_to_point(current)
        for move in moves:
            neighbor = tuple(current[i] + move[i] for i in range(3))
            if any(neighbor[i] < 0 or neighbor[i] >= shape[i] for i in range(3)):
                continue
            neighbor_point = index_to_point(neighbor)
            if not environment.point_is_free(neighbor_point):
                continue
            if not environment.segment_is_free(current_point, neighbor_point):
                continue
            tentative = g_score[current] + float(np.linalg.norm(neighbor_point - current_point))
            if tentative >= g_score.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            heuristic = float(np.linalg.norm(environment.goal - neighbor_point))
            heapq.heappush(queue, (tentative + heuristic, next(counter), neighbor))
    return PlanResult("A*", False, [], time.perf_counter() - started, len(closed), [], "No path found")


@dataclass
class _TreeNode:
    point: Point
    parent: int | None
    cost: float


def _sample(environment: Environment, rng: np.random.Generator, goal_bias: float) -> Point:
    if rng.random() < goal_bias:
        return environment.goal.copy()
    return np.asarray([rng.uniform(low, high) for low, high in environment.bounds])


def _steer(source: Point, target: Point, step_size: float) -> Point:
    vector = target - source
    distance = float(np.linalg.norm(vector))
    if distance <= step_size:
        return target.copy()
    return source + vector / distance * step_size


def _tree_path(nodes: list[_TreeNode], index: int, goal: Point | None = None) -> list[Point]:
    path = [goal.copy()] if goal is not None else []
    while index is not None:
        path.append(nodes[index].point.copy())
        index = nodes[index].parent
    path.reverse()
    return path


def _tree_edges(nodes: list[_TreeNode]) -> list[tuple[Point, Point]]:
    return [(nodes[node.parent].point.copy(), node.point.copy()) for node in nodes if node.parent is not None]


def _shift_descendant_costs(nodes: list[_TreeNode], root: int, delta: float) -> None:
    """Keep descendant costs consistent after an RRT* rewire."""
    frontier = [root]
    while frontier:
        parent = frontier.pop()
        for index, node in enumerate(nodes):
            if node.parent == parent:
                node.cost += delta
                frontier.append(index)


def rrt_3d(
    environment: Environment,
    seed: int = 0,
    max_iterations: int = 3500,
    step_size: float = 5.0,
    goal_bias: float = 0.12,
    goal_tolerance: float = 6.0,
) -> PlanResult:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    nodes = [_TreeNode(environment.start.copy(), None, 0.0)]
    for _ in range(max_iterations):
        sample = _sample(environment, rng, goal_bias)
        points = np.asarray([node.point for node in nodes])
        nearest = int(np.argmin(np.linalg.norm(points - sample, axis=1)))
        new_point = _steer(nodes[nearest].point, sample, step_size)
        if not environment.point_is_free(new_point) or not environment.segment_is_free(nodes[nearest].point, new_point):
            continue
        new_cost = nodes[nearest].cost + float(np.linalg.norm(new_point - nodes[nearest].point))
        nodes.append(_TreeNode(new_point, nearest, new_cost))
        new_index = len(nodes) - 1
        if np.linalg.norm(new_point - environment.goal) <= goal_tolerance and environment.segment_is_free(new_point, environment.goal):
            return PlanResult(
                "RRT", True, _tree_path(nodes, new_index, environment.goal),
                time.perf_counter() - started, len(nodes), _tree_edges(nodes),
            )
    return PlanResult("RRT", False, [], time.perf_counter() - started, len(nodes), _tree_edges(nodes), "Iteration limit reached")


def rrt_star_3d(
    environment: Environment,
    seed: int = 0,
    max_iterations: int = 4500,
    step_size: float = 5.0,
    goal_bias: float = 0.10,
    goal_tolerance: float = 6.0,
    neighbor_radius: float = 12.0,
) -> PlanResult:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    nodes = [_TreeNode(environment.start.copy(), None, 0.0)]
    goal_candidates: list[int] = []
    for _ in range(max_iterations):
        sample = _sample(environment, rng, goal_bias)
        points = np.asarray([node.point for node in nodes])
        nearest = int(np.argmin(np.linalg.norm(points - sample, axis=1)))
        new_point = _steer(nodes[nearest].point, sample, step_size)
        if not environment.point_is_free(new_point) or not environment.segment_is_free(nodes[nearest].point, new_point):
            continue
        distances = np.linalg.norm(points - new_point, axis=1)
        near_indices = np.where(distances <= neighbor_radius)[0].tolist()
        best_parent = nearest
        best_cost = nodes[nearest].cost + float(distances[nearest])
        for index in near_indices:
            cost = nodes[index].cost + float(distances[index])
            if cost < best_cost and environment.segment_is_free(nodes[index].point, new_point):
                best_parent, best_cost = index, cost
        nodes.append(_TreeNode(new_point, best_parent, best_cost))
        new_index = len(nodes) - 1
        for index in near_indices:
            candidate_cost = best_cost + float(np.linalg.norm(nodes[index].point - new_point))
            if candidate_cost < nodes[index].cost and environment.segment_is_free(new_point, nodes[index].point):
                delta = candidate_cost - nodes[index].cost
                nodes[index].parent = new_index
                nodes[index].cost = candidate_cost
                _shift_descendant_costs(nodes, index, delta)
        if np.linalg.norm(new_point - environment.goal) <= goal_tolerance and environment.segment_is_free(new_point, environment.goal):
            goal_candidates.append(new_index)
        if goal_candidates and len(nodes) >= 900:
            break
    if goal_candidates:
        best = min(goal_candidates, key=lambda i: nodes[i].cost + float(np.linalg.norm(nodes[i].point - environment.goal)))
        return PlanResult(
            "RRT*", True, _tree_path(nodes, best, environment.goal),
            time.perf_counter() - started, len(nodes), _tree_edges(nodes),
        )
    return PlanResult("RRT*", False, [], time.perf_counter() - started, len(nodes), _tree_edges(nodes), "Iteration limit reached")

