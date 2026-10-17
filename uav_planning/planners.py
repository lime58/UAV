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


@dataclass(frozen=True)
class FlightConstraints:
    maximum_altitude: float
    minimum_enroute_altitude: float
    approach_radius: float
    maximum_climb_angle_deg: float
    maximum_turn_angle_deg: float


def _inside_approach(point: Point, environment: Environment, radius: float) -> bool:
    return min(
        float(np.linalg.norm((point - environment.start)[:2])),
        float(np.linalg.norm((point - environment.goal)[:2])),
    ) <= radius


def transition_is_valid(
    environment: Environment,
    start: Point,
    end: Point,
    previous_direction: Point | None,
    constraints: FlightConstraints,
) -> bool:
    """Validate one flight transition for every constrained planner."""
    if max(float(start[2]), float(end[2])) > constraints.maximum_altitude + 1e-9:
        return False
    # segment_is_free checks both endpoints against the environment bounds and
    # checks the complete segment against safety-margin-expanded buildings.
    if not environment.segment_is_free(start, end):
        return False

    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return False
    horizontal = float(np.linalg.norm(direction[:2]))
    vertical_takeoff_or_landing = (
        horizontal < 1e-9
        and _inside_approach(start, environment, constraints.approach_radius)
        and _inside_approach(end, environment, constraints.approach_radius)
    )
    if horizontal < 1e-9 and not vertical_takeoff_or_landing:
        return False
    if not vertical_takeoff_or_landing:
        climb_angle = math.degrees(math.atan2(abs(float(direction[2])), horizontal))
        if climb_angle > constraints.maximum_climb_angle_deg + 1e-9:
            return False

    # Sample the segment so a long edge cannot leave an approach zone below
    # the minimum route altitude and only climb after crossing its boundary.
    sample_count = max(2, int(math.ceil(length / 5.0)) + 1)
    for fraction in np.linspace(0.0, 1.0, sample_count):
        point = start + fraction * direction
        if not _inside_approach(point, environment, constraints.approach_radius):
            if point[2] < constraints.minimum_enroute_altitude - 1e-9:
                return False

    if previous_direction is not None and not vertical_takeoff_or_landing:
        if _angle_deg(previous_direction, direction) > constraints.maximum_turn_angle_deg + 1e-9:
            return False
    return True


def validate_flight_path(
    path: list[Point], environment: Environment, constraints: FlightConstraints,
) -> bool:
    """Validate a complete constrained route through the shared transition check."""
    if len(path) < 2:
        return False
    previous_direction: Point | None = None
    for start, end in zip(path, path[1:]):
        if not transition_is_valid(environment, start, end, previous_direction, constraints):
            return False
        previous_direction = end - start
    return True


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _overflight_fraction(environment: Environment, start: Point, end: Point) -> float:
    """Conservative building-overflight indicator used by the urban risk cost."""
    midpoint = 0.5 * (start + end)
    for obstacle in environment.obstacles:
        low = np.asarray(obstacle.minimum, dtype=float) - environment.safety_margin
        high = np.asarray(obstacle.maximum, dtype=float) + environment.safety_margin
        if low[0] <= midpoint[0] <= high[0] and low[1] <= midpoint[1] <= high[1] and midpoint[2] > high[2]:
            return 1.0
    return 0.0


def constrained_astar_3d(
    environment: Environment,
    horizontal_resolution: float = 10.0,
    vertical_resolution: float = 2.5,
    maximum_altitude: float = 50.0,
    minimum_enroute_altitude: float = 12.0,
    approach_radius: float = 75.0,
    maximum_climb_angle_deg: float = 15.0,
    maximum_turn_angle_deg: float = 90.0,
    climb_cost_per_m: float = 4.0,
    turn_cost_per_deg: float = 0.08,
    overflight_cost_per_m: float = 12.0,
    heuristic_weight: float = 2.0,
) -> PlanResult:
    """Direction-state A* for a flyable low-altitude route.

    Unlike a geometric 26-neighbour demo, the state includes the incoming
    direction.  This allows the search itself to reject excessive climb and
    heading changes instead of trying to repair an infeasible route afterward.
    Pure vertical moves are deliberately excluded.
    """
    started = time.perf_counter()
    constraints = FlightConstraints(
        maximum_altitude, minimum_enroute_altitude, approach_radius,
        maximum_climb_angle_deg, maximum_turn_angle_deg,
    )
    low = np.asarray([axis[0] for axis in environment.bounds], dtype=float)
    high = np.asarray([axis[1] for axis in environment.bounds], dtype=float)
    low[2] = min(float(environment.start[2]), float(environment.goal[2]))
    high[2] = min(high[2], maximum_altitude)
    step = np.asarray([horizontal_resolution, horizontal_resolution, vertical_resolution], dtype=float)
    shape = tuple(int(math.floor((high[i] - low[i]) / step[i])) + 1 for i in range(3))

    def point_to_index(point: Point) -> tuple[int, int, int]:
        raw = np.rint((point - low) / step).astype(int)
        return tuple(int(np.clip(raw[i], 0, shape[i] - 1)) for i in range(3))

    def index_to_point(index: tuple[int, int, int]) -> Point:
        return low + step * np.asarray(index, dtype=float)

    start_index, goal_index = point_to_index(environment.start), point_to_index(environment.goal)
    moves = [
        move for move in itertools.product((-1, 0, 1), repeat=3)
        if move != (0, 0, 0)
    ]

    State = tuple[tuple[int, int, int], tuple[int, int, int] | None]
    start_state: State = (start_index, None)
    queue: list[tuple[float, int, State]] = []
    counter = itertools.count()
    heapq.heappush(queue, (0.0, next(counter), start_state))
    g_score: dict[State, float] = {start_state: 0.0}
    came_from: dict[State, State] = {}
    search_edges: list[tuple[Point, Point]] = []
    closed: set[State] = set()
    goal_state: State | None = None

    while queue:
        _, _, current_state = heapq.heappop(queue)
        if current_state in closed:
            continue
        closed.add(current_state)
        current, previous_move = current_state
        if current == goal_index:
            goal_state = current_state
            break
        current_point = environment.start.copy() if current_state == start_state else index_to_point(current)
        for move in moves:
            neighbor = tuple(current[i] + move[i] for i in range(3))
            if any(neighbor[i] < 0 or neighbor[i] >= shape[i] for i in range(3)):
                continue
            neighbor_point = environment.goal.copy() if neighbor == goal_index else index_to_point(neighbor)
            previous_direction = None if previous_move is None else step * np.asarray(previous_move, dtype=float)
            if not transition_is_valid(environment, current_point, neighbor_point, previous_direction, constraints):
                continue
            turn_angle = 0.0 if previous_direction is None else _angle_deg(previous_direction, neighbor_point - current_point)
            segment_length = float(np.linalg.norm(neighbor_point - current_point))
            climb = max(0.0, float(neighbor_point[2] - current_point[2]))
            transition = (
                segment_length
                + climb_cost_per_m * climb
                + turn_cost_per_deg * turn_angle
                + overflight_cost_per_m * segment_length * _overflight_fraction(environment, current_point, neighbor_point)
            )
            neighbor_state: State = (neighbor, move)
            tentative = g_score[current_state] + transition
            if tentative >= g_score.get(neighbor_state, float("inf")):
                continue
            came_from[neighbor_state] = current_state
            search_edges.append((current_point.copy(), neighbor_point.copy()))
            g_score[neighbor_state] = tentative
            delta = environment.goal - neighbor_point
            heuristic = float(np.linalg.norm(delta)) + climb_cost_per_m * max(0.0, float(delta[2]))
            heapq.heappush(queue, (tentative + heuristic_weight * heuristic, next(counter), neighbor_state))

    if goal_state is None:
        return PlanResult("Constrained A*", False, [], time.perf_counter() - started, len(closed), search_edges, "No feasible path")
    states = [goal_state]
    while states[-1] in came_from:
        states.append(came_from[states[-1]])
    states.reverse()
    path = [index_to_point(state[0]) for state in states]
    path[0] = environment.start.copy()
    path[-1] = environment.goal.copy()
    if not validate_flight_path(path, environment, constraints):
        return PlanResult("Constrained A*", False, [], time.perf_counter() - started, len(closed), search_edges, "Final path constraint validation failed")
    return PlanResult("Constrained A*", True, path, time.perf_counter() - started, len(closed), search_edges)


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


def _incoming_direction(nodes: list[_TreeNode], index: int) -> Point | None:
    parent = nodes[index].parent
    return None if parent is None else nodes[index].point - nodes[parent].point


def _children_remain_valid(
    environment: Environment,
    nodes: list[_TreeNode],
    node_index: int,
    new_incoming: Point,
    constraints: FlightConstraints,
) -> bool:
    return all(
        transition_is_valid(
            environment, nodes[node_index].point, child.point, new_incoming, constraints
        )
        for child in nodes
        if child.parent == node_index
    )


def _constrained_tree_root(environment: Environment, constraints: FlightConstraints) -> list[_TreeNode]:
    nodes = [_TreeNode(environment.start.copy(), None, 0.0)]
    if environment.start[2] < constraints.minimum_enroute_altitude:
        staging = environment.start.copy()
        staging[2] = constraints.minimum_enroute_altitude
        if transition_is_valid(environment, environment.start, staging, None, constraints):
            nodes.append(_TreeNode(staging, 0, float(np.linalg.norm(staging - environment.start))))
    return nodes


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


def constrained_rrt_3d(
    environment: Environment,
    constraints: FlightConstraints,
    seed: int = 0,
    max_iterations: int = 8000,
    step_size: float = 30.0,
    goal_bias: float = 0.18,
    goal_tolerance: float = 35.0,
) -> PlanResult:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    nodes = _constrained_tree_root(environment, constraints)
    for _ in range(max_iterations):
        sample = _sample(environment, rng, goal_bias)
        points = np.asarray([node.point for node in nodes])
        nearest = int(np.argmin(np.linalg.norm(points - sample, axis=1)))
        new_point = _steer(nodes[nearest].point, sample, step_size)
        if not transition_is_valid(
            environment, nodes[nearest].point, new_point,
            _incoming_direction(nodes, nearest), constraints,
        ):
            continue
        new_cost = nodes[nearest].cost + float(np.linalg.norm(new_point - nodes[nearest].point))
        nodes.append(_TreeNode(new_point, nearest, new_cost))
        new_index = len(nodes) - 1
        if np.linalg.norm(new_point - environment.goal) <= goal_tolerance and transition_is_valid(
            environment, new_point, environment.goal,
            _incoming_direction(nodes, new_index), constraints,
        ):
            path = _tree_path(nodes, new_index, environment.goal)
            if validate_flight_path(path, environment, constraints):
                return PlanResult(
                    "Constrained RRT", True, path,
                    time.perf_counter() - started, len(nodes), _tree_edges(nodes),
                )
    return PlanResult(
        "Constrained RRT", False, [], time.perf_counter() - started,
        len(nodes), _tree_edges(nodes), "Iteration limit reached",
    )


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
    goal_candidates = [
        index for index in goal_candidates
        if transition_is_valid(
            environment, nodes[index].point, environment.goal,
            _incoming_direction(nodes, index), constraints,
        )
    ]
    if goal_candidates:
        best = min(goal_candidates, key=lambda i: nodes[i].cost + float(np.linalg.norm(nodes[i].point - environment.goal)))
        return PlanResult(
            "RRT*", True, _tree_path(nodes, best, environment.goal),
            time.perf_counter() - started, len(nodes), _tree_edges(nodes),
        )
    return PlanResult("RRT*", False, [], time.perf_counter() - started, len(nodes), _tree_edges(nodes), "Iteration limit reached")


def constrained_rrt_star_3d(
    environment: Environment,
    constraints: FlightConstraints,
    seed: int = 0,
    max_iterations: int = 10000,
    step_size: float = 30.0,
    goal_bias: float = 0.16,
    goal_tolerance: float = 35.0,
    neighbor_radius: float = 75.0,
    minimum_nodes_after_goal: int = 700,
) -> PlanResult:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    nodes = _constrained_tree_root(environment, constraints)
    goal_candidates: list[int] = []
    for _ in range(max_iterations):
        sample = _sample(environment, rng, goal_bias)
        points = np.asarray([node.point for node in nodes])
        nearest = int(np.argmin(np.linalg.norm(points - sample, axis=1)))
        new_point = _steer(nodes[nearest].point, sample, step_size)
        distances = np.linalg.norm(points - new_point, axis=1)
        near_indices = np.where(distances <= neighbor_radius)[0].tolist()
        if nearest not in near_indices:
            near_indices.append(nearest)

        valid_parents: list[tuple[float, int]] = []
        for index in near_indices:
            if transition_is_valid(
                environment, nodes[index].point, new_point,
                _incoming_direction(nodes, index), constraints,
            ):
                valid_parents.append((nodes[index].cost + float(distances[index]), index))
        if not valid_parents:
            continue
        best_cost, best_parent = min(valid_parents)
        nodes.append(_TreeNode(new_point, best_parent, best_cost))
        new_index = len(nodes) - 1
        new_incoming = _incoming_direction(nodes, new_index)
        assert new_incoming is not None

        for index in near_indices:
            if index == best_parent or index == 0:
                continue
            edge_length = float(np.linalg.norm(nodes[index].point - new_point))
            candidate_cost = best_cost + edge_length
            if candidate_cost >= nodes[index].cost:
                continue
            candidate_incoming = nodes[index].point - new_point
            if not transition_is_valid(
                environment, new_point, nodes[index].point, new_incoming, constraints,
            ):
                continue
            if not _children_remain_valid(
                environment, nodes, index, candidate_incoming, constraints,
            ):
                continue
            delta = candidate_cost - nodes[index].cost
            nodes[index].parent = new_index
            nodes[index].cost = candidate_cost
            _shift_descendant_costs(nodes, index, delta)

        if np.linalg.norm(new_point - environment.goal) <= goal_tolerance and transition_is_valid(
            environment, new_point, environment.goal, new_incoming, constraints,
        ):
            goal_candidates.append(new_index)
        if goal_candidates and len(nodes) >= minimum_nodes_after_goal:
            break
    valid_candidates = [
        index for index in goal_candidates
        if transition_is_valid(
            environment, nodes[index].point, environment.goal,
            _incoming_direction(nodes, index), constraints,
        )
    ]
    for best in sorted(
        valid_candidates,
        key=lambda index: nodes[index].cost + float(np.linalg.norm(nodes[index].point - environment.goal)),
    ):
        path = _tree_path(nodes, best, environment.goal)
        if validate_flight_path(path, environment, constraints):
            return PlanResult(
                "Constrained RRT*", True, path,
                time.perf_counter() - started, len(nodes), _tree_edges(nodes),
            )
    return PlanResult(
        "Constrained RRT*", False, [], time.perf_counter() - started,
        len(nodes), _tree_edges(nodes), "Iteration limit reached",
    )

