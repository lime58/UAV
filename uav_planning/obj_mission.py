from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .analysis import path_length, smooth_path, validate_path
from .environment import Environment
from .mesh_adapter import load_obj, mesh_to_column_obstacles, mesh_to_component_obstacles, write_route_obj
from .planners import astar_3d, rrt_3d, rrt_star_3d
from .visualization import plot_paths, plot_topdown


@dataclass
class ObjMissionReport:
    mission_id: str
    mission_name: str
    status: str
    algorithm: str
    model_file: str
    mesh_vertices: int
    mesh_faces: int
    obstacle_volumes: int
    waypoint_count: int
    route_length_m: float
    planning_time_s: float
    expanded_nodes: int
    collision_free: bool
    voxel_size_m: float
    safety_margin_m: float


def run_obj_mission(config_path: str | Path) -> tuple[ObjMissionReport, list[np.ndarray], Path]:
    config_path = Path(config_path)
    root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_path = root / config["model"]
    output = root / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)

    mesh = load_obj(model_path)
    voxel_size = float(config["mesh"]["voxel_size_m"])
    obstacle_model = config["mesh"].get("obstacle_model", "voxel_columns")
    if obstacle_model == "component_boxes":
        obstacles = mesh_to_component_obstacles(mesh)
    elif obstacle_model == "voxel_columns":
        obstacles = mesh_to_column_obstacles(mesh, voxel_size)
    else:
        raise ValueError("mesh.obstacle_model must be component_boxes or voxel_columns")
    environment = Environment.from_values(
        config["mission_name"],
        tuple(tuple(pair) for pair in config["bounds"]),
        config["start"],
        config["goal"],
        obstacles,
        float(config["flight"]["safety_margin_m"]),
    )
    planner = config["planner"]
    algorithm = config["algorithm"].lower()
    if algorithm == "astar":
        result = astar_3d(environment, float(planner["resolution_m"]))
    elif algorithm == "rrt":
        result = rrt_3d(environment, seed=int(config.get("random_seed", 0)))
    elif algorithm == "rrt_star":
        result = rrt_star_3d(environment, seed=int(config.get("random_seed", 0)))
    else:
        raise ValueError("algorithm must be astar, rrt, or rrt_star")
    if not result.success:
        raise RuntimeError(result.message)
    route = smooth_path(result.path, environment)
    collision_free = validate_path(route, environment)
    if not collision_free:
        raise RuntimeError("Smoothed OBJ route failed collision validation")

    report = ObjMissionReport(
        config["mission_id"], config["mission_name"], "SUCCESS", result.algorithm,
        str(config["model"]), len(mesh.vertices), len(mesh.faces), len(obstacles),
        len(route), path_length(route), result.planning_time, result.expanded_nodes,
        collision_free, voxel_size, environment.safety_margin,
    )
    (output / "obj_mission_report.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "obj_route.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "x_m", "y_m", "z_m"])
        for index, point in enumerate(route):
            writer.writerow([index, *[round(float(value), 3) for value in point]])
    write_route_obj(output / "obj_route.obj", route)
    plot_paths(environment, {result.algorithm: route}, output / "obj_route_3d.png", config["mission_name"])
    plot_topdown(environment, route, output / "obj_route_topdown.png", config["mission_name"])
    return report, route, output
