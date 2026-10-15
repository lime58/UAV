from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uav_planning.analysis import maximum_climb_angle, maximum_turn_angle, path_length, validate_path
from uav_planning.mission import minimum_clearance
from uav_planning.planners import (
    FlightConstraints, constrained_astar_3d, constrained_rrt_3d,
    constrained_rrt_star_3d,
)
from uav_planning.scenarios import load_geojson_scenario
from uav_planning.visualization import plot_paths


def metric(result, environment, seed):
    path = result.path
    return {
        "algorithm": result.algorithm, "seed": seed, "success": result.success,
        "collision_free": validate_path(path, environment) if result.success else False,
        "path_length_m": path_length(path) if result.success else None,
        "planning_time_s": result.planning_time,
        "waypoint_count": len(path),
        "minimum_clearance_m": minimum_clearance(path, environment.obstacles) if result.success else None,
        "maximum_altitude_m": max((float(point[2]) for point in path), default=None),
        "maximum_climb_angle_deg": maximum_climb_angle(path) if result.success else None,
        "maximum_turn_angle_deg": maximum_turn_angle(path) if result.success else None,
        "expanded_generated_nodes": result.expanded_nodes,
    }


def main() -> None:
    environment = load_geojson_scenario("data/cases/hhu_jiangning.geojson")
    environment.safety_margin = 4.0
    constraints = FlightConstraints(50.0, 12.0, 75.0, 15.0, 90.0)
    output = Path("outputs/hhu_constrained_comparison")
    output.mkdir(parents=True, exist_ok=True)
    rows, paths = [], {}

    astar = constrained_astar_3d(
        environment, 10.0, 2.5, 50.0, 12.0, 75.0, 15.0, 90.0,
        4.0, 0.08, 12.0, 2.0,
    )
    rows.append(metric(astar, environment, 0))
    paths[astar.algorithm] = astar.path

    for seed in range(5):
        rrt = constrained_rrt_3d(
            environment, constraints, seed, 12000, 40.0, 0.25, 50.0,
        )
        rows.append(metric(rrt, environment, seed))
        if rrt.success and rrt.algorithm not in paths:
            paths[rrt.algorithm] = rrt.path
        rrt_star = constrained_rrt_star_3d(
            environment, constraints, seed, 8000, 40.0, 0.20, 50.0, 70.0, 250,
        )
        rows.append(metric(rrt_star, environment, seed))
        if rrt_star.success and rrt_star.algorithm not in paths:
            paths[rrt_star.algorithm] = rrt_star.path

    fields = list(rows[0])
    with (output / "raw_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (output / "raw_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = []
    for algorithm in ("Constrained A*", "Constrained RRT", "Constrained RRT*"):
        subset = [row for row in rows if row["algorithm"] == algorithm]
        successful = [row for row in subset if row["success"] and row["collision_free"]]
        item = {"algorithm": algorithm, "runs": len(subset), "success_rate": len(successful) / len(subset)}
        for key in fields[4:]:
            values = [row[key] for row in successful if row[key] is not None]
            item[f"mean_{key}"] = mean(values) if values else None
            item[f"std_{key}"] = pstdev(values) if len(values) > 1 else 0.0 if values else None
        summary.append(item)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_paths(environment, paths, output / "constrained_paths_3d.png", "Constrained planner comparison")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
