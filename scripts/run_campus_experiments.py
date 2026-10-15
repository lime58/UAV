"""Reproducible campus experiments for the literature-guided planner.

The resolution sweep is the evidence used to select the demonstration grid;
it is intentionally kept separate from the mission configuration.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uav_planning.analysis import maximum_climb_angle, maximum_turn_angle, path_length, validate_path
from uav_planning.mission import minimum_clearance
from uav_planning.planners import constrained_astar_3d
from uav_planning.scenarios import load_geojson_scenario
from uav_planning.visualization import plot_paths


def main() -> None:
    environment = load_geojson_scenario("data/cases/hhu_jiangning.geojson")
    environment.safety_margin = 4.0
    output = Path("outputs/hhu_experiments")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    paths = {}
    for horizontal_resolution, vertical_resolution in ((10.0, 2.5), (20.0, 5.0), (30.0, 5.0), (40.0, 5.0)):
        result = constrained_astar_3d(
            environment,
            horizontal_resolution=horizontal_resolution,
            vertical_resolution=vertical_resolution,
            maximum_altitude=50.0,
            minimum_enroute_altitude=12.0,
            approach_radius=75.0,
            maximum_climb_angle_deg=15.0,
            maximum_turn_angle_deg=90.0,
            climb_cost_per_m=4.0,
            turn_cost_per_deg=0.08,
            overflight_cost_per_m=12.0,
            heuristic_weight=2.0,
        )
        path = result.path
        label = f"Constrained A* ({horizontal_resolution:.0f} m)"
        paths[label] = path
        rows.append({
            "horizontal_resolution_m": horizontal_resolution,
            "vertical_resolution_m": vertical_resolution,
            "success": result.success,
            "collision_free": validate_path(path, environment),
            "planning_time_s": result.planning_time,
            "expanded_nodes": result.expanded_nodes,
            "waypoint_count": len(path),
            "path_length_m": path_length(path),
            "minimum_clearance_m": minimum_clearance(path, environment.obstacles),
            "maximum_altitude_m": max(float(p[2]) for p in path),
            "maximum_climb_angle_deg": maximum_climb_angle(path),
            "maximum_turn_angle_deg": maximum_turn_angle(path),
        })
    with (output / "resolution_sensitivity.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "resolution_sensitivity.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_paths(environment, paths, output / "resolution_paths_3d.png", "Campus resolution sensitivity")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
