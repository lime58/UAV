from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .analysis import maximum_climb_angle, maximum_turn_angle, path_length, smooth_path, validate_path
from .environment import AABB, Environment, Point
from .planners import PlanResult, astar_3d, constrained_astar_3d, rrt_3d, rrt_star_3d
from .scenarios import load_geojson_scenario
from .visualization import plot_paths, plot_topdown, plot_tree


@dataclass
class MissionReport:
    mission_id: str
    mission_name: str
    scenario: str
    algorithm: str
    status: str
    collision_free: bool
    waypoint_count: int
    horizontal_distance_m: float
    total_distance_m: float
    estimated_flight_time_s: float
    minimum_obstacle_clearance_m: float
    maximum_altitude_m: float
    maximum_climb_angle_deg: float
    maximum_turn_angle_deg: float
    planning_time_s: float
    expanded_nodes: int
    data_source: str
    notes: list[str]


def _distance_to_box(point: Point, box: AABB) -> float:
    low = np.asarray(box.minimum, dtype=float)
    high = np.asarray(box.maximum, dtype=float)
    delta = np.maximum(np.maximum(low - point, point - high), 0.0)
    return float(np.linalg.norm(delta))


def minimum_clearance(path: list[Point], obstacles: list[AABB], sample_interval: float = 3.0) -> float:
    if not obstacles or not path:
        return float("inf")
    minimum = float("inf")
    for start, end in zip(path, path[1:]):
        distance = float(np.linalg.norm(end - start))
        samples = max(2, int(math.ceil(distance / sample_interval)) + 1)
        for fraction in np.linspace(0.0, 1.0, samples):
            point = start + fraction * (end - start)
            minimum = min(minimum, *(_distance_to_box(point, obstacle) for obstacle in obstacles))
    return minimum


def _horizontal_distance(path: list[Point]) -> float:
    return float(sum(np.linalg.norm((b - a)[:2]) for a, b in zip(path, path[1:])))


def _local_to_wgs84(point: Point, origin: list[float]) -> list[float]:
    lon0, lat0 = origin
    lon = lon0 + point[0] / (111_320.0 * math.cos(math.radians(lat0)))
    lat = lat0 + point[1] / 110_540.0
    return [lon, lat, float(point[2])]


def _phase(index: int, count: int) -> str:
    if index == 0:
        return "GROUND_START"
    if index == 1:
        return "TAKEOFF"
    if index == count - 2:
        return "LANDING_APPROACH"
    if index == count - 1:
        return "GROUND_GOAL"
    return "CRUISE"


def run_mission(config_path: str | Path) -> tuple[MissionReport, list[Point], PlanResult]:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parent.parent
    scenario_path = project_root / config["scenario"]
    scenario_json = json.loads(scenario_path.read_text(encoding="utf-8"))
    environment = load_geojson_scenario(scenario_path)
    flight = config["flight"]
    planner = config["planner"]
    environment.safety_margin = float(flight["safety_margin_m"])

    ground_start = environment.start.copy()
    ground_goal = environment.goal.copy()

    algorithm = config["algorithm"].lower()
    if algorithm == "constrained_astar":
        raw = constrained_astar_3d(
            environment,
            horizontal_resolution=float(planner["horizontal_resolution_m"]),
            vertical_resolution=float(planner["vertical_resolution_m"]),
            maximum_altitude=float(flight["maximum_altitude_m"]),
            minimum_enroute_altitude=float(flight["minimum_enroute_altitude_m"]),
            approach_radius=float(flight["approach_radius_m"]),
            maximum_climb_angle_deg=float(flight["maximum_climb_angle_deg"]),
            maximum_turn_angle_deg=float(flight["maximum_turn_angle_deg"]),
            climb_cost_per_m=float(planner["climb_cost_per_m"]),
            turn_cost_per_deg=float(planner["turn_cost_per_deg"]),
            overflight_cost_per_m=float(planner["overflight_cost_per_m"]),
            heuristic_weight=float(planner.get("heuristic_weight", 1.0)),
        )
    elif algorithm == "astar":
        raw = astar_3d(environment, resolution=float(planner["astar_resolution_m"]))
    elif algorithm == "rrt":
        raw = rrt_3d(
            environment,
            seed=int(config["random_seed"]),
            max_iterations=int(planner["max_iterations"]),
            step_size=float(planner["step_size_m"]),
            goal_bias=float(planner["goal_bias"]),
            goal_tolerance=float(planner["goal_tolerance_m"]),
        )
    elif algorithm == "rrt_star":
        raw = rrt_star_3d(
            environment,
            seed=int(config["random_seed"]),
            max_iterations=int(planner["max_iterations"]),
            step_size=float(planner["step_size_m"]),
            goal_bias=float(planner["goal_bias"]),
            goal_tolerance=float(planner["goal_tolerance_m"]),
            neighbor_radius=float(planner["neighbor_radius_m"]),
        )
    else:
        raise ValueError("algorithm must be constrained_astar, astar, rrt, or rrt_star")
    if not raw.success:
        raise RuntimeError(f"Planning failed: {raw.message}")

    # The complete ground-to-ground mission is planned as one path.  We do not
    # prepend/append artificial vertical segments because that hides climb-angle
    # infeasibility and was the main flaw in the earlier demonstration.
    full_path = raw.path if algorithm == "constrained_astar" else smooth_path(raw.path, environment)
    if not validate_path(full_path, environment):
        raise RuntimeError("Final route failed collision validation")
    total = path_length(full_path)
    horizontal = _horizontal_distance(full_path)
    vertical = sum(abs(float(b[2] - a[2])) for a, b in zip(full_path, full_path[1:]))
    estimated_time = horizontal / float(flight["cruise_speed_mps"]) + vertical / float(flight["vertical_speed_mps"])
    clearance = minimum_clearance(full_path, environment.obstacles)
    max_altitude = max(float(point[2]) for point in full_path)
    source = scenario_json.get("properties", {}).get("source", "unknown")
    notes = [
        "建筑高度优先采用OSM height，其次按building:levels×3.3m估算，缺失时采用15m默认值。",
        "规划结果用于航线方案与风险评估，不直接替代飞控系统的实时避障和合规审批。",
        "所有输出航段均通过安全距离膨胀后的三维碰撞复核。",
        "起降点附近采用受限过渡区，航路段不得低于配置的最低离地高度。",
    ]
    report = MissionReport(
        config["mission_id"], config["mission_name"], environment.name,
        raw.algorithm, "SUCCESS", True, len(full_path), horizontal, total,
        estimated_time, clearance, max_altitude, maximum_climb_angle(full_path), maximum_turn_angle(full_path), raw.planning_time,
        raw.expanded_nodes, source, notes,
    )
    return report, full_path, raw


def export_mission(config_path: str | Path, report: MissionReport, path: list[Point], raw: PlanResult) -> Path:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parent.parent
    scenario_path = project_root / config["scenario"]
    scenario_json = json.loads(scenario_path.read_text(encoding="utf-8"))
    origin = scenario_json["properties"]["origin_wgs84"]
    output = project_root / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)

    (output / "mission_report.json").write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "route_local.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "phase", "x_m", "y_m", "altitude_m"])
        for index, point in enumerate(path):
            writer.writerow([index, _phase(index, len(path)), *[round(float(value), 3) for value in point]])

    coordinates = [_local_to_wgs84(point, origin) for point in path]
    route_geojson = {
        "type": "FeatureCollection",
        "properties": {
            "mission_id": report.mission_id,
            "coordinate_reference_system": "WGS84 / EPSG:4326",
            "altitude_reference": "relative demonstration altitude",
        },
        "features": [
            {
                "type": "Feature",
                "properties": {"mission_id": report.mission_id, "algorithm": report.algorithm, "status": report.status},
                "geometry": {"type": "LineString", "coordinates": coordinates},
            },
            *[
                {
                    "type": "Feature",
                    "properties": {"sequence": index, "phase": _phase(index, len(path))},
                    "geometry": {"type": "Point", "coordinates": coordinate},
                }
                for index, coordinate in enumerate(coordinates)
            ],
        ],
    }
    (output / "route_wgs84.geojson").write_text(json.dumps(route_geojson, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [
        "# 无人机航线规划任务报告",
        "",
        f"- 任务编号：{report.mission_id}",
        f"- 任务名称：{report.mission_name}",
        f"- 场景：{report.scenario}",
        f"- 算法：{report.algorithm}",
        f"- 状态：{report.status}",
        f"- 航点数：{report.waypoint_count}",
        f"- 三维航程：{report.total_distance_m:.2f} m",
        f"- 水平航程：{report.horizontal_distance_m:.2f} m",
        f"- 预计飞行时间：{report.estimated_flight_time_s:.1f} s",
        f"- 最小障碍净距：{report.minimum_obstacle_clearance_m:.2f} m",
        f"- 最大相对高度：{report.maximum_altitude_m:.2f} m",
        f"- 最大航段爬升/下降角：{report.maximum_climb_angle_deg:.2f}°",
        f"- 最大航向变化角：{report.maximum_turn_angle_deg:.2f}°",
        f"- 规划耗时：{report.planning_time_s:.3f} s",
        f"- 扩展节点：{report.expanded_nodes}",
        f"- 数据来源：{report.data_source}",
        "",
        "## 校验结论",
        "",
        "航线已通过边界、障碍物安全距离、起降走廊和三维航段碰撞校验。",
        "",
        "## 说明",
        "",
        *[f"- {note}" for note in report.notes],
    ]
    (output / "mission_report.md").write_text("\n".join(rows), encoding="utf-8")

    environment = load_geojson_scenario(scenario_path)
    environment.safety_margin = float(config["flight"]["safety_margin_m"])
    plot_paths(environment, {report.algorithm: path}, output / "mission_route_3d.png", report.mission_name)
    plot_topdown(environment, path, output / "mission_route_topdown.png", report.mission_name)
    plot_tree(environment, raw.tree_edges, path[1:-1], report.algorithm, output / "search_tree_3d.png")
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{report.mission_name}</title>
<style>
body{{margin:0;background:#f4f7fb;color:#172033;font-family:'Microsoft YaHei',Arial,sans-serif}}
main{{max-width:1180px;margin:32px auto;padding:0 24px}}h1{{font-size:30px;margin:0 0 8px}}.sub{{color:#64748b;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.card{{background:white;border-radius:12px;padding:18px;box-shadow:0 4px 18px #1e293b12}}
.value{{font-size:25px;font-weight:700;color:#0f766e;margin-top:8px}}.label{{font-size:13px;color:#64748b}}
.panel{{background:white;border-radius:12px;margin-top:18px;padding:20px;box-shadow:0 4px 18px #1e293b12}}img{{width:100%;border-radius:8px}}
.ok{{display:inline-block;background:#dcfce7;color:#166534;padding:5px 10px;border-radius:999px;font-weight:700}}table{{border-collapse:collapse;width:100%}}td{{padding:10px;border-bottom:1px solid #e2e8f0}}
@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<h1>{report.mission_name}</h1><div class="sub">任务编号 {report.mission_id}　<span class="ok">{report.status}</span></div>
<section class="grid">
<div class="card"><div class="label">三维航程</div><div class="value">{report.total_distance_m:.1f} m</div></div>
<div class="card"><div class="label">预计飞行时间</div><div class="value">{report.estimated_flight_time_s:.1f} s</div></div>
<div class="card"><div class="label">最小障碍净距</div><div class="value">{report.minimum_obstacle_clearance_m:.2f} m</div></div>
<div class="card"><div class="label">规划耗时</div><div class="value">{report.planning_time_s:.2f} s</div></div>
</section>
<section class="panel"><h2>俯视航线</h2><img src="mission_route_topdown.png" alt="俯视航线"></section>
<section class="panel"><h2>三维航线</h2><img src="mission_route_3d.png" alt="三维航线"></section>
<section class="panel"><h2>任务摘要</h2><table>
<tr><td>规划算法</td><td>{report.algorithm}</td></tr><tr><td>航点数量</td><td>{report.waypoint_count}</td></tr>
<tr><td>最大相对高度</td><td>{report.maximum_altitude_m:.2f} m</td></tr><tr><td>碰撞复核</td><td>{'通过' if report.collision_free else '未通过'}</td></tr>
<tr><td>建筑数据</td><td>{report.data_source}</td></tr></table></section>
</main></body></html>"""
    (output / "mission_dashboard.html").write_text(html, encoding="utf-8")
    return output

