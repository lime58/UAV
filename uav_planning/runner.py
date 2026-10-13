from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from .analysis import Metrics, calculate_metrics
from .environment import Environment
from .planners import astar_3d, rrt_3d, rrt_star_3d
from .visualization import plot_paths, plot_summary, plot_tree


def run_once(environment: Environment, seed: int = 7, resolution: float = 2.0) -> tuple[list[Metrics], dict, dict]:
    raw_results = {
        "A*": astar_3d(environment, resolution=resolution),
        "RRT": rrt_3d(environment, seed=seed),
        "RRT*": rrt_star_3d(environment, seed=seed),
    }
    metrics, paths = [], {}
    for name, result in raw_results.items():
        metric, path = calculate_metrics(result, environment, seed)
        metrics.append(metric)
        paths[name] = path
    return metrics, paths, raw_results


def benchmark(environment: Environment, runs: int, output_dir: str | Path, resolution: float = 2.0) -> tuple[list[Metrics], list[dict]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[Metrics] = []
    representative_paths = {}
    representative_raw = {}
    astar_result = astar_3d(environment, resolution=resolution)
    astar_metric, astar_path = calculate_metrics(astar_result, environment, 0)
    all_metrics.append(astar_metric)
    representative_paths["A*"] = astar_path
    representative_raw["A*"] = astar_result
    for seed in range(runs):
        for planner in (rrt_3d, rrt_star_3d):
            result = planner(environment, seed=seed)
            metric, path = calculate_metrics(result, environment, seed)
            all_metrics.append(metric)
            if result.algorithm not in representative_paths and result.success:
                representative_paths[result.algorithm] = path
                representative_raw[result.algorithm] = result

    fields = list(all_metrics[0].to_dict())
    with (output_dir / "raw_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(metric.to_dict() for metric in all_metrics)

    summary = []
    for algorithm in ("A*", "RRT", "RRT*"):
        subset = [m for m in all_metrics if m.algorithm == algorithm]
        successful = [m for m in subset if m.success and m.collision_free]
        lengths = [m.path_length_m for m in successful if m.path_length_m is not None]
        times = [m.planning_time_s for m in subset]
        turns = [m.turning_angle_deg for m in successful if m.turning_angle_deg is not None]
        summary.append({
            "scenario": environment.name,
            "algorithm": algorithm,
            "runs": len(subset),
            "success_rate": len(successful) / len(subset),
            "mean_planning_time_s": mean(times),
            "std_planning_time_s": pstdev(times) if len(times) > 1 else 0.0,
            "mean_path_length_m": mean(lengths) if lengths else None,
            "std_path_length_m": pstdev(lengths) if len(lengths) > 1 else 0.0,
            "mean_turning_angle_deg": mean(turns) if turns else None,
        })
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)

    plot_paths(environment, representative_paths, output_dir / "path_comparison.png")
    for name in ("RRT", "RRT*"):
        if name in representative_raw:
            plot_tree(environment, representative_raw[name].tree_edges, representative_paths[name], name, output_dir / f"{name.replace('*', '_star')}_tree.png")
    plot_summary(summary, output_dir / "metrics_comparison.png")
    return all_metrics, summary

