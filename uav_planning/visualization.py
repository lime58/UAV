from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np

from .environment import Environment, Point


COLORS = {"A*": "#1677ff", "Constrained A*": "#0f766e", "RRT": "#f97316", "RRT*": "#16a34a"}

_font_path = Path(__file__).resolve().parent.parent / "assets" / "NotoSansSC.ttf"
if _font_path.exists():
    font_manager.fontManager.addfont(str(_font_path))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(_font_path)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def _draw_environment(ax, environment: Environment) -> None:
    for obstacle in environment.obstacles:
        x, y, z = obstacle.minimum
        dx = obstacle.maximum[0] - x
        dy = obstacle.maximum[1] - y
        dz = obstacle.maximum[2] - z
        ax.bar3d(x, y, z, dx, dy, dz, color="#64748b", alpha=0.42, shade=True)
    ax.scatter(*environment.start, s=70, c="#22c55e", marker="o", label="Start")
    ax.scatter(*environment.goal, s=90, c="#ef4444", marker="*", label="Goal")
    ax.set_xlim(*environment.bounds[0]); ax.set_ylim(*environment.bounds[1]); ax.set_zlim(*environment.bounds[2])
    ax.set_xlabel("X / m"); ax.set_ylabel("Y / m"); ax.set_zlabel("Z / m")
    ax.view_init(elev=27, azim=-58)


def plot_paths(environment: Environment, paths: dict[str, list[Point]], output: str | Path, title: str | None = None) -> None:
    fig = plt.figure(figsize=(10, 7), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    _draw_environment(ax, environment)
    for algorithm, path in paths.items():
        if not path:
            continue
        points = np.asarray(path)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=COLORS.get(algorithm), linewidth=2.3, marker="o", markersize=2.5, label=algorithm)
    ax.set_title(title or f"3D UAV path planning - {environment.name}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_tree(environment: Environment, edges: list[tuple[Point, Point]], path: list[Point], algorithm: str, output: str | Path) -> None:
    fig = plt.figure(figsize=(10, 7), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    _draw_environment(ax, environment)
    for start, end in edges[:: max(1, len(edges) // 700)]:
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color="#94a3b8", alpha=0.24, linewidth=0.45)
    if path:
        points = np.asarray(path)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=COLORS.get(algorithm, "#1677ff"), linewidth=2.6, label=f"{algorithm} path")
    ax.set_title(f"{algorithm} search tree - {environment.name}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_topdown(environment: Environment, path: list[Point], output: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), dpi=180)
    for obstacle in environment.obstacles:
        x, y, _ = obstacle.minimum
        width = obstacle.maximum[0] - x
        height = obstacle.maximum[1] - y
        ax.add_patch(Rectangle((x, y), width, height, facecolor="#64748b", edgecolor="#334155", alpha=0.46, linewidth=0.5))
    points = np.asarray(path)
    ax.plot(points[:, 0], points[:, 1], color="#16a34a", linewidth=2.8, marker="o", markersize=4, label="Planned route")
    ax.scatter(path[0][0], path[0][1], s=85, c="#22c55e", label="Start", zorder=4)
    ax.scatter(path[-1][0], path[-1][1], s=100, c="#ef4444", marker="*", label="Goal", zorder=4)
    ax.set_xlim(*environment.bounds[0]); ax.set_ylim(*environment.bounds[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Local X / m"); ax.set_ylabel("Local Y / m")
    ax.set_title(title); ax.grid(alpha=0.18); ax.legend(loc="upper left")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary: list[dict], output: str | Path) -> None:
    algorithms = [row["algorithm"] for row in summary]
    lengths = [row.get("mean_path_length_m", 0) or 0 for row in summary]
    times = [row.get("mean_planning_time_s", 0) or 0 for row in summary]
    success = [row.get("success_rate", 0) * 100 for row in summary]
    colors = [COLORS.get(name, "#64748b") for name in algorithms]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=180)
    for ax, values, title, ylabel in zip(
        axes,
        (lengths, times, success),
        ("Mean path length", "Mean planning time", "Success rate"),
        ("m", "s", "%"),
    ):
        ax.bar(algorithms, values, color=colors, width=0.58)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.22)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

