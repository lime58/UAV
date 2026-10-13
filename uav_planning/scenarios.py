from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .environment import AABB, Environment


def _simple() -> Environment:
    return Environment.from_values(
        "simple",
        ((0, 60), (0, 60), (0, 40)),
        (2, 2, 2),
        (56, 56, 32),
        [
            AABB((18, 12, 0), (30, 30, 24), "building_1"),
            AABB((34, 34, 0), (45, 48, 30), "building_2"),
        ],
        1.5,
    )


def _medium() -> Environment:
    return Environment.from_values(
        "medium",
        ((0, 80), (0, 80), (0, 50)),
        (2, 4, 4),
        (76, 74, 42),
        [
            AABB((12, 10, 0), (24, 28, 30), "building_1"),
            AABB((30, 0, 0), (42, 25, 22), "building_2"),
            AABB((31, 38, 0), (49, 55, 38), "building_3"),
            AABB((52, 18, 0), (66, 38, 32), "building_4"),
            AABB((58, 52, 0), (70, 67, 44), "building_5"),
        ],
        2.0,
    )


def _complex() -> Environment:
    return Environment.from_values(
        "complex",
        ((0, 100), (0, 100), (0, 60)),
        (2, 4, 4),
        (96, 94, 52),
        [
            AABB((10, 8, 0), (23, 30, 32), "building_1"),
            AABB((28, 0, 0), (42, 24, 42), "building_2"),
            AABB((27, 35, 0), (43, 57, 28), "building_3"),
            AABB((47, 16, 0), (61, 42, 36), "building_4"),
            AABB((50, 55, 0), (66, 80, 46), "building_5"),
            AABB((69, 4, 0), (82, 31, 30), "building_6"),
            AABB((74, 42, 0), (90, 63, 50), "building_7"),
            AABB((18, 67, 0), (39, 87, 38), "building_8"),
            AABB((73, 73, 0), (88, 88, 34), "building_9"),
        ],
        2.0,
    )


SCENARIOS = {"simple": _simple, "medium": _medium, "complex": _complex}


def get_scenario(name: str) -> Environment:
    try:
        return SCENARIOS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown scenario {name!r}; choose from {sorted(SCENARIOS)}") from exc


def _polygon_bounds(coordinates: list) -> tuple[float, float, float, float]:
    ring = coordinates[0]
    xs = [float(p[0]) for p in ring]
    ys = [float(p[1]) for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def load_geojson_scenario(path: str | Path) -> Environment:
    """Load projected/local-metric GeoJSON and extrude polygons into boxes.

    Polygon properties: role=obstacle, height, base_height, name.
    Point properties: role=start or role=goal; optional altitude.
    Collection properties: bounds=[[xmin,xmax],[ymin,ymax],[zmin,zmax]], safety_margin.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    obstacles: list[AABB] = []
    start = goal = None
    for index, feature in enumerate(data.get("features", []), start=1):
        geometry = feature.get("geometry") or {}
        props = feature.get("properties") or {}
        role = str(props.get("role", "obstacle")).lower()
        geom_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if geom_type == "Point" and role in {"start", "goal"}:
            altitude = float(props.get("altitude", coordinates[2] if len(coordinates) > 2 else 2.0))
            point = np.asarray([coordinates[0], coordinates[1], altitude], dtype=float)
            if role == "start":
                start = point
            else:
                goal = point
        elif geom_type in {"Polygon", "MultiPolygon"}:
            polygons = [coordinates] if geom_type == "Polygon" else coordinates
            for part, polygon in enumerate(polygons, start=1):
                xmin, ymin, xmax, ymax = _polygon_bounds(polygon)
                base = float(props.get("base_height", 0.0))
                height = float(props.get("height", 20.0))
                obstacles.append(
                    AABB(
                        (xmin, ymin, base),
                        (xmax, ymax, base + height),
                        f"{props.get('name', f'obstacle_{index}')}_{part}",
                    )
                )
    if start is None or goal is None:
        raise ValueError("GeoJSON must include Point features with role=start and role=goal")
    collection_props = data.get("properties") or {}
    if "bounds" in collection_props:
        bounds = tuple(tuple(float(v) for v in axis) for axis in collection_props["bounds"])
    else:
        all_x = [start[0], goal[0]] + [v for box in obstacles for v in (box.minimum[0], box.maximum[0])]
        all_y = [start[1], goal[1]] + [v for box in obstacles for v in (box.minimum[1], box.maximum[1])]
        max_z = max([start[2], goal[2], 30.0] + [box.maximum[2] for box in obstacles])
        pad = 5.0
        bounds = ((min(all_x) - pad, max(all_x) + pad), (min(all_y) - pad, max(all_y) + pad), (0.0, max_z + pad))
    return Environment.from_values(
        collection_props.get("name", Path(path).stem),
        bounds, start, goal, obstacles,
        float(collection_props.get("safety_margin", 1.5)),
    )

