from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EARTH_M_PER_DEG_LAT = 110_540.0


def local_xy(lon: float, lat: float, lon0: float, lat0: float) -> tuple[float, float]:
    x = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * EARTH_M_PER_DEG_LAT
    return x, y


def parse_height(tags: dict) -> tuple[float, str]:
    value = str(tags.get("height", "")).lower().replace("meters", "").replace("meter", "").replace("m", "").strip()
    try:
        return max(3.0, float(value)), "height"
    except ValueError:
        pass
    try:
        return max(3.0, float(tags["building:levels"]) * 3.3), "building:levels×3.3m"
    except (KeyError, ValueError, TypeError):
        return 15.0, "default_15m"


def point_in_polygon(point: tuple[float, float], ring: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def choose_open_points(
    boundary: list[tuple[float, float]],
    building_boxes: list[tuple[float, float, float, float]],
    bounds: tuple[float, float, float, float],
    margin: float = 10.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    min_x, min_y, max_x, max_y = bounds
    candidates = []
    x = math.ceil(min_x / 10.0) * 10.0
    while x <= max_x:
        y = math.ceil(min_y / 10.0) * 10.0
        while y <= max_y:
            if point_in_polygon((x, y), boundary) and all(
                not (xmin - margin <= x <= xmax + margin and ymin - margin <= y <= ymax + margin)
                for xmin, ymin, xmax, ymax in building_boxes
            ):
                candidates.append((x, y))
            y += 10.0
        x += 10.0
    if len(candidates) < 2:
        raise ValueError("Could not find two open takeoff/landing points")
    start = min(candidates, key=lambda point: point[0] + point[1])
    distant = [point for point in candidates if math.dist(point, start) >= 500.0]
    goal = max(distant or candidates, key=lambda point: math.dist(point, start))
    return start, goal


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OSM Overpass buildings to a local-metric mission case")
    parser.add_argument("--input", default="data/raw/hhu_buildings_osm.json")
    parser.add_argument("--boundary", default="data/raw/hhu_campus_boundary.geojson")
    parser.add_argument("--output", default="data/cases/hhu_jiangning.geojson")
    args = parser.parse_args()

    osm = json.loads(Path(args.input).read_text(encoding="utf-8"))
    boundary = json.loads(Path(args.boundary).read_text(encoding="utf-8"))
    boundary_ring = boundary["features"][0]["geometry"]["coordinates"][0]
    lon0 = sum(point[0] for point in boundary_ring) / len(boundary_ring)
    lat0 = sum(point[1] for point in boundary_ring) / len(boundary_ring)
    local_boundary = [local_xy(point[0], point[1], lon0, lat0) for point in boundary_ring]
    min_x = min(point[0] for point in local_boundary)
    max_x = max(point[0] for point in local_boundary)
    min_y = min(point[1] for point in local_boundary)
    max_y = max(point[1] for point in local_boundary)

    features = []
    building_boxes = []
    height_sources: dict[str, int] = {}
    for element in osm.get("elements", []):
        geometry = element.get("geometry") or []
        if len(geometry) < 4:
            continue
        ring = [local_xy(point["lon"], point["lat"], lon0, lat0) for point in geometry]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        building_boxes.append((
            min(point[0] for point in ring), min(point[1] for point in ring),
            max(point[0] for point in ring), max(point[1] for point in ring),
        ))
        tags = element.get("tags") or {}
        height, height_source = parse_height(tags)
        height_sources[height_source] = height_sources.get(height_source, 0) + 1
        features.append({
            "type": "Feature",
            "properties": {
                "role": "obstacle",
                "name": tags.get("name", f"OSM building {element['id']}"),
                "height": round(height, 2),
                "height_source": height_source,
                "osm_type": element.get("type"),
                "osm_id": element.get("id"),
            },
            "geometry": {"type": "Polygon", "coordinates": [[list(point) for point in ring]]},
        })

    start_xy, goal_xy = choose_open_points(
        local_boundary,
        building_boxes,
        (min_x, min_y, max_x, max_y),
    )
    for role, name, point_xy in (
        ("start", "西南起降点", start_xy),
        ("goal", "东北任务点", goal_xy),
    ):
        features.append({
            "type": "Feature",
            "properties": {"role": role, "name": name, "altitude": 2.0},
            "geometry": {"type": "Point", "coordinates": list(point_xy)},
        })

    output = {
        "type": "FeatureCollection",
        "properties": {
            "name": "河海大学江宁校区低空航线规划案例",
            "bounds": [[math.floor(min_x - 20), math.ceil(max_x + 20)], [math.floor(min_y - 20), math.ceil(max_y + 20)], [0, 90]],
            "safety_margin": 4.0,
            "origin_wgs84": [lon0, lat0],
            "coordinate_system": "local_equirectangular_metre",
            "source": "OpenStreetMap Nominatim and Overpass API",
            "source_url": "https://www.openstreetmap.org/way/92364518",
            "licence": "OpenStreetMap contributors, ODbL 1.0",
            "retrieved_at": "2026-10-13",
            "building_count": len(features) - 2,
            "height_sources": height_sources,
        },
        "features": features,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["properties"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

