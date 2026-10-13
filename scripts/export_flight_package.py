from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uav_planning.flight_export import export_flight_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Export validated route as mission JSON and KML/KMZ")
    parser.add_argument("--route", default="outputs/hhu_delivery_mission/route_wgs84.geojson")
    parser.add_argument("--report", default="outputs/hhu_delivery_mission/mission_report.json")
    parser.add_argument("--output", default="outputs/hhu_delivery_mission/flight_package")
    parser.add_argument("--speed", type=float, default=8.0)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    json_path, kmz_path = export_flight_package(
        args.route, args.output, report["mission_id"], report["mission_name"], args.speed
    )
    print(f"Mission JSON: {json_path}")
    print(f"KML/KMZ: {kmz_path}")


if __name__ == "__main__":
    main()
