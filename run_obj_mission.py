from __future__ import annotations

import argparse

from uav_planning.obj_mission import run_obj_mission


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an OBJ-backed 3D UAV planning mission")
    parser.add_argument("--config", default="config/obj_demo_mission.json")
    args = parser.parse_args()
    report, _, output = run_obj_mission(args.config)
    print(f"{report.status}: {report.waypoint_count} waypoints, {report.route_length_m:.2f} m")
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
