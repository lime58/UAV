from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from uav_planning.mission import export_mission, run_mission


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an end-to-end UAV route-planning mission")
    parser.add_argument("--config", default="config/hhu_delivery_mission.json")
    args = parser.parse_args()
    report, path, raw = run_mission(args.config)
    output = export_mission(args.config, report, path, raw)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()

