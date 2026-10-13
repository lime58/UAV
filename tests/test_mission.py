import json
import unittest
from pathlib import Path

from uav_planning.scenarios import load_geojson_scenario
from uav_planning.mission import run_mission


class MissionDataTests(unittest.TestCase):
    def test_real_case_metadata_and_obstacles(self):
        case_path = Path("data/cases/hhu_jiangning.geojson")
        data = json.loads(case_path.read_text(encoding="utf-8"))
        self.assertIn("OpenStreetMap", data["properties"]["source"])
        environment = load_geojson_scenario(case_path)
        self.assertGreaterEqual(len(environment.obstacles), 70)
        self.assertTrue(environment.point_is_free(environment.start))
        self.assertTrue(environment.point_is_free(environment.goal))

    def test_end_to_end_route_is_safe(self):
        report, path, _ = run_mission("config/hhu_delivery_mission.json")
        self.assertEqual(report.status, "SUCCESS")
        self.assertTrue(report.collision_free)
        self.assertGreater(report.minimum_obstacle_clearance_m, 4.0)
        self.assertGreaterEqual(len(path), 4)


if __name__ == "__main__":
    unittest.main()

