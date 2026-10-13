import json
import tempfile
import unittest
from pathlib import Path

from uav_planning.flight_export import export_flight_package
from uav_planning.mesh_adapter import load_obj, mesh_to_component_obstacles
from uav_planning.obj_mission import run_obj_mission
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

    def test_obj_model_pipeline(self):
        mesh = load_obj("data/cases/obj_building_demo.obj")
        self.assertEqual(len(mesh.vertices), 24)
        self.assertEqual(len(mesh_to_component_obstacles(mesh)), 3)
        report, route, _ = run_obj_mission("config/obj_demo_mission.json")
        self.assertEqual(report.status, "SUCCESS")
        self.assertTrue(report.collision_free)
        self.assertGreaterEqual(len(route), 3)

    def test_flight_package_export(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path, kmz_path = export_flight_package(
                "outputs/hhu_delivery_mission/route_wgs84.geojson",
                directory,
                "TEST-001",
                "测试航线",
            )
            mission = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(mission["schema"], "uav-mission-interchange/v1")
            self.assertGreaterEqual(len(mission["waypoints"]), 4)
            self.assertTrue(kmz_path.exists())


if __name__ == "__main__":
    unittest.main()

