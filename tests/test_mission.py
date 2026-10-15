import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from uav_planning.flight_export import export_flight_package
from uav_planning.mesh_adapter import load_obj, mesh_to_component_obstacles
from uav_planning.obj_mission import run_obj_mission
from uav_planning.scenarios import load_geojson_scenario
from uav_planning.mission import run_mission
from uav_planning.analysis import maximum_climb_angle, maximum_turn_angle, validate_path
from uav_planning.planners import (
    FlightConstraints, constrained_astar_3d, constrained_rrt_3d,
    constrained_rrt_star_3d, transition_is_valid,
)


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
        self.assertLessEqual(report.maximum_climb_angle_deg, 15.0)
        self.assertLessEqual(report.maximum_turn_angle_deg, 90.0)
        self.assertLessEqual(report.maximum_altitude_m, 50.0)
        self.assertGreaterEqual(report.maximum_altitude_m, 12.0)

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

    def test_constrained_planners_share_and_satisfy_constraints(self):
        environment = load_geojson_scenario("data/cases/hhu_jiangning.geojson")
        environment.safety_margin = 4.0
        constraints = FlightConstraints(50.0, 12.0, 75.0, 15.0, 90.0)
        planners = (
            lambda: constrained_astar_3d(environment, 20.0, 5.0, 50.0, 12.0, 75.0, 15.0, 90.0),
            lambda: constrained_rrt_3d(environment, constraints, 0, 12000, 40.0, 0.25, 50.0),
            lambda: constrained_rrt_star_3d(environment, constraints, 0, 8000, 40.0, 0.20, 50.0, 70.0, 250),
        )
        for planner in planners:
            with self.subTest(planner=planner):
                with patch("uav_planning.planners.transition_is_valid", wraps=transition_is_valid) as shared_check:
                    result = planner()
                self.assertGreater(shared_check.call_count, 0)
                self.assertTrue(result.success, result.message)
                self.assertTrue(validate_path(result.path, environment))
                self.assertLessEqual(max(point[2] for point in result.path), constraints.maximum_altitude)
                self.assertLessEqual(maximum_climb_angle(result.path), constraints.maximum_climb_angle_deg + 1e-6)
                self.assertLessEqual(maximum_turn_angle(result.path), constraints.maximum_turn_angle_deg + 1e-6)
                previous = None
                for start, end in zip(result.path, result.path[1:]):
                    self.assertTrue(transition_is_valid(environment, start, end, previous, constraints))
                    previous = end - start


if __name__ == "__main__":
    unittest.main()

