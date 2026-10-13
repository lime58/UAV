from __future__ import annotations

import json
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
from zipfile import ZIP_DEFLATED, ZipFile


def _route_coordinates(route_geojson: dict) -> list[list[float]]:
    for feature in route_geojson.get("features", []):
        geometry = feature.get("geometry", {})
        if geometry.get("type") == "LineString":
            coordinates = geometry.get("coordinates", [])
            if len(coordinates) < 2:
                raise ValueError("Route LineString must contain at least two coordinates")
            for lon, lat, altitude in coordinates:
                if not (-180 <= lon <= 180 and -90 <= lat <= 90 and altitude >= 0):
                    raise ValueError("Route contains an invalid WGS84 waypoint")
            return coordinates
    raise ValueError("No LineString route found in GeoJSON")


def export_flight_package(
    route_path: str | Path,
    output_directory: str | Path,
    mission_id: str,
    mission_name: str,
    cruise_speed_mps: float = 8.0,
) -> tuple[Path, Path]:
    """Export a vendor-neutral mission JSON and standard KML/KMZ package.

    The package is intentionally not sent to a live aircraft. Vendor adapters
    can translate this validated interchange model to a specific SDK later.
    """
    if cruise_speed_mps <= 0:
        raise ValueError("cruise_speed_mps must be positive")
    route_json = json.loads(Path(route_path).read_text(encoding="utf-8"))
    coordinates = _route_coordinates(route_json)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    waypoints = []
    for index, (lon, lat, altitude) in enumerate(coordinates):
        phase = "TAKEOFF" if index == 0 else "LAND" if index == len(coordinates) - 1 else "TRANSIT"
        waypoints.append({
            "sequence": index,
            "longitude": lon,
            "latitude": lat,
            "relative_altitude_m": altitude,
            "speed_mps": cruise_speed_mps,
            "phase": phase,
            "actions": [],
        })
    mission = {
        "schema": "uav-mission-interchange/v1",
        "mission_id": mission_id,
        "mission_name": mission_name,
        "coordinate_reference_system": "EPSG:4326",
        "altitude_reference": "relative",
        "execution": {"mode": "waypoint", "breakpoint_resume": True},
        "waypoints": waypoints,
    }
    json_path = output / "flight_mission.json"
    json_path.write_text(json.dumps(mission, ensure_ascii=False, indent=2), encoding="utf-8")

    kml = Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = SubElement(kml, "Document")
    SubElement(document, "name").text = mission_name
    placemark = SubElement(document, "Placemark")
    SubElement(placemark, "name").text = mission_id
    line = SubElement(placemark, "LineString")
    SubElement(line, "altitudeMode").text = "relativeToGround"
    SubElement(line, "coordinates").text = " ".join(
        f"{lon:.8f},{lat:.8f},{altitude:.3f}" for lon, lat, altitude in coordinates
    )
    kml_bytes = b'<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(kml, encoding="utf-8")
    kml_path = output / "flight_route.kml"
    kml_path.write_bytes(kml_bytes)
    kmz_path = output / "flight_route.kmz"
    with ZipFile(kmz_path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("doc.kml", kml_bytes)
    return json_path, kmz_path
