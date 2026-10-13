from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .environment import AABB


@dataclass(frozen=True)
class ObjMesh:
    """Minimal OBJ geometry required by the planning pipeline."""

    vertices: np.ndarray
    faces: tuple[tuple[int, ...], ...]

    @property
    def minimum(self) -> np.ndarray:
        return self.vertices.min(axis=0)

    @property
    def maximum(self) -> np.ndarray:
        return self.vertices.max(axis=0)


def load_obj(path: str | Path) -> ObjMesh:
    """Read OBJ vertices and polygon faces without requiring a rendering library."""
    vertices: list[list[float]] = []
    faces: list[tuple[int, ...]] = []
    for raw_line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f" and len(parts) >= 4:
            indices = []
            for token in parts[1:]:
                index = int(token.split("/", 1)[0])
                indices.append(index - 1 if index > 0 else len(vertices) + index)
            faces.append(tuple(indices))
    if not vertices or not faces:
        raise ValueError(f"OBJ file has no usable vertices/faces: {path}")
    vertex_array = np.asarray(vertices, dtype=float)
    if any(index < 0 or index >= len(vertex_array) for face in faces for index in face):
        raise ValueError("OBJ face references a vertex outside the file")
    return ObjMesh(vertex_array, tuple(faces))


def mesh_to_column_obstacles(
    mesh: ObjMesh,
    voxel_size: float,
    base_altitude: float | None = None,
    name_prefix: str = "mesh-column",
) -> list[AABB]:
    """Create conservative vertical obstacle columns from a building mesh.

    Every face is projected to an XY voxel range. The highest intersecting face
    defines the column height, and the column is filled down to the scene base.
    This deliberately overestimates occupied space and is intended for building
    or urban-scene meshes, not caves or overhanging terrain.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    minimum = mesh.minimum
    base = float(minimum[2] if base_altitude is None else base_altitude)
    heights: dict[tuple[int, int], float] = {}
    for face in mesh.faces:
        polygon = mesh.vertices[list(face)]
        face_min = polygon.min(axis=0)
        face_max = polygon.max(axis=0)
        ix0 = int(np.floor((face_min[0] - minimum[0]) / voxel_size))
        ix1 = int(np.floor((face_max[0] - minimum[0]) / voxel_size))
        iy0 = int(np.floor((face_min[1] - minimum[1]) / voxel_size))
        iy1 = int(np.floor((face_max[1] - minimum[1]) / voxel_size))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                heights[(ix, iy)] = max(heights.get((ix, iy), base), float(face_max[2]))

    obstacles = []
    for number, ((ix, iy), top) in enumerate(sorted(heights.items())):
        x0 = float(minimum[0] + ix * voxel_size)
        y0 = float(minimum[1] + iy * voxel_size)
        obstacles.append(
            AABB(
                (x0, y0, base),
                (x0 + voxel_size, y0 + voxel_size, max(base + voxel_size, top)),
                f"{name_prefix}-{number}",
            )
        )
    return obstacles


def mesh_to_component_obstacles(mesh: ObjMesh, name_prefix: str = "mesh-component") -> list[AABB]:
    """Collapse disconnected mesh components into conservative bounding volumes."""
    parent = list(range(len(mesh.vertices)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    used: set[int] = set()
    for face in mesh.faces:
        used.update(face)
        for index in face[1:]:
            union(face[0], index)
    components: dict[int, list[int]] = {}
    for index in used:
        components.setdefault(find(index), []).append(index)
    obstacles = []
    for number, indices in enumerate(components.values()):
        points = mesh.vertices[indices]
        obstacles.append(AABB(tuple(points.min(axis=0)), tuple(points.max(axis=0)), f"{name_prefix}-{number}"))
    return obstacles


def write_route_obj(path: str | Path, route: list[np.ndarray]) -> None:
    """Write a route as an OBJ polyline that can be opened with the scene mesh."""
    rows = ["# UAV route polyline"]
    rows.extend(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}" for point in route)
    rows.append("l " + " ".join(str(index) for index in range(1, len(route) + 1)))
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")
