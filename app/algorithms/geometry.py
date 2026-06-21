from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import MultiPoint, Point, Polygon, box, mapping

from .colors import load_ratio_color
from .types import FacilityData


def _finite_voronoi_polygons(vor: Voronoi, radius: float | None = None) -> tuple[list[list[int]], np.ndarray]:
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi input must be 2D.")

    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = float(np.ptp(vor.points, axis=0).max() * 2)

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(vertex >= 0 for vertex in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [vertex for vertex in vertices if vertex >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[vertex] for vertex in new_region])
        centroid = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - centroid[1], vs[:, 0] - centroid[0])
        new_region = [vertex for _, vertex in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def bounds_polygon(bounds: dict[str, float]) -> Polygon:
    return box(bounds["min_lon"], bounds["min_lat"], bounds["max_lon"], bounds["max_lat"])


def build_voronoi_geojson(
    facilities: FacilityData,
    bounds: dict[str, float],
    values: list[float] | np.ndarray | None = None,
    value_label: str = "population",
) -> dict[str, Any]:
    clip_polygon = bounds_polygon(bounds)
    points = np.column_stack([facilities.lon, facilities.lat]).astype(float)
    average = float(np.mean(values)) if values is not None and len(values) else 0.0
    features = []

    if len(points) == 0:
        return {"type": "FeatureCollection", "features": []}

    if len(points) < 3 or MultiPoint(points).minimum_clearance == 0:
        for idx, point in enumerate(points):
            geom = Point(point).buffer(0.01).intersection(clip_polygon)
            value = float(values[idx]) if values is not None and idx < len(values) else 0.0
            features.append(_feature(facilities, idx, geom, value, average, value_label))
        return {"type": "FeatureCollection", "features": features}

    try:
        vor = Voronoi(points)
        regions, vertices = _finite_voronoi_polygons(vor)
    except Exception:
        for idx, point in enumerate(points):
            geom = Point(point).buffer(0.01).intersection(clip_polygon)
            value = float(values[idx]) if values is not None and idx < len(values) else 0.0
            features.append(_feature(facilities, idx, geom, value, average, value_label))
        return {"type": "FeatureCollection", "features": features}

    for idx, region in enumerate(regions[: len(points)]):
        polygon = Polygon(vertices[region])
        clipped = polygon.intersection(clip_polygon)
        if clipped.is_empty:
            continue
        value = float(values[idx]) if values is not None and idx < len(values) else 0.0
        features.append(_feature(facilities, idx, clipped, value, average, value_label))

    return {"type": "FeatureCollection", "features": features}


def _feature(facilities: FacilityData, idx: int, geom, value: float, average: float, value_label: str) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {
            "facility_id": facilities.ids[idx],
            "name": facilities.names[idx],
            value_label: value,
            "color": load_ratio_color(value, average),
        },
        "geometry": mapping(geom),
    }
