from typing import Any

import numpy as np

from .assignment import assign_population
from .types import AssignmentResult, FacilityData, PopulationData


def compute_scores(
    population: PopulationData,
    facilities: FacilityData,
    assignments: AssignmentResult | None = None,
    acceptable_radius: float = 0.05,
    fuzzy_radius: float = 0.0,
    omega: float = 0.2,
    outside_radius_weight: float = 1.0,
) -> list[dict[str, Any]]:
    if assignments is None:
        assignments = assign_population(population, facilities)

    n_facilities = len(facilities.ids)
    regular = assignments.population_by_facility.astype(float)
    outside = np.zeros(n_facilities, dtype=float)
    external = np.zeros(n_facilities, dtype=float)

    outside_mask = assignments.distance_by_population > acceptable_radius
    if outside_mask.any():
        outside = np.bincount(
            assignments.facility_index_by_population[outside_mask],
            weights=population.weight[outside_mask],
            minlength=n_facilities,
        ).astype(float)

    if fuzzy_radius > 0:
        pop_xy = population.coordinates.astype(float)
        facility_xy = facilities.coordinates.astype(float)
        deltas = pop_xy[:, None, :] - facility_xy[None, :, :]
        distances = np.sqrt(np.sum(deltas * deltas, axis=2))
        fuzzy_hits = distances <= fuzzy_radius
        for point_index, assigned_facility in enumerate(assignments.facility_index_by_population):
            for facility_index in np.where(fuzzy_hits[point_index])[0]:
                if facility_index != assigned_facility:
                    external[facility_index] += population.weight[point_index]

    scores = facilities.risk_factor * (regular + outside_radius_weight * outside + omega * external)

    rows: list[dict[str, Any]] = []
    for idx, facility_id in enumerate(facilities.ids):
        rows.append(
            {
                "facility_id": facility_id,
                "name": facilities.names[idx],
                "regular_population": float(regular[idx]),
                "outside_radius_population": float(outside[idx]),
                "external_fuzzy_population": float(external[idx]),
                "risk_factor": float(facilities.risk_factor[idx]),
                "score": float(scores[idx]),
            }
        )
    return rows
