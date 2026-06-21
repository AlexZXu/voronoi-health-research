from __future__ import annotations

from typing import Any

import numpy as np

from .assignment import assign_population
from .metrics import compute_metrics
from .scoring import compute_scores
from .types import AlgorithmState, FacilityData, PopulationData, ScenarioInput


DEFAULT_PARAMETERS = {
    "max_iterations": 50,
    "tolerance": 1e-5,
    "learning_rate": 1.0,
    "acceptable_radius": 0.05,
    "fuzzy_radius": 0.0,
    "omega": 0.2,
    "outside_radius_weight": 1.0,
    "respect_fixed_facilities": True,
    "empty_cluster_strategy": "keep",
}


def merged_parameters(parameters: dict[str, Any] | None) -> dict[str, Any]:
    merged = DEFAULT_PARAMETERS.copy()
    if parameters:
        merged.update(parameters)
    return merged


def initialize_state(scenario: ScenarioInput) -> AlgorithmState:
    params = merged_parameters(scenario.parameters)
    assignments = assign_population(scenario.population, scenario.facilities)
    scores = compute_scores(
        scenario.population,
        scenario.facilities,
        assignments,
        acceptable_radius=float(params["acceptable_radius"]),
        fuzzy_radius=float(params["fuzzy_radius"]),
        omega=float(params["omega"]),
        outside_radius_weight=float(params["outside_radius_weight"]),
    )
    metrics = compute_metrics(scenario.population, assignments, float(params["acceptable_radius"]))
    return AlgorithmState(
        iteration=0,
        facilities=scenario.facilities,
        assignments=assignments,
        facility_scores=scores,
        metrics=metrics,
    )


def step_state(population: PopulationData, state: AlgorithmState, bounds: dict[str, float], parameters: dict[str, Any]) -> AlgorithmState:
    params = merged_parameters(parameters)
    facilities = state.facilities
    assignments = assign_population(population, facilities)
    target_coords = _weighted_centroids(population, facilities, assignments, params)
    old_coords = facilities.coordinates
    learning_rate = float(params["learning_rate"])
    new_coords = old_coords + learning_rate * (target_coords - old_coords)

    if bool(params["respect_fixed_facilities"]):
        new_coords[facilities.fixed] = old_coords[facilities.fixed]

    new_coords[:, 0] = np.clip(new_coords[:, 0], bounds["min_lat"], bounds["max_lat"])
    new_coords[:, 1] = np.clip(new_coords[:, 1], bounds["min_lon"], bounds["max_lon"])
    movement_by_facility = np.sqrt(np.sum((new_coords - old_coords) ** 2, axis=1))
    movement_norm = float(np.max(movement_by_facility)) if len(movement_by_facility) else 0.0
    next_facilities = facilities.with_coordinates(new_coords)
    next_assignments = assign_population(population, next_facilities)
    scores = compute_scores(
        population,
        next_facilities,
        next_assignments,
        acceptable_radius=float(params["acceptable_radius"]),
        fuzzy_radius=float(params["fuzzy_radius"]),
        omega=float(params["omega"]),
        outside_radius_weight=float(params["outside_radius_weight"]),
    )
    metrics = compute_metrics(population, next_assignments, float(params["acceptable_radius"]))
    iteration = state.iteration + 1
    converged = movement_norm < float(params["tolerance"]) or iteration >= int(params["max_iterations"])

    return AlgorithmState(
        iteration=iteration,
        facilities=next_facilities,
        assignments=next_assignments,
        facility_scores=scores,
        metrics=metrics,
        movement_norm=movement_norm,
        converged=converged,
    )


def _weighted_centroids(population: PopulationData, facilities: FacilityData, assignments, params: dict[str, Any]) -> np.ndarray:
    n_facilities = len(facilities.ids)
    sums = np.zeros((n_facilities, 2), dtype=float)
    weights = np.zeros(n_facilities, dtype=float)
    coords = population.coordinates.astype(float)
    pop_weight = population.weight.astype(float) * population.risk_weight.astype(float)

    for facility_index in range(n_facilities):
        mask = assignments.facility_index_by_population == facility_index
        if mask.any():
            local_weights = pop_weight[mask]
            sums[facility_index] += np.sum(coords[mask] * local_weights[:, None], axis=0)
            weights[facility_index] += np.sum(local_weights)

    acceptable_radius = float(params["acceptable_radius"])
    outside_weight = float(params["outside_radius_weight"])
    if outside_weight > 0:
        outside_mask = assignments.distance_by_population > acceptable_radius
        for facility_index in range(n_facilities):
            mask = outside_mask & (assignments.facility_index_by_population == facility_index)
            if mask.any():
                local_weights = outside_weight * pop_weight[mask]
                sums[facility_index] += np.sum(coords[mask] * local_weights[:, None], axis=0)
                weights[facility_index] += np.sum(local_weights)

    fuzzy_radius = float(params["fuzzy_radius"])
    omega = float(params["omega"])
    if fuzzy_radius > 0 and omega > 0:
        facility_coords = facilities.coordinates.astype(float)
        deltas = coords[:, None, :] - facility_coords[None, :, :]
        distances = np.sqrt(np.sum(deltas * deltas, axis=2))
        for point_index in range(len(population.ids)):
            for facility_index in np.where(distances[point_index] <= fuzzy_radius)[0]:
                if facility_index != assignments.facility_index_by_population[point_index]:
                    weighted = omega * pop_weight[point_index]
                    sums[facility_index] += coords[point_index] * weighted
                    weights[facility_index] += weighted

    centroids = facilities.coordinates.copy()
    non_empty = weights > 0
    centroids[non_empty] = sums[non_empty] / weights[non_empty, None]

    if params["empty_cluster_strategy"] == "largest_error_point":
        empty = np.where(~non_empty)[0]
        largest_error_order = np.argsort(-assignments.distance_by_population * population.weight)
        for idx, facility_index in enumerate(empty):
            if idx < len(largest_error_order):
                centroids[facility_index] = coords[largest_error_order[idx]]

    return centroids
