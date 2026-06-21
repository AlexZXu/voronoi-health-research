from typing import Any

import numpy as np

from .types import AssignmentResult, PopulationData


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(values == 0):
        return 0.0
    sorted_values = np.sort(values)
    n = values.size
    numerator = np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_values)
    denominator = n * np.sum(sorted_values)
    return float(numerator / denominator)


def compute_metrics(
    population: PopulationData,
    assignments: AssignmentResult,
    acceptable_radius: float,
) -> dict[str, Any]:
    loads = assignments.population_by_facility.astype(float)
    total_population = float(np.sum(population.weight))
    mean_load = float(np.mean(loads)) if loads.size else 0.0
    outside_population = float(np.sum(population.weight[assignments.distance_by_population > acceptable_radius]))
    weighted_distance_sum = float(np.sum(population.weight * assignments.distance_by_population))
    mean_weighted_distance = weighted_distance_sum / total_population if total_population else 0.0
    max_weighted_distance = float(np.max(population.weight * assignments.distance_by_population)) if len(population.weight) else 0.0
    load_std = float(np.std(loads)) if loads.size else 0.0

    return {
        "total_population": total_population,
        "facility_count": int(loads.size),
        "mean_load": mean_load,
        "load_variance": float(np.var(loads)) if loads.size else 0.0,
        "load_coefficient_of_variation": load_std / mean_load if mean_load else 0.0,
        "max_load": float(np.max(loads)) if loads.size else 0.0,
        "min_load": float(np.min(loads)) if loads.size else 0.0,
        "outside_radius_population": outside_population,
        "outside_radius_percent": outside_population / total_population if total_population else 0.0,
        "mean_weighted_distance": mean_weighted_distance,
        "max_population_weighted_distance": max_weighted_distance,
        "load_gini": gini(loads),
    }
