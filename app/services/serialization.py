from __future__ import annotations

from typing import Any

import numpy as np

from app.algorithms.types import AlgorithmState, FacilityData, PopulationData, ScenarioInput


def population_to_dict(population: PopulationData) -> dict[str, Any]:
    return {
        "ids": population.ids,
        "names": population.names,
        "lat": population.lat.tolist(),
        "lon": population.lon.tolist(),
        "weight": population.weight.tolist(),
        "risk_weight": population.risk_weight.tolist(),
        "metadata": population.metadata,
    }


def population_from_dict(data: dict[str, Any]) -> PopulationData:
    return PopulationData(
        ids=list(data["ids"]),
        names=list(data["names"]),
        lat=np.asarray(data["lat"], dtype=float),
        lon=np.asarray(data["lon"], dtype=float),
        weight=np.asarray(data["weight"], dtype=float),
        risk_weight=np.asarray(data.get("risk_weight", [1.0] * len(data["ids"])), dtype=float),
        metadata=list(data.get("metadata", [{} for _ in data["ids"]])),
    )


def facilities_to_dict(facilities: FacilityData) -> dict[str, Any]:
    return {
        "ids": facilities.ids,
        "names": facilities.names,
        "lat": facilities.lat.tolist(),
        "lon": facilities.lon.tolist(),
        "risk_factor": facilities.risk_factor.tolist(),
        "fixed": facilities.fixed.astype(bool).tolist(),
        "capacity": None if facilities.capacity is None else facilities.capacity.tolist(),
        "metadata": facilities.metadata,
    }


def facilities_from_dict(data: dict[str, Any]) -> FacilityData:
    return FacilityData(
        ids=list(data["ids"]),
        names=list(data["names"]),
        lat=np.asarray(data["lat"], dtype=float),
        lon=np.asarray(data["lon"], dtype=float),
        risk_factor=np.asarray(data.get("risk_factor", [1.0] * len(data["ids"])), dtype=float),
        fixed=np.asarray(data.get("fixed", [False] * len(data["ids"])), dtype=bool),
        capacity=None if data.get("capacity") is None else np.asarray(data["capacity"], dtype=float),
        metadata=list(data.get("metadata", [{} for _ in data["ids"]])),
    )


def scenario_to_input(scenario: dict[str, Any]) -> ScenarioInput:
    return ScenarioInput(
        population=population_from_dict(scenario["population"]),
        facilities=facilities_from_dict(scenario["facilities"]),
        bounds=scenario["bounds"],
        parameters=scenario.get("parameters", {}),
    )


def state_to_dict(state: AlgorithmState) -> dict[str, Any]:
    return {
        "iteration": state.iteration,
        "facilities": facilities_to_dict(state.facilities),
        "assignments": {
            "facility_index_by_population": state.assignments.facility_index_by_population.tolist(),
            "distance_by_population": state.assignments.distance_by_population.tolist(),
            "population_by_facility": state.assignments.population_by_facility.tolist(),
        },
        "facility_scores": state.facility_scores,
        "metrics": state.metrics,
        "movement_norm": state.movement_norm,
        "converged": state.converged,
    }
