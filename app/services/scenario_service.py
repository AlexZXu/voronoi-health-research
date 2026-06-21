from __future__ import annotations

from typing import Any

from app.algorithms.assignment import assign_population
from app.algorithms.geometry import build_voronoi_geojson
from app.algorithms.lloyd import initialize_state, merged_parameters
from app.algorithms.metrics import compute_metrics
from app.algorithms.scoring import compute_scores
from app.algorithms.types import ScenarioInput
from app.services.serialization import facilities_from_dict, facilities_to_dict, population_to_dict, state_to_dict


def create_scenario_payload(population, facilities, bounds, name: str = "Atlanta sample", parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "population": population_to_dict(population),
        "facilities": facilities_to_dict(facilities),
        "bounds": bounds,
        "parameters": merged_parameters(parameters or {}),
    }


def baseline_payload(scenario_input: ScenarioInput) -> dict[str, Any]:
    params = merged_parameters(scenario_input.parameters)
    assignments = assign_population(scenario_input.population, scenario_input.facilities)
    scores = compute_scores(
        scenario_input.population,
        scenario_input.facilities,
        assignments,
        acceptable_radius=float(params["acceptable_radius"]),
        fuzzy_radius=float(params["fuzzy_radius"]),
        omega=float(params["omega"]),
        outside_radius_weight=float(params["outside_radius_weight"]),
    )
    metrics = compute_metrics(scenario_input.population, assignments, float(params["acceptable_radius"]))
    values = [row["score"] for row in scores]
    return {
        "metrics": metrics,
        "facility_scores": scores,
        "voronoi": build_voronoi_geojson(scenario_input.facilities, scenario_input.bounds, values, "score"),
    }


def scenario_response(scenario: dict[str, Any], scenario_input: ScenarioInput) -> dict[str, Any]:
    baseline = baseline_payload(scenario_input)
    return {
        "id": scenario["id"],
        "name": scenario["name"],
        "bounds": scenario["bounds"],
        "parameters": scenario.get("parameters", {}),
        "population": population_points_response(scenario_input),
        "facilities": facilities_response(scenario_input.facilities),
        "baseline": baseline,
    }


def population_points_response(scenario_input: ScenarioInput) -> list[dict[str, Any]]:
    population = scenario_input.population
    assignments = assign_population(population, scenario_input.facilities)
    return [
        {
            "id": population.ids[idx],
            "name": population.names[idx],
            "lat": float(population.lat[idx]),
            "lon": float(population.lon[idx]),
            "population": float(population.weight[idx]),
            "facility_index": int(assignments.facility_index_by_population[idx]),
        }
        for idx in range(len(population.ids))
    ]


def facilities_response(facilities) -> list[dict[str, Any]]:
    return [
        {
            "id": facilities.ids[idx],
            "name": facilities.names[idx],
            "lat": float(facilities.lat[idx]),
            "lon": float(facilities.lon[idx]),
            "risk_factor": float(facilities.risk_factor[idx]),
            "fixed": bool(facilities.fixed[idx]),
        }
        for idx in range(len(facilities.ids))
    ]


def update_facilities_from_payload(scenario: dict[str, Any], facility_rows: list[dict[str, Any]]) -> dict[str, Any]:
    facilities = facilities_from_dict(scenario["facilities"])
    by_id = {facility_id: idx for idx, facility_id in enumerate(facilities.ids)}
    for row in facility_rows:
        idx = by_id.get(str(row["id"]))
        if idx is None:
            continue
        facilities.lat[idx] = float(row["lat"])
        facilities.lon[idx] = float(row["lon"])
        if "fixed" in row:
            facilities.fixed[idx] = bool(row["fixed"])
        if "risk_factor" in row:
            facilities.risk_factor[idx] = float(row["risk_factor"])
    scenario["facilities"] = facilities_to_dict(facilities)
    return scenario


def initial_run_state(scenario_input: ScenarioInput) -> dict[str, Any]:
    return state_to_dict(initialize_state(scenario_input))
