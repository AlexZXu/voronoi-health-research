from __future__ import annotations

from typing import Any

from app.algorithms.geometry import build_voronoi_geojson
from app.algorithms.lloyd import initialize_state, merged_parameters, step_state
from app.algorithms.types import AlgorithmState
from app.services.serialization import facilities_from_dict, population_from_dict, state_to_dict


def create_run_payload(scenario: dict[str, Any], algorithm: str, parameters: dict[str, Any]) -> dict[str, Any]:
    scenario_input = _scenario_input(scenario, parameters)
    state = initialize_state(scenario_input)
    return {
        "scenario_id": scenario["id"],
        "algorithm": algorithm,
        "status": "pending",
        "parameters": merged_parameters(parameters),
        "iterations": [state_to_dict(state)],
        "latest": state_to_dict(state),
    }


def step_run(scenario: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    scenario_input = _scenario_input(scenario, run.get("parameters", {}))
    latest = run["latest"]
    state = _state_from_latest(latest)
    next_state = step_state(scenario_input.population, state, scenario_input.bounds, run.get("parameters", {}))
    serialized = state_to_dict(next_state)
    run["iterations"].append(serialized)
    run["latest"] = serialized
    run["status"] = "complete" if next_state.converged else "running"
    return run


def run_to_convergence(scenario: dict[str, Any], run: dict[str, Any], max_steps: int | None = None) -> dict[str, Any]:
    params = merged_parameters(run.get("parameters", {}))
    limit = max_steps or int(params["max_iterations"])
    for _ in range(limit):
        if run.get("latest", {}).get("converged"):
            break
        run = step_run(scenario, run)
    return run


def run_response(run: dict[str, Any], scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    latest = run["latest"]
    response = {
        "id": run["id"],
        "scenario_id": run["scenario_id"],
        "algorithm": run["algorithm"],
        "status": run["status"],
        "parameters": run["parameters"],
        "latest": latest,
        "iteration_count": len(run.get("iterations", [])),
    }
    if scenario:
        facilities = facilities_from_dict(latest["facilities"])
        values = [row["score"] for row in latest["facility_scores"]]
        response["voronoi"] = build_voronoi_geojson(facilities, scenario["bounds"], values, "score")
    return response


def _scenario_input(scenario: dict[str, Any], parameters: dict[str, Any]):
    from app.algorithms.types import ScenarioInput

    return ScenarioInput(
        population=population_from_dict(scenario["population"]),
        facilities=facilities_from_dict(scenario["facilities"]),
        bounds=scenario["bounds"],
        parameters=merged_parameters(parameters or scenario.get("parameters", {})),
    )


def _state_from_latest(latest: dict[str, Any]) -> AlgorithmState:
    from app.algorithms.types import AssignmentResult
    import numpy as np

    assignments_data = latest["assignments"]
    assignments = AssignmentResult(
        facility_index_by_population=np.asarray(assignments_data["facility_index_by_population"], dtype=int),
        distance_by_population=np.asarray(assignments_data["distance_by_population"], dtype=float),
        squared_distance_by_population=np.asarray(assignments_data["distance_by_population"], dtype=float) ** 2,
        population_by_facility=np.asarray(assignments_data["population_by_facility"], dtype=float),
    )
    return AlgorithmState(
        iteration=int(latest["iteration"]),
        facilities=facilities_from_dict(latest["facilities"]),
        assignments=assignments,
        facility_scores=latest["facility_scores"],
        metrics=latest["metrics"],
        movement_norm=float(latest.get("movement_norm", 0.0)),
        converged=bool(latest.get("converged", False)),
    )
