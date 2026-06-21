import numpy as np

from app.algorithms.lloyd import initialize_state, step_state
from app.algorithms.types import FacilityData, PopulationData, ScenarioInput


def test_lloyd_moves_unfixed_facility_toward_weighted_centroid():
    population = PopulationData(
        ids=["a", "b"],
        names=["a", "b"],
        lat=np.array([0.0, 0.0]),
        lon=np.array([0.0, 2.0]),
        weight=np.array([1.0, 3.0]),
        risk_weight=np.ones(2),
    )
    facilities = FacilityData(
        ids=["f1"],
        names=["f1"],
        lat=np.array([0.0]),
        lon=np.array([0.0]),
        risk_factor=np.ones(1),
        fixed=np.array([False]),
    )
    bounds = {"min_lat": -1.0, "max_lat": 1.0, "min_lon": -1.0, "max_lon": 3.0}
    params = {"acceptable_radius": 10.0, "fuzzy_radius": 0.0, "learning_rate": 1.0}

    state = initialize_state(ScenarioInput(population, facilities, bounds, params))
    next_state = step_state(population, state, bounds, params)

    assert next_state.facilities.lat[0] == 0.0
    assert next_state.facilities.lon[0] == 1.5
    assert next_state.metrics["total_population"] == 4.0


def test_fixed_facility_does_not_move():
    population = PopulationData(
        ids=["a"],
        names=["a"],
        lat=np.array([0.0]),
        lon=np.array([2.0]),
        weight=np.array([10.0]),
        risk_weight=np.ones(1),
    )
    facilities = FacilityData(
        ids=["f1"],
        names=["f1"],
        lat=np.array([0.0]),
        lon=np.array([0.0]),
        risk_factor=np.ones(1),
        fixed=np.array([True]),
    )
    bounds = {"min_lat": -1.0, "max_lat": 1.0, "min_lon": -1.0, "max_lon": 3.0}
    params = {"acceptable_radius": 10.0, "respect_fixed_facilities": True}

    state = initialize_state(ScenarioInput(population, facilities, bounds, params))
    next_state = step_state(population, state, bounds, params)

    assert next_state.facilities.lon[0] == 0.0
