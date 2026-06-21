import numpy as np

from app.algorithms.assignment import assign_population
from app.algorithms.types import FacilityData, PopulationData


def test_assignment_conserves_population():
    population = PopulationData(
        ids=["a", "b", "c"],
        names=["a", "b", "c"],
        lat=np.array([0.0, 0.0, 10.0]),
        lon=np.array([0.0, 1.0, 10.0]),
        weight=np.array([10.0, 20.0, 30.0]),
        risk_weight=np.ones(3),
    )
    facilities = FacilityData(
        ids=["f1", "f2"],
        names=["f1", "f2"],
        lat=np.array([0.0, 10.0]),
        lon=np.array([0.0, 10.0]),
        risk_factor=np.ones(2),
        fixed=np.array([False, False]),
    )

    result = assign_population(population, facilities)

    assert result.facility_index_by_population.tolist() == [0, 0, 1]
    assert result.population_by_facility.tolist() == [30.0, 30.0]
    assert result.population_by_facility.sum() == population.weight.sum()
