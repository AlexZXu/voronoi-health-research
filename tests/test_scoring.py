import numpy as np

from app.algorithms.assignment import assign_population
from app.algorithms.scoring import compute_scores
from app.algorithms.types import FacilityData, PopulationData


def test_score_components_include_outside_and_fuzzy_population():
    population = PopulationData(
        ids=["a", "b"],
        names=["a", "b"],
        lat=np.array([0.0, 0.0]),
        lon=np.array([0.0, 1.0]),
        weight=np.array([10.0, 20.0]),
        risk_weight=np.ones(2),
    )
    facilities = FacilityData(
        ids=["f1", "f2"],
        names=["f1", "f2"],
        lat=np.array([0.0, 0.0]),
        lon=np.array([0.0, 0.9]),
        risk_factor=np.ones(2),
        fixed=np.array([False, False]),
    )
    assignments = assign_population(population, facilities)

    scores = compute_scores(
        population,
        facilities,
        assignments,
        acceptable_radius=0.05,
        fuzzy_radius=0.2,
        omega=0.5,
        outside_radius_weight=1.0,
    )

    assert scores[1]["regular_population"] == 20.0
    assert scores[1]["outside_radius_population"] == 20.0
    assert scores[1]["external_fuzzy_population"] == 0.0
    assert scores[1]["score"] == 40.0
