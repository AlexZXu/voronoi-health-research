from pathlib import Path

from app.services.data_import import load_atlanta_sample


def test_atlanta_sample_imports():
    root = Path(__file__).resolve().parent.parent
    population, facilities, bounds = load_atlanta_sample(root / "data_sets")

    assert len(population.ids) == 36
    assert len(facilities.ids) == 10
    assert population.weight.sum() > 0
    assert bounds["min_lat"] < bounds["max_lat"]
    assert bounds["min_lon"] < bounds["max_lon"]
