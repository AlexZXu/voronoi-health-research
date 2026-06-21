from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PopulationData:
    ids: list[str]
    names: list[str]
    lat: np.ndarray
    lon: np.ndarray
    weight: np.ndarray
    risk_weight: np.ndarray
    metadata: list[dict[str, Any]] = field(default_factory=list)

    @property
    def coordinates(self) -> np.ndarray:
        return np.column_stack([self.lat, self.lon])


@dataclass
class FacilityData:
    ids: list[str]
    names: list[str]
    lat: np.ndarray
    lon: np.ndarray
    risk_factor: np.ndarray
    fixed: np.ndarray
    capacity: np.ndarray | None = None
    metadata: list[dict[str, Any]] = field(default_factory=list)

    @property
    def coordinates(self) -> np.ndarray:
        return np.column_stack([self.lat, self.lon])

    def with_coordinates(self, coordinates: np.ndarray) -> "FacilityData":
        return FacilityData(
            ids=list(self.ids),
            names=list(self.names),
            lat=coordinates[:, 0].astype(float),
            lon=coordinates[:, 1].astype(float),
            risk_factor=self.risk_factor.copy(),
            fixed=self.fixed.copy(),
            capacity=None if self.capacity is None else self.capacity.copy(),
            metadata=list(self.metadata),
        )


@dataclass
class AssignmentResult:
    facility_index_by_population: np.ndarray
    distance_by_population: np.ndarray
    squared_distance_by_population: np.ndarray
    population_by_facility: np.ndarray


@dataclass
class AlgorithmState:
    iteration: int
    facilities: FacilityData
    assignments: AssignmentResult
    facility_scores: list[dict[str, Any]]
    metrics: dict[str, Any]
    movement_norm: float = 0.0
    converged: bool = False


@dataclass
class ScenarioInput:
    population: PopulationData
    facilities: FacilityData
    bounds: dict[str, float]
    parameters: dict[str, Any]
