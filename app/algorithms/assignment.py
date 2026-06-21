import numpy as np

from .types import AssignmentResult, FacilityData, PopulationData


def assign_population(population: PopulationData, facilities: FacilityData) -> AssignmentResult:
    if len(facilities.ids) == 0:
        raise ValueError("At least one facility is required for assignment.")

    pop_xy = population.coordinates.astype(float)
    facility_xy = facilities.coordinates.astype(float)

    deltas = pop_xy[:, None, :] - facility_xy[None, :, :]
    squared_distances = np.sum(deltas * deltas, axis=2)
    facility_index = np.argmin(squared_distances, axis=1)
    squared_distance = squared_distances[np.arange(len(pop_xy)), facility_index]
    distance = np.sqrt(squared_distance)
    population_by_facility = np.bincount(
        facility_index,
        weights=population.weight,
        minlength=len(facilities.ids),
    ).astype(float)

    return AssignmentResult(
        facility_index_by_population=facility_index,
        distance_by_population=distance,
        squared_distance_by_population=squared_distance,
        population_by_facility=population_by_facility,
    )
