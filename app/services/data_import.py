from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.algorithms.types import FacilityData, PopulationData


POPULATION_ALIASES = {
    "id": ["id", "zip", "zipcode", "zipcodes", "tract", "geoid"],
    "name": ["name", "label", "area"],
    "lat": ["lat", "latitude", "y"],
    "lon": ["lon", "lng", "longitude", "x"],
    "weight": ["population", "pop", "demand", "weight"],
    "risk_weight": ["risk", "risk_weight", "risk weight"],
}

FACILITY_ALIASES = {
    "id": ["id", "facility_id"],
    "name": ["name", "facility", "clinic", "hospital"],
    "lat": ["lat", "latitude", "y"],
    "lon": ["lon", "lng", "longitude", "x"],
    "risk_factor": ["risk", "risk_factor", "risk factor"],
    "capacity": ["capacity", "cap"],
    "fixed": ["fixed", "locked"],
}


def guess_mapping(columns: list[str], dataset_type: str) -> dict[str, str]:
    aliases = POPULATION_ALIASES if dataset_type == "population" else FACILITY_ALIASES
    normalized = {column.lower().strip().replace("_", " "): column for column in columns}
    mapping: dict[str, str] = {}
    for semantic, names in aliases.items():
        for name in names:
            if name in normalized:
                mapping[semantic] = normalized[name]
                break
    return mapping


def read_population_csv(path: str | Path, mapping: dict[str, str] | None = None) -> PopulationData:
    df = pd.read_csv(path)
    mapping = mapping or guess_mapping(list(df.columns), "population")
    _require(mapping, ["lat", "lon", "weight"], "population")
    errors = validate_numeric_columns(df, mapping, ["lat", "lon", "weight"])
    if errors:
        raise ValueError("; ".join(errors[:5]))

    ids = _string_column(df, mapping.get("id"), prefix="pop")
    names = _string_column(df, mapping.get("name") or mapping.get("id"), prefix="Population")
    risk = _float_column(df, mapping.get("risk_weight"), default=1.0)
    population = PopulationData(
        ids=ids,
        names=names,
        lat=pd.to_numeric(df[mapping["lat"]]).to_numpy(dtype=float),
        lon=pd.to_numeric(df[mapping["lon"]]).to_numpy(dtype=float),
        weight=pd.to_numeric(df[mapping["weight"]]).to_numpy(dtype=float),
        risk_weight=risk,
        metadata=_metadata(df, set(mapping.values())),
    )
    _validate_population(population)
    return population


def read_facilities_csv(path: str | Path, mapping: dict[str, str] | None = None) -> FacilityData:
    df = pd.read_csv(path)
    mapping = mapping or guess_mapping(list(df.columns), "facilities")
    _require(mapping, ["lat", "lon"], "facilities")
    errors = validate_numeric_columns(df, mapping, ["lat", "lon"])
    if errors:
        raise ValueError("; ".join(errors[:5]))

    ids = _string_column(df, mapping.get("id") or mapping.get("name"), prefix="fac")
    names = _string_column(df, mapping.get("name") or mapping.get("id"), prefix="Facility")
    facilities = FacilityData(
        ids=ids,
        names=names,
        lat=pd.to_numeric(df[mapping["lat"]]).to_numpy(dtype=float),
        lon=pd.to_numeric(df[mapping["lon"]]).to_numpy(dtype=float),
        risk_factor=_float_column(df, mapping.get("risk_factor"), default=1.0),
        fixed=_bool_column(df, mapping.get("fixed"), default=False),
        capacity=None if "capacity" not in mapping else _float_column(df, mapping.get("capacity"), default=np.nan),
        metadata=_metadata(df, set(mapping.values())),
    )
    _validate_facilities(facilities)
    return facilities


def validate_numeric_columns(df: pd.DataFrame, mapping: dict[str, str], keys: list[str]) -> list[str]:
    errors: list[str] = []
    for key in keys:
        column = mapping.get(key)
        if not column or column not in df:
            errors.append(f"Missing column for {key}")
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        bad_rows = numeric[numeric.isna()].index.tolist()
        if bad_rows:
            errors.append(f"{column} has non-numeric values at rows {', '.join(str(row + 2) for row in bad_rows[:5])}")
    return errors


def compute_bounds(population: PopulationData, facilities: FacilityData, padding_ratio: float = 0.08) -> dict[str, float]:
    lat = np.concatenate([population.lat, facilities.lat])
    lon = np.concatenate([population.lon, facilities.lon])
    lat_span = max(float(lat.max() - lat.min()), 0.01)
    lon_span = max(float(lon.max() - lon.min()), 0.01)
    return {
        "min_lat": float(lat.min() - lat_span * padding_ratio),
        "max_lat": float(lat.max() + lat_span * padding_ratio),
        "min_lon": float(lon.min() - lon_span * padding_ratio),
        "max_lon": float(lon.max() + lon_span * padding_ratio),
    }


def load_atlanta_sample(data_dir: Path) -> tuple[PopulationData, FacilityData, dict[str, float]]:
    population = read_population_csv(data_dir / "atlanta-zip-coords.csv")
    facilities = read_facilities_csv(data_dir / "atlanta-health-clinics.csv")
    return population, facilities, compute_bounds(population, facilities)


def _require(mapping: dict[str, str], keys: list[str], dataset_type: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{dataset_type} CSV is missing semantic columns: {', '.join(missing)}")


def _string_column(df: pd.DataFrame, column: str | None, prefix: str) -> list[str]:
    if column and column in df:
        return [str(value) for value in df[column].fillna("")]
    return [f"{prefix}-{idx + 1}" for idx in range(len(df))]


def _float_column(df: pd.DataFrame, column: str | None, default: float) -> np.ndarray:
    if column and column in df:
        return pd.to_numeric(df[column], errors="coerce").fillna(default).to_numpy(dtype=float)
    return np.full(len(df), default, dtype=float)


def _bool_column(df: pd.DataFrame, column: str | None, default: bool) -> np.ndarray:
    if not column or column not in df:
        return np.full(len(df), default, dtype=bool)
    return df[column].astype(str).str.lower().isin(["1", "true", "yes", "y", "fixed", "locked"]).to_numpy(dtype=bool)


def _metadata(df: pd.DataFrame, mapped_columns: set[str]) -> list[dict[str, Any]]:
    extra_columns = [column for column in df.columns if column not in mapped_columns]
    return df[extra_columns].fillna("").to_dict(orient="records")


def _validate_population(population: PopulationData) -> None:
    if len(population.ids) == 0:
        raise ValueError("Population dataset must contain at least one row.")
    if np.any((population.lat < -90) | (population.lat > 90)):
        raise ValueError("Population latitude must be between -90 and 90.")
    if np.any((population.lon < -180) | (population.lon > 180)):
        raise ValueError("Population longitude must be between -180 and 180.")
    if np.any(population.weight < 0):
        raise ValueError("Population values must be nonnegative.")


def _validate_facilities(facilities: FacilityData) -> None:
    if len(facilities.ids) == 0:
        raise ValueError("Facility dataset must contain at least one row.")
    if np.any((facilities.lat < -90) | (facilities.lat > 90)):
        raise ValueError("Facility latitude must be between -90 and 90.")
    if np.any((facilities.lon < -180) | (facilities.lon > 180)):
        raise ValueError("Facility longitude must be between -180 and 180.")
