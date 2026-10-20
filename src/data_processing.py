"""Feature engineering for the tractor fuel consumption dataset.

Two raw sources are turned into one row of features each:

* ``process_trajectories`` - one GPS trace per intervention (time, lat, lon,
  speed) becomes duration, distance and speed/acceleration summaries.
* ``process_parcels`` - one polygon per parcel (lat/lon vertices) becomes
  area, perimeter and vertex count.

Distances use ``geopy.distance.geodesic`` rather than a vectorised haversine
so that re-running this module reproduces the values stored in
``data/processed/merged_data.csv`` exactly.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import pyproj
from geopy.distance import geodesic
from shapely.geometry import Polygon
from shapely.ops import transform

TRAJECTORY_COLUMNS = ("Temps", "Latitude", "Longitude", "Vitesse")
PARCEL_COLUMNS = ("Latitude", "Longitude")

# Columns of the raw trajectory files that measure fuel use directly. They are
# the target in another unit and must never reach the feature table.
TARGET_PROXY_COLUMNS = ("Consommation (ml)",)


def _sorted_csvs(folder_path: str) -> list[str]:
    """Return CSV filenames in a deterministic order.

    ``os.listdir`` order is filesystem dependent, which makes row order in the
    output frame vary between machines. Sorting removes that.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"No such folder: {folder_path}")
    return sorted(f for f in os.listdir(folder_path) if f.endswith(".csv"))


def _path_distance_km(latitudes: Iterable[float], longitudes: Iterable[float]) -> float:
    """Total geodesic length of a sequence of points, in kilometres."""
    points = list(zip(latitudes, longitudes))
    return sum(
        geodesic(points[i - 1], points[i]).km for i in range(1, len(points))
    )


def process_trajectories(folder_path: str) -> pd.DataFrame:
    """Summarise every GPS trajectory CSV in ``folder_path``.

    Each file is expected to hold one intervention, named ``<ID>.csv``, with
    the columns in :data:`TRAJECTORY_COLUMNS`. Rows are sorted by time before
    any difference is taken, so a file saved out of chronological order still
    yields correct durations and accelerations.

    Returns
    -------
    pd.DataFrame
        One row per trajectory, keyed by ``ID``. Empty files, files missing
        required columns and fixes without coordinates are skipped.
    """
    records = []

    for filename in _sorted_csvs(folder_path):
        df = pd.read_csv(os.path.join(folder_path, filename))

        missing = set(TRAJECTORY_COLUMNS) - set(df.columns)
        if df.empty or missing:
            continue

        # A fix without coordinates or a timestamp cannot contribute to a
        # distance or a duration (geodesic() raises on NaN).
        df = df.dropna(subset=["Temps", "Latitude", "Longitude"])
        if len(df) < 2:
            continue

        df["Temps"] = pd.to_datetime(df["Temps"], format="%Y-%m-%d %H:%M:%S")
        df = df.sort_values("Temps").reset_index(drop=True)

        # Instantaneous acceleration in km/h per hour.
        elapsed_s = df["Temps"].diff().dt.total_seconds()
        speed_delta = df["Vitesse"].diff()
        acceleration = (speed_delta / elapsed_s.replace(0, np.nan)) * 3600

        duration_min = (
            df["Temps"].iloc[-1] - df["Temps"].iloc[0]
        ).total_seconds() / 60

        records.append(
            {
                "ID": os.path.splitext(filename)[0],
                "Duree_mn": duration_min,
                "Distance_km": _path_distance_km(df["Latitude"], df["Longitude"]),
                "Vitesse_moy_kmph": df["Vitesse"].mean(),
                "Vitesse_med_kmph": df["Vitesse"].median(),
                "Vitesse_max_kmph": df["Vitesse"].max(),
                "Accélération_moy_kmph2": acceleration.mean(),
                "Accélération_max_kmph2": acceleration.max(),
            }
        )

    features = pd.DataFrame(records)
    leaked = set(TARGET_PROXY_COLUMNS) & set(features.columns)
    if leaked:
        raise ValueError(f"target proxy column(s) in features: {sorted(leaked)}")
    return features


def _area_and_perimeter(latitudes: list[float], longitudes: list[float]) -> tuple[float, float]:
    """Projected area (m²) and perimeter (m) of a lat/lon polygon.

    The polygon is reprojected from WGS84 to the local UTM zone before
    measuring, because degrees are not a unit of area. ``buffer(0)`` repairs
    self-intersecting rings, which otherwise report a nonsensical area.
    """
    polygon = Polygon(zip(longitudes, latitudes))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area == 0:
        raise ValueError("degenerate parcel outline: zero area after repair")

    centroid = polygon.centroid
    utm_zone = int((centroid.x + 180) // 6) + 1
    epsg = 32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone

    project = pyproj.Transformer.from_crs(
        pyproj.CRS("EPSG:4326"), pyproj.CRS(f"EPSG:{epsg}"), always_xy=True
    ).transform
    projected = transform(project, polygon)

    return projected.area, projected.length


def process_parcels(folder_path: str) -> pd.DataFrame:
    """Summarise every parcel boundary CSV in ``folder_path``.

    Returns
    -------
    pd.DataFrame
        One row per parcel with ``Surface_ha``, ``Perimetre_km`` and
        ``Complexite`` (the number of boundary vertices, a crude proxy for how
        irregular the parcel outline is). Polygons with fewer than three
        vertices are skipped, and so are outlines with zero area once repaired
        (with a warning).
    """
    records = []

    for filename in _sorted_csvs(folder_path):
        df = pd.read_csv(os.path.join(folder_path, filename))

        if set(PARCEL_COLUMNS) - set(df.columns):
            continue
        df = df.dropna(subset=list(PARCEL_COLUMNS))
        if len(df) < 3:
            continue

        latitudes = df["Latitude"].tolist()
        longitudes = df["Longitude"].tolist()
        try:
            area_m2, perimeter_m = _area_and_perimeter(latitudes, longitudes)
        except ValueError as exc:
            warnings.warn(f"{filename}: {exc}; parcel skipped")
            continue

        records.append(
            {
                "Parcelle": os.path.splitext(filename)[0],
                "Surface_ha": area_m2 / 10_000,
                "Perimetre_km": perimeter_m / 1_000,
                "Complexite": len(latitudes),
            }
        )

    return pd.DataFrame(records)
