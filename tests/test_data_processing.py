import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_processing import process_parcels, process_trajectories  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")


def _write_trace(folder, name, rows):
    pd.DataFrame(rows, columns=["Temps", "Latitude", "Longitude", "Vitesse",
                                "Consommation (ml)"]).to_csv(folder / name, index=False)


def test_trajectory_summary_and_time_ordering(tmp_path):
    # Rows saved out of order: durations and accelerations must not care.
    _write_trace(tmp_path, "1.csv", [
        ("2024-01-01 08:10:00", 34.001, -5.0, 10, 30.0),
        ("2024-01-01 08:00:00", 34.000, -5.0, 0, 0.0),
        ("2024-01-01 08:20:00", 34.002, -5.0, 10, 30.0),
    ])
    out = process_trajectories(str(tmp_path))

    row = out.iloc[0]
    assert row["ID"] == "1"
    assert row["Duree_mn"] == pytest.approx(20)
    assert row["Distance_km"] == pytest.approx(0.222, abs=0.002)   # 0.002° of latitude
    assert row["Vitesse_max_kmph"] == 10
    assert row["Accélération_max_kmph2"] == pytest.approx(60)      # +10 km/h in 1/6 h


def test_fixes_without_coordinates_are_ignored(tmp_path):
    _write_trace(tmp_path, "2.csv", [
        ("2024-01-01 08:00:00", 34.000, -5.0, 5, 0.0),
        ("2024-01-01 08:05:00", np.nan, np.nan, 5, 0.0),
        ("2024-01-01 08:10:00", 34.001, -5.0, 5, 0.0),
    ])
    out = process_trajectories(str(tmp_path))
    assert out.iloc[0]["Distance_km"] == pytest.approx(0.111, abs=0.002)


def test_fuel_column_never_becomes_a_feature(tmp_path):
    _write_trace(tmp_path, "3.csv", [
        ("2024-01-01 08:00:00", 34.000, -5.0, 5, 10.0),
        ("2024-01-01 08:05:00", 34.001, -5.0, 5, 10.0),
    ])
    out = process_trajectories(str(tmp_path))
    assert not any("Consommation" in c for c in out.columns)


def test_parcel_area_of_a_known_square(tmp_path):
    # ~100 m x ~100 m square near the farm's latitude.
    lat0, lon0 = 34.35, -5.95
    dlat = 100 / 111_000
    dlon = 100 / (111_000 * np.cos(np.radians(lat0)))
    pd.DataFrame({"Latitude": [lat0, lat0, lat0 + dlat, lat0 + dlat],
                  "Longitude": [lon0, lon0 + dlon, lon0 + dlon, lon0]}).to_csv(
        tmp_path / "sq.csv", index=False)

    row = process_parcels(str(tmp_path)).iloc[0]
    assert row["Surface_ha"] == pytest.approx(1.0, rel=0.02)
    assert row["Perimetre_km"] == pytest.approx(0.4, rel=0.02)
    assert row["Complexite"] == 4


def test_degenerate_parcel_is_skipped_with_a_warning(tmp_path):
    pd.DataFrame({"Latitude": [34.0, 34.0, 34.0],
                  "Longitude": [-5.0, -5.001, -5.002]}).to_csv(tmp_path / "line.csv", index=False)
    with pytest.warns(UserWarning, match="degenerate"):
        out = process_parcels(str(tmp_path))
    assert out.empty


def test_pipeline_reproduces_the_stored_feature_table():
    stored = pd.read_csv(os.path.join(ROOT, "data", "processed", "merged_data.csv"))
    stored["ID"] = stored["ID"].astype(str)
    recomputed = process_trajectories(os.path.join(RAW, "trajets_train"))
    check = stored.merge(recomputed, on="ID", suffixes=("_stored", "_new"))

    assert len(check) == len(recomputed) > 0
    for col in ["Duree_mn", "Distance_km", "Vitesse_moy_kmph", "Accélération_max_kmph2"]:
        np.testing.assert_allclose(check[f"{col}_stored"], check[f"{col}_new"], atol=1e-9)
