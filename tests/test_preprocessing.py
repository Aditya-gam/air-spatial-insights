"""
tests/test_preprocessing.py

This module contains comprehensive tests for the functions in the preprocessing module:
    - clean_monitors()
    - filter_pollutant()
    - remove_outliers()
    - apply_boxcox()

The tests use synthetic GeoDataFrames created in-memory as well as test subsets
from the data generated by the create_test_data.py script (located in data/test_data).
The tests cover general cases, edge cases, and scenarios with invalid or missing data.

To run the tests with verbose output, you can either:
    • Run from the repository root:
          pytest -v --maxfail=1 --disable-warnings
    • Execute this file directly:
          python tests/test_preprocessing.py
"""

# Insert the project root into sys.path so that absolute imports work properly.
from scripts.preprocessing import clean_monitors, filter_pollutant, remove_outliers, apply_boxcox
from shapely.geometry import Point, Polygon
import numpy as np
import geopandas as gpd
import pandas as pd
import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


# Import functions to be tested from the preprocessing module.


def create_monitors_gdf(data=None):
    """
    Helper function to create a sample monitors GeoDataFrame with the following columns:
      - monitor_id, Latitude, Longitude, value_mean, pollutant_name, Year, geometry.

    If no data is provided, default values are used.

    Returns:
        GeoDataFrame: A sample GeoDataFrame with test monitor data.
    """
    if data is None:
        data = {
            "monitor_id": [1, 2, 3, 4, 5],
            "Latitude": [37.0, 37.1, 37.2, 37.3, 37.4],
            "Longitude": [-121.0, -121.1, -121.2, -121.3, -121.4],
            "value_mean": [10.0, 15.0, 20.0, 25.0, 30.0],
            "pollutant_name": ["Ozone", "Ozone", "PM2.5 - Local Conditions", "Ozone", "Ozone"],
            "Year": [2024, 2024, 2024, 2023, 2024]
        }
    df = pd.DataFrame(data)
    df["geometry"] = df.apply(lambda row: Point(
        row["Longitude"], row["Latitude"]), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf

# ----------------------------------------
# Tests for clean_monitors()
# ----------------------------------------


def test_clean_monitors_removes_missing_geometry():
    """
    Verify that clean_monitors() drops rows with missing geometry.
    """
    gdf = create_monitors_gdf()
    gdf.loc[0, "geometry"] = None
    cleaned = clean_monitors(gdf.copy())
    assert len(cleaned) == len(gdf) - \
        1, "Rows with missing geometry should be removed."


def test_clean_monitors_removes_duplicates():
    """
    Verify that duplicate rows (based on monitor_id, Latitude, Longitude) are removed.
    """
    gdf = create_monitors_gdf()
    duplicate = gdf.iloc[0].copy()
    gdf = pd.concat([gdf, duplicate.to_frame().T], ignore_index=True)
    cleaned = clean_monitors(gdf.copy())
    assert len(cleaned) == len(gdf) - 1, "Duplicate rows should be removed."


def test_clean_monitors_filters_non_positive_values():
    """
    Verify that rows with non-positive 'value_mean' are filtered out.
    """
    gdf = create_monitors_gdf()
    gdf.loc[1, "value_mean"] = 0.0
    cleaned = clean_monitors(gdf.copy())
    assert 0.0 not in cleaned["value_mean"].values, "Rows with non-positive value_mean should be filtered out."


def test_clean_monitors_missing_value_mean_raises_error():
    """
    Verify that if the 'value_mean' column is missing, clean_monitors() raises a ValueError.
    """
    gdf = create_monitors_gdf()
    gdf = gdf.drop(columns=["value_mean"])
    with pytest.raises(ValueError):
        clean_monitors(gdf.copy())


def test_clean_monitors_empty_gdf_returns_empty():
    """
    Verify that an empty GeoDataFrame remains empty after cleaning.
    """
    gdf = gpd.GeoDataFrame(columns=["monitor_id", "Latitude", "Longitude", "value_mean", "pollutant_name", "Year", "geometry"],
                           crs="EPSG:4326")
    cleaned = clean_monitors(gdf.copy())
    assert cleaned.empty, "Empty GeoDataFrame should remain empty."

# ----------------------------------------
# Tests for filter_pollutant()
# ----------------------------------------


def test_filter_pollutant_case_insensitive():
    """
    Verify that filtering for a pollutant is case-insensitive and trims whitespace.

    After normalization, all remaining records should match the pollutant 'ozone'.
    """
    gdf = create_monitors_gdf()
    gdf["pollutant_name"] = gdf["pollutant_name"].apply(
        lambda x: f"  {x.upper()}  ")
    filtered = filter_pollutant(gdf.copy(), pollutant="ozone", year=2024)
    for val in filtered["pollutant_name"]:
        assert val.strip().lower() == "ozone"


def test_filter_pollutant_no_matching_records():
    """
    Verify that filtering with a pollutant that doesn't exist returns an empty GeoDataFrame.
    """
    gdf = create_monitors_gdf()
    filtered = filter_pollutant(gdf.copy(), pollutant="Nonexistent", year=2024)
    assert filtered.empty, "Should return an empty GeoDataFrame if no records match."

# ----------------------------------------
# Tests for remove_outliers()
# ----------------------------------------


def test_remove_outliers_removes_extreme_values():
    """
    Verify that remove_outliers() removes extreme values outside the specified quantile range.
    """
    gdf = create_monitors_gdf()
    new_row = {
        "monitor_id": 6,
        "Latitude": 37.5,
        "Longitude": -121.5,
        "value_mean": 1000.0,
        "pollutant_name": "Ozone",
        "Year": 2024,
        "geometry": Point(-121.5, 37.5)
    }
    gdf = pd.concat([gdf, pd.DataFrame([new_row])], ignore_index=True)
    filtered = remove_outliers(
        gdf.copy(), col="value_mean", lower_q=0.01, upper_q=0.99)
    assert 1000.0 not in filtered["value_mean"].values, "Extreme outlier should be removed."


def test_remove_outliers_constant_column():
    """
    Verify that if the column values are constant, no rows are removed.
    """
    gdf = create_monitors_gdf()
    gdf["value_mean"] = 50.0
    filtered = remove_outliers(
        gdf.copy(), col="value_mean", lower_q=0.01, upper_q=0.99)
    assert len(filtered) == len(
        gdf), "No rows should be removed if all values are the same."


def test_remove_outliers_invalid_column():
    """
    Verify that an invalid column name causes remove_outliers() to raise a ValueError.
    """
    gdf = create_monitors_gdf()
    with pytest.raises(ValueError):
        remove_outliers(gdf.copy(), col="nonexistent",
                        lower_q=0.01, upper_q=0.99)

# ----------------------------------------
# Tests for apply_boxcox()
# ----------------------------------------


def test_apply_boxcox_valid():
    """
    Verify that apply_boxcox() transforms the column and returns the transformed GeoDataFrame
    along with a lambda value.
    """
    gdf = create_monitors_gdf()
    # Ensure all values are positive by taking absolute value and adding a constant.
    gdf["value_mean"] = gdf["value_mean"].abs() + 1.0
    transformed_gdf, lam = apply_boxcox(gdf.copy(), col="value_mean")
    assert "value_mean_boxcox" in transformed_gdf.columns, "Transformed column should be added."
    assert isinstance(lam, float), "Lambda should be of type float."


def test_apply_boxcox_non_positive_raises():
    """
    Verify that apply_boxcox() raises a ValueError if any value is zero or negative.
    """
    gdf = create_monitors_gdf()
    gdf.loc[0, "value_mean"] = 0.0
    with pytest.raises(ValueError):
        apply_boxcox(gdf.copy(), col="value_mean")


def test_apply_boxcox_invalid_column_raises():
    """
    Verify that applying the Box-Cox transformation on a nonexistent column raises a ValueError.
    """
    gdf = create_monitors_gdf()
    with pytest.raises(ValueError):
        apply_boxcox(gdf.copy(), col="nonexistent")


# ----------------------------------------
# Main block to run tests with verbose output when executed directly.
# ----------------------------------------
if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", "--maxfail=1", "--disable-warnings"]))
