"""
tests/test_analysis.py

This module contains comprehensive tests for the functions in the analysis module:
    - spatial_join_and_aggregate()
    - dbscan_clustering()
    - moran_global_local()
    - compute_geometry_stats()

The tests use synthetic GeoDataFrames created in-memory as well as subsets of the test data
generated by create_test_data.py (located in data/test_data). The tests cover general cases,
edge cases, and outlier cases.

To run these tests, execute the following command from the repository root:
    pytest --maxfail=1 --disable-warnings -q
"""

import os
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon

from scripts.analysis import (
    spatial_join_and_aggregate,
    dbscan_clustering,
    moran_global_local,
    compute_geometry_stats
)
import libpysal

# -----------------------------
# Helper functions for synthetic data
# -----------------------------


def create_synthetic_tracts(num=5):
    """
    Create a synthetic GeoDataFrame of census tracts as simple square polygons.

    Each tract will have a unique GEOID and a polygon geometry.
    """
    geoms = []
    geoid_list = []
    for i in range(num):
        # Create a square polygon shifted by i units in x direction.
        poly = Polygon([(i, 0), (i+1, 0), (i+1, 1), (i, 1)])
        geoms.append(poly)
        geoid_list.append(f"TRACT_{i}")
    df = pd.DataFrame({"GEOID": geoid_list})
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
    return gdf


def create_synthetic_monitors(intersect=True):
    """
    Create a synthetic GeoDataFrame for monitors.

    If intersect=True, points will be placed inside the tracts created by create_synthetic_tracts().
    If False, they will be placed far away (so they do not intersect).
    """
    data = {
        "monitor_id": [1, 2, 3, 4],
        "Latitude": [],
        "Longitude": [],
        "value_mean": [10.0, 20.0, 30.0, 40.0],
        "pollutant_name": ["Ozone", "Ozone", "Ozone", "Ozone"],
        "Year": [2024, 2024, 2024, 2024],
    }
    if intersect:
        # Place points inside the first three tracts (squares with x coordinates 0,1,2)
        # Last point deliberately far away
        data["Latitude"] = [0.5, 0.5, 0.5, 10.0]
        data["Longitude"] = [0.5, 1.5, 2.5, 10.0]
    else:
        # All points outside the synthetic tracts (e.g., at coordinates far away)
        data["Latitude"] = [10.0, 10.1, 10.2, 10.3]
        data["Longitude"] = [10.0, 10.1, 10.2, 10.3]
    df = pd.DataFrame(data)
    df["geometry"] = df.apply(lambda row: Point(
        row["Longitude"], row["Latitude"]), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf

# -----------------------------
# Tests for spatial_join_and_aggregate()
# -----------------------------


def test_spatial_join_and_aggregate_valid():
    """Test with valid intersecting monitors and tracts and verify new 'Ozone' column exists."""
    tracts = create_synthetic_tracts(num=5)
    # Create monitors that intersect with first three tracts
    monitors = create_synthetic_monitors(intersect=True)
    # Perform spatial join and aggregation using the "value_mean" column.
    result = spatial_join_and_aggregate(monitors, tracts, col="value_mean")
    # Check that the resulting GeoDataFrame contains a new "Ozone" column.
    assert "Ozone" in result.columns, "'Ozone' column should be present in the aggregated output."
    # Check that tracts that received monitors have the correct mean.
    # For the first tract, monitor at (0.5,0.5) with value 10.0 should be included.
    tract0 = result[result["GEOID"] == "TRACT_0"]
    if not tract0.empty:
        np.testing.assert_almost_equal(tract0.iloc[0]["Ozone"], 10.0)


def test_spatial_join_and_aggregate_no_intersection():
    """Test with monitors that do not intersect any tract.

    The returned GeoDataFrame should have many tracts with NaN in the 'Ozone' column.
    """
    tracts = create_synthetic_tracts(num=5)
    # Create monitors that do not intersect (points placed far away)
    monitors = create_synthetic_monitors(intersect=False)
    result = spatial_join_and_aggregate(monitors, tracts, col="value_mean")
    # Expect that no tract gets a monitor value.
    assert result["Ozone"].isna().all(
    ), "All tracts should have NaN for 'Ozone' when no monitors intersect."


def test_spatial_join_and_aggregate_mismatched_crs():
    """Test that the function reprojects monitors to match tracts if CRS differ."""
    tracts = create_synthetic_tracts(num=5)
    monitors = create_synthetic_monitors(intersect=True)
    # Reproject monitors to a different CRS
    monitors = monitors.to_crs(epsg=3857)
    result = spatial_join_and_aggregate(monitors, tracts, col="value_mean")
    # The output should be in tracts' CRS (EPSG:4326)
    assert result.crs.to_string() == "EPSG:4326", "Resulting CRS should be EPSG:4326."


def test_spatial_join_and_aggregate_invalid_column():
    """Test that providing an invalid column name returns a GeoDataFrame
    where aggregation may fail (resulting in NaNs)."""
    tracts = create_synthetic_tracts(num=5)
    monitors = create_synthetic_monitors(intersect=True)
    result = spatial_join_and_aggregate(monitors, tracts, col="nonexistent")
    # Expect that the "Ozone" column exists but contains all NaNs.
    assert "Ozone" in result.columns, "'Ozone' column must be present even if computed values are NaN."
    assert result["Ozone"].isna().all(
    ), "All values in 'Ozone' should be NaN when using an invalid column."

# -----------------------------
# Tests for dbscan_clustering()
# -----------------------------


def test_dbscan_clustering_valid():
    """Test that dbscan_clustering() adds a 'cluster' column and that clusters are computed."""
    monitors = create_synthetic_monitors(intersect=True)
    result = dbscan_clustering(
        monitors, col="value_mean", eps=0.1, min_samples=1)
    assert "cluster" in result.columns, "Cluster column should be added."
    # When min_samples=1, each point should form its own cluster (if eps is small)
    unique_clusters = set(result["cluster"])
    assert -1 not in unique_clusters, "There should be no noise points when min_samples=1 and eps is small."


def test_dbscan_clustering_eps_zero():
    """Test that eps=0 leads to all points marked as noise."""
    monitors = create_synthetic_monitors(intersect=True)
    result = dbscan_clustering(
        monitors, col="value_mean", eps=0.0, min_samples=2)
    # When eps is zero, no two points are within zero distance, so all should be noise (-1)
    unique_clusters = set(result["cluster"])
    assert unique_clusters == {-1}, "All points should be noise when eps=0."


def test_dbscan_clustering_high_eps():
    """Test that a very high eps results in one cluster."""
    monitors = create_synthetic_monitors(intersect=True)
    result = dbscan_clustering(
        monitors, col="value_mean", eps=1000.0, min_samples=2)
    unique_clusters = set(result["cluster"])
    # Excluding noise, expect one cluster
    non_noise = unique_clusters - {-1}
    assert len(non_noise) == 1, "Should be exactly one cluster with very high eps."


def test_dbscan_clustering_invalid_column():
    """Test that an invalid numeric column raises an error."""
    monitors = create_synthetic_monitors(intersect=True)
    with pytest.raises(AttributeError):
        dbscan_clustering(monitors, col="nonexistent", eps=0.5, min_samples=2)

# -----------------------------
# Tests for moran_global_local()
# -----------------------------


def create_synthetic_tracts_with_values(values, crs="EPSG:4326"):
    """
    Create a synthetic GeoDataFrame of tracts with a given array of pollutant values.
    Each tract is a simple square and a GEOID is assigned.
    """
    num = len(values)
    geoms = []
    geoids = []
    for i in range(num):
        poly = Polygon([(i, 0), (i+1, 0), (i+1, 1), (i, 1)])
        geoms.append(poly)
        geoids.append(f"TRACT_{i}")
    df = pd.DataFrame({"GEOID": geoids, "Ozone": values})
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=crs)
    return gdf


def test_moran_global_local_valid():
    """Test that moran_global_local() computes global Moran's I and assigns LISA clusters."""
    # Create synthetic tracts with varying Ozone values.
    values = [1, 2, 3, 4, 5]
    tracts = create_synthetic_tracts_with_values(values)
    global_I, p_val, clusters = moran_global_local(tracts, col="Ozone")
    # Global Moran's I should be a float; p_val should be between 0 and 1.
    assert isinstance(global_I, float)
    assert 0 <= p_val <= 1
    # Clusters should be a pandas Series of same length as the filtered tracts.
    assert isinstance(clusters, pd.Series)
    # Since values vary, some clusters may not be "Not significant".
    assert len(clusters) == len(
        tracts), "Clusters series length should match number of tracts (after component filtering)."


def test_moran_global_local_constant_values():
    """Test with a GeoDataFrame where the pollutant column is constant (zero variance)."""
    values = [5, 5, 5, 5, 5]
    tracts = create_synthetic_tracts_with_values(values)
    global_I, p_val, clusters = moran_global_local(tracts, col="Ozone")
    # Global Moran's I may be undefined (nan) in zero variance case.
    assert np.isnan(global_I) or global_I == 0


def test_moran_global_local_non_numeric():
    """Test that a non-numeric column causes an error or returns NaN for Moran's I."""
    tracts = create_synthetic_tracts(num=5)
    tracts["Ozone"] = ["a", "b", "c", "d", "e"]
    with pytest.raises(Exception):
        moran_global_local(tracts, col="Ozone")


def test_moran_global_local_missing_values():
    """Test with a GeoDataFrame where some tracts have missing pollutant data."""
    values = [1, 2, None, 4, 5]
    tracts = create_synthetic_tracts_with_values(values)
    # Drop tracts with missing values (simulate realistic behavior)
    tracts = tracts.dropna(subset=["Ozone"])
    global_I, p_val, clusters = moran_global_local(tracts, col="Ozone")
    assert len(clusters) == len(
        tracts), "All tracts with valid data should have an assigned cluster."

# -----------------------------
# Tests for compute_geometry_stats()
# -----------------------------


def test_compute_geometry_stats_valid():
    """Test that compute_geometry_stats() returns a dictionary with correct keys and reasonable values."""
    tracts = create_synthetic_tracts(num=5)
    stats = compute_geometry_stats(tracts)
    expected_keys = {
        "num_features", "avg_vertices", "min_vertices", "max_vertices",
        "min_area_sq_km", "max_area_sq_km", "avg_area_sq_km"
    }
    assert set(
        stats.keys()) == expected_keys, "Returned stats dictionary keys are incorrect."
    assert stats["num_features"] == 5


def test_compute_geometry_stats_mixed_geometries():
    """Test compute_geometry_stats() with a GeoDataFrame that includes both Polygons and MultiPolygons."""
    # Create one simple polygon and one multipolygon
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    multipoly = MultiPolygon([poly1, poly2])
    df = pd.DataFrame({
        "GEOID": ["A", "B"],
    })
    gdf = gpd.GeoDataFrame(df, geometry=[poly1, multipoly], crs="EPSG:4326")
    stats = compute_geometry_stats(gdf)
    # Check that number of features is 2
    assert stats["num_features"] == 2


def test_compute_geometry_stats_empty():
    """Test that an empty GeoDataFrame returns a dictionary with NaN or zero values appropriately."""
    gdf = gpd.GeoDataFrame(columns=["GEOID", "geometry"], crs="EPSG:4326")
    stats = compute_geometry_stats(gdf)
    # For an empty GeoDataFrame, num_features should be 0 and other values may be NaN.
    assert stats["num_features"] == 0

# -----------------------------
# To run these tests, execute:
#    pytest --maxfail=1 --disable-warnings -q
# -----------------------------
