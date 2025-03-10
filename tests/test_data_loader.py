"""
tests/test_data_loader.py

This module contains comprehensive tests for the data_loader module using pytest.
It tests load_census_tracts() and load_monitors() using valid input files as well
as edge and error cases. Test files are assumed to be created in 'data/test_data'
by the create_test_data.py script.
"""
from scripts.data_loader import load_census_tracts, load_monitors
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


# Define the directory for test data (created by create_test_data.py)
TEST_DATA_DIR = os.path.join("data", "test_data")


# ----------------------------------------
# Tests for load_census_tracts
# ----------------------------------------

def test_load_census_tracts_valid():
    """Test loading a valid census tract shapefile."""
    valid_path = os.path.join(TEST_DATA_DIR, "tracts_valid.shp")
    gdf = load_census_tracts(valid_path, expected_crs="EPSG:4326")
    # Check that the returned GeoDataFrame is not empty and has expected CRS.
    assert not gdf.empty, "GeoDataFrame should not be empty."
    assert gdf.crs.to_string() == "EPSG:4326", "CRS should be EPSG:4326."


def test_load_census_tracts_diff_crs():
    """
    Test loading a shapefile that is in a different CRS.
    The function should reproject it to EPSG:4326.
    """
    diff_crs_path = os.path.join(TEST_DATA_DIR, "tracts_diff_crs.shp")
    gdf = load_census_tracts(diff_crs_path, expected_crs="EPSG:4326")
    # The returned GeoDataFrame should be reprojected to EPSG:4326.
    assert gdf.crs.to_string() == "EPSG:4326", "CRS should be reprojected to EPSG:4326."


def test_load_census_tracts_file_not_found():
    """Test that a non-existent file raises a FileNotFoundError."""
    non_existent_path = os.path.join(TEST_DATA_DIR, "nonexistent.shp")
    with pytest.raises(FileNotFoundError):
        load_census_tracts(non_existent_path)


def test_load_census_tracts_invalid_file(tmp_path):
    """
    Test that passing a file that is not a valid shapefile raises an exception.
    We create a temporary text file.
    """
    bad_file = tmp_path / "bad_file.txt"
    bad_file.write_text("This is not a shapefile")
    with pytest.raises(FileNotFoundError):
        load_census_tracts(str(bad_file))


def test_load_census_tracts_empty_file():
    """
    Test that an empty shapefile raises a ValueError.
    We use the test subset 'tracts_empty.shp' created by create_test_data.py.
    """
    empty_path = os.path.join(TEST_DATA_DIR, "tracts_empty.shp")
    with pytest.raises(ValueError):
        load_census_tracts(empty_path)


def test_load_census_tracts_no_crs():
    """
    Test that a shapefile with no CRS defined gets its CRS set to EPSG:4326.
    """
    no_crs_path = os.path.join(TEST_DATA_DIR, "tracts_no_crs.shp")
    gdf = load_census_tracts(no_crs_path, expected_crs="EPSG:4326")
    # The function should have set the CRS to EPSG:4326.
    assert gdf.crs is not None, "CRS should be set."
    assert gdf.crs.to_string() == "EPSG:4326", "CRS should be EPSG:4326."


def test_load_census_tracts_wrong_format():
    """
    Test that passing a file in the wrong format (e.g., a CSV instead of a shapefile)
    results in an error.
    """
    wrong_format_path = os.path.join(TEST_DATA_DIR, "tracts_no_geometry.csv")
    with pytest.raises(Exception):
        load_census_tracts(wrong_format_path)


# ----------------------------------------
# Tests for load_monitors
# ----------------------------------------

def test_load_monitors_valid():
    """Test loading a valid monitors CSV file."""
    valid_path = os.path.join(TEST_DATA_DIR, "monitors_valid.csv")
    gdf = load_monitors(valid_path, expected_crs="EPSG:4326")
    # Check that the returned GeoDataFrame is not empty and has a geometry column.
    assert not gdf.empty, "GeoDataFrame should not be empty."
    assert "geometry" in gdf.columns, "GeoDataFrame must contain a geometry column."
    assert gdf.crs.to_string() == "EPSG:4326", "CRS should be EPSG:4326."


def test_load_monitors_file_not_found():
    """Test that a non-existent CSV file raises a FileNotFoundError."""
    non_existent_path = os.path.join(TEST_DATA_DIR, "nonexistent.csv")
    with pytest.raises(FileNotFoundError):
        load_monitors(non_existent_path)


def test_load_monitors_missing_required_columns():
    """
    Test that a CSV file missing required columns (e.g., Latitude/Longitude)
    raises a ValueError.
    """
    missing_cols_path = os.path.join(
        TEST_DATA_DIR, "monitors_missing_cols.csv")
    with pytest.raises(ValueError):
        load_monitors(missing_cols_path)


def test_load_monitors_out_of_range():
    """
    Test that a CSV with out-of-range coordinate values drops those rows.
    We compare the number of rows in the returned GeoDataFrame
    with the original valid subset.
    """
    valid_path = os.path.join(TEST_DATA_DIR, "monitors_valid.csv")
    gdf_valid = load_monitors(valid_path, expected_crs="EPSG:4326")
    out_of_range_path = os.path.join(
        TEST_DATA_DIR, "monitors_out_of_range.csv")
    gdf_out = load_monitors(out_of_range_path, expected_crs="EPSG:4326")
    # Since two rows were set to invalid in the test data, the returned number should be lower.
    assert len(gdf_out) < len(gdf_valid.head(10)), (
        "Rows with out-of-range coordinates should be dropped.")


def test_load_monitors_empty():
    """
    Test that an empty CSV file results in a ValueError due to empty GeoDataFrame.
    """
    empty_path = os.path.join(TEST_DATA_DIR, "monitors_empty.csv")
    with pytest.raises(ValueError):
        load_monitors(empty_path)


def test_load_monitors_malformed(tmp_path):
    """
    Test that a malformed CSV file raises an exception.
    We use the malformed CSV created in test data.
    """
    malformed_path = os.path.join(TEST_DATA_DIR, "monitors_malformed.csv")
    with pytest.raises(Exception):
        load_monitors(malformed_path)


def test_load_monitors_diff_crs():
    """
    Test that a monitors CSV whose coordinates are in a different CRS
    is reprojected to the expected CRS (EPSG:4326).
    """
    diff_crs_path = os.path.join(TEST_DATA_DIR, "monitors_diff_crs.csv")
    gdf = load_monitors(diff_crs_path, expected_crs="EPSG:4326")
    # The returned GeoDataFrame should have CRS EPSG:4326.
    assert gdf.crs.to_string() == "EPSG:4326", "CRS should be EPSG:4326 after reprojection."


# ----------------------------------------
# Command to run these tests:
# In your repository root, run:
#     pytest --maxfail=1 --disable-warnings -q
# or simply:
#     pytest
# ----------------------------------------
