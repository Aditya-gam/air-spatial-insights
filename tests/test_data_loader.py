"""
tests/test_data_loader.py

This module contains comprehensive tests for the data_loader module using pytest.
It tests the functions load_census_tracts() and load_monitors() using valid input files
as well as edge and error cases. Test data is assumed to be created in 'data/test_data'
by the create_test_data.py script.

To run the tests with verbose output, you can either:
    • Run from the repository root:
          pytest -v --maxfail=1 --disable-warnings
    • Execute this file directly:
          python tests/test_data_loader.py
"""

# Insert the project root into sys.path so that absolute imports work properly.
from scripts.data_loader import load_census_tracts, load_monitors
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


# Import functions to be tested from the data_loader module.

# Define the directory for test data (created by create_test_data.py)
TEST_DATA_DIR = os.path.join("data", "test_data")

# ----------------------------------------
# Test Cases for load_census_tracts
# ----------------------------------------


def test_load_census_tracts_valid():
    """
    Test loading a valid census tract shapefile.

    Verifies that:
      - The returned GeoDataFrame is not empty.
      - The CRS is set to the expected 'EPSG:4326'.
    """
    valid_path = os.path.join(TEST_DATA_DIR, "tracts_valid.shp")
    gdf = load_census_tracts(valid_path, expected_crs="EPSG:4326")
    assert not gdf.empty, "GeoDataFrame should not be empty."
    assert gdf.crs.to_string() == "EPSG:4326", "CRS should be EPSG:4326."


def test_load_census_tracts_diff_crs():
    """
    Test loading a shapefile that is in a different CRS.

    The function should reproject the GeoDataFrame to EPSG:4326.
    """
    diff_crs_path = os.path.join(TEST_DATA_DIR, "tracts_diff_crs.shp")
    gdf = load_census_tracts(diff_crs_path, expected_crs="EPSG:4326")
    assert gdf.crs.to_string() == "EPSG:4326", "CRS should be reprojected to EPSG:4326."


def test_load_census_tracts_file_not_found():
    """
    Test that a non-existent shapefile raises a FileNotFoundError.
    """
    non_existent_path = os.path.join(TEST_DATA_DIR, "nonexistent.shp")
    with pytest.raises(FileNotFoundError):
        load_census_tracts(non_existent_path)


def test_load_census_tracts_invalid_file(tmp_path):
    """
    Test that passing an invalid file (not a shapefile) raises an error.

    A temporary text file is created to simulate an invalid shapefile.
    """
    bad_file = tmp_path / "bad_file.txt"
    bad_file.write_text("This is not a shapefile")
    with pytest.raises(FileNotFoundError):
        load_census_tracts(str(bad_file))


def test_load_census_tracts_empty_file():
    """
    Test that an empty shapefile raises a ValueError.
    """
    empty_path = os.path.join(TEST_DATA_DIR, "tracts_empty.shp")
    with pytest.raises(ValueError):
        load_census_tracts(empty_path)


def test_load_census_tracts_no_crs():
    """
    Test that a shapefile with no defined CRS is assigned 'EPSG:4326'.
    """
    no_crs_path = os.path.join(TEST_DATA_DIR, "tracts_no_crs.shp")
    gdf = load_census_tracts(no_crs_path, expected_crs="EPSG:4326")
    assert gdf.crs is not None, "CRS should be set."
    assert gdf.crs.to_string() == "EPSG:4326", "CRS should be EPSG:4326."


def test_load_census_tracts_wrong_format():
    """
    Test that providing a file in the wrong format (e.g., CSV instead of shapefile)
    results in an error.
    """
    wrong_format_path = os.path.join(TEST_DATA_DIR, "tracts_no_geometry.csv")
    with pytest.raises(Exception):
        load_census_tracts(wrong_format_path)

# ----------------------------------------
# Test Cases for load_monitors
# ----------------------------------------


def test_load_monitors_valid():
    """
    Test loading a valid monitors CSV file.

    Checks that:
      - The returned GeoDataFrame is not empty.
      - A geometry column is present.
      - The CRS is set to EPSG:4326.
    """
    valid_path = os.path.join(TEST_DATA_DIR, "monitors_valid.csv")
    gdf = load_monitors(valid_path, expected_crs="EPSG:4326")
    assert not gdf.empty, "GeoDataFrame should not be empty."
    assert "geometry" in gdf.columns, "GeoDataFrame must contain a geometry column."
    assert gdf.crs.to_string() == "EPSG:4326", "CRS should be EPSG:4326."


def test_load_monitors_file_not_found():
    """
    Test that a non-existent CSV file raises a FileNotFoundError.
    """
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
    Test that a CSV file with out-of-range coordinate values results in
    rows being dropped.

    Compares the number of rows in the returned GeoDataFrame with a valid subset.
    """
    valid_path = os.path.join(TEST_DATA_DIR, "monitors_valid.csv")
    gdf_valid = load_monitors(valid_path, expected_crs="EPSG:4326")
    out_of_range_path = os.path.join(
        TEST_DATA_DIR, "monitors_out_of_range.csv")
    gdf_out = load_monitors(out_of_range_path, expected_crs="EPSG:4326")
    assert len(gdf_out) < len(gdf_valid.head(
        10)), "Rows with out-of-range coordinates should be dropped."


def test_load_monitors_empty():
    """
    Test that an empty CSV file leads to a ValueError due to an empty GeoDataFrame.
    """
    empty_path = os.path.join(TEST_DATA_DIR, "monitors_empty.csv")
    with pytest.raises(ValueError):
        load_monitors(empty_path)


def test_load_monitors_malformed(tmp_path):
    """
    Test that a malformed CSV file raises an exception.
    """
    malformed_path = os.path.join(TEST_DATA_DIR, "monitors_malformed.csv")
    with pytest.raises(Exception):
        load_monitors(malformed_path)


def test_load_monitors_diff_crs():
    """
    Test that a monitors CSV with coordinates in a different CRS
    is reprojected to the expected CRS (EPSG:4326).
    """
    diff_crs_path = os.path.join(TEST_DATA_DIR, "monitors_diff_crs.csv")
    gdf = load_monitors(diff_crs_path, expected_crs="EPSG:4326")
    assert gdf.crs.to_string() == "EPSG:4326", "CRS should be EPSG:4326 after reprojection."


# ----------------------------------------
# Main block to run tests with verbose output when executed directly.
# ----------------------------------------
if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", "--maxfail=1", "--disable-warnings"]))
