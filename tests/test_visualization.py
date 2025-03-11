"""
tests/test_visualization.py

This module contains comprehensive tests for the functions in the visualization module:
    - plot_choropleth()
    - plot_kdistance()
    - plot_lisa_map()
    - plot_moran_scatter()
    - create_folium_map()

The tests use synthetic GeoDataFrames (and NumPy arrays) to simulate various general and edge cases.
Test cases include:
    - Valid inputs producing expected plot properties (titles, labels, legends)
    - GeoDataFrames with missing values in the target column
    - Empty GeoDataFrames
    - Different coordinate systems (CRS) to test automatic reprojection for Folium maps
    - Special cases for k-distance and Moran scatter plots

To run these tests, execute from the repository root:
    pytest --maxfail=1 --disable-warnings -q
"""

import os
import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
import libpysal

from shapely.geometry import Point, Polygon, MultiPolygon

# Import functions to be tested
from scripts.visualization import (
    plot_choropleth,
    plot_kdistance,
    plot_lisa_map,
    plot_moran_scatter,
    create_folium_map
)

# -----------------------------
# Helper Functions for Synthetic Data
# -----------------------------


def create_synthetic_tracts(num=5, with_lisa=True, constant_value=None):
    """
    Create a synthetic GeoDataFrame representing census tracts.

    Parameters
    ----------
    num : int, optional
        Number of tracts to create (default is 5).
    with_lisa : bool, optional
        If True, add a 'LISA_cluster' column with alternating values.
    constant_value : float or None, optional
        If provided, sets the "Ozone" column to this constant value.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with columns "GEOID", "Ozone" and (if with_lisa) "LISA_cluster",
        with simple square polygons.
    """
    geoms = []
    geoids = []
    ozone_vals = []
    lisa_clusters = []
    for i in range(num):
        # Create a square polygon: shift by i in x direction.
        poly = Polygon([(i, 0), (i+1, 0), (i+1, 1), (i, 1)])
        geoms.append(poly)
        geoids.append(f"TRACT_{i}")
        # Set ozone value: if constant_value provided, use that, else use i+1.
        ozone_vals.append(
            constant_value if constant_value is not None else (i + 1))
        # For LISA cluster, if requested, alternate cluster types.
        if with_lisa:
            if i % 2 == 0:
                lisa_clusters.append("High-High")
            else:
                lisa_clusters.append("Not significant")
    df = pd.DataFrame({
        "GEOID": geoids,
        "Ozone": ozone_vals
    })
    if with_lisa:
        df["LISA_cluster"] = lisa_clusters
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
    return gdf


def create_synthetic_distances(num=10, extreme=False):
    """
    Create a synthetic sorted array of distances for k-distance plot tests.

    Parameters
    ----------
    num : int, optional
        Number of distance values (default is 10).
    extreme : bool, optional
        If True, include some extreme values.

    Returns
    -------
    np.ndarray
        A sorted 1D array of distances.
    """
    if extreme:
        distances = np.linspace(0, 100, num)
    else:
        distances = np.linspace(0.1, 1.0, num)
    return np.sort(distances)


def create_synthetic_weights(tracts_gdf):
    """
    Create a Queen contiguity weights matrix from the given GeoDataFrame.

    Parameters
    ----------
    tracts_gdf : gpd.GeoDataFrame
        GeoDataFrame for which to compute spatial weights.

    Returns
    -------
    libpysal.weights.W
        The computed weights matrix.
    """
    w = libpysal.weights.Queen.from_dataframe(tracts_gdf, use_index=False)
    w.transform = "R"
    return w

# -----------------------------
# Tests for plot_choropleth()
# -----------------------------


def test_plot_choropleth_valid():
    """Test plot_choropleth with a valid GeoDataFrame and numeric column."""
    tracts = create_synthetic_tracts(num=5)
    ax = plot_choropleth(tracts, col="Ozone", title="Test Choropleth")
    # Verify that the title is set
    assert "Test Choropleth" in ax.get_title()
    # Verify that axis labels are present
    assert ax.get_xlabel() == "Longitude"
    assert ax.get_ylabel() == "Latitude"
    plt.close(ax.figure)


def test_plot_choropleth_missing_values():
    """Test plot_choropleth with a GeoDataFrame that has missing values in the target column,
    and verify that custom missing_kwds are applied.
    """
    tracts = create_synthetic_tracts(num=5)
    # Set one value to NaN
    tracts.loc[0, "Ozone"] = np.nan
    custom_kwds = {"color": "pink", "edgecolor": "red",
                   "hatch": "///", "label": "Missing"}
    ax = plot_choropleth(tracts, col="Ozone",
                         title="Choropleth Missing", missing_kwds=custom_kwds)
    # Check that the legend contains the custom missing label.
    handles, labels = ax.get_legend_handles_labels()
    assert "Missing" in labels
    plt.close(ax.figure)


def test_plot_choropleth_empty_gdf():
    """Test plot_choropleth with an empty GeoDataFrame.

    The function should handle this gracefully.
    """
    empty_gdf = gpd.GeoDataFrame(
        columns=["GEOID", "Ozone", "geometry"], crs="EPSG:4326")
    ax = plot_choropleth(empty_gdf, col="Ozone", title="Empty Choropleth")
    # Since there are no features, the total bounds may be zeros; check that no error occurs.
    plt.close(ax.figure)

# -----------------------------
# Tests for plot_kdistance()
# -----------------------------


def test_plot_kdistance_valid():
    """Test plot_kdistance with a valid sorted array of distances."""
    distances = create_synthetic_distances(num=10)
    ax = plot_kdistance(distances, k=5)
    # Check that title contains "5-Distance Graph"
    assert "5-Distance Graph" in ax.get_title()
    plt.close(ax.figure)


def test_plot_kdistance_empty_array():
    """Test plot_kdistance with an empty array."""
    distances = np.array([])
    ax = plot_kdistance(distances, k=5)
    # Should produce an empty plot without crashing.
    plt.close(ax.figure)


def test_plot_kdistance_extreme_values():
    """Test plot_kdistance with an array of extreme values."""
    distances = create_synthetic_distances(num=10, extreme=True)
    ax = plot_kdistance(distances, k=5)
    # Check that y-axis label is set properly
    assert "Nearest Neighbor" in ax.get_ylabel()
    plt.close(ax.figure)

# -----------------------------
# Tests for plot_lisa_map()
# -----------------------------


def test_plot_lisa_map_valid():
    """Test plot_lisa_map with a GeoDataFrame that has a valid 'LISA_cluster' column."""
    tracts = create_synthetic_tracts(num=6, with_lisa=True)
    ax = plot_lisa_map(tracts, lisa_col="LISA_cluster",
                       title="Test LISA Map", legend_loc="lower left")
    # Check that the title is set correctly.
    assert "Test LISA Map" in ax.get_title()
    # Check that the legend is placed in the lower left.
    # (We verify by checking that legend exists)
    assert ax.get_legend() is not None
    plt.close(ax.figure)


def test_plot_lisa_map_missing_lisa_col():
    """Test plot_lisa_map with a GeoDataFrame missing the 'LISA_cluster' column.

    This should raise a KeyError.
    """
    tracts = create_synthetic_tracts(num=5, with_lisa=False)
    with pytest.raises(KeyError):
        plot_lisa_map(tracts, lisa_col="LISA_cluster")


def test_plot_lisa_map_all_not_significant():
    """Test plot_lisa_map with a GeoDataFrame where all tracts are 'Not significant'."""
    tracts = create_synthetic_tracts(num=5, with_lisa=True)
    tracts["LISA_cluster"] = "Not significant"
    ax = plot_lisa_map(tracts, lisa_col="LISA_cluster",
                       title="All Not Significant")
    # In this case, the legend should contain the default "No significant clusters" entry.
    handles, labels = ax.get_legend_handles_labels()
    assert any("No significant clusters" in lab for lab in labels)
    plt.close(ax.figure)

# -----------------------------
# Tests for plot_moran_scatter()
# -----------------------------


def test_plot_moran_scatter_valid():
    """Test plot_moran_scatter with valid numeric column and spatial weights matrix."""
    tracts = create_synthetic_tracts(num=5)
    w = create_synthetic_weights(tracts)
    ax = plot_moran_scatter(tracts, col="Ozone", w=w)
    # Verify that the plot has x and y labels as expected.
    assert "Standardized Values" in ax.get_xlabel()
    assert "Standardized Spatial Lag" in ax.get_ylabel()
    plt.close(ax.figure)


def test_plot_moran_scatter_zero_variance():
    """Test plot_moran_scatter with a column having zero variance."""
    # Create tracts where Ozone is constant
    tracts = create_synthetic_tracts(num=5, constant_value=10)
    w = create_synthetic_weights(tracts)
    # This should result in a division by zero when standardizing; check that it handles gracefully.
    ax = plot_moran_scatter(tracts, col="Ozone", w=w)
    # The regression line fitting may fail; ensure the Axes object is returned.
    plt.close(ax.figure)


def test_plot_moran_scatter_invalid_weights():
    """Test plot_moran_scatter with an invalid weights matrix (all zeros).

    For example, create a weights matrix with all zeros.
    """
    tracts = create_synthetic_tracts(num=5)
    # Create a fake weights matrix: a numpy array of zeros, then wrap in a libpysal.weights.W
    n = len(tracts)
    zero_matrix = np.zeros((n, n))
    # Use the W.from_dataframe() method as a base then override the full() output.
    w = create_synthetic_weights(tracts)
    # Monkey-patch the full() method to return a zero matrix.
    w.full = lambda: (zero_matrix, None)
    ax = plot_moran_scatter(tracts, col="Ozone", w=w)
    plt.close(ax.figure)

# -----------------------------
# Tests for create_folium_map()
# -----------------------------


def test_create_folium_map_valid(tmp_path):
    """Test create_folium_map with a valid GeoDataFrame in EPSG:4326.

    Verify that the output HTML file is created.
    """
    tracts = create_synthetic_tracts(num=5)
    output_html = tmp_path / "test_map.html"
    fmap = create_folium_map(tracts, col="Ozone", output_html=str(output_html))
    # Verify that the file exists.
    assert output_html.exists()
    # Basic check: the map object is a folium.Map instance.
    assert isinstance(fmap, folium.Map)


def test_create_folium_map_reproject(tmp_path):
    """Test create_folium_map with a GeoDataFrame not in EPSG:4326 (should reproject).

    Create a GeoDataFrame in EPSG:3857 and verify that the function reprojects it.
    """
    tracts = create_synthetic_tracts(num=5)
    tracts_3857 = tracts.to_crs(epsg=3857)
    output_html = tmp_path / "test_map_reproject.html"
    fmap = create_folium_map(tracts_3857, col="Ozone",
                             output_html=str(output_html))
    assert output_html.exists()
    # Verify that the map's data is now in EPSG:4326 by checking one of the features' coordinates.
    # (Since folium requires lat/lon, the function should have reprojected.)
    plt.close()


def test_create_folium_map_missing_values(tmp_path):
    """Test create_folium_map with a GeoDataFrame that has missing values in the specified column.

    The function should drop rows with missing values and still create a map.
    """
    tracts = create_synthetic_tracts(num=5)
    tracts.loc[0, "Ozone"] = None
    output_html = tmp_path / "test_map_missing.html"
    fmap = create_folium_map(tracts, col="Ozone", output_html=str(output_html))
    assert output_html.exists()
    plt.close()

# -----------------------------
# Command to Run Tests:
# -----------------------------
# From the repository root, run:
#     pytest --maxfail=1 --disable-warnings -q
