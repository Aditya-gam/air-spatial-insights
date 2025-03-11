"""
visualization.py

This module contains functions for visualizing spatial data and analysis results.
It provides routines to plot static choropleth maps, k-distance elbow plots, LISA
cluster maps, Moran scatter plots, and interactive Folium maps.

Functions:
    plot_choropleth(tracts_gdf, col, title, missing_kwds) -> plt.Axes
    plot_kdistance(distances, k) -> plt.Axes
    plot_lisa_map(tracts_gdf, lisa_col, title, legend_loc) -> plt.Axes
    plot_moran_scatter(tracts_gdf, col, w) -> plt.Axes
    create_folium_map(tracts_gdf, col, output_html) -> folium.Map
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import folium
from esda import Moran_Local
import libpysal

# Set up logger
logger = logging.getLogger(__name__)


def plot_choropleth(
    tracts_gdf: gpd.GeoDataFrame,
    col: str,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    missing_kwds: Optional[dict] = None
) -> plt.Axes:
    """
    Plot a static choropleth map of census tracts colored by the specified column.

    Parameters
    ----------
    tracts_gdf : gpd.GeoDataFrame
        GeoDataFrame containing census tracts and a numeric column.
    col : str
        The name of the column to color the tracts (e.g., "Ozone").
    title : str, optional
        Title for the plot.
    ax : plt.Axes, optional
        Pre-existing matplotlib Axes to plot on; if None, a new figure and axes will be created.
    missing_kwds : dict, optional
        A dictionary of plotting options for missing data (passed to GeoPandas plot).

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the choropleth map.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    if missing_kwds is None:
        missing_kwds = {"color": "lightgrey",
                        "edgecolor": "black", "hatch": "", "label": "No Data"}

    tracts_gdf.plot(
        column=col,
        cmap="OrRd",
        legend=True,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
        missing_kwds=missing_kwds
    )
    ax.set_title(
        title if title is not None else f"Census Tracts by {col}", fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    # Adjust axis limits to the full extent of the data
    minx, miny, maxx, maxy = tracts_gdf.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # Override the legend to include a custom patch for missing data if a label is provided.
    if missing_kwds is not None and "label" in missing_kwds:
        missing_patch = mpatches.Patch(facecolor=missing_kwds.get("color", "lightgrey"),
                                       edgecolor=missing_kwds.get(
                                           "edgecolor", "black"),
                                       hatch=missing_kwds.get("hatch", ""),
                                       label=missing_kwds["label"])
        # Create a new legend with only the missing data patch.
        ax.legend(handles=[missing_patch], loc="best", fontsize=12)

    logger.info(
        f"Choropleth map for column '{col}' plotted with {len(tracts_gdf)} tracts.")
    return ax


def plot_kdistance(distances: np.ndarray, k: int) -> plt.Axes:
    """
    Plot the k-distance elbow graph used for DBSCAN parameter selection.

    Parameters
    ----------
    distances : np.ndarray
        Sorted array of distances to the k-th nearest neighbor for each point.
    k : int
        The k value used (i.e., the k-th nearest neighbor).

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the k-distance graph.
    """
    _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(distances, lw=2)
    ax.set_title(f"{k}-Distance Graph for DBSCAN", fontsize=14)
    ax.set_xlabel("Points (sorted by distance to k-th neighbor)", fontsize=12)
    ax.set_ylabel(f"Distance to {k}-th Nearest Neighbor", fontsize=12)
    logger.info("K-distance graph plotted.")

    return ax


def plot_lisa_map(
    tracts_gdf: gpd.GeoDataFrame,
    lisa_col: str = "LISA_cluster",
    title: Optional[str] = None,
    legend_loc: str = "upper right"
) -> plt.Axes:
    """
    Plot a Local Moran's I (LISA) cluster map.

    The function expects the GeoDataFrame to have a column (default "LISA_cluster")
    with cluster classification labels.

    Parameters
    ----------
    tracts_gdf : gpd.GeoDataFrame
        GeoDataFrame containing census tracts with LISA cluster classifications.
    lisa_col : str, optional
        The column name with LISA cluster labels (default is "LISA_cluster").
    title : str, optional
        Title for the plot.
    legend_loc : str, optional
        Location for the legend (default is "upper right").

    Returns
    -------
    plt.Axes
        The matplotlib Axes object with the LISA cluster map.
    """
    # Ensure the data is in a projected CRS (e.g., EPSG:3857) for proper area representation.
    tracts_proj = tracts_gdf.to_crs(epsg=3857)
    # Define a color mapping for cluster types.
    color_map = {
        "High-High": "red",
        "Low-Low": "blue",
        "High-Low": "orange",
        "Low-High": "lightblue",
        "Not significant": "lightgray"
    }
    _, ax = plt.subplots(figsize=(8, 6))
    # Plot all tracts in a light background.
    tracts_proj.plot(color="lightgray", ax=ax, edgecolor="white")
    # Plot each cluster type separately and collect legend handles.
    legend_handles = []
    for cluster_type, color in color_map.items():
        if cluster_type == "Not significant":
            continue
        subset = tracts_proj[tracts_proj[lisa_col] == cluster_type]
        if not subset.empty:
            subset.plot(color=color, ax=ax,
                        edgecolor="white", label=cluster_type)
            legend_handles.append(mpatches.Patch(
                color=color, label=cluster_type))
    if not legend_handles:
        legend_handles.append(mpatches.Patch(
            color="lightgray", label="No significant clusters"))
    ax.set_title(
        title if title else "Local Moran's I Cluster Map", fontsize=14)
    ax.set_aspect("equal")
    ax.legend(handles=legend_handles, loc=legend_loc, fontsize=12)
    logger.info("LISA cluster map plotted.")

    return ax


def plot_moran_scatter(
    tracts_gdf: gpd.GeoDataFrame, col: str, w: libpysal.weights.W
) -> plt.Axes:
    """
    Create a Moran scatter plot with a regression line.

    This function:
      - Standardizes the values in the specified column.
      - Computes the spatial lag using the full weights matrix.
      - Plots the standardized values (Z) against the standardized spatial lag.
      - Draws horizontal and vertical reference lines at 0 and attempts to fit a regression line.

    Parameters
    ----------
    tracts_gdf : gpd.GeoDataFrame
        GeoDataFrame containing census tracts with the numeric column to analyze.
    col : str
        The name of the numeric column to standardize and analyze (e.g., "Ozone").
    w : libpysal.weights.W
        The spatial weights matrix corresponding to the tracts.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the Moran scatter plot.
    """
    # Extract the values and compute their standard scores.
    y = tracts_gdf[col].values
    z = (y - y.mean()) / y.std()
    # Compute the spatial lag.
    w_matrix = w.full()[0]
    lag = w_matrix @ y
    lag_z = (lag - y.mean()) / y.std()

    _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z, lag_z, c="gray", alpha=0.6)
    ax.axhline(0, color="black", linestyle="--")
    ax.axvline(0, color="black", linestyle="--")

    # Fit a regression line if possible.
    try:
        m, b = np.polyfit(z, lag_z, 1)
        ax.plot(z, m * z + b, color="green", label="Regression line")
        logger.info("Regression line fitted in Moran scatter plot.")
    except np.linalg.LinAlgError as e:
        logger.warning(f"Regression line fitting failed: {e}")

    ax.set_title("Moran Scatter Plot", fontsize=14)
    ax.set_xlabel("Standardized Values (Z)", fontsize=12)
    ax.set_ylabel("Standardized Spatial Lag", fontsize=12)
    ax.legend()
    logger.info("Moran scatter plot created.")

    return ax


def create_folium_map(
    tracts_gdf: gpd.GeoDataFrame, col: str, output_html: str
) -> folium.Map:
    """
    Create an interactive Folium choropleth map of the census tracts.

    The function:
      - Assumes that the GeoDataFrame is in EPSG:4326.
      - Drops rows with missing values in the specified column.
      - Creates a Folium Map centered on the data.
      - Adds a Choropleth layer based on the specified column.
      - Saves the map to an HTML file.

    Parameters
    ----------
    tracts_gdf : gpd.GeoDataFrame
        GeoDataFrame containing census tracts with the numeric column to map.
    col : str
        The column used for the choropleth (e.g., "Ozone").
    output_html : str
        The file path where the interactive map HTML should be saved.

    Returns
    -------
    folium.Map
        The interactive Folium Map object.
    """
    # Ensure the GeoDataFrame is in lat/lon CRS.
    if tracts_gdf.crs.to_string() != "EPSG:4326":
        logger.info("Reprojecting tracts to EPSG:4326 for Folium mapping.")
        tracts_gdf = tracts_gdf.to_crs(epsg=4326)
    # Drop rows with missing values in the specified column.
    gdf = tracts_gdf.dropna(subset=[col, "geometry"]).copy()
    logger.info(f"{len(gdf)} tracts will be mapped with column '{col}'.")

    # Compute the center of the data.
    bounds = gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    # Create a Folium map centered on the data.
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=6)
    # Add a choropleth layer.
    folium.Choropleth(
        geo_data=gdf,
        name=col,
        data=gdf,
        columns=["GEOID", col],
        key_on="feature.properties.GEOID",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"Mean {col}"
    ).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    fmap.save(output_html)
    logger.info(f"Folium map created and saved to {output_html}.")

    return fmap


if __name__ == "__main__":
    # For debugging purposes only; in production, these functions will be imported by main.py.
    logging.basicConfig(level=logging.INFO)
    logger.info("Visualization module loaded and ready.")
