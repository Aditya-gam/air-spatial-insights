"""
analysis.py

This module implements spatial analysis functions for our air quality project.
It includes functions for spatial joining and aggregation of monitor data,
DBSCAN clustering of monitor records, computation of global and local Moran's I,
and basic geometry statistics for census tracts.
"""

import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal
from esda import Moran, Moran_Local
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

# Set up logger
logger = logging.getLogger(__name__)


def spatial_join_and_aggregate(
    ozone_gdf: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame, col: str = "value_mean"
) -> gpd.GeoDataFrame:
    """
    Spatially join the ozone monitors with census tracts and aggregate pollutant metrics.

    This function:
      - Ensures that the monitors GeoDataFrame is in the same CRS as the tracts.
      - Performs a spatial join using an 'intersects' predicate.
      - Groups the joined data by GEOID and computes the mean of the specified column.
      - If the specified column is not found in the joined data, aggregation fails gracefully,
        resulting in NaN values for all tracts.
      - Merges the aggregated pollutant values back to the original census tracts.

    Parameters
    ----------
    ozone_gdf : gpd.GeoDataFrame
        GeoDataFrame containing monitor records with a numeric column to aggregate.
    tracts : gpd.GeoDataFrame
        GeoDataFrame of census tracts.
    col : str, optional
        The column name to aggregate (default is "value_mean").

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame of census tracts with a new column "Ozone" representing the aggregated
        mean pollutant measurement. If the specified column is invalid, "Ozone" will contain NaNs.
    """
    # Ensure both GeoDataFrames share the same CRS.
    if ozone_gdf.crs != tracts.crs:
        logger.info("Reprojecting ozone monitors to match census tracts CRS.")
        ozone_gdf = ozone_gdf.to_crs(tracts.crs)

    # Perform spatial join.
    joined = gpd.sjoin(ozone_gdf, tracts, how="inner", predicate="intersects")
    logger.info(f"Spatial join complete: {len(joined)} records obtained.")

    # Group by GEOID and compute mean of the specified column.
    if col in joined.columns:
        aggregated = joined.groupby("GEOID")[col].mean().reset_index()
    else:
        logger.warning(
            f"Column '{col}' not found in joined data. Aggregated values will be set to NaN.")
        # Create an aggregated DataFrame with NaN values for each tract.
        aggregated = pd.DataFrame(
            {"GEOID": tracts["GEOID"], col: [float("nan")] * len(tracts)})

    aggregated.rename(columns={col: "Ozone"}, inplace=True)
    logger.info("Aggregation of pollutant data complete.")

    # Merge aggregated data back to the census tracts.
    tracts_agg = tracts.merge(aggregated, on="GEOID", how="left")
    num_with_data = tracts_agg["Ozone"].notna().sum()
    logger.info(
        f"Merged aggregated data: {num_with_data} tracts have pollutant data out of {len(tracts_agg)}.")

    return tracts_agg


def dbscan_clustering(
    monitors_gdf: gpd.GeoDataFrame, col: str, eps: float, min_samples: int
) -> gpd.GeoDataFrame:
    """
    Apply DBSCAN clustering to monitor data using spatial coordinates and a pollutant measurement.

    The function extracts the (x, y) coordinates and the specified numeric column into a feature matrix,
    runs DBSCAN clustering, and adds a new column 'cluster' to the input GeoDataFrame.

    Parameters
    ----------
    monitors_gdf : gpd.GeoDataFrame
        GeoDataFrame containing monitor records with valid geometry.
    col : str
        The name of the numeric column to use for clustering (e.g., "value_boxcox").
    eps : float
        The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples : int
        The number of samples in a neighborhood for a point to be considered a core point.

    Returns
    -------
    gpd.GeoDataFrame
        The input GeoDataFrame with an additional column 'cluster' containing DBSCAN cluster labels.
    """
    # Extract spatial coordinates and the measurement column.
    coords = np.array(
        list(zip(monitors_gdf.geometry.x, monitors_gdf.geometry.y)))
    values = monitors_gdf[col].values.reshape(-1, 1)
    # Combine the coordinates and measurement into one feature matrix.
    X = np.hstack([coords, values])
    logger.info(f"Constructed feature matrix for DBSCAN with shape: {X.shape}")

    # Apply DBSCAN.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    monitors_clustered = monitors_gdf.copy()
    monitors_clustered["cluster"] = labels
    n_clusters = len(set(labels) - {-1})
    noise_count = list(labels).count(-1)
    logger.info(
        f"DBSCAN clustering complete: {n_clusters} clusters found with {noise_count} noise points.")

    return monitors_clustered


def moran_global_local(
    tracts_gdf: gpd.GeoDataFrame, col: str
) -> Tuple[float, float, pd.Series]:
    """
    Compute Global and Local Moran's I for a pollutant in the census tracts and classify clusters.

    This function:
      - Constructs a Queen contiguity spatial weights matrix.
      - Removes islands (tracts with no neighbors) and selects the largest connected component.
      - Computes Global Moran's I and its p-value.
      - Computes Local Moran's I and p-values.
      - Classifies each tract based on its value and spatial lag into categories:
        "High-High", "Low-Low", "High-Low", "Low-High", or "Not significant".
      - Adds the classification to a new column 'LISA_cluster'.

    Parameters
    ----------
    tracts_gdf : gpd.GeoDataFrame
        GeoDataFrame containing census tracts with a numeric column `col`.
    col : str
        The name of the numeric column to analyze (e.g., "Ozone").

    Returns
    -------
    Tuple[float, float, pd.Series]
        A tuple containing:
         - Global Moran's I value (float),
         - Global Moran's I p-value (float),
         - A Pandas Series of cluster labels ("LISA_cluster") for each tract.
    """
    # Build Queen contiguity spatial weights.
    w = libpysal.weights.Queen.from_dataframe(tracts_gdf, use_index=False)
    w.transform = "R"
    logger.info("Constructed Queen contiguity spatial weights.")

    # Remove island tracts.
    islands = w.islands
    if islands:
        logger.info(f"Removing island tracts with indices: {islands}")
        tracts_gdf = tracts_gdf.drop(index=islands).copy()
        w = libpysal.weights.Queen.from_dataframe(tracts_gdf, use_index=False)
        w.transform = "R"
        logger.info("Recomputed spatial weights after island removal.")

    # Select the largest connected component.
    components = w.component_labels
    comp_series = pd.Series(components)
    largest_component = comp_series.value_counts().idxmax()
    logger.info(
        f"Largest connected component label: {largest_component} with {comp_series.value_counts()[largest_component]} tracts.")
    mask = comp_series == largest_component
    tracts_largest = tracts_gdf.iloc[mask].copy()
    w = libpysal.weights.Queen.from_dataframe(tracts_largest, use_index=False)
    w.transform = "R"

    # Extract the pollutant values.
    y = tracts_largest[col].values

    # Compute Global Moran's I.
    global_moran = Moran(y, w)
    logger.info(
        f"Global Moran's I: {global_moran.I:.3f}, p-value: {global_moran.p_sim:.3f}")

    # Compute Local Moran's I.
    lisa = Moran_Local(y, w)
    p_vals = lisa.p_sim
    signif = p_vals < 0.05

    # Classify tracts into LISA clusters.
    clusters = []
    for i, val in enumerate(y):
        if not signif[i]:
            clusters.append("Not significant")
        else:
            if val > y.mean() and lisa.y_z[i] > 0:
                clusters.append("High-High")
            elif val < y.mean() and lisa.y_z[i] < 0:
                clusters.append("Low-Low")
            elif val > y.mean() and lisa.y_z[i] < 0:
                clusters.append("High-Low")
            elif val < y.mean() and lisa.y_z[i] > 0:
                clusters.append("Low-High")
            else:
                clusters.append("Not significant")
    tracts_largest["LISA_cluster"] = clusters
    logger.info("Assigned LISA cluster classifications to tracts.")

    return global_moran.I, global_moran.p_sim, pd.Series(clusters, index=tracts_largest.index)


def compute_geometry_stats(tracts_gdf: gpd.GeoDataFrame) -> Dict[str, float]:
    """
    Compute basic geometry statistics for the census tracts.

    Parameters
    ----------
    tracts_gdf : gpd.GeoDataFrame
        GeoDataFrame of census tracts.

    Returns
    -------
    Dict[str, float]
        A dictionary containing:
            - Total number of features,
            - Average number of vertices per feature,
            - Minimum and maximum vertices per feature,
            - Minimum, maximum, and average area (in square kilometers).
    """
    num_features = len(tracts_gdf)

    def count_vertices(geom) -> int:
        if geom.geom_type == "Polygon":
            return len(geom.exterior.coords)
        elif geom.geom_type == "MultiPolygon":
            return sum(len(part.exterior.coords) for part in geom.geoms)
        else:
            return 0

    tracts_gdf["num_vertices"] = tracts_gdf.geometry.apply(count_vertices)
    avg_vertices = tracts_gdf["num_vertices"].mean()
    min_vertices = tracts_gdf["num_vertices"].min()
    max_vertices = tracts_gdf["num_vertices"].max()

    # Reproject to EPSG:3310 for accurate area calculations.
    tracts_proj = tracts_gdf.to_crs(epsg=3310)
    areas_sq_km = tracts_proj.geometry.area / 1e6
    min_area = areas_sq_km.min()
    max_area = areas_sq_km.max()
    avg_area = areas_sq_km.mean()

    stats = {
        "num_features": num_features,
        "avg_vertices": avg_vertices,
        "min_vertices": min_vertices,
        "max_vertices": max_vertices,
        "min_area_sq_km": min_area,
        "max_area_sq_km": max_area,
        "avg_area_sq_km": avg_area,
    }
    logger.info("Computed geometry statistics for census tracts.")

    return stats


if __name__ == "__main__":
    # For debugging purposes only.
    # In production, these functions will be imported and used by main.py.
    logging.basicConfig(level=logging.INFO)
    logger.info("Analysis module loaded.")
    # Example dummy calls (replace with actual data loading in main.py)
    # from data_loader import load_census_tracts, load_monitors
    # tracts_gdf = load_census_tracts("data/tl_2024_06_tract.zip")
    # ozone_gdf = filter_pollutant(load_monitors("data/annual_conc_by_monitor_2024.csv"), "Ozone", 2024)
    logger.info("Analysis module ready for use.")
