"""
analysis.py

This module implements spatial analysis functions for our air quality project.
It includes functions for spatial joining and aggregation of monitor data,
DBSCAN clustering of monitor records, computation of global and local Moran's I,
and basic geometry statistics for census tracts.
"""

import logging
import warnings
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

    If eps is less than or equal to zero, the function assigns all points as noise (-1)
    to avoid violating DBSCAN parameter constraints.

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
    # Extract spatial coordinates.
    coords = np.array(
        list(zip(monitors_gdf.geometry.x, monitors_gdf.geometry.y)))
    # Attempt to extract the measurement column; if not found, raise AttributeError.
    try:
        values = monitors_gdf[col].values.reshape(-1, 1)
    except KeyError as e:
        raise AttributeError(
            f"Column '{col}' not found in monitors GeoDataFrame.") from e

    # Combine the coordinates and measurement into one feature matrix.
    X = np.hstack([coords, values])
    logger.info(f"Constructed feature matrix for DBSCAN with shape: {X.shape}")

    # If eps is less than or equal to zero, assign all points as noise.
    if eps <= 0.0:
        logger.warning(
            "eps <= 0 provided. Assigning all points as noise (-1).")
        monitors_clustered = monitors_gdf.copy()
        monitors_clustered["cluster"] = -1
        return monitors_clustered

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
      - Constructs a Queen contiguity spatial weights matrix on the full dataset.
      - For each connected component with at least 3 tracts, computes Local Moran's I and assigns
        clusters based on significance and the relationship between tract values and their spatial lag.
      - For components with fewer than 3 tracts, assigns the cluster "Not significant".
      - Computes Global Moran's I and its p-value on the largest connected component (with at least 3 tracts);
        if no such component exists, returns NaN.
      - Returns a Pandas Series of cluster labels for all tracts.

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
         - Global Moran's I value (float) computed on the largest connected component (or NaN),
         - Global Moran's I p-value (float) (or NaN),
         - A Pandas Series of cluster labels ("LISA_cluster") for each tract in the input.
    """
    # Suppress the user warning about disconnected weights.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The weights matrix is not fully connected")
        w = libpysal.weights.Queen.from_dataframe(tracts_gdf, use_index=False)
    w.transform = "R"
    logger.info("Constructed Queen contiguity spatial weights.")

    # Get connected component labels (indexed by tracts_gdf index)
    comp_labels = pd.Series(w.component_labels, index=tracts_gdf.index)
    clusters = pd.Series(index=tracts_gdf.index, dtype=object)

    # Process each connected component
    for comp in comp_labels.unique():
        comp_idx = comp_labels[comp_labels == comp].index
        if len(comp_idx) >= 3:
            comp_gdf = tracts_gdf.loc[comp_idx].copy()
            y = comp_gdf[col].values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                w_comp = libpysal.weights.Queen.from_dataframe(
                    comp_gdf, use_index=False)
            w_comp.transform = "R"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                moran_local = Moran_Local(y, w_comp)
            comp_clusters = []
            for i, val in enumerate(y):
                if moran_local.p_sim[i] >= 0.05:
                    comp_clusters.append("Not significant")
                else:
                    if val > y.mean() and moran_local.y_z[i] > 0:
                        comp_clusters.append("High-High")
                    elif val < y.mean() and moran_local.y_z[i] < 0:
                        comp_clusters.append("Low-Low")
                    elif val > y.mean() and moran_local.y_z[i] < 0:
                        comp_clusters.append("High-Low")
                    elif val < y.mean() and moran_local.y_z[i] > 0:
                        comp_clusters.append("Low-High")
                    else:
                        comp_clusters.append("Not significant")
            clusters.loc[comp_idx] = comp_clusters
        else:
            # For components with fewer than 3 tracts, assign "Not significant"
            clusters.loc[comp_idx] = "Not significant"

    # Compute Global Moran's I on the largest connected component with at least 3 tracts.
    valid_comps = [comp for comp in comp_labels.unique() if (
        comp_labels == comp).sum() >= 3]
    if valid_comps:
        largest_comp = max(valid_comps, key=lambda c: (comp_labels == c).sum())
        comp_idx = comp_labels[comp_labels == largest_comp].index
        comp_gdf = tracts_gdf.loc[comp_idx].copy()
        y = comp_gdf[col].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            w_comp = libpysal.weights.Queen.from_dataframe(
                comp_gdf, use_index=False)
        w_comp.transform = "R"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            moran_obj = Moran(y, w_comp)
        global_I = moran_obj.I
        p_val = moran_obj.p_sim
    else:
        global_I = np.nan
        p_val = np.nan

    return global_I, p_val, clusters


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
    if tracts_gdf.empty:
        return {
            "num_features": 0,
            "avg_vertices": np.nan,
            "min_vertices": np.nan,
            "max_vertices": np.nan,
            "min_area_sq_km": np.nan,
            "max_area_sq_km": np.nan,
            "avg_area_sq_km": np.nan,
        }

    num_features = len(tracts_gdf)

    def count_vertices(geom) -> int:
        if geom.geom_type == "Polygon":
            return len(geom.exterior.coords)
        elif geom.geom_type == "MultiPolygon":
            return sum(len(part.exterior.coords) for part in geom.geoms)
        else:
            return 0

    # Force the output of count_vertices to be numeric.
    vertices = tracts_gdf.geometry.apply(count_vertices).astype(float)
    avg_vertices = vertices.mean()
    min_vertices = vertices.min()
    max_vertices = vertices.max()

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
