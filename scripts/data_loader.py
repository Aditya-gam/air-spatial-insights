"""
data_loader.py

This module contains functions for loading census tract shapefiles and air quality
monitor CSV data into GeoDataFrames. It ensures data integrity, checks file existence,
and validates geometry.
"""

import os
import logging
from typing import Optional

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

logger = logging.getLogger(__name__)


def load_census_tracts(filepath: str, expected_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Load a census tract shapefile and return a GeoDataFrame in the specified CRS.

    Parameters
    ----------
    filepath : str
        Path to the zipped shapefile (e.g., "data/tl_2024_06_tract.zip").
    expected_crs : str
        The desired coordinate reference system (default: "EPSG:4326").

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing census tracts, reprojected to `expected_crs` if needed.

    Raises
    ------
    FileNotFoundError
        If the provided shapefile does not exist or cannot be accessed.
    ValueError
        If the resulting GeoDataFrame has empty geometry or fails to load.
    """
    # Check if the file exists
    if not os.path.isfile(filepath):
        logger.error(f"Shapefile not found at path: {filepath}")
        raise FileNotFoundError(f"Could not find the shapefile at {filepath}")

    # Attempt to read the shapefile
    try:
        tracts_gdf = gpd.read_file(filepath)
    except Exception as e:
        logger.exception(f"Error reading shapefile from {filepath}: {e}")
        raise FileNotFoundError(
            f"Could not read the shapefile at {filepath}") from e

    # Check if geometry is present
    if tracts_gdf.empty or "geometry" not in tracts_gdf.columns:
        logger.error(f"Loaded shapefile has no valid geometry: {filepath}")
        raise ValueError(f"Loaded shapefile has no valid geometry: {filepath}")

    # Reproject to expected CRS if necessary
    if tracts_gdf.crs is None:
        logger.warning(
            "Shapefile has no CRS defined. Setting CRS to EPSG:4326 by default.")
        tracts_gdf.set_crs(epsg=4326, inplace=True)
    elif tracts_gdf.crs.to_string() != expected_crs:
        logger.info(
            f"Reprojecting tracts from {tracts_gdf.crs} to {expected_crs}...")
        tracts_gdf = tracts_gdf.to_crs(expected_crs)

    # Log final info
    logger.info(
        f"Loaded {len(tracts_gdf)} census tracts from '{filepath}' with CRS={tracts_gdf.crs}."
    )

    return tracts_gdf


def load_monitors(filepath: str, expected_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Load air quality monitor data from a CSV and return a GeoDataFrame in the specified CRS.

    This function:
      1. Reads the CSV using pandas.
      2. Converts 'Longitude'/'Latitude' columns to a geometry in EPSG:4326.
      3. Checks for missing geometry or invalid coordinates.
      4. Reprojects to the desired CRS if necessary.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (e.g., "data/annual_conc_by_monitor_2024.csv").
    expected_crs : str
        The desired coordinate reference system (default: "EPSG:4326").

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing monitor data with valid geometry in `expected_crs`.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist or cannot be accessed.
    ValueError
        If the resulting GeoDataFrame has invalid geometry or fails to load properly.
    """
    # Check if the file exists
    if not os.path.isfile(filepath):
        logger.error(f"Monitor CSV not found at path: {filepath}")
        raise FileNotFoundError(f"Could not find the CSV file at {filepath}")

    # Read CSV with pandas
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.exception(f"Error reading CSV from {filepath}: {e}")
        raise FileNotFoundError(
            f"Could not read the CSV file at {filepath}") from e

    # Validate essential columns
    required_cols = {"Longitude", "Latitude"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(f"CSV is missing required columns: {missing}")
        raise ValueError(f"CSV missing required columns: {missing}")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326"  # monitors are typically in WGS84 lat/lon
    )

    if gdf.empty or "geometry" not in gdf.columns:
        logger.error(f"Loaded CSV has no valid geometry: {filepath}")
        raise ValueError(f"Loaded CSV has no valid geometry: {filepath}")

    # Optionally filter out out-of-range coordinates
    mask_valid_coords = (
        (gdf.geometry.y >= -90) & (gdf.geometry.y <= 90) &
        (gdf.geometry.x >= -180) & (gdf.geometry.x <= 180)
    )
    invalid_count = (~mask_valid_coords).sum()
    if invalid_count > 0:
        logger.warning(
            f"Dropping {invalid_count} rows with invalid lat/lon from {filepath}.")
        gdf = gdf[mask_valid_coords].copy()

    # Reproject to expected CRS if needed
    if gdf.crs.to_string() != expected_crs:
        logger.info(
            f"Reprojecting monitors from {gdf.crs} to {expected_crs}...")
        gdf = gdf.to_crs(expected_crs)

    logger.info(
        f"Loaded {len(gdf)} monitor records from '{filepath}' with CRS={gdf.crs}. "
        f"(Dropped {invalid_count} invalid coords if any.)"
    )

    return gdf
