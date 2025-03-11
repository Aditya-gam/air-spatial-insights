"""
preprocessing.py

This module contains functions to preprocess air quality monitor data.
It provides routines to clean the monitors data, filter by pollutant and year,
remove outliers, and apply a Box–Cox transformation.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import boxcox

from scripts.data_loader import load_census_tracts, load_monitors

logger = logging.getLogger(__name__)


def clean_monitors(monitors_gdf: gpd.GeoDataFrame,
                   filter_invalid: bool = True) -> gpd.GeoDataFrame:
    """
    Clean the air quality monitors GeoDataFrame.

    This function performs the following steps:
      - Drops rows with missing geometry.
      - Removes duplicate rows based on 'monitor_id', 'Latitude', and 'Longitude'.
      - Optionally, filters out rows with non-positive values in 'value_mean'.

    Parameters
    ----------
    monitors_gdf : gpd.GeoDataFrame
        The input GeoDataFrame containing monitor data.
    filter_invalid : bool, optional
        If True, drop rows where the 'value_mean' column is zero or negative.
        Default is True.

    Returns
    -------
    gpd.GeoDataFrame
        A cleaned GeoDataFrame of monitor data.

    Raises
    ------
    ValueError
        If the required column 'value_mean' is missing when filter_invalid is True.
    """
    # Drop rows with missing geometry
    initial_count = len(monitors_gdf)
    monitors_gdf = monitors_gdf[monitors_gdf.geometry.notna()].copy()
    logger.info(
        f"Dropped {initial_count - len(monitors_gdf)} rows with missing geometry.")

    # Remove duplicates based on monitor_id and coordinates
    before_dup = len(monitors_gdf)
    monitors_gdf = monitors_gdf.drop_duplicates(
        subset=["monitor_id", "Latitude", "Longitude"])
    logger.info(
        f"Removed {before_dup - len(monitors_gdf)} duplicate rows based on monitor_id, Latitude, and Longitude.")

    # Optionally, filter out rows with non-positive value_mean if required
    if filter_invalid:
        if "value_mean" not in monitors_gdf.columns:
            logger.error("Column 'value_mean' is missing from monitors_gdf.")
            raise ValueError(
                "Column 'value_mean' is required for filtering invalid values.")
        before_invalid = len(monitors_gdf)
        monitors_gdf = monitors_gdf[monitors_gdf["value_mean"] > 0.0].copy()
        logger.info(
            f"Filtered out {before_invalid - len(monitors_gdf)} rows with non-positive 'value_mean'.")

    return monitors_gdf


def filter_pollutant(monitors_gdf: gpd.GeoDataFrame,
                     pollutant: str, year: int) -> gpd.GeoDataFrame:
    """
    Filter the monitors GeoDataFrame by pollutant name and year.

    Parameters
    ----------
    monitors_gdf : gpd.GeoDataFrame
        The input GeoDataFrame containing monitor data.
    pollutant : str
        The pollutant name to filter by (e.g., "Ozone" or "PM2.5 - Local Conditions").
    year : int
        The year to filter on.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame filtered for the specified pollutant and year.
    """
    filtered_gdf = monitors_gdf[
        (monitors_gdf["pollutant_name"].str.strip().str.lower() == pollutant.lower()) &
        (monitors_gdf["Year"] == year)
    ].copy()

    logger.info(
        f"Filtered monitors: {len(filtered_gdf)} records for pollutant '{pollutant}' in year {year}.")

    return filtered_gdf


def remove_outliers(monitors_gdf: gpd.GeoDataFrame,
                    col: str,
                    lower_q: float = 0.01,
                    upper_q: float = 0.99) -> gpd.GeoDataFrame:
    """
    Remove outliers from a numeric column in the monitors GeoDataFrame.

    The function computes the lower and upper quantile thresholds for the specified column
    and removes rows with values outside these thresholds.

    Parameters
    ----------
    monitors_gdf : gpd.GeoDataFrame
        The input GeoDataFrame containing monitor data.
    col : str
        The name of the numeric column to filter out outliers.
    lower_q : float, optional
        The lower quantile threshold (default: 0.01).
    upper_q : float, optional
        The upper quantile threshold (default: 0.99).

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with outliers removed.
    """
    if col not in monitors_gdf.columns:
        logger.error(f"Column '{col}' not found in monitors GeoDataFrame.")
        raise ValueError(f"Column '{col}' is required for outlier removal.")

    lower_bound = monitors_gdf[col].quantile(lower_q)
    upper_bound = monitors_gdf[col].quantile(upper_q)
    before_count = len(monitors_gdf)

    filtered_gdf = monitors_gdf[(monitors_gdf[col] >= lower_bound) & (
        monitors_gdf[col] <= upper_bound)].copy()

    logger.info(
        f"Removed {before_count - len(filtered_gdf)} outlier rows from column '{col}' "
        f"using quantiles [{lower_q}, {upper_q}]: lower_bound={lower_bound}, upper_bound={upper_bound}."
    )

    return filtered_gdf


def apply_boxcox(monitors_gdf: gpd.GeoDataFrame,
                 col: str) -> Tuple[gpd.GeoDataFrame, float]:
    """
    Apply the Box–Cox transformation to a numeric column in the monitors GeoDataFrame.

    This function transforms the specified column (which must have all positive values)
    and stores the result in a new column with suffix '_boxcox'. It returns the updated
    GeoDataFrame along with the lambda value from the Box–Cox transformation.

    Parameters
    ----------
    monitors_gdf : gpd.GeoDataFrame
        The input GeoDataFrame containing monitor data.
    col : str
        The name of the numeric column to transform.

    Returns
    -------
    Tuple[gpd.GeoDataFrame, float]
        A tuple containing the updated GeoDataFrame and the Box–Cox lambda value.

    Raises
    ------
    ValueError
        If the specified column contains non-positive values.
    """
    if col not in monitors_gdf.columns:
        logger.error(f"Column '{col}' not found in monitors GeoDataFrame.")
        raise ValueError(
            f"Column '{col}' is required for Box-Cox transformation.")

    # Ensure all values are positive (Box-Cox requires positive values)
    if (monitors_gdf[col] <= 0).any():
        logger.error(
            "Box-Cox transformation requires all values to be positive.")
        raise ValueError(
            "Column values must be positive for Box-Cox transformation.")

    # Apply Box-Cox transformation
    try:
        transformed_values, lam = boxcox(monitors_gdf[col])
        monitors_gdf[f"{col}_boxcox"] = transformed_values
        logger.info(
            f"Applied Box-Cox transformation on '{col}' with lambda = {lam:.6f}.")
    except Exception as e:
        logger.exception("Error applying Box-Cox transformation.")
        raise e

    return monitors_gdf, lam


if __name__ == "__main__":
    # For debugging purposes only: run each function with sample file paths.
    # In production, these functions would be called from main.py.
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting preprocessing...")

    try:
        tracts_gdf = load_census_tracts("data/tl_2024_06_tract.zip")
        monitors_gdf = load_monitors("data/annual_conc_by_monitor_2024.csv")
        monitors_clean = clean_monitors(monitors_gdf)
        monitors_filtered = filter_pollutant(monitors_clean, "Ozone", 2024)
        monitors_no_outliers = remove_outliers(monitors_filtered, "value_mean")
        monitors_boxcox, lam_val = apply_boxcox(
            monitors_no_outliers, "value_mean")

        logger.info("Preprocessing completed successfully.")
    except Exception as err:
        logger.error(f"An error occurred during preprocessing: {err}")
