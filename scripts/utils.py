"""
utils.py

This module provides utility functions used throughout the air quality spatial analysis project.
Functions include:
    - setup_logging: Configures a standardized logger for the project.
    - check_crs: Verifies that a GeoDataFrame’s CRS matches an expected CRS.
    - count_vertices: Computes the number of vertices in a Polygon or MultiPolygon geometry.
"""

import logging
from typing import Dict, Any, Optional

import geopandas as gpd
from shapely.geometry.base import BaseGeometry


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure the logger for the project.

    This function configures the logging settings with a standardized format and log level.
    It returns a logger that can be used throughout the project.

    Parameters
    ----------
    level : int, optional
        The logging level (default: logging.INFO).

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # Configure the basic logging settings
    logging.basicConfig(level=level, format=log_format)
    logger = logging.getLogger("air_spatial_insights")
    logger.setLevel(level)
    logger.info("Logging is set up.")
    return logger


def check_crs(gdf: gpd.GeoDataFrame, expected_crs: str = "EPSG:4326") -> None:
    """
    Check if a GeoDataFrame has the expected CRS.

    If the GeoDataFrame has no CRS defined, a ValueError is raised. If the CRS is defined
    but does not match the expected CRS, a warning is logged.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame whose CRS is to be checked.
    expected_crs : str, optional
        The expected coordinate reference system (default: "EPSG:4326").

    Raises
    ------
    ValueError
        If the GeoDataFrame has no CRS defined.
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")
    current_crs = gdf.crs.to_string()
    if current_crs != expected_crs:
        logging.warning(
            f"GeoDataFrame CRS ({current_crs}) does not match the expected CRS ({expected_crs})."
        )


def count_vertices(geom: BaseGeometry) -> int:
    """
    Count the number of vertices in a geometry.

    This function works for both Polygon and MultiPolygon geometries.
    For a Polygon, it returns the number of coordinates in the exterior ring.
    For a MultiPolygon, it returns the sum of the number of coordinates in each exterior ring.
    For other geometry types, it returns 0.

    Parameters
    ----------
    geom : BaseGeometry
        The geometry (Polygon or MultiPolygon) to count vertices for.

    Returns
    -------
    int
        The number of vertices in the geometry. Returns 0 if the geometry is neither
        a Polygon nor a MultiPolygon.
    """
    if geom.geom_type == "Polygon":
        return len(geom.exterior.coords)
    elif geom.geom_type == "MultiPolygon":
        return sum(len(part.exterior.coords) for part in geom.geoms)
    else:
        return 0


# Optionally, add more utility functions here as needed.

if __name__ == "__main__":
    # For debugging purposes only.
    logger = setup_logging()
    # Create a dummy GeoDataFrame to test the utilities.
    import geopandas as gpd
    from shapely.geometry import Point

    dummy_gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
    try:
        check_crs(dummy_gdf, "EPSG:4326")
        logger.info("CRS check passed.")
    except ValueError as e:
        logger.error(e)

    num_vertices = count_vertices(dummy_gdf.iloc[0].geometry)
    logger.info(f"Number of vertices in dummy geometry: {num_vertices}")
