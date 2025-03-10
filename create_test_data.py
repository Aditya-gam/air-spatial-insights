#!/usr/bin/env python
"""
create_test_data.py

This script creates test subsets from the main datasets located in "data/".
It creates both valid and deliberately problematic/malformed datasets for testing purposes.
The output files are saved in "data/test_data/".

Subsets for the census tract data include:
  - A small valid subset ("tracts_valid.shp")
  - A subset reprojected to a different CRS ("tracts_diff_crs.shp")
  - An “empty” GeoDataFrame saved as a shapefile ("tracts_empty.shp")
  - A version with no CRS defined ("tracts_no_crs.shp")
  - A CSV version with no geometry column ("tracts_no_geometry.csv")

Subsets for the monitors data include:
  - A small valid subset ("monitors_valid.csv")
  - A CSV with missing required columns (Latitude/Longitude removed) ("monitors_missing_cols.csv")
  - A CSV with out-of-range coordinate values ("monitors_out_of_range.csv")
  - An empty CSV ("monitors_empty.csv")
  - A malformed CSV file (not in proper CSV format) ("monitors_malformed.csv")
  - A CSV version that will later need reprojection ("monitors_diff_crs.csv")

Make sure that the original datasets are located in the "data/" folder:
  - "data/tl_2024_06_tract.zip" (census tract shapefile)
  - "data/annual_conc_by_monitor_2024.csv" (air quality monitor data)

Usage:
    python create_test_data.py
"""

import os
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Configure basic logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("create_test_data")

# Define file paths for original data
TRACTS_PATH = "data/raw/tl_2024_06_tract.zip"
MONITORS_PATH = "data/raw/annual_conc_by_monitor_2024.csv"

# Define output directory for test data
TEST_DATA_DIR = "data/test_data"


def create_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")
    else:
        logger.info(f"Directory already exists: {path}")


def create_tracts_subsets():
    logger.info("Creating census tracts test subsets...")
    # Load full tracts data
    tracts = gpd.read_file(TRACTS_PATH)

    # 1. Valid subset: first 10 features
    tracts_valid = tracts.head(10)
    tracts_valid.to_file(os.path.join(TEST_DATA_DIR, "tracts_valid.shp"))
    logger.info("Saved valid tracts subset (tracts_valid.shp).")

    # 2. Reprojected subset: first 10 features to EPSG:3857 (different CRS)
    tracts_diff_crs = tracts.head(10).to_crs(epsg=3857)
    tracts_diff_crs.to_file(os.path.join(TEST_DATA_DIR, "tracts_diff_crs.shp"))
    logger.info("Saved reprojected tracts subset (tracts_diff_crs.shp).")

    # 3. Empty GeoDataFrame with same schema
    tracts_empty = tracts.iloc[0:0].copy()
    tracts_empty.to_file(os.path.join(TEST_DATA_DIR, "tracts_empty.shp"))
    logger.info("Saved empty tracts shapefile (tracts_empty.shp).")

    # 4. Tracts with no CRS defined: remove CRS from valid subset and save as shapefile
    tracts_no_crs = tracts_valid.copy()
    tracts_no_crs.crs = None
    tracts_no_crs.to_file(os.path.join(TEST_DATA_DIR, "tracts_no_crs.shp"))
    logger.info("Saved tracts shapefile with no CRS (tracts_no_crs.shp).")

    # 5. Attributes-only version (CSV) with no geometry column
    tracts_no_geometry = tracts_valid.drop(columns=["geometry"])
    tracts_no_geometry.to_csv(os.path.join(
        TEST_DATA_DIR, "tracts_no_geometry.csv"), index=False)
    logger.info(
        "Saved tracts attributes without geometry (tracts_no_geometry.csv).")


def create_monitors_subsets():
    logger.info("Creating monitors test subsets...")
    # Load full monitors CSV into DataFrame
    monitors_df = pd.read_csv(MONITORS_PATH)
    # Create a GeoDataFrame for the valid case
    monitors_df["geometry"] = pd.Series(
        [Point(x, y)
         for x, y in zip(monitors_df["Longitude"], monitors_df["Latitude"])]
    )
    monitors_valid = gpd.GeoDataFrame(monitors_df, crs="EPSG:4326")

    # 1. Valid subset: first 10 records
    monitors_valid_subset = monitors_valid.head(10)
    monitors_valid_subset.to_csv(os.path.join(
        TEST_DATA_DIR, "monitors_valid.csv"), index=False)
    logger.info("Saved valid monitors subset (monitors_valid.csv).")

    # 2. CSV missing required columns: remove Latitude and Longitude
    monitors_missing_cols = monitors_valid_subset.drop(
        columns=["Latitude", "Longitude"])
    monitors_missing_cols.to_csv(os.path.join(
        TEST_DATA_DIR, "monitors_missing_cols.csv"), index=False)
    logger.info(
        "Saved monitors CSV with missing columns (monitors_missing_cols.csv).")

    # 3. CSV with out-of-range coordinate values: modify some coordinates
    monitors_out_of_range = monitors_valid_subset.copy()
    # Set first two rows' Latitude and Longitude to invalid values
    # > 90
    monitors_out_of_range.loc[monitors_out_of_range.index[0],
                              "Latitude"] = 100.0
    # < -180
    monitors_out_of_range.loc[monitors_out_of_range.index[1],
                              "Longitude"] = -200.0
    monitors_out_of_range.to_csv(os.path.join(
        TEST_DATA_DIR, "monitors_out_of_range.csv"), index=False)
    logger.info(
        "Saved monitors CSV with out-of-range coordinates (monitors_out_of_range.csv).")

    # 4. Empty CSV: create an empty DataFrame with the same columns
    monitors_empty = monitors_valid_subset.iloc[0:0].copy()
    monitors_empty.to_csv(os.path.join(
        TEST_DATA_DIR, "monitors_empty.csv"), index=False)
    logger.info("Saved empty monitors CSV (monitors_empty.csv).")

    # 5. Malformed CSV: write a text file that is not proper CSV
    malformed_content = "This is not a CSV file.\nIt does not contain proper data."
    with open(os.path.join(TEST_DATA_DIR, "monitors_malformed.csv"), "w") as f:
        f.write(malformed_content)
    logger.info("Saved malformed monitors file (monitors_malformed.csv).")

    # 6. CSV with different CRS: For monitors, we simulate this by writing a CSV
    #    that will later be interpreted as having a different CRS.
    #    (Since CSV does not have a CRS, we simulate it by including coordinate values
    #     that represent a different system, e.g., EPSG:3857 coordinates.)
    monitors_diff_crs = monitors_valid_subset.copy()
    # Reproject the geometry to EPSG:3857 and extract x,y as new coordinate columns
    monitors_diff_crs = monitors_diff_crs.to_crs(epsg=3857)
    monitors_diff_crs["Longitude"] = monitors_diff_crs.geometry.x
    monitors_diff_crs["Latitude"] = monitors_diff_crs.geometry.y
    monitors_diff_crs.to_csv(os.path.join(
        TEST_DATA_DIR, "monitors_diff_crs.csv"), index=False)
    logger.info(
        "Saved monitors CSV with coordinates in EPSG:3857 (monitors_diff_crs.csv).")


def main():
    create_directory(TEST_DATA_DIR)
    create_tracts_subsets()
    create_monitors_subsets()
    logger.info("All test data subsets created successfully.")


if __name__ == "__main__":
    main()
