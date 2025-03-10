"""
main.py

This is the main orchestrator script for the Air Spatial Insights project.
It sequentially loads the data, preprocesses the monitor records, performs spatial
analysis (including spatial joining, DBSCAN clustering, and Moran’s I calculations),
visualizes the results, and saves outputs to disk.

Usage:
    python -m scripts.main
"""

from visualization import (
    plot_choropleth,
    create_folium_map,
    plot_kdistance,
    plot_lisa_map,
    plot_moran_scatter,
)
from analysis import (
    spatial_join_and_aggregate,
    dbscan_clustering,
    moran_global_local,
)
from preprocessing import (
    clean_monitors,
    filter_pollutant,
    remove_outliers,
    apply_boxcox,
)
from data_loader import load_census_tracts, load_monitors
import os
import logging

# Set up logging.
from utils import setup_logging
logger = setup_logging(level=logging.INFO)

# Data loading.

# Preprocessing.

# Analysis.

# Visualization.


def main():
    logger.info("Starting Air Spatial Insights workflow.")

    # -----------------------
    # Load Data
    # -----------------------
    tracts = load_census_tracts("data/tl_2024_06_tract.zip")
    monitors_gdf = load_monitors("data/annual_conc_by_monitor_2024.csv")
    logger.info(
        f"Loaded {len(tracts)} census tracts and {len(monitors_gdf)} monitor records.")

    # -----------------------
    # Preprocessing
    # -----------------------
    monitors_gdf = clean_monitors(monitors_gdf)
    monitors_gdf = filter_pollutant(monitors_gdf, pollutant="Ozone", year=2024)
    monitors_gdf = remove_outliers(
        monitors_gdf, col="value_mean", lower_q=0.01, upper_q=0.99)
    monitors_gdf, lambda_val = apply_boxcox(monitors_gdf, col="value_mean")
    logger.info(f"Preprocessing complete. Box-Cox lambda: {lambda_val:.6f}")

    # -----------------------
    # Analysis
    # -----------------------
    # Spatial join and aggregation: compute mean Ozone per tract.
    tracts_ozone = spatial_join_and_aggregate(
        monitors_gdf, tracts, col="value_mean")

    # DBSCAN clustering on the preprocessed monitors (using the Box-Cox transformed values).
    monitors_clustered = dbscan_clustering(
        monitors_gdf, col="value_boxcox", eps=0.52, min_samples=6)

    # Compute Global and Local Moran's I on the aggregated tract data.
    global_moran, p_value, lisa_clusters = moran_global_local(
        tracts_ozone, col="Ozone")
    logger.info(
        f"Global Moran's I: {global_moran:.3f}, p-value: {p_value:.3f}")

    # -----------------------
    # Visualization
    # -----------------------
    # Plot static choropleth map using Matplotlib.
    fig_choro = plot_choropleth(
        tracts_ozone, col="Ozone", title="Census Tracts by Mean Ozone")
    fig_choro.figure.savefig("results/choropleth_ozone.png", dpi=300)
    logger.info("Static choropleth map saved to 'results/choropleth_ozone.png'.")

    # Create interactive Folium map.
    folium_map = create_folium_map(
        tracts_ozone, col="Ozone", output_html="maps/ozone_map.html")
    logger.info(
        "Interactive Folium map created and saved to 'maps/ozone_map.html'.")

    # Optionally, if you have computed the k-distance array (e.g., kth_distances) during DBSCAN clustering,
    # you could plot the k-distance graph. (Assuming kth_distances is available)
    # For example:
    # fig_kdist = plot_kdistance(kth_distances, k=5)
    # fig_kdist.figure.savefig("results/k_distance.png", dpi=300)

    # Plot the LISA cluster map.
    fig_lisa = plot_lisa_map(
        tracts_ozone, lisa_col="LISA_cluster", title="Local Moran's I Cluster Map")
    fig_lisa.figure.savefig("results/lisa_cluster_map.png", dpi=300)
    logger.info("LISA cluster map saved to 'results/lisa_cluster_map.png'.")

    # Compute a Queen contiguity weights matrix for the Moran scatter plot.
    import libpysal
    w = libpysal.weights.Queen.from_dataframe(tracts_ozone, use_index=False)
    w.transform = "R"
    fig_moran = plot_moran_scatter(tracts_ozone, col="Ozone", w=w)
    fig_moran.figure.savefig("results/moran_scatter.png", dpi=300)
    logger.info("Moran scatter plot saved to 'results/moran_scatter.png'.")

    # -----------------------
    # Save Results
    # -----------------------
    os.makedirs("results", exist_ok=True)
    tracts_ozone.to_file("results/census_tracts_pollution.shp")
    monitors_clustered.to_file("results/air_quality_monitors_clusters.shp")
    logger.info("Shapefiles saved to the 'results' directory.")

    logger.info("Air Spatial Insights workflow completed successfully.")


if __name__ == "__main__":
    main()
