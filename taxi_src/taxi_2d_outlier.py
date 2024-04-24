import calendar
import pandas as pd
import numpy as np
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from shapely import wkt
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timedelta
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from collections import defaultdict
from pyproj import Transformer
from geopy.geocoders import Photon
from geopy.extra.rate_limiter import RateLimiter

# Initialize Nominatim API
geolocator = Photon(user_agent="measurements")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# Bounds of NYC in wgs84 format
nyc_bounds_wgs84 = {
    'west': -74.255735,
    'south': 40.496044,
    'east': -73.700272,
    'north': 40.915256,
}
# Transform coordinates of NYC bounds
nyc_bounds_mercator = {
    'west': transformer.transform(nyc_bounds_wgs84['west'], nyc_bounds_wgs84['south'])[0],
    'south': transformer.transform(nyc_bounds_wgs84['west'], nyc_bounds_wgs84['south'])[1],
    'east': transformer.transform(nyc_bounds_wgs84['east'], nyc_bounds_wgs84['north'])[0],
    'north': transformer.transform(nyc_bounds_wgs84['east'], nyc_bounds_wgs84['north'])[1],
}


class Taxi_2D_Outlier:
    @staticmethod
    def find_outlier(p_X, p_Psi, p_Y, p_W, p_Phi, p_percentage, p_count, p_month) -> None:
        """
        Plots outliers based on magnitude from the residual of X-(Ψ * p_Y * p_W * p_Φ).
        Args:
            p_X: Original temporal signal input
            p_Psi: Graph dictionary Ψ
            p_Y: Encoding matrix to reconstruct p_X
            p_W: Encoding matrix to reconstruct p_X
            p_Phi: Time series p_Φ
            p_percentage: Top percentile of outliers that the user wishes to plot
            p_count: Number of subplots to display, one for each outlier in p_count. Maximum count of 10.

        """
        # Compute the residual and the reconstructed X
        res = p_X - (p_Psi @ p_Y @ p_W @ p_Phi)
        reconstructed_X = p_Psi @ p_Y @ p_W @ p_Phi

        location_data = pd.read_csv('../Taxi_prep/taxi+_zone_lookup.csv', header=None,
                                    names=['ID', 'Borough', 'Neighborhood', 'ZoneType'])

        # Used for indexing
        flatten_residual = res.flatten()
        sorted_values = np.argsort(np.abs(flatten_residual))[::-1]

        # Find the percentage of how many outliers there will be based on input
        num_outliers_percentage = int(len(flatten_residual) * p_percentage / 100)
        outlier_indices_percentage = sorted_values[:num_outliers_percentage]
        row_indices_percentage, col_indices_percentage = np.unravel_index(outlier_indices_percentage, p_X.shape)

        # Determine the indices of the top fixed number of outliers
        num_outliers_for_subplots = min(p_count, 10)
        outlier_indices_fixed = sorted_values[:num_outliers_for_subplots]
        row_indices_fixed, col_indices_fixed = np.unravel_index(outlier_indices_fixed, p_X.shape)

        # Use magnitude for X and reconstructed X
        X_magnitude = np.abs(p_X)
        reconstructed_X = np.abs(reconstructed_X)

        # Plotting!
        fig = plt.figure(figsize=(18, 4 * num_outliers_for_subplots))
        # 3 "grids"
        gs = GridSpec(1, 3, figure=fig, width_ratios=[3, 1, 1])

        # Subplot 1: Original Data with Percentage-Based Outliers
        ax_matrix = fig.add_subplot(gs[0])
        ax_matrix.imshow(X_magnitude, cmap='gray')
        ax_matrix.scatter(col_indices_percentage, row_indices_percentage, color='red', s=50,
                          label=f"Top {p_percentage}Percentage Outliers")
        ax_matrix.set_xlabel('Column Index')
        ax_matrix.set_ylabel('Row Index')
        ax_matrix.set_title(f'Original Data X with Top {p_percentage}% Outliers')

        # Subplot 2: Local Neighborhood of Outliers Corresponding to the Time Series
        # Groups the outliers (row, col) by row index so that they can be plotted on the same subplot together
        outliers_by_series = defaultdict(list)
        for row_idx, col_idx in zip(row_indices_fixed, col_indices_fixed):
            Phi_row_idx = row_idx % p_Phi.shape[0]
            outliers_by_series[Phi_row_idx].append((row_idx, col_idx))

        # Left hand side will have the local neighborhood of the outliers with the time series
        left_column_rhs = GridSpecFromSubplotSpec(min(len(outlier_indices_fixed), 10), 1, subplot_spec=gs[1])
        # Right hand side will have the fit
        right_column_rhs = GridSpecFromSubplotSpec(10, 1, subplot_spec=gs[2])  # Assuming 10 new subplots

        time_points = np.arange(p_Phi.shape[1])
        label_local_neighborhood_displayed = False
        label_reconstructed_neighborhood_displayed = False
        label_actual_outlier_displayed = False
        label_reconstructed_outlier_displayed = False
        for i, (phi_row_idx, indices) in enumerate(outliers_by_series.items()):
            # Add vertical subplots down each column
            ax_ts = fig.add_subplot(left_column_rhs[i])
            ax_ts_right = fig.add_subplot(right_column_rhs[i])
            # Fetches the time series at that particular index (by row)
            time_series = p_Phi[phi_row_idx, :]

            for row_idx, col_idx in indices:
                # Finds the local neighborhood
                start = max(col_idx - num_outliers_for_subplots, 0)
                end = min(col_idx + num_outliers_for_subplots + 1, len(time_series))

                # Plots the local neighborhood (time series) along with the corresponding X
                ax_ts.plot(time_points[start:end], time_series[start:end], label='Local Neighborhood' if i == 0 else "")
                ax_ts.scatter(time_points[col_idx], time_series[col_idx], color='red', zorder=5,
                              label='Outlier' if i == 0 else "")
                name = location_data.loc[row_idx + 1, 'Neighborhood']
                date_time_for_index = datetime(2017, p_month, 1) + timedelta(hours=int(col_idx))
                ax_ts.set_title(f"{name} | {date_time_for_index.strftime('%B %d, %Y, %H:%M')}")
                ax_ts.annotate(f'({row_idx}, {col_idx})',
                               (time_points[col_idx], time_series[col_idx]),
                               textcoords="axes fraction",
                               xytext=(0.95, 0.05),
                               ha='right',
                               va='bottom',
                               fontweight='bold',
                               bbox=dict(facecolor='white', alpha=0.5),
                               fontsize=8)

                # Subplot 3: Fit of the Outlier Corresponding to X in Local Neighborhood
                # For the same neighborhood, plot X values against reconstructed values
                if not label_local_neighborhood_displayed:
                    ax_ts_right.plot(time_points[start:end], X_magnitude[row_idx, start:end], 'b-',
                                     label='Local Neighborhood of X')
                    label_local_neighborhood_displayed = True
                else:
                    ax_ts_right.plot(time_points[start:end], X_magnitude[row_idx, start:end], 'b-')

                if not label_reconstructed_neighborhood_displayed:
                    ax_ts_right.plot(time_points[start:end], reconstructed_X[row_idx, start:end], 'g--',
                                     label='Reconstructed Neighborhood of X')
                    label_reconstructed_neighborhood_displayed = True
                else:
                    ax_ts_right.plot(time_points[start:end], reconstructed_X[row_idx, start:end], 'g--')

                if not label_actual_outlier_displayed:
                    ax_ts_right.scatter(col_idx, X_magnitude[row_idx, col_idx], color='blue', zorder=5,
                                        label='Actual Outlier (X)')
                    label_actual_outlier_displayed = True
                else:
                    ax_ts_right.scatter(col_idx, X_magnitude[row_idx, col_idx], color='blue', zorder=5)

                if not label_reconstructed_outlier_displayed:
                    ax_ts_right.scatter(col_idx, reconstructed_X[row_idx, col_idx], color='green', zorder=5,
                                        label='Reconstructed Outlier (X)')
                    label_reconstructed_outlier_displayed = True
                else:
                    ax_ts_right.scatter(col_idx, reconstructed_X[row_idx, col_idx], color='green', zorder=5)

                # Sets the x-axis for each subplot
                ax_ts.set_xlim(start, end)
                ax_ts_right.set_xlim(start, end)

            # Right hand side: legend, since no specific point is necessary but just the representation
            if i == 0:
                handles, labels = ax_ts_right.get_legend_handles_labels()
                ax_ts_right.legend(handles, labels, loc='upper right', bbox_to_anchor=(2.15, 1.3), fontsize='small')

            # Sets y-axis for each subplot
            y_min, y_max = time_series[start:end].min(), time_series[start:end].max()
            ax_ts.set_ylim(y_min, y_max)
            y_min_right, y_max_right = np.min(X_magnitude), np.max(X_magnitude)
            ax_ts_right.set_ylim(y_min_right, y_max_right)

            # For the last (bottom) subplot only
            if i == num_outliers_for_subplots - 1:
                ax_ts.set_xlabel('Time (Hr)')
                ax_ts.set_ylabel('Volume against Time')

            ax_ts.grid(True)
            ax_ts_right.grid(True)

        # Adjust as needed for visualization
        plt.subplots_adjust(hspace=1.2, bottom=0.1, right=0.9, wspace=0.5)
        plt.show()

    @staticmethod
    def find_row_outlier(p_X, p_Psi, p_Y, p_W, p_Phi, p_count, p_month, p_method):
        """
        Plots row outliers based on average magnitude from the residual of X-(Ψ * p_Y * p_W * p_Φ).
        Args:
            p_X: Original temporal signal input
            p_Psi: Graph dictionary Ψ
            p_Y: Encoding matrix to reconstruct p_X
            p_W: Encoding matrix to reconstruct p_X
            p_Phi: Time series p_Φ
            p_count: Number of subplots to display, one for each row outlier in p_count. Maximum count of 10.
        """
        # Calculate the residual and reconstruction of X
        res = p_X - (p_Psi @ p_Y @ p_W @ p_Phi)
        reconstructed_X = (p_Psi @ p_Y @ p_W @ p_Phi)
        magnitude_X = abs(p_X)
        reconstructed_X = abs(reconstructed_X)

        # Take the average of each row
        row_avg = np.abs(res).mean(axis=1)

        # Determine the number of rows to plot based on user input and sort average value
        sorted_rows = np.argsort(row_avg)[::-1]
        p_count = min(p_count, 10)
        outlier_rows = sorted_rows[:p_count]
        num_plots = len(outlier_rows)
        location_data = pd.read_csv('../Taxi_prep/taxi+_zone_lookup.csv', header=None,
                                    names=['ID', 'Borough', 'Neighborhood', 'ZoneType'])

        # Define grid
        fig = plt.figure(figsize=(15, 3 * p_count))
        gs = GridSpec(num_plots, 2)  # Define grid layout for the figure
        ax = fig.add_subplot(gs[p_count // 4:p_count - (p_count // 4) + 1, 0])

        # Read centroids csv to gather locations for IDs
        df = pd.read_csv("../Taxi_prep/NHoodNameCentroids.csv")
        # New dataframe column for geometry
        df['geometry'] = df['the_geom'].apply(wkt.loads)
        # Add CRS to new GDF
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        # Determine outlier row IDs and fetch their
        object_ids = [r + 1 for r in outlier_rows]
        # New GDF containing only the outlier IDs, despite incorrect name mappings (we'll deal with that below)
        gdf_outliers = gdf[gdf['OBJECTID'].isin(object_ids)]
        gdf_outliers.crs = "EPSG:4326"
        # Define CRS
        gdf_outliers = gdf_outliers.to_crs(epsg=3857)
        new_object_id = []
        add_these_separately = []
        # The mapping from zone_lookup.csv to NHoodNameCentroids is not 1:1, so we need to convert the mapping to
        # properly obtain the zone ID and zone name to plot

        for idx, row in gdf_outliers.iterrows():
            zone_name = location_data.loc[idx + 1, 'Neighborhood']  # Get the real zone name corresponding to the outlier
            try:
                # Get the object ID in the dataframe for that zone
                object_id = df.loc[df['Name'] == zone_name, 'OBJECTID'].iloc[0]
                # Re-map
                new_object_id.append(object_id)
            except IndexError:
                # Can't find it in the dataframe (either due to a misspelling/data error/etc.
                # Make call to geocode API to gather coordinates (use sparingly, otherwise rate limited)
                location = geocode(zone_name)
                if location:
                    # Build these on to the GDF separately
                    point = Point(location.longitude, location.latitude)
                    add_these_separately.append((zone_name, point))

        # Build new GDF
        add_these_separately_gdf = pd.DataFrame(add_these_separately, columns=['Zone', 'geometry'])
        add_these_separately_gdf = gpd.GeoDataFrame(add_these_separately_gdf, geometry='geometry', crs="EPSG:4326")

        # New GDF of API-retrieved location, concatenate to the old GDF
        proper_outlier_gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        proper_outlier = proper_outlier_gdf[proper_outlier_gdf['OBJECTID'].isin(new_object_id)]
        proper_outlier = gpd.GeoDataFrame(pd.concat([proper_outlier, add_these_separately_gdf], ignore_index=True),
                                          crs="EPSG:4326")

        # Find all unique zones
        unique_names = proper_outlier['Name'].dropna().unique()
        unique_zones = proper_outlier['Zone'].dropna().unique()
        all_unique_zones = np.concatenate((unique_names, unique_zones))
        # Subplot 1: Plotting Locations of Outliers on Map of NYC
        # Plot zones to colors
        colors = plt.colormaps['hsv']
        gen_unique_colors = [colors(i / len(all_unique_zones)) for i in range(len(all_unique_zones))]
        zone_color_map = {zone: gen_unique_colors[i] for i, zone in enumerate(all_unique_zones)}

        # Convert CRS
        proper_outlier = proper_outlier.to_crs(epsg=3857)

        # Plot locations on map of NYC
        for idx, row in proper_outlier.iterrows():
            zone_name = row.Name if pd.notnull(row['Name']) else row.Zone
            color = zone_color_map[zone_name]
            ax.plot(row.geometry.x, row.geometry.y, 'o', color=color, markersize=10, label=zone_name)

        # Base map, bounds, and other settings
        ax.set_xlim([nyc_bounds_mercator['west'], nyc_bounds_mercator['east']])
        ax.set_ylim([nyc_bounds_mercator['south'], nyc_bounds_mercator['north']])
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        ax.set_axis_off()
        m = calendar.month_name[p_month]
        s = "NYC Taxi Pickups" if p_method == "pickup" else "NYC Taxi Dropoffs"
        ax.set_title(f"{m} {s}")
        ax.set_aspect('equal')
        # Only plot unique labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # This removes duplicates
        plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=8)

        # Iterate through each row index to build grid
        for i, row_idx in enumerate(outlier_rows):
            # Subplot 2: Plot the Row Values Against Their Reconstructed Values (Fit)
            ax_compare = fig.add_subplot(gs[i, 1])
            ax_compare.plot(magnitude_X[row_idx, :], 'b-', label='X')
            ax_compare.plot(reconstructed_X[row_idx, :], 'g--', label='Reconstructed X')

            borough = location_data.loc[row_idx + 1, 'Borough']
            neighborhood = location_data.loc[row_idx + 1, 'Neighborhood']
            ax_compare.set_title(f"{borough}: {neighborhood}")
            ax_compare.set_ylim(np.min(magnitude_X), np.max(magnitude_X))

            # Add legend to first subplot
            if i == 0:
                handles, labels = ax_compare.get_legend_handles_labels()
                ax_compare.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1.3), fontsize='small')

            # Enable the xlabel only for bottom subplot
            if i == num_plots - 1:
                # ax.set_xlabel('Time Index')
                ax_compare.set_xlabel('Time Index')
            else:
                # ax.tick_params(labelbottom=False)
                ax_compare.tick_params(labelbottom=False)

            for index in range(0, 744, 24):
                ax_compare.axvline(x=index, color='red', linestyle='-',
                                   linewidth=0.5)  # For the subplot of reconstructed values
            # ax.grid(True)
            ax_compare.grid(True)

        plt.subplots_adjust(hspace=0.5, bottom=0.1, right=0.9)
        plt.show()

    @staticmethod
    def find_col_outlier(p_X, p_Psi, p_Y, p_W, p_Phi, p_count, p_month, p_method):
        """
        Plots column outliers based on average magnitude from the residual of X-(Ψ * p_Y * p_W * p_Φ).
        Args:
            p_X: Original temporal signal input
            p_Psi: Graph dictionary Ψ
            p_Y: Encoding matrix to reconstruct p_X
            p_W: Encoding matrix to reconstruct p_X
            p_Phi: Time series p_Φ
            p_count: Number of subplots to display, one for each column outlier in p_count. Maximum count of 10.
        """
        # Calculate the residual and reconstruction of X
        res = p_X - (p_Psi @ p_Y @ p_W @ p_Phi)
        magnitude_X = abs(p_X)
        reconstructed_X = p_Psi @ p_Y @ p_W @ p_Phi
        reconstructed_X = abs(reconstructed_X)

        # Take the average of each column
        col_avg = np.abs(res).mean(axis=0)
        sorted_columns = np.argsort(col_avg)[::-1]
        outlier_columns = sorted_columns[:p_count]

        # Antioutlier
        sorted_columns = np.argsort(col_avg)  # Do not reverse the order
        antianomaly_columns = sorted_columns[:p_count]

        # Determine number of time series to plot
        num_series_to_plot = min(p_Phi.shape[0], 10)  # Plot up to the first 10 time series
        fig = plt.figure(figsize=(15, 3 * p_count))
        gs = GridSpec(num_series_to_plot, 2)  # Define grid layout for the figure

        for i, col_idx in enumerate(antianomaly_columns):
            # Subplot 1: Plot the AntiAnomaly Columns Against Their Reconstructed Values (Fit)
            ax = fig.add_subplot(gs[i, 0])
            ax.plot(magnitude_X[:, col_idx], 'b-', label='X')
            ax.plot(reconstructed_X[:, col_idx], 'g--', label='Reconstructed X')

            # Set y-axis
            ax.set_ylim(min(np.min(magnitude_X[:, col_idx]), np.min(reconstructed_X[:, col_idx])),
                        max(np.max(magnitude_X[:, col_idx]), np.max(reconstructed_X[:, col_idx])))
            date_time_for_index = datetime(2017, p_month, 1) + timedelta(hours=int(col_idx))

            ax.set_title(f"Day/Time: {date_time_for_index.strftime('%B %d, %Y, %H:%M')}")

            # Enable the xlabel only for bottom subplot
            if i == num_series_to_plot - 1:
                if p_method == "pickup":
                    s = "Pickup"
                else:
                    s = "Dropoff"
                ax.set_xlabel('Node Index')
                ax.set_ylabel(f'Volume {s}')
            else:
                # ax.tick_params(labelbottom=False)
                ax.tick_params(labelbottom=False)

        # Iterate through each column outlier index
        for i, col_idx in enumerate(outlier_columns):
            # Subplot 2: Plot the Column Values Against Their Reconstructed Values (Fit)
            ax_compare = fig.add_subplot(gs[i, 1])
            ax_compare.plot(magnitude_X[:, col_idx], 'b-', label='X')
            ax_compare.plot(reconstructed_X[:, col_idx], 'g--', label='Reconstructed X')

            # Set y-axis
            ax_compare.set_ylim(min(np.min(magnitude_X[:, col_idx]), np.min(reconstructed_X[:, col_idx])),
                                max(np.max(magnitude_X[:, col_idx]), np.max(reconstructed_X[:, col_idx])))
            date_time_for_index = datetime(2017, p_month, 1) + timedelta(hours=int(col_idx))

            ax_compare.set_title(f"Day/Time: {date_time_for_index.strftime('%B %d, %Y, %H:%M')}")

            # Add legend to first subplot
            if i == 0:
                handles, labels = ax_compare.get_legend_handles_labels()
                ax_compare.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1.3), fontsize='small')

            # Enable the xlabel only for bottom subplot
            if i == num_series_to_plot - 1:
                if p_method == "pickup":
                    s = "Pickup"
                else:
                    s = "Dropoff"

                ax_compare.set_xlabel('Node Index')
                ax_compare.set_ylabel(f'Volume {s}')
            else:
                # ax.tick_params(labelbottom=False)
                ax_compare.tick_params(labelbottom=False)

            ax_compare.grid(True)

        # Adjust as needed for visualization
        plt.subplots_adjust(hspace=0.5, bottom=0.1, right=0.9)
        plt.show()
