import calendar
import csv
from tgsd_clustering import cluster

import pandas as pd
import numpy as np
import scipy.io
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from shapely import wkt

from tgsd import tgsd, gen_gft_new, gen_rama, find_outlier, find_col_outlier, find_row_outlier
from mdtd import mdtd, mdtd_load_config, mdtd_format_numpy_to_csv, mdtd_format_csv_to_numpy, mdtd_find_outlier, mdtd_clustering

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


def load_lat_long():
    """
    Loads the latitude and longitude Matlab file
    Returns:
        Numpy array of lat_long_info.mat
    """
    return scipy.io.loadmat("Taxi_prep/lat_long_info.mat")


def clean_taxi(month, method, perspective):
    """
    Performs TGSD or MDTD algorithms on taxi data for a specific month in 2017, along with various downstream tasks
    Args:
        month: Integer value of month [1, 12]
        method: {"pickup", "dropoff", "both"}. TGSD utilizes either pickup or dropoff information while MDTD will use both.
        perspective: {"point", "row", "col"}. Choice of display for graphs.
    """
    def generate_pickup_or_dropoff_adj(adj_matrix, all_data, method, date_range):
        """
        Generates a square adjacency matrix given some data for a particular month.
        Entry (i, j) in the adjacency matrix is the sum of passengers picked up or dropped off at i or j from i or j
        Args:
            adj_matrix: Some empty adjacency matrix of a given size to fill.
            all_data: pandas DataFrame of all necessary location/time/pickup/dropoff data
            method: {"pickup", "dropoff", "both"}. TGSD utilizes either pickup or dropoff information while MDTD will use both.
            date_range: Given days in some particular month.

        Returns:
            Numpy array of the formatted data and symmetrical adjacency matrix.
        """
        if method == "pickup":
            aggregated_data = all_data.groupby(['time_bin', 'PULocationID']).size().reset_index(name='trip_count')
            pivot_table = aggregated_data.pivot(index='time_bin', columns='PULocationID', values='trip_count')
            trip_counts = all_data.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='trip_count')
            adj_matrix[trip_counts['PULocationID'] - 1, trip_counts['DOLocationID'] - 1] = trip_counts['trip_count']
        elif method == "dropoff":
            aggregated_data = all_data.groupby(['time_bin', 'DOLocationID']).size().reset_index(name='trip_count')
            pivot_table = aggregated_data.pivot(index='time_bin', columns='DOLocationID', values='trip_count')
            trip_counts = all_data.groupby(['DOLocationID', 'PULocationID']).size().reset_index(name='trip_count')
            adj_matrix[trip_counts['DOLocationID'] - 1, trip_counts['PULocationID'] - 1] = trip_counts['trip_count']

        adj_matrix += adj_matrix.T
        np.fill_diagonal(adj_matrix, adj_matrix.diagonal() // 2)

        pivot_table = pivot_table.reindex(date_range, fill_value=0).reindex(columns=np.arange(1, 266), fill_value=0)
        pivot_table.fillna(0, inplace=True)
        return np.array(pivot_table.values.T), adj_matrix

    def extract_adj_matrix_csv(adj_matrix, pickup_or_dropoff):
        """
        Helper method to extract a .csv of pickup or dropoff data from an adjacency matrix as a numpy array
        Args:
            adj_matrix: Numpy array of an adjacency matrix
            pickup_or_dropoff: Either pickup or dropoff data to extract
        """
        rows, cols = np.nonzero(adj_matrix)
        data = list(zip(rows + 1, cols + 1))  # Adding 1 to convert from 0-based to 1-based indexing, if necessary
        df_nonzero = pd.DataFrame(data, columns=['r', 'c'])
        df_nonzero.to_csv(f'{pickup_or_dropoff}_adjacency.csv', index=False, header=False)

    def extract_tensor_csv(all_data, date_range, num_locations):
        """
        Helper method to extract a .csv of pickup and dropoff data from a tensor as a numpy array
        Args:
            all_data: pandas DataFrame of all necessary location/time/pickup/dropoff data
            date_range: Given days in some particular month.
        """
        time_bins = all_data['time_bin'].sort_values().unique()
        time_bin_to_index = {time: index for index, time in enumerate(time_bins)}
        all_data['time_index'] = all_data['time_bin'].apply(lambda x: time_bin_to_index[x])
        trip_counts = all_data.groupby(['PULocationID', 'DOLocationID', 'time_index']).size().reset_index(name='count')
        shape = (num_locations, num_locations * len(date_range))
        trip_counts['PULocationID'] -= 1
        trip_counts['DOLocationID'] -= 1
        row = trip_counts['PULocationID']
        col = trip_counts['DOLocationID'] * len(date_range) + trip_counts['time_index']
        data = trip_counts['count']
        sparse_matrix = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
        tensor = sparse_matrix.toarray().reshape((num_locations, num_locations, len(date_range)))
        mdtd_format_numpy_to_csv(tensor)

    cols = ['PULocationID', 'DOLocationID', 'formatted_time']
    month_st = "0" + str(month) if month < 10 else str(month)
    all_data = pd.read_csv(f'Taxi_prep/tensor_data/sec_yellow_tripdata_2017-{month_st}', header=None, usecols=[1, 2, 3],
                           names=cols)

    # Convert formatted_time to datetime, adjust to time_bin
    all_data['time_bin'] = pd.to_timedelta(all_data['formatted_time'], unit='s') + pd.to_datetime('2017-01-01')
    all_data['time_bin'] = all_data['time_bin'].dt.floor('H')

    end_days_map = {"02": "28", "04": "30", "06": "30", "09": "30", "11": "30"}
    end_days = end_days_map.get(month_st, "31")

    date_range = pd.date_range(start=f'2017-{month_st}-01', end=f'2017-{month_st}-{end_days} 23:00:00', freq='H')
    num_locations = 265  # Total number of unique locations
    adj_template = np.zeros((num_locations, num_locations), dtype=int)

    if method == "pickup" or method == "dropoff":
        # TGSD
        # Generate respective data and adjacency matrix given method
        d, adj_matrix = generate_pickup_or_dropoff_adj(adj_template, all_data, method, date_range)
        Psi_GFT = gen_gft_new(adj_matrix, False)
        Psi_GFT = Psi_GFT[0]  # eigenvectors
        ram = gen_rama(t=d.shape[1], max_period=24)  # t = 2nd dim of Ramanujan dictionary
        mask = np.random.randint(0, 65536, size=(1, 3500), dtype=np.uint16)
        # Perform TGSD
        Y, W = tgsd(d, Psi_GFT, ram, mask, iterations=100, k=7, lambda_1=.1, lambda_2=.1, lambda_3=1,
                    rho_1=.01, rho_2=.01, type="rand")

        # Downstream tasks
        if perspective == "point":
            find_outlier(d, Psi_GFT, Y, W, ram, .1, 30, p_month=month)
        elif perspective == "row":
            find_row_outlier(d, Psi_GFT, Y, W, ram, 10, p_month=month, p_method=method)
        else:
            find_col_outlier(d, Psi_GFT, Y, W, ram, 10, p_month=month, p_method=method)

        # K-Means
        cluster(Psi_GFT, Y)

    else:
        # MDTD
        # _, adj_matrix_pickup = pickup_or_dropoff(adj_template, all_data, "pickup", date_range)
        # _, adj_matrix_dropoff = pickup_or_dropoff(adj_template, all_data, "dropoff", date_range)
        # extract_adj_matrix_csv(adj_matrix_dropoff, "dropoff")
        # extract_adj_matrix_csv(adj_matrix_pickup, "pickup")
        # extract_tensor_csv(all_data, date_range, num_locations)

        X, adj_1, adj_2, mask, count_nnz, num_iters_check, lam, K, epsilon = mdtd_load_config()
        # Perform MDTD
        return_X, recon_X, phi_y = mdtd(is_syn=False, X=X, adj1=adj_1, adj2=adj_2, mask=mask, count_nnz=count_nnz, num_iters_check=num_iters_check,
                                 lam=lam, K=K, epsilon=epsilon)
        # Downstream tasks
        if perspective == "row":
            mdtd_find_outlier(return_X, recon_X, 10, "x")
        elif perspective == "col":
            mdtd_find_outlier(return_X, recon_X, 10, "y")
        else:
            mdtd_find_outlier(return_X, recon_X, 10, "z")
        # K-Means
        mdtd_clustering(phi_y, 7)

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

    location_data = pd.read_csv('Taxi_prep/taxi+_zone_lookup.csv', header=None,
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
    location_data = pd.read_csv('Taxi_prep/taxi+_zone_lookup.csv', header=None,
                                names=['ID', 'Borough', 'Neighborhood', 'ZoneType'])

    # Define grid
    fig = plt.figure(figsize=(15, 3 * p_count))
    gs = GridSpec(num_plots, 2)  # Define grid layout for the figure
    ax = fig.add_subplot(gs[p_count // 4:p_count - (p_count // 4) + 1, 0])

    # Read centroids csv to gather locations for IDs
    df = pd.read_csv("Taxi_prep/NHoodNameCentroids.csv")
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


load_taxi_data = load_lat_long()
mapping = load_taxi_data['Id_and_lat_long']
# Month = Integer value of month, [1,12]]
# Method = "pickup" or "dropoff" or "both"
# Perspective = "point" or "row" or "column"
clean_taxi(month=5, method="both", perspective="row")
