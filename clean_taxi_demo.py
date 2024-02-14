import pandas as pd
import numpy
import numpy as np
import scipy.io
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from shapely import wkt

from tgsd import tgsd, gen_gft_new, gen_rama, find_outlier, find_col_outlier, find_row_outlier
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

# To prevent spamming the service, wrap the geolocator with a rate limiter
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
nyc_bounds_wgs84 = {
    'west': -74.255735,
    'south': 40.496044,
    'east': -73.700272,
    'north': 40.915256,
}
nyc_bounds_mercator = {
    'west': transformer.transform(nyc_bounds_wgs84['west'], nyc_bounds_wgs84['south'])[0],
    'south': transformer.transform(nyc_bounds_wgs84['west'], nyc_bounds_wgs84['south'])[1],
    'east': transformer.transform(nyc_bounds_wgs84['east'], nyc_bounds_wgs84['north'])[0],
    'north': transformer.transform(nyc_bounds_wgs84['east'], nyc_bounds_wgs84['north'])[1],
}


def load_lat_long():
    return scipy.io.loadmat("Taxi_prep/lat_long_info.mat")


def clean_taxi(month, method, perspective, mapping):
    cols = ['PULocationID', 'DOLocationID', 'formatted_time']
    all_data = pd.read_csv(f'Taxi_prep/tensor_data/sec_yellow_tripdata_2017-{month}', header=None, usecols=[1, 2, 3],
                           names=cols)

    # Convert formatted_time to datetime, adjust to time_bin
    if month < 10:
        month_st = "0" + str(month)
    else:
        month_st = str(month)

    all_data['time_bin'] = pd.to_timedelta(all_data['formatted_time'], unit='s') + pd.to_datetime(f'2017-01-01')
    all_data['time_bin'] = all_data['time_bin'].dt.floor('H')

    if month_st == "02":
        end_days = "28"
    elif month_st == "09" or month_st == "11" or month_st == "04" or month_st == "06":
        end_days = "30"
    else:
        end_days = "31"

    date_range = pd.date_range(start=f'2017-{month_st}-01', end=f'2017-{month_st}-{end_days} 23:00:00', freq='H')

    if method == "pickup":
        aggregated_data = all_data.groupby(['time_bin', 'PULocationID']).size().reset_index(name='trip_count')
        pivot_table = aggregated_data.pivot(index='time_bin', columns='PULocationID', values='trip_count')

    elif method == "dropoff":
        aggregated_data = all_data.groupby(['time_bin', 'DOLocationID']).size().reset_index(name='trip_count')
        pivot_table = aggregated_data.pivot(index='time_bin', columns='DOLocationID', values='trip_count')

    pivot_table = pivot_table.reindex(date_range, fill_value=0).reindex(columns=np.arange(1, 266), fill_value=0)
    pivot_table.fillna(0, inplace=True)
    pivot_table_array = pivot_table.values.T
    d = numpy.array(pivot_table_array)

    # Initialize a square matrix of zeros
    num_locations = 265  # Total number of unique locations
    adj_matrix = np.zeros((num_locations, num_locations))

    total_pickups_by_location = aggregated_data.groupby('PULocationID')['trip_count'].sum() if method == "pickup" else \
        aggregated_data.groupby('DOLocationID')['trip_count'].sum()

    for i in range(1, num_locations + 1):
        for j in range(1, num_locations + 1):
            pickups_i = total_pickups_by_location.get(i, 0)
            pickups_j = total_pickups_by_location.get(j, 0)
            adj_matrix[i - 1, j - 1] = pickups_i + pickups_j if i != j else pickups_i

    Psi_GFT = gen_gft_new(adj_matrix, False)
    Psi_GFT = Psi_GFT[0]  # eigenvectors
    ram = gen_rama(t=d.shape[1], max_period=24)  # 200x10
    mask = np.random.randint(0, 65536, size=(1, 3500), dtype=np.uint16)
    Y, W = tgsd(d, Psi_GFT, ram, mask, iterations=100, k=7, lambda_1=.1, lambda_2=.1, lambda_3=1,
                rho_1=.01, rho_2=.01, type="rand")

    if perspective == "point":
        find_outlier(d, Psi_GFT, Y, W, ram, .1, 30)
    elif perspective == "row":
        find_row_outlier(d, Psi_GFT, Y, W, ram, 10, mapping=mapping)
    else:
        find_col_outlier(d, Psi_GFT, Y, W, ram, 10, p_month=month)


def find_outlier(p_X, p_Psi, p_Y, p_W, p_Phi, p_percentage, p_count) -> None:
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
                      label=f'Top {p_percentage}% Outliers')
    ax_matrix.set_xlabel('Column Index')
    ax_matrix.set_ylabel('Row Index')
    ax_matrix.set_title('Original Data X with Top % Outliers')

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
            ax_ts_right.plot(time_points[start:end], X_magnitude[row_idx, start:end], 'b-',
                             label='Local Neighborhood of X' if i == 0 else "")
            ax_ts_right.plot(time_points[start:end], reconstructed_X[row_idx, start:end], 'g--',
                             label='Reconstructed Neighborhood of X' if i == 0 else "")
            ax_ts_right.scatter(col_idx, X_magnitude[row_idx, col_idx], color='blue', zorder=5,
                                label='Actual Outlier (X)' if i == 0 else "")
            ax_ts_right.scatter(col_idx, reconstructed_X[row_idx, col_idx], color='green', zorder=5,
                                label='Reconstructed Outlier (X)' if i == 0 else "")

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
            ax_ts.set_xlabel('Time Index')
            ax_ts_right.set_xlabel('Time Index')

        ax_ts.grid(True)
        ax_ts_right.grid(True)

    # Adjust as needed for visualization
    plt.subplots_adjust(hspace=0.5, bottom=0.1, right=0.9)
    plt.show()


def find_row_outlier(p_X, p_Psi, p_Y, p_W, p_Phi, p_count, mapping):
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

    # !!! Plot coordinate points
    proper_outlier.to_crs(epsg=3857).plot(ax=ax, marker='o', color='red', markersize=50)

    # Sets bounds of the plot to NYC coordinates
    ax.set_xlim([nyc_bounds_mercator['west'], nyc_bounds_mercator['east']])
    ax.set_ylim([nyc_bounds_mercator['south'], nyc_bounds_mercator['north']])

    # Base map of NYC
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    ax.set_axis_off()  # Remove axis for a cleaner map
    ax.set_aspect('equal')

    # Annotations
    for idx, row in proper_outlier.iterrows():
        x, y = row.geometry.x, row.geometry.y
        zone_name = row.Name if row.Name is not None else row.Zone
        # Add annotation with a small offset from the point
        ax.annotate(zone_name, xy=(x, y), xytext=(x, y), textcoords="offset points",
                    fontsize=12, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='black', alpha=0.9))

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


def find_col_outlier(p_X, p_Psi, p_Y, p_W, p_Phi, p_count, p_month):
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

    # Determine number of time series to plot
    num_series_to_plot = min(p_Phi.shape[0], 10)  # Plot up to the first 10 time series
    fig = plt.figure(figsize=(15, 3 * p_count))
    gs = GridSpec(num_series_to_plot, 2)  # Define grid layout for the figure

    # Iterate through each column outlier index
    for i, col_idx in enumerate(outlier_columns):
        # Subplot 1: Average Value of the Column Plotted Against the Time Series
        ax = fig.add_subplot(gs[i, 0])
        # Extracting the vertical time series for the outlier column
        vertical_series = p_Phi[:, col_idx]
        ax.plot(vertical_series, color='blue')

        # Plot the column index
        ax.annotate(f'Column {col_idx}', xy=(0.0, 0.95), xycoords='axes fraction', ha='left', va='top',
                    fontweight='bold', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white', alpha=0.5))
        ax.invert_yaxis()  # Invert y-axis to have the top of the plot as the start of the series

        # Set y-axis
        y_min, y_max = vertical_series.min(), vertical_series.max()
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, p_Phi.shape[0])

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
            ax.set_xlabel('Time Index')
            ax_compare.set_xlabel('Time Index')
        else:
            # ax.tick_params(labelbottom=False)
            ax_compare.tick_params(labelbottom=False)

        ax.grid(True)
        ax_compare.grid(True)

    # Adjust as needed for visualization
    plt.subplots_adjust(hspace=0.5, bottom=0.1, right=0.9)
    plt.show()


load_taxi_data = load_lat_long()
mapping = load_taxi_data['Id_and_lat_long']
clean_taxi(7, method="dropoff", perspective="column", mapping=mapping)
