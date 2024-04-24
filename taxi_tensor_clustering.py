from matplotlib.patches import Circle
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from shapely import wkt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

class Taxi_Tensor_Clustering:
    @staticmethod
    def find_clusters(p_PhiY, n_clusters):
        """
        Performs k-means clustering on PhiY
        Args:
            p_PhiY: List of matrices of Phi*Y from MDTD algorithm
            n_clusters: Number of clusters, pre-defined
        """
        def find_closest_data_indices(matrix, centroids):
            """
            Finds the closest index in the matrix given centroids following K-Means
            Args:
                matrix: Some matrix, in this case, Phi_i * Y_i
                centroids: Centroids corresponding to this matrix after K-Means

            Returns:
                List of indices closest to the centroid
            """
            closest_indices = []
            for centroid in centroids:
                # Calculate the Euclidean distance from each point in the matrix to the centroid
                distances = np.sqrt(((matrix - centroid) ** 2).sum(axis=1))
                # Find the index of the closest point
                closest_index = np.argmin(distances)
                closest_indices.append(closest_index)
            return closest_indices

        def filter_and_convert_gdf(gdf, object_ids, original_crs="EPSG:4326", target_crs="EPSG:3857"):
            """
            Filters a GeoDataFrame by OBJECTID, sets the CRS, and converts to a new CRS.
            Args:
                gdf: Original GeoDataFrame
                object_ids: List of OBJECTIDs to filter by
                original_crs: Original CRS of the GeoDataFrame
                target_crs: Target CRS of the GeoDataFrame

            Returns:
                New GeoDataFrame filtered by OBJECTID and converted to target CRS
            """
            filtered_gdf = gdf[gdf['OBJECTID'].isin(object_ids)] # Filter GDF by object ID
            filtered_gdf.crs = original_crs
            return filtered_gdf.to_crs(target_crs)

        def remap_coords_to_zone(df, gdf_clusters, location_data):
            """
            Remaps outliers based on zone names, handles missing data with geocoding, and builds separate lists
            for direct inclusion into a GeoDataFrame.
            Args:
                df: DataFrame with mapping name :: OBJECTID
                gdf_clusters: GeoFrame with cluster data
                location_data: DataFrame with location data (coordinates)

            Returns:
                List of remapped OBJECTIDs and a list of tuples for missing data
            """
            new_object_ids = []
            add_these_separately = []

            for idx, row in gdf_clusters.iterrows():
                zone_name = location_data.loc[idx + 1, 'Neighborhood']  # Get the real zone name corresponding to the outlier
                try:
                    # Get the object ID in the dataframe for that zone
                    object_id = df.loc[df['Name'] == zone_name, 'OBJECTID'].iloc[0]
                    # Re-map
                    new_object_ids.append(object_id)
                except IndexError:
                    # Can't find it in the dataframe (either due to a misspelling/data error/etc.
                    # Make call to geocode API to gather coordinates (use sparingly, otherwise rate limited)
                    location = geocode(zone_name)
                    if location:
                        # Build these on to the GDF separately
                        point = Point(location.longitude, location.latitude)
                        add_these_separately.append((zone_name, point))
            return new_object_ids, add_these_separately

        def build_gdf_from_tuples(data_tuples, columns=['Zone', 'geometry'], crs="EPSG:4326"):
            """
            Convert list of tuples to GeoDataFrame
            Args:
                data_tuples: List of tuples where each tuple contains data for the GeoDataFrame's row
                columns: List of column names for the GeoDataFrame
                crs: CRS to set for the GeoDataFrame

            Returns:
                 GeoDataFrame constructed from the tuples
            """
            df = pd.DataFrame(data_tuples, columns=columns)
            return gpd.GeoDataFrame(df, geometry='geometry', crs=crs)

        def filter_concat_gdf(original_gdf, object_ids, additional_gdf):
            """
            Filters a GeoDataFrame by OBJECTID, concatenates it with an additional GeoDataFrame,
            and ensures the resulting GeoDataFrame has a consistent CRS
            Args:
                original_gdf: Original GeoDataFrame
                object_ids: OBJECTID values to filter by
                additional_gdf: Concatenate with the original GeoDataFrame

            Returns:
                Concatenated GeoDataFrame with consistent CRS
            """
            # Filter the original GeoDataFrame by OBJECTID
            filtered_gdf = original_gdf[original_gdf['OBJECTID'].isin(object_ids)]
            # Concatenate the filtered GDF with the additional GDF
            concatenated_gdf = pd.concat([filtered_gdf, additional_gdf], ignore_index=True)
            # Ensure the concatenated GeoDataFrame has a consistent CRS
            result_gdf = gpd.GeoDataFrame(concatenated_gdf, crs=original_gdf.crs)
            return result_gdf

        # Define lists for centroid indices and radii (pickup/dropoff/time)
        centroid_list_pickup = []
        centroid_list_dropoff = []
        centroid_list_time = []
        radii_pickup = []
        radii_dropoff = []
        radii_time = []
        farthest_point_pickup = []
        farthest_point_dropoff = []

        # Iterate through list of PhiY and perform KMeans on each
        for i, matrix in enumerate(p_PhiY):
            # Number of points is first dimension of X
            # Perform K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(matrix)
            # Retrieve indices from centroids
            closest_indices = find_closest_data_indices(matrix, kmeans.cluster_centers_)
            radii = []
            farthest_point = []
            for j in range(n_clusters):
                cluster_points = matrix[kmeans.labels_ == j]
                center = kmeans.cluster_centers_[j]
                distances = np.linalg.norm(cluster_points - center, axis=1)
                radius = np.max(distances)
                radii.append(radius)
                farthest_point_index = np.argmax(distances)
                # Convert local cluster_points index to global matrix index
                global_index = np.where(kmeans.labels_ == j)[0][farthest_point_index]
                farthest_point.append(global_index)

            if i == 0:
                centroid_list_pickup.extend(closest_indices)
                radii_pickup.extend(radii)
                farthest_point_pickup.extend(farthest_point)
            elif i == 1:
                centroid_list_dropoff.extend(closest_indices)
                radii_dropoff.extend(radii)
                farthest_point_dropoff.extend(farthest_point)
            else:
                centroid_list_time.extend(closest_indices)
                radii_time.extend(radii)

        # Define location data from CSV
        location_data = pd.read_csv('Taxi_prep/taxi+_zone_lookup.csv', header=None,
                                    names=['ID', 'Borough', 'Neighborhood', 'ZoneType'])

        # Define grid
        fig = plt.figure(figsize=(15, 3 * n_clusters))
        gs = GridSpec(n_clusters, 1)
        ax = fig.add_subplot(gs[n_clusters // 4:n_clusters - (n_clusters // 4) + 1, 0])

        # Read centroids csv to gather locations for IDs
        df = pd.read_csv("Taxi_prep/NHoodNameCentroids.csv")
        # New dataframe column for geometry
        df['geometry'] = df['the_geom'].apply(wkt.loads)

        # New GDF containing only the outlier IDs, despite incorrect name mappings (we'll deal with that below)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

        # Retrieve GDFs for each centroid and radius location
        gdf_pickup_centroids = filter_and_convert_gdf(gdf, [r + 1 for r in centroid_list_pickup])
        gdf_dropoff_centroids = filter_and_convert_gdf(gdf, [r + 1 for r in centroid_list_dropoff])
        gdf_pickup_centroid_radius = filter_and_convert_gdf(gdf, [r + 1 for r in farthest_point_pickup])
        gdf_dropoff_centroid_radius = filter_and_convert_gdf(gdf, [r + 1 for r in farthest_point_dropoff])

        # The mapping from zone_lookup.csv to NHoodNameCentroids is not 1:1, so we need to convert the mapping to
        # properly obtain the zone ID and zone name to plot. Will be stored in a separate list if there are "missing" values,
        # from which API call is made to retrieve proper coordinates
        new_pickup_id, missing_pickup = remap_coords_to_zone(df, gdf_pickup_centroids, location_data)
        new_dropoff_id, missing_dropoff = remap_coords_to_zone(df, gdf_dropoff_centroids, location_data)
        new_pickup_radius_id, missing_pickup_radius = remap_coords_to_zone(df, gdf_pickup_centroid_radius, location_data)
        new_dropoff_radius_id, missing_dropoff_radius = remap_coords_to_zone(df, gdf_dropoff_centroid_radius, location_data)

        # Build new GDF from missing locations
        missing_pickup_gdf = build_gdf_from_tuples(missing_pickup)
        missing_dropoff_gdf = build_gdf_from_tuples(missing_dropoff)
        missing_pickup_radius_gdf = build_gdf_from_tuples(missing_pickup_radius)
        missing_dropoff_radius_gdf = build_gdf_from_tuples(missing_dropoff_radius)

        # Concatenate GDF with any missing location values
        correct_pickup_centroid = filter_concat_gdf(gdf, new_pickup_id, missing_pickup_gdf)
        correct_dropoff_centroid = filter_concat_gdf(gdf, new_dropoff_id, missing_dropoff_gdf)
        correct_pickup_radius = filter_concat_gdf(gdf, new_pickup_radius_id, missing_pickup_radius_gdf)
        correct_dropoff_radius = filter_concat_gdf(gdf, new_dropoff_radius_id, missing_dropoff_radius_gdf)

        # Find all unique zones
        unique_pickup_names = correct_pickup_centroid['Name'].dropna().unique()
        unique_pickup_zones = correct_pickup_centroid['Zone'].dropna().unique()
        all_unique_pickup_zones = np.concatenate((unique_pickup_names, unique_pickup_zones))

        unique_dropoff_names = correct_dropoff_centroid['Name'].dropna().unique()
        unique_dropoff_zones = correct_dropoff_centroid['Zone'].dropna().unique()
        all_unique_dropoff_zones = np.concatenate((unique_dropoff_names, unique_dropoff_zones))

        all_unique_zones = np.concatenate((all_unique_pickup_zones, all_unique_dropoff_zones))
        all_unique = np.unique(all_unique_zones)

        # Subplot 1: Plotting Locations of Outliers on Map of NYC
        # Plot zones to colors
        colors = plt.colormaps['hsv']
        gen_unique_colors = [colors(i / len(all_unique)) for i in range(len(all_unique))]
        global_zone_color_map = {zone: gen_unique_colors[i] for i, zone in enumerate(all_unique)}

        # Now, extract specific color maps for pickup and dropoff from the global map
        pickup_zone_color_map = {zone: global_zone_color_map[zone] for zone in np.concatenate((unique_pickup_names, unique_pickup_zones))}
        dropoff_zone_color_map = {zone: global_zone_color_map[zone] for zone in np.concatenate((unique_dropoff_names, unique_dropoff_zones))}

        # Convert CRS
        correct_pickup_centroid = correct_pickup_centroid.to_crs(epsg=3857)
        correct_dropoff_centroid = correct_dropoff_centroid.to_crs(epsg=3857)
        correct_pickup_radius = correct_pickup_radius.to_crs(epsg=3857)
        correct_dropoff_radius = correct_dropoff_radius.to_crs(epsg=3857)

        # Plot locations on map of NYC
        for idx, row in correct_pickup_centroid.iterrows():
            zone_name = row.Name if pd.notnull(row['Name']) else row.Zone
            radius_point_row = correct_pickup_radius.loc[idx]
            radius = row.geometry.distance(radius_point_row.geometry)
            color = pickup_zone_color_map[zone_name]
            ax.plot(row.geometry.x, row.geometry.y, 'o', color=color, markersize=6, label=f'pickup_{zone_name}')
            circle = Circle((row.geometry.x, row.geometry.y), radius, edgecolor=color, facecolor='none', linewidth=2, label=f'radius_{zone_name}')
            ax.add_patch(circle)
            # use proper_outlier_pickup_radius to plot radii

        for idx, row in correct_dropoff_centroid.iterrows():
            zone_name = row.Name if pd.notnull(row['Name']) else row.Zone
            radius_point_row = correct_dropoff_radius.loc[idx]
            radius = row.geometry.distance(radius_point_row.geometry)
            color = dropoff_zone_color_map[zone_name]
            ax.plot(row.geometry.x, row.geometry.y, 'o', color=color, markersize=6, label=f'dropoff_{zone_name}')
            circle = Circle((row.geometry.x, row.geometry.y), radius, edgecolor=color, facecolor='none', linewidth=2, label=f'radius_{zone_name}')
            ax.add_patch(circle)
            # use proper_outlier_dropoff_radius to plot radii

        # Base map, bounds, and other settings
        ax.set_xlim([nyc_bounds_mercator['west'], nyc_bounds_mercator['east']])
        ax.set_ylim([nyc_bounds_mercator['south'], nyc_bounds_mercator['north']])
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        ax.set_axis_off()
        #m = calendar.month_name[p_month]
        #s = "NYC Taxi Pickups" if p_method == "pickup" else "NYC Taxi Dropoffs"
        #ax.set_title(f"{m} {s}")
        ax.set_aspect('equal')
        # Only plot unique labels
        handles, labels = ax.get_legend_handles_labels()

        # Filter for pickups and dropoffs
        pickup_handles = [h for h, l in zip(handles, labels) if 'pickup_' in l]
        pickup_labels = [l.replace('pickup_', '') for h, l in zip(handles, labels) if 'pickup_' in l]
        dropoff_handles = [h for h, l in zip(handles, labels) if 'dropoff_' in l]
        dropoff_labels = [l.replace('dropoff_', '') for h, l in zip(handles, labels) if 'dropoff_' in l]

        # Create two legends
        pickup_legend = ax.legend(pickup_handles, pickup_labels, loc='upper left', title="Pickup Locations")
        dropoff_legend = ax.legend(dropoff_handles, dropoff_labels, loc='upper right', title="Dropoff Locations")

        # Add the pickup legend manually to the current Axes, dropoff_legend is automatically added
        ax.add_artist(pickup_legend)
        ax.set_title("April NYC Taxi Pickups and Dropoffs by Volume")
        plt.show()
