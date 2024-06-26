import pandas as pd
import numpy as np
import scipy.io
import dictionary_generation
from mdtd_src import mdtd_outlier, mdtd_data_process, mdtd_home
from taxi_src import taxi_2d_outlier, taxi_tensor_clustering
from tgsd_src import tgsd_clustering, tgsd_smartsearch, tgsd_home, tgsd_screening
import json

GenDict = dictionary_generation.GenerateDictionary
MDTD_Data_Process = mdtd_data_process.MDTD_Data_Process
MDTD_Outlier = mdtd_outlier.MDTD_Outlier
Taxi_2D_Outlier = taxi_2d_outlier.Taxi_2D_Outlier
TwoD_Clustering = tgsd_clustering.TGSD_Cluster
Taxi_Tensor_Clustering = taxi_tensor_clustering.Taxi_Tensor_Clustering

class Taxi_Demo:
    def __init__(self, month, method, perspective, auto, optimizer_method, screening_bool):
        self.month = month
        self.method = method
        self.perspective = perspective
        self.auto = auto
        self.optimizer_method = optimizer_method
        self.screening = screening_bool

    def load_lat_long(self):
        """
        Loads the latitude and longitude Matlab file
        Returns:
            Numpy array of lat_long_info.mat
        """
        return scipy.io.loadmat("Taxi_prep/lat_long_info.mat")

    def clean_and_run(self):
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

        def extract_adj_matrix_csv(adj_matrix, pickup_or_dropoff, month):
            """
            Helper method to extract a .csv of pickup or dropoff data from an adjacency matrix as a numpy array
            Args:
                adj_matrix: Numpy array of an adjacency matrix
                pickup_or_dropoff: Either pickup or dropoff data to extract
            """
            rows, cols = np.nonzero(adj_matrix)
            data = list(zip(rows + 1, cols + 1))  # Adding 1 to convert from 0-based to 1-based indexing, if necessary
            df_nonzero = pd.DataFrame(data, columns=['r', 'c'])
            df_nonzero.to_csv(f'Taxi_tensor_data/{month}_{pickup_or_dropoff}_adjacency.csv', index=False, header=False)

        def extract_tensor_csv(all_data, date_range, num_locations, month):
            """
            Helper method to extract a .csv of pickup and dropoff data from a tensor as a numpy array
            Args:
                all_data: pandas DataFrame of all necessary location/time/pickup/dropoff data
                date_range: Given days in some particular month.
            """
            time_bins = all_data['time_bin'].sort_values().unique()
            time_bin_to_index = {time: index for index, time in enumerate(time_bins)}
            all_data['time_index'] = all_data['time_bin'].apply(lambda x: time_bin_to_index[x])
            trip_counts = all_data.groupby(['PULocationID', 'DOLocationID', 'time_index']).size().reset_index(
                name='count')
            shape = (num_locations, num_locations * len(date_range))
            trip_counts['PULocationID'] -= 1
            trip_counts['DOLocationID'] -= 1
            row = trip_counts['PULocationID']
            col = trip_counts['DOLocationID'] * len(date_range) + trip_counts['time_index']
            data = trip_counts['count']

            try:
                sparse_matrix = scipy.sparse.coo_matrix((data, (row, col)))
            except Exception as e:
                print("Failed to create a sparse matrix for this tensor. Please check the dimensions over again.")
            sparse_matrix_csr = sparse_matrix.tocsr()
            trimmed_sparse_matrix_csr = sparse_matrix_csr[:, :shape[1]]
            trimmed_sparse_matrix_coo = trimmed_sparse_matrix_csr.tocoo()
            tensor = trimmed_sparse_matrix_coo.toarray().reshape((num_locations, num_locations, len(date_range)))
            MDTD_Data_Process.mdtd_format_numpy_to_csv(tensor, f"Taxi_tensor_data/{month}_tensor_data")

        cols = ['PULocationID', 'DOLocationID', 'formatted_time']
        month_st = "0" + str(self.month) if self.month < 10 else str(self.month)
        all_data = pd.read_csv(f'Taxi_prep/tensor_data/sec_yellow_tripdata_2017-{month_st}', header=None,
                               usecols=[1, 2, 3],
                               names=cols)

        # Convert formatted_time to datetime, adjust to time_bin
        all_data['time_bin'] = pd.to_timedelta(all_data['formatted_time'], unit='s') + pd.to_datetime('2017-01-01')
        all_data['time_bin'] = all_data['time_bin'].dt.floor('H')

        end_days_map = {"02": "28", "04": "30", "06": "30", "09": "30", "11": "30"}
        end_days = end_days_map.get(month_st, "31")

        date_range = pd.date_range(start=f'2017-{month_st}-01', end=f'2017-{month_st}-{end_days} 23:00:00', freq='H')
        num_locations = 265  # Total number of unique locations
        adj_template = np.zeros((num_locations, num_locations), dtype=int)

        # Use this if you need to rebuild tensor .csv
        #extract_tensor_csv(all_data, date_range, num_locations, month_st)

        if self.method == "pickup" or self.method == "dropoff":
            # Generate respective data and adjacency matrix given method
            d, adj_matrix = generate_pickup_or_dropoff_adj(adj_template, all_data, self.method, date_range)
            # Perform TGSD
            # Load in config file and override X/Psi/Phi/Mask with taxi data instead
            TGSD_Driver = tgsd_home.TGSD_Home("tgsd_src/config.json", screening_flag=self.screening)
            TGSD_Driver.X = d
            TGSD_Driver.Psi_D = GenDict.gen_gft_new(adj_matrix, False)[0]
            TGSD_Driver.Phi_D = GenDict.gen_rama(t=d.shape[1], max_period=24)
            size_y = int(.1*adj_matrix.shape[0]*adj_matrix.shape[1])
            TGSD_Driver.mask = np.random.randint(0, 65536, size=(1, size_y), dtype=np.uint16)

            if self.screening:
                TGSD_Screening = tgsd_screening.TGSD_Screening(x=TGSD_Driver.X, left=TGSD_Driver.Psi_D, right=TGSD_Driver.Phi_D)
                TGSD_Screening.find_lam_max()
                TGSD_Screening.ST2_2D(0.8 * TGSD_Screening.lam_max)
                TGSD_Driver.Psi_D, TGSD_Driver.Phi_D = TGSD_Screening.make_dictonary()

            if not self.auto:
                Y, W = TGSD_Driver.tgsd(TGSD_Driver.X, TGSD_Driver.Psi_D, TGSD_Driver.Phi_D, TGSD_Driver.mask, iterations=100, optimizer_method=self.optimizer_method, learning_rate=0.00000001)
                print(f"Imputing {TGSD_Driver.mask.shape[1]} masked values...")
                TGSD_Driver.return_missing_values(TGSD_Driver.mask, TGSD_Driver.Psi_D, TGSD_Driver.Phi_D, Y, W)
                # Returns missing values, downloads new CSV and displays graph of imputed values
                print(f".csv of {TGSD_Driver.mask.shape[1]} imputed values downloaded to imputed_values.csv.")
            else:
                # Run smart search using thresholds of 0.3 and 0.3. Change accordingly.
                # If demo is changed to False, default values of X/Psi/Phi/mask will be loaded in.
                # This is *not* what we want for the demo on the taxi dataset.
                Smart_Search = tgsd_smartsearch.CustomEncoder(config_path="tgsd_src/config.json",
                                                              demo=True, demo_X=TGSD_Driver.X, demo_Psi=TGSD_Driver.Psi_D, demo_Phi=TGSD_Driver.Phi_D, demo_mask=TGSD_Driver.mask,
                                                              coefficient_threshold=0.3,
                                                              residual_threshold=0.3)
                Smart_Search.run_smart_search()
                # Obtain the encoding matrices using optimized hyperparameters.
                Y, W = Smart_Search.get_Y_W()

            # Downstream tasks
            if self.perspective == "point":
                Taxi_2D_Outlier.find_outlier(TGSD_Driver.X, TGSD_Driver.Psi_D, Y, W, TGSD_Driver.Phi_D, .1, 5,
                                             p_month=self.month)
            elif self.perspective == "row":
                Taxi_2D_Outlier.find_row_outlier(TGSD_Driver.X, TGSD_Driver.Psi_D, Y, W, TGSD_Driver.Phi_D, 5,
                                                 p_month=self.month, p_method=self.method)
            else:
                Taxi_2D_Outlier.find_col_outlier(TGSD_Driver.X, TGSD_Driver.Psi_D, Y, W, TGSD_Driver.Phi_D, 5,
                                                 p_month=self.month, p_method=self.method)
            # K-Means
            TwoD_Clustering.cluster(TGSD_Driver.Psi_D, Y)

        else:
            # MDTD
            # , adj_matrix_pickup = generate_pickup_or_dropoff_adj(adj_template, all_data, "pickup", date_range)
            # _, adj_matrix_dropoff = generate_pickup_or_dropoff_adj(adj_template, all_data, "dropoff", date_range)
            # extract_adj_matrix_csv(adj_matrix_dropoff, "dropoff", month_st)
            # extract_adj_matrix_csv(adj_matrix_pickup, "pickup", month_st)
            # extract_tensor_csv(all_data, date_range, num_locations, month_st)

            with open("mdtd_src/mdtd_config.json", 'r') as file:
                tensor_data = json.load(file)
            tensor_data["X"] = f"Taxi_tensor_data/{month_st}_tensor_data"
            tensor_data["adj-1"] = f"Taxi_tensor_data/{month_st}_pickup_adjacency.csv"
            tensor_data["adj-2"] = f"Taxi_tensor_data/{month_st}_dropoff_adjacency.csv"
            with open("mdtd_src/mdtd_config.json", 'w') as file:
                json.dump(tensor_data, file, indent=4)
            MDTD_Driver = mdtd_home.MDTD_Home(
                "mdtd_src/mdtd_config.json")  # Modified config file to fit the month of taxi dropoff.
            # Perform MDTD
            return_X, recon_X, phi_y = MDTD_Driver.mdtd(False, MDTD_Driver.X, MDTD_Driver.adj_1, MDTD_Driver.adj_2,
                                                        MDTD_Driver.mask, MDTD_Driver.count_nnz,
                                                        MDTD_Driver.num_iters_check, MDTD_Driver.lam, MDTD_Driver.K,
                                                        MDTD_Driver.epsilon, model=self.optimizer_method)

            # Returns missing values, downloads new CSV and displays graph of imputed values
            MDTD_Driver.return_missing_values(MDTD_Driver.mask, recon_X)
            print(f".csv of {len(MDTD_Driver.mask)} imputed values downloaded to tensor_imputed_values.csv.")
            # Downstream tasks
            if self.perspective == "row":
                MDTD_Outlier.mdtd_find_outlier(return_X, recon_X, 5, "x")
            elif self.perspective == "col":
                MDTD_Outlier.mdtd_find_outlier(return_X, recon_X, 5, "y")
            else:
                MDTD_Outlier.mdtd_find_outlier(return_X, recon_X, 5, "z")

            Taxi_Tensor_Clustering.find_clusters(phi_y, 5)

# load_taxi_data = load_lat_long()
# mapping = load_taxi_data['Id_and_lat_long']
