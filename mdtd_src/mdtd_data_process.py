import numpy as np
import pandas as pd

class MDTD_Data_Process:

    @staticmethod
    def syn_data_to_numpy():
        """
        Saves a .csv file with synthetic data to a numpy array
        Returns:
            Numpy arrays for MDTD algorithm: Phi, PhiYg, P
        """
        load_phi_array = []
        for i in range(3):  # Assuming you have 3 files to load
            arr = np.loadtxt(f'mdtd_demo_data/Phi_{i}.csv', delimiter=',')
            load_phi_array.append(arr)
        phi_array = np.empty((1, 3), dtype='object')
        for i, arr in enumerate(load_phi_array):
            phi_array[0, i] = arr

        load_phi_yg_array = []
        for i in range(3):  # Assuming you have 3 files to load
            arr = np.loadtxt(f'mdtd_demo_data/PhiYg_{i}.csv', delimiter=',')
            load_phi_yg_array.append(arr)
        phi_yg_array = np.empty((1, 3), dtype='object')
        for i, arr in enumerate(load_phi_yg_array):
            phi_yg_array[0, i] = arr

        list_string = pd.read_csv(f'mdtd_demo_data/P.csv', header=None).iloc[0]
        p_array_of_arrays = np.empty((list_string.size,), dtype=object)
        for i, item in enumerate(list_string):
            p_array_of_arrays[i] = np.array([item], dtype=np.uint16)

        # Reshape to 1xN where each element is a numpy array
        p_array_of_arrays = p_array_of_arrays.reshape(1, -1)

        return phi_array, phi_yg_array, p_array_of_arrays

    @staticmethod
    def mdtd_format_csv_to_numpy(p_data_csv):
        """
        Formats a .csv file to a numpy array, compatible for MDTD algorithm
        Args:
            p_data_csv: Some .csv file of format (r, c, d, v) for each line
        Returns:
            Reconstructed numpy array from .csv file
        """
        df = pd.read_csv(p_data_csv)
        max_row = df['Row'].max() + 1
        max_col = df['Column'].max() + 1
        max_depth = df['Depth'].max() + 1
        reconstructed_array = np.full((max_row, max_col, max_depth), np.nan, dtype=np.longdouble)
        rows = df['Row'].values
        cols = df['Column'].values
        depths = df['Depth'].values
        values = df['Value'].values
        reconstructed_array[rows, cols, depths] = values
        return reconstructed_array

    @staticmethod
    def mdtd_format_numpy_to_csv(p_data, filepath):
        """
        Formats a numpy array to a .csv file to download
        Args:
            p_data: Some numpy array of the format (r, c, d, v)
        Returns:
            .csv file where each line is of the format r, c, d, v
        """
        row_indices, column_indices, depth_indices = np.indices(p_data.shape)
        rows = row_indices.flatten()
        columns = column_indices.flatten()
        depths = depth_indices.flatten()
        values = p_data.flatten()
        df = pd.DataFrame({
            'Row': rows,
            'Column': columns,
            'Depth': depths,
            'Value': values
        })
        df.to_csv(filepath, index=False)
