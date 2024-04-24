from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np


class TGSD_Outlier:
    @staticmethod
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
                          label=f'Top {p_percentage} % Outliers')
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

    @staticmethod
    def find_row_outlier(p_X, p_Psi, p_Y, p_W, p_Phi, p_count) -> None:
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
        sorted_smallest = np.argsort(row_avg)
        p_count = min(p_count, 10)
        outlier_rows = sorted_rows[:p_count]
        antioutlier_rows = sorted_smallest[:p_count]
        num_plots = len(outlier_rows)

        # Define grid
        fig = plt.figure(figsize=(15, 3 * p_count))
        gs = GridSpec(num_plots, 2)  # Define grid layout for the figure

        # Iterate through each row index to build grid
        for i, row_idx in enumerate(antioutlier_rows):
            # Subplot 1: Plot the "antianomaly" rows, i.e., rows with least deviation
            ax = fig.add_subplot(gs[i, 0])
            # Assuming the logic for what to plot remains the same, but now focusing on antianomaly rows
            ax.plot(magnitude_X[row_idx, :], 'b-', label='X')
            ax.plot(reconstructed_X[row_idx, :], 'g--', label='Reconstructed X')
            ax.set_ylim(np.min(magnitude_X), np.max(magnitude_X))
            ax.set_title(f'Antianomaly Row {row_idx}')  # Subplot title

            # Add legend to first subplot
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1.3), fontsize='small')

            # Enable the xlabel only for bottom subplot
            if i == num_plots - 1:
                ax.set_xlabel('Time Index')
                ax.set_xlabel('Time Index')
            else:
                ax.tick_params(labelbottom=False)
                ax.tick_params(labelbottom=False)

        for i, row_idx in enumerate(outlier_rows):
            # Subplot 2: Plot the Row Values Against Their Reconstructed Values (Fit)
            ax_compare = fig.add_subplot(gs[i, 1])
            ax_compare.plot(magnitude_X[row_idx, :], 'b-', label='X')
            ax_compare.plot(reconstructed_X[row_idx, :], 'g--', label='Reconstructed X')
            ax_compare.set_title(f'Anomaly Row {row_idx}')  # Subplot title

            # Set y-axis
            ax_compare.set_ylim(np.min(magnitude_X), np.max(magnitude_X))

            # Add legend
            if i == 0:
                handles, labels = ax_compare.get_legend_handles_labels()
                ax_compare.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1.3), fontsize='small')

            # Enable the xlabel only for bottom subplot
            if i == num_plots - 1:
                ax.set_xlabel('Time Index')
                ax_compare.set_xlabel('Time Index')
            else:
                ax.tick_params(labelbottom=False)
                ax_compare.tick_params(labelbottom=False)

        ax.grid(True)
        ax_compare.grid(True)

        plt.figtext(0.5, 0.01, "Comparative Analysis of Antianomalies and Anomalies", ha="center", fontsize=14, fontweight='bold')
        plt.subplots_adjust(hspace=0.5, bottom=0.1, right=0.9)
        plt.show()

    @staticmethod
    def find_col_outlier(p_X, p_Psi, p_Y, p_W, p_Phi, p_count) -> None:
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
        # Antioutlier
        sorted_columns = np.argsort(col_avg)  # Do not reverse the order
        antianomaly_columns = sorted_columns[:p_count]

        for i, col_idx in enumerate(antianomaly_columns):
            # Subplot 1: Plot the AntiAnomaly Columns Against Their Reconstructed Values (Fit)
            ax = fig.add_subplot(gs[i, 0])
            ax.plot(magnitude_X[:, col_idx], 'b-', label='X')
            ax.plot(reconstructed_X[:, col_idx], 'g--', label='Reconstructed X')

            # Set y-axis
            ax.set_ylim(min(np.min(magnitude_X[:, col_idx]), np.min(reconstructed_X[:, col_idx])),
                        max(np.max(magnitude_X[:, col_idx]), np.max(reconstructed_X[:, col_idx])))
            ax.set_title(f"Column {col_idx}")
            if i == num_series_to_plot - 1:
                ax.set_xlabel('Arbitrary Time Index')
                ax.set_ylabel('Value on Time Series')
            else:
                # ax.tick_params(labelbottom=False)
                ax.tick_params(labelbottom=False)
            ax.grid(True)

        # Iterate through each column outlier index
        for i, col_idx in enumerate(outlier_columns):
            # Subplot 2: Plot the Column Values Against Their Reconstructed Values (Fit)
            ax_compare = fig.add_subplot(gs[i, 1])
            ax_compare.plot(magnitude_X[:, col_idx], 'b-', label='X')
            ax_compare.plot(reconstructed_X[:, col_idx], 'g--', label='Reconstructed X')

            # Set y-axis
            ax_compare.set_ylim(min(np.min(magnitude_X[:, col_idx]), np.min(reconstructed_X[:, col_idx])),
                                max(np.max(magnitude_X[:, col_idx]), np.max(reconstructed_X[:, col_idx])))
            ax_compare.set_title(f"Column {col_idx}")
            # Add legend to first subplot
            if i == 0:
                handles, labels = ax_compare.get_legend_handles_labels()
                ax_compare.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1.3), fontsize='small')

            # Enable the xlabel only for bottom subplot
            if i == num_series_to_plot - 1:
                ax_compare.set_xlabel('Arbitrary Time Index')
                ax_compare.set_ylabel('Value on Time Series')
            else:
                ax_compare.tick_params(labelbottom=False)

            ax_compare.grid(True)

        # Adjust as needed for visualization
        plt.figtext(0.5, 0.01, "Comparative Analysis of Antianomalies and Anomalies", ha="center", fontsize=14, fontweight='bold')
        plt.subplots_adjust(hspace=0.6, bottom=0.2, right=0.9)
        plt.show()
