import numpy as np
import matplotlib.pyplot as plt

class MDTD_Outlier:

    @staticmethod
    def mdtd_find_outlier(p_X, p_recon, p_count, p_slice) -> None:
        """
        Plots outliers based on magnitude from the residual of p_X-(p_recon).
        Args:
            p_X: Original temporal signal tensor
            p_recon: Reconstruction of p_X from S ⊡ Φ1Y1, Φ2Y2, Φ3Y3
            p_percentage: Top percentile of outliers that the user wishes to plot
            p_count: Number of subplots to display, one for each outlier in p_count. Maximum count of 10.
            p_slice: x/y/z plane to visualize tensor.
        """
        # TODO
        outlier_indices = []
        avg_magnitude_indices = []
        p_X, p_recon = np.abs(p_X), np.abs(p_recon)
        residual = p_X - p_recon
        p_count = min(p_count, 10)

        def arbitrary():
            for i in range(p_X.shape[2]):
                slice_flattened = residual[:, :, i].flatten()
                sorted_values = np.argsort(slice_flattened)[::-1]
                outlier_indices_fixed = sorted_values[:p_count]
                row_indices_fixed, col_indices_fixed = np.unravel_index(outlier_indices_fixed, p_X[:, :, i].shape)

                for r, c in zip(row_indices_fixed, col_indices_fixed):
                    outlier_indices.append((r, c, i))

            pane_max_outlier_magnitude = {}
            for (r, c, pane) in outlier_indices:
                magnitude = p_X[r, c, pane]

                if pane in pane_max_outlier_magnitude:
                    pane_max_outlier_magnitude[pane] = max(pane_max_outlier_magnitude[pane], magnitude)
                else:
                    pane_max_outlier_magnitude[pane] = magnitude

            # Top 25 panes (arbitrary number) with the largest magnitude of outliers
            top_panes = sorted(pane_max_outlier_magnitude, key=pane_max_outlier_magnitude.get, reverse=True)[:25]

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            for pane_idx in top_panes:
                pane_outliers = [(r, c, z) for r, c, z in outlier_indices if z == pane_idx]
                magnitudes = [p_X[r, c, z] for r, c, z in pane_outliers]
                sizes = [(magnitude / max(magnitudes)) * 100 for magnitude in magnitudes]
                rows, cols, zs = zip(*pane_outliers)
                ax.scatter(rows, cols, zs=pane_idx, s=sizes, depthshade=True, label=f'Pane {pane_idx}')

            # Set labels and title
            ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_xlabel('Row Index')
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.set_ylabel('Column Index')

            ax.set_zlabel('Pane Index')
            ax.set_title('Top 25 Panes with 10 Largest Outliers')

            # Set the ticks for the z-axis (pane index) to correspond to the indices of the panes
            ax.set_xticks([0, p_X.shape[0]])
            ax.set_yticks([0, p_X.shape[1]])
            ax.set_zticks([min(top_panes), max(top_panes)])

            # Adjust the view angle for better visualization
            # elev = height
            # azim = L to R
            ax.view_init(elev=30, azim=120)
            ax.legend(loc='center left', bbox_to_anchor=(1.10, 0.5), title='Pane Number')

            plt.show()

        def x_slice():
            """
            Calculates and plots the x-axis outlier slice
            """
            average_magnitude_per_slice = np.mean(residual, axis=(1, 2))
            top_slices_indices = np.argsort(average_magnitude_per_slice)[::-1][:p_count]
            outlier_indices_sorted_by_pane = sorted(top_slices_indices)

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            # Define the range for y and z
            y_range = np.arange(p_X.shape[1])
            z_range = np.arange(p_X.shape[2])

            Y_mesh, Z_mesh = np.meshgrid(y_range, z_range)
            normalized_magnitudes = average_magnitude_per_slice[top_slices_indices] / np.max(
                average_magnitude_per_slice[top_slices_indices])
            # Plot meshgrids for the top 10 panes by average magnitude
            for i, slice_index in enumerate(outlier_indices_sorted_by_pane):
                X_mesh = np.full(Y_mesh.shape, slice_index)
                alpha = normalized_magnitudes[i] * 0.9 + 0.1

                ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=alpha, label=f'X Slice #{slice_index}')

            # Set the labels and title
            ax.set_xlabel('Row Index')
            ax.set_ylabel('Column Index')
            ax.set_zlabel('Pane Index')
            ax.set_title(f'Top {p_count} X Panes by Average Magnitude')

            # Adjust the legend and the view angle
            ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), title='Pane Number')
            ax.view_init(elev=10, azim=-70)  # Adjust the view angle

            plt.show()

        def y_slice():
            """
            Calculates and plots the y-axis outlier slice
            """
            average_magnitude_per_slice = np.mean(residual, axis=(0, 2))
            top_slices_indices = np.argsort(average_magnitude_per_slice)[::-1][:p_count]
            outlier_indices_sorted_by_pane = sorted(top_slices_indices)

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            # Define the range for x and z
            x_range = np.arange(p_X.shape[0])
            z_range = np.arange(p_X.shape[2])
            # Create a meshgrid for x and z
            X_mesh, Z_mesh = np.meshgrid(x_range, z_range)
            normalized_magnitudes = average_magnitude_per_slice[top_slices_indices] / np.max(
                average_magnitude_per_slice[top_slices_indices])
            # Plot meshgrids for the top 10 panes by average magnitude
            for i, slice_index in enumerate(outlier_indices_sorted_by_pane):
                Y_mesh = np.full(X_mesh.shape, slice_index)
                alpha = normalized_magnitudes[i] * 0.9 + 0.1

                ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=alpha, label=f'Y Slice #{slice_index}')

            # Set the labels and title
            ax.set_xlabel('Row Index')
            ax.set_ylabel('Column Index')
            ax.set_zlabel('Pane Index')
            ax.set_title(f'Top {p_count} Y Panes by Average Magnitude')

            # Adjust the legend and the view angle
            ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), title='Pane Number')
            ax.view_init(elev=10, azim=-70)  # Adjust the view angle

            plt.show()

        def z_slice():
            """
            Calculates and plots the z-axis outlier slice
            """
            average_magnitude_per_slice = np.mean(residual, axis=(0, 1))
            top_slices_indices = np.argsort(average_magnitude_per_slice)[::-1][:p_count]
            outlier_indices_sorted_by_pane = sorted(top_slices_indices)

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            # Define the range for x and y
            x_range = np.arange(p_X.shape[0])
            y_range = np.arange(p_X.shape[1])

            # Create a meshgrid for x and y
            X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
            normalized_magnitudes = average_magnitude_per_slice[top_slices_indices] / np.max(
                average_magnitude_per_slice[top_slices_indices])

            # Plot meshgrids for the top 10 panes by average magnitude
            for i, slice_index in enumerate(outlier_indices_sorted_by_pane):
                # Z position of the pane in the 3D plot
                Z_mesh = np.full(X_mesh.shape, slice_index)
                alpha = normalized_magnitudes[i] * 0.9 + 0.1  # Scale alpha between 0.2 and 1.0

                # Plot the meshgrid as a semi-transparent plane
                ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=alpha, label=f'Z Slice #{slice_index}')

            # Set the labels and title
            ax.set_xlabel('Row Index')
            ax.set_ylabel('Column Index')
            ax.set_zlabel('Pane Index')
            ax.set_title(f'Top {p_count} Z Panes by Average Magnitude')

            # Adjust the legend and the view angle
            ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), title='Pane Number')
            ax.view_init(elev=10, azim=-70)  # Adjust the view angle

            plt.show()

        # Slice method
        if p_slice == "x":
            x_slice()
        elif p_slice == "y":
            y_slice()
        else:
            z_slice()
