import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


class MDTD_Cluster:

    @staticmethod
    def mdtd_clustering(p_PhiY, n_clusters):
        """
        Performs k-means clustering on PhiY
        Args:
            p_PhiY: List of matrices of Phi*Y from MDTD algorithm
            n_clusters: Number of clusters, pre-defined
        """
        fig = plt.figure(figsize=(18, 6))

        cmap = matplotlib.cm.viridis
        norm = matplotlib.colors.Normalize(vmin=0, vmax=n_clusters - 1)

        tensor_shape = (p_PhiY[0].shape[0], p_PhiY[1].shape[0], p_PhiY[2].shape[0])

        for i, matrix in enumerate(p_PhiY):
            # Run K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(matrix)
            labels = kmeans.labels_

            ax = fig.add_subplot(1, 3, i + 1, projection='3d')
            ax.set_title(f'Clusters Along {"XYZ"[i]}-axis')

            # Set limits based on the tensor shape to reflect its actual dimensions
            ax.set_xlim([0, tensor_shape[0]])
            ax.set_ylim([0, tensor_shape[1]])
            ax.set_zlim([0, tensor_shape[2]])

            for cluster_id in range(n_clusters):
                cluster_indices = np.where(labels == cluster_id)[0]
                color = cmap(norm(cluster_id))
                # Determine which axis the slices are along and plot accordingly
                for slice_index in cluster_indices:
                    if i == 0:  # X-axis slices
                        ax.bar3d(slice_index, 0, 0, 1, tensor_shape[1], tensor_shape[2], color=color, alpha=0.5)
                    elif i == 1:  # Y-axis slices
                        ax.bar3d(0, slice_index, 0, tensor_shape[0], 1, tensor_shape[2], color=color, alpha=0.5)
                    else:  # Z-axis slices
                        ax.bar3d(0, 0, slice_index, tensor_shape[0], tensor_shape[1], 1, color=color, alpha=0.5)

            ax.view_init(elev=20, azim=-35)  # Adjust the view angle for better visibility

        plt.tight_layout()

        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=fig.axes, shrink=0.5, aspect=5, location='right')
        cbar.set_label('Cluster ID')
        cbar.set_ticks(np.arange(n_clusters))

        plt.show()
