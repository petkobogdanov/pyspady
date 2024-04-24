import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class TGSD_Cluster:
    @staticmethod
    def cluster(p_Psi, p_Y):
        """
        Performs k-means clustering along psi*Y for TGSD
        Args:
            p_Psi: Graph dictionary
            p_Y: Encoding matrix
        """
        psiy = numpy.abs(p_Psi) @ numpy.abs(p_Y)
        columns = [f'Col{i+1}' for i in range(psiy.shape[1])]
        df = pd.DataFrame(psiy, columns=columns)

        scaler = StandardScaler()
        scaled_columns = [f'{col}_T' for col in columns]
        df[scaled_columns] = scaler.fit_transform(df[columns])

        def run_kmeans(df, scaled_columns):
            max_clusters = min(8, len(scaled_columns))  # Adjust the maximum number of clusters if needed
            kmeans_columns = {}  # Store intermediate results here

            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(df[scaled_columns])
                kmeans_columns[f'KMeans_{k}'] = kmeans.labels_

            df_kmeans = pd.DataFrame(kmeans_columns)
            df = pd.concat([df, df_kmeans], axis=1)
            nrows = (max_clusters // 4) + (0 if max_clusters % 4 == 0 else 1)
            ncols = min(max_clusters, 4)
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

            for i, ax in enumerate(fig.axes):
                ax.scatter(x=df[scaled_columns[0]], y=df[scaled_columns[1]], c=df[f'KMeans_{i+1}'])
                ax.set_title(f'N Clusters: {i+1}')
                if i + 1 == max_clusters:
                    break
            plt.tight_layout()
            plt.show()

        run_kmeans(df, scaled_columns)

