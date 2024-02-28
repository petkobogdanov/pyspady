import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster(p_Psi, p_Y):
    """
    Performs k-means clustering along psi*Y for TGSD
    Args:
        p_Psi: Graph dictionary
        p_Y: Encoding matrix
    """
    psiy = numpy.abs(p_Psi) @ numpy.abs(p_Y)
    df = pd.DataFrame(psiy, columns=['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7'])

    # Add the scaled attributes to the same DF
    scaler = StandardScaler()
    df[['Col1_T', 'Col2_T', 'Col3_T', 'Col4_T', 'Col5_T', 'Col6_T', 'Col7_T']] = scaler.fit_transform(
        df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7']])

    def run_kmeans(df):
        for k in range(1, 9):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(df[['Col1_T', 'Col2_T']])
            df[f'KMeans_{k}'] = kmeans.labels_

        print(df)

        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 5))

        for i, ax in enumerate(fig.axes, start=1):
            ax.scatter(x=df['Col1'], y=df['Col2'], c=df[f'KMeans_{i}'])
            ax.set_title(f'N Clusters: {i}')
        plt.show()

    run_kmeans(df)
