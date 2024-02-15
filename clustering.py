import numpy
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tgsd import tgsd
# from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as ss
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
def cluster(p_Psi, p_Y):
    psiy = numpy.abs(p_Psi) @ numpy.abs(p_Y)
    df = pd.DataFrame(psiy, columns=['Col1','Col2','Col3','Col4','Col5','Col6','Col7'])

    #Add the scaled attributes to the same DF
    scaler = StandardScaler()
    df[['Col1_T','Col2_T','Col3_T','Col4_T','Col5_T','Col6_T','Col7_T']] = scaler.fit_transform(df[['Col1','Col2','Col3','Col4','Col5','Col6','Col7']])

    scaled_df = df.loc[:,'Col1_T':'Col7_T']
    unscaled_df = df.loc[:,'Col1':'Col7']

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


