
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
# mpl.use("tkagg")
import matplotlib.pyplot as plt
# from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as ss
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
# import seaborn as sns

df = pd.read_csv('PsiYRealdata.csv')

df.columns = ['Col1','Col2','Col3','Col4','Col5','Col6','Col7']

print(df)
print(df.describe())

#Add the scaled attributes to the same DF
scaler = StandardScaler()
df[['Col1_T','Col2_T','Col3_T','Col4_T','Col5_T','Col6_T','Col7_T']] = scaler.fit_transform(df[['Col1','Col2','Col3','Col4','Col5','Col6','Col7']])
print(df)

scaled_df = df.loc[:,'Col1_T':'Col7_T']
unscaled_df = df.loc[:,'Col1':'Col7']


print("optimise")

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
if __name__ == '__main__':
    run_kmeans(df)
    print("run kmeans")
#c# plt.show()