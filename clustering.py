
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as ss
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
import seaborn as sns

df = pd.read_csv('/Users/josephregan/PycharmProjects/pyspady/PsiYRealdata.csv')
df.columns = ['Col1','Col2','Col3','Col4','Col5','Col6','Col7']

print(df)
print(df.describe())
# kmeans_kwargs ={
#     "init":"random",
#     "n_init":10,
#     "max_iter":300,
#     "random_state":42,
# }
# sse=[]
#
# features, true_labels = make_blobs(
#     n_samples=200,
#     centers=3,
#     cluster_std=2.75,
#     random_state=42
# )
#
#Add the scaled attributes to the same DF
scaler = StandardScaler()
df[['Col1_T','Col2_T','Col3_T','Col4_T','Col5_T','Col6_T','Col7_T']] = scaler.fit_transform(df[['Col1','Col2','Col3','Col4','Col5','Col6','Col7']])
print(df)

scaled_df = df.loc[:,'Col1_T':'Col7_T']
unscaled_df = df.loc[:,'Col1':'Col7']

# corrmat = unscaled_df.corr()
# sns.heatmap(corrmat)
# plt.show()
#Identify the optmimal numbr of clusters
def optimise_k_means(data,max_k):
    means = []
    inertias = []

    for k in range (1,max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)
    #generate the elbow plot
    fig = plt.subplots(figsize=(10,5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

optimise_k_means(df[['Col1_T','Col2_T']],10)

#c
kmeans = KMeans(n_clusters=7)
kmeans.fit(df[['Col1_T','Col2_T']])
df['kmeans_4'] = kmeans.labels_
# ss(kmeans,df['kmeans_4'])

plt.scatter(x=df['Col1'],y=df['Col2'],c=df['kmeans_4'])

plt.show()
for k in range(1,9):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(df[['Col1_T','Col2_T']])
    df[f'KMeans_{k}'] = kmeans.labels_

print(df)

fig, axs = plt.subplots(nrows=2,ncols=4,figsize=(20,5))

for i, ax in enumerate(fig.axes,start=1):
    ax.scatter(x=df['Col1'], y=df['Col2'], c=df[f'KMeans_{i}'])
    ax.set_title(f'N Clusters: {i}')
plt.show()

#
# kmeans = KMeans(
#     init="random",
#     n_clusters=3,
#     n_init=10,
#     max_iter=300,
#     random_state=42
# )
# for k in range(1,11):
#     kmeans=KMeans(n_clusters=k,**kmeans_kwargs)
#     kmeans.fit(scaled_features)
#     sse.append(kmeans.inertia_)
#
# plt.style.use("fivethirtyeight")
# plt.plot(range(1, 11), sse)
# plt.xticks(range(1, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.show()
# kl = KneeLocator(
#     range(1, 11), sse, curve="convex", direction="decreasing"
# )
#
# kl.elbow
# # A list holds the silhouette coefficients for each k
# silhouette_coefficients = []
#
# # Notice you start at 2 clusters for silhouette coefficient
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(scaled_features)
#     score = silhouette_score(scaled_features, kmeans.labels_)
#     silhouette_coefficients.append(score)
#
# plt.style.use("fivethirtyeight")
# plt.plot(range(2, 11), silhouette_coefficients)
# plt.xticks(range(2, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Coefficient")
# plt.show()
#
# # # Final locations of the centroid
# # print("Cluster Center",kmeans.cluster_centers_)
# # # The number of iterations required to converge
# # print("The number of iters:", kmeans.n_iter_)
# #
# # print(features[:5])
# # print(true_labels[:5])
# # print(scaled_features[:5])
# features, true_labels = make_moons(
#     n_samples=250, noise=0.05, random_state=42
# )
# scaled_features = scaler.fit_transform(features)
# # Instantiate k-means and dbscan algorithms
# kmeans = KMeans(n_clusters=2)
# dbscan = DBSCAN(eps=0.3)
#
# # Fit the algorithms to the features
# kmeans.fit(scaled_features)
# dbscan.fit(scaled_features)
#
# # Compute the silhouette scores for each algorithm
# kmeans_silhouette = silhouette_score(
#     scaled_features, kmeans.labels_
# ).round(2)
# dbscan_silhouette = silhouette_score(
#    scaled_features, dbscan.labels_
# ).round (2)
#
# kmeans_silhouette
#
#
# dbscan_silhouette
# # Plot the data and cluster silhouette comparison
# fig, (ax1, ax2) = plt.subplots(
#     1, 2, figsize=(8, 6), sharex=True, sharey=True
# )
# fig.suptitle(f"Clustering Algorithm Comparison: Crescents", fontsize=16)
# fte_colors = {
#     0: "#008fd5",
#     1: "#fc4f30",
# }
# # The k-means plot
# km_colors = [fte_colors[label] for label in kmeans.labels_]
# ax1.scatter(scaled_features[:, 0], scaled_features[:, 1], c=km_colors)
# ax1.set_title(
#     f"k-means\nSilhouette: {kmeans_silhouette}", fontdict={"fontsize": 12}
# )
#
# # The dbscan plot
# db_colors = [fte_colors[label] for label in dbscan.labels_]
# ax2.scatter(scaled_features[:, 0], scaled_features[:, 1], c=db_colors)
# ax2.set_title(
#     f"DBSCAN\nSilhouette: {dbscan_silhouette}", fontdict={"fontsize": 12}
# )
# plt.show()
# kmeans_kwargs ={
#     "init":"random",
#     "n_init":10,
#     "max_iter":300,
#     "random_state":42,
# }
# sse=[]
#
# features, true_labels = make_blobs(
#     n_samples=200,
#     centers=3,
#     cluster_std=2.75,
#     random_state=42
# )
#
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)
#
# kmeans = KMeans(
#     init="random",
#     n_clusters=3,
#     n_init=10,
#     max_iter=300,
#     random_state=42
# )
# for k in range(1,11):
#     kmeans=KMeans(n_clusters=k,**kmeans_kwargs)
#     kmeans.fit(scaled_features)
#     sse.append(kmeans.inertia_)
#
# plt.style.use("fivethirtyeight")
# plt.plot(range(1, 11), sse)
# plt.xticks(range(1, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.show()
# kl = KneeLocator(
#     range(1, 11), sse, curve="convex", direction="decreasing"
# )
#
# kl.elbow
# # A list holds the silhouette coefficients for each k
# silhouette_coefficients = []
#
# # Notice you start at 2 clusters for silhouette coefficient
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(scaled_features)
#     score = silhouette_score(scaled_features, kmeans.labels_)
#     silhouette_coefficients.append(score)
#
# plt.style.use("fivethirtyeight")
# plt.plot(range(2, 11), silhouette_coefficients)
# plt.xticks(range(2, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Coefficient")
# plt.show()
#
# # # Final locations of the centroid
# # print("Cluster Center",kmeans.cluster_centers_)
# # # The number of iterations required to converge
# # print("The number of iters:", kmeans.n_iter_)
# #
# # print(features[:5])
# # print(true_labels[:5])
# # print(scaled_features[:5])
# features, true_labels = make_moons(
#     n_samples=250, noise=0.05, random_state=42
# )
# scaled_features = scaler.fit_transform(features)
# # Instantiate k-means and dbscan algorithms
# kmeans = KMeans(n_clusters=2)
# dbscan = DBSCAN(eps=0.3)
#
# # Fit the algorithms to the features
# kmeans.fit(scaled_features)
# dbscan.fit(scaled_features)
#
# # Compute the silhouette scores for each algorithm
# kmeans_silhouette = silhouette_score(
#     scaled_features, kmeans.labels_
# ).round(2)
# dbscan_silhouette = silhouette_score(
#    scaled_features, dbscan.labels_
# ).round (2)
#
# kmeans_silhouette
#
#
# dbscan_silhouette
# # Plot the data and cluster silhouette comparison
# fig, (ax1, ax2) = plt.subplots(
#     1, 2, figsize=(8, 6), sharex=True, sharey=True
# )
# fig.suptitle(f"Clustering Algorithm Comparison: Crescents", fontsize=16)
# fte_colors = {
#     0: "#008fd5",
#     1: "#fc4f30",
# }
# # The k-means plot
# km_colors = [fte_colors[label] for label in kmeans.labels_]
# ax1.scatter(scaled_features[:, 0], scaled_features[:, 1], c=km_colors)
# ax1.set_title(
#     f"k-means\nSilhouette: {kmeans_silhouette}", fontdict={"fontsize": 12}
# )
#
# # The dbscan plot
# db_colors = [fte_colors[label] for label in dbscan.labels_]
# ax2.scatter(scaled_features[:, 0], scaled_features[:, 1], c=db_colors)
# ax2.set_title(
#     f"DBSCAN\nSilhouette: {dbscan_silhouette}", fontdict={"fontsize": 12}
# )
# plt.show()