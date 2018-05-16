#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# 28 clusters

dataset = pd.read_csv("SFWeather.csv",
                      usecols=range(1, 7))  # Same code but with all the original types of Weather, so 28 clusters
print(str(dataset.head()))

enc = LabelEncoder()
enc.fit(dataset.Weather)
dataset.Weather = enc.transform(dataset.Weather)

dataset = dataset.dropna()

features = ['Humidity', 'Pressure', 'Temperature']
train_ds = dataset[features]

normalized_ds = MinMaxScaler().fit_transform(train_ds)

model = KMeans(n_clusters=28).fit(normalized_ds)

plt.figure()
plt.scatter(model.labels_, dataset.values[:, 2], c=model.labels_)
plt.title("k = 28, Temperatures and Clusters")
plt.xlabel("Cluster")
plt.ylabel("Temperature")

plt.figure()
plt.scatter(model.labels_, dataset.values[:, 0], c=model.labels_)
plt.title("k = 28, Humidity and Clusters")
plt.xlabel("Cluster")
plt.ylabel("Humidity")

plt.figure()
plt.scatter(model.labels_, dataset.values[:, 3], c=model.labels_)
plt.title("k = 28, Weathers and Clusters")
plt.xlabel("Cluster")
plt.ylabel("Weather")
plt.show()

# 8 and 4 cluster

dataset = pd.read_csv("SFWeather clean.csv", usecols=range(1, 7))  # load the dataset
print(str(dataset.head()))  # just print the first 5 rows to check the data

dataset = dataset.dropna()  # removes rows that have invalid values

features = ['Humidity', 'Pressure', 'Temperature']
train_ds = dataset[features]  # select the columns to use

normalized_ds = MinMaxScaler().fit_transform(train_ds)  # The dataset is normalized

for k in [4, 8]:
    km = KMeans(n_clusters=k, n_jobs=-1)
    model = KMeans(n_clusters=k, n_jobs=-1, algorithm="auto").fit(
        normalized_ds)  # KMeans is trained and run with 8 clusters

    plt.figure()
    plt.scatter(model.labels_, dataset.values[:, 2],
                c=model.labels_)  # Plot showing Temperatures distributed by clusters
    plt.title("k = %d, Temperatures and Clusters" % k)
    plt.xlabel("Cluster")
    plt.ylabel("Temperature")

    plt.figure()
    plt.scatter(model.labels_, dataset.values[:, 0], c=model.labels_)  # Plot showing Humidity distributed by clusters
    plt.title("k = %d, Humidity and Clusters" % k)
    plt.xlabel("Cluster")
    plt.ylabel("Humidity")

    plt.figure()
    plt.scatter(model.labels_, dataset.values[:, 3], c=model.labels_)  # Plot showing Weather distributed by clusters
    plt.title("k = %d, Weathers and Clusters" % k)
    plt.xlabel("Cluster")
    plt.ylabel("Weather")

plt.show()

centroids = model.cluster_centers_
f = plt.figure()
plot3d = f.add_subplot(111, projection='3d')
plot3d.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
               marker='x', s=169, linewidths=3,
               color='b', zorder=10)
plt.title("Centroids")
plt.show()

print(str(centroids))

# EM

for k in [4, 8]:

    gem = GaussianMixture(n_components=k)
    gem.fit(normalized_ds)
    model = gem.predict(normalized_ds
                        )
    plt.figure()
    plt.scatter(model, dataset.values[:, 2],
                c=model)  # Plot showing Temperatures distributed by clusters
    plt.title("Gaussian EM k = %d, Temperatures and Clusters" % k)
    plt.xlabel("Cluster")
    plt.ylabel("Temperature")

    plt.figure()
    plt.scatter(model, dataset.values[:, 0], c=model)  # Plot showing Humidity distributed by clusters
    plt.title("Gaussian EM k = %d, Humidity and Clusters" % k)
    plt.xlabel("Cluster")
    plt.ylabel("Humidity")

    plt.figure()
    plt.scatter(model, dataset.values[:, 3], c=model)  # Plot showing Weather distributed by clusters
    plt.title("Gaussian EM k = %d, Weathers and Clusters" % k)
    plt.xlabel("Cluster")
    plt.ylabel("Weather")

plt.show()

# Agglomerative clustering

model = AgglomerativeClustering(linkage="average", n_clusters=8, affinity='mahalanobis')

model.fit(train_ds)

plt.figure()
plt.scatter(model.labels_, dataset.values[:, 3], c=model.labels_)  # Plot showing Humidity distributed by clusters
plt.title("Agglomerative Clustering, Weather and Clusters")
plt.xlabel("Cluster")
plt.ylabel("Temperature")
plt.show()
