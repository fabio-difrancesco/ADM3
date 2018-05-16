#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D


# Huge dataset

# dataset = pd.read_csv("minute_weather.csv")  # load the dataset
#
# sampled_df = dataset[(dataset['rowID'] % 30) == 0]
# sampled_df.shape
#
# sampled_df = sampled_df.dropna()
#
# features = ['air_pressure', 'air_temp', 'avg_wind_direction', 'avg_wind_speed', 'max_wind_direction',
#         'max_wind_speed','relative_humidity']
# select_df = sampled_df[features]
#
# X = MinMaxScaler().fit_transform(select_df)

# for k in range(3, 12):
#
#     model = KMeans(n_clusters=k, n_jobs=-1, algorithm="auto").fit_predict(X)
#
#     plt.figure()
#     plt.scatter(select_df.values[:, 3], model, c=model)  # Plot showing Temperatures distributed by clusters
#     plt.title("Temperatures and Clusters, k = " + str(k))
#
#     plt.figure()
#     plt.scatter(select_df.values[:, 6], model, c=model)  # Plot showing Humidity distributed by clusters
#     plt.title("Humidity and Clusters, k = " + str(k))
#
# plt.show()

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

#EM

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

for k in [4, 8]:
    model = KMeans(n_clusters=k, n_jobs=-1, algorithm="full").fit(
        normalized_ds)  # KMeans is trained and run with 8 clusters

    plt.figure()
    plt.scatter(model.labels_, dataset.values[:, 2],
                c=model.labels_)  # Plot showing Temperatures distributed by clusters
    plt.title("EM k = %d, Temperatures and Clusters" % k)
    plt.xlabel("Cluster")
    plt.ylabel("Temperature")

    plt.figure()
    plt.scatter(model.labels_, dataset.values[:, 0], c=model.labels_)  # Plot showing Humidity distributed by clusters
    plt.title("EM k = %d, Humidity and Clusters" % k)
    plt.xlabel("Cluster")
    plt.ylabel("Humidity")

    plt.figure()
    plt.scatter(model.labels_, dataset.values[:, 3], c=model.labels_)  # Plot showing Weather distributed by clusters
    plt.title("EM k = %d, Weathers and Clusters" % k)
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

# knn

enc = LabelEncoder()
enc.fit(dataset.Weather)
dataset.Weather = enc.transform(
    dataset.Weather)  # a LabelEncoder is used to transform categorical values of the types of weathers into numerical values

data_train, data_test, y_train, y_test = train_test_split(dataset.values, dataset.values[:, 3], test_size=0.3)

knn_train_data = np.delete(data_train, 3, 1)
knn_test_data = np.delete(data_test, 3, 1)

clf = KNeighborsClassifier(15)
clf.fit(knn_train_data, data_train[:, 3])

knn_result = clf.predict(knn_test_data)

# cross validation knn
scores = []
data_without_target = np.delete(dataset.values, 3, 1)
target = dataset.values[:, 3]
k_fold = KFold(n_splits=10)
for train_indices, test_indices in k_fold.split(target):
    scores.append(
        clf.fit(data_without_target[train_indices], target[train_indices]).score(data_without_target[test_indices],
                                                                                 target[test_indices]))

print("knn accuracy: %0.4f (+/- %0.4f)" % (np.mean(scores), np.std(scores) * 2))
