#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

dataset = pd.read_csv("SFWeather clean.csv", usecols=range(1, 7))  # load the dataset
print(str(dataset.head()))  # just print the first 5 rows to check the data

dataset = dataset.dropna()  # removes rows that have invalid values

features = ['Humidity', 'Pressure', 'Temperature']
train_ds = dataset[features]  # select the columns to use

normalized_ds = MinMaxScaler().fit_transform(train_ds)  # The dataset is normalized

prediction = KMeans(n_clusters=8).fit_predict(normalized_ds)  # KMeans is trained and run with 8 clusters

plt.figure()
plt.scatter(dataset.values[:, 2], prediction, c=prediction)  # Plot showing Temperatures distributed by clusters
plt.title("Temperatures and Clusters")

plt.figure()
plt.scatter(dataset.values[:, 0], prediction, c=prediction)  # Plot showing Humidity distributed by clusters
plt.title("Humidity and Clusters")

plt.figure()
plt.scatter(dataset.values[:, 3], prediction, c=prediction)  # Plot showing Weather distributed by clusters
plt.title("Weathers and Clusters")
plt.show()

# 28 clusters

dataset = pd.read_csv("SFWeather.csv", usecols=range(1, 7))  # Same code but with all the original types of Weather, so 28 clusters
print(str(dataset.head()))

enc = LabelEncoder()
enc.fit(dataset.Weather)
dataset.Weather = enc.transform(dataset.Weather)

dataset = dataset.dropna()

features = ['Humidity', 'Pressure', 'Temperature']
train_ds = dataset[features]

normalized_ds = MinMaxScaler().fit_transform(train_ds)

prediction = KMeans(n_clusters=28).fit_predict(normalized_ds)

plt.figure()
plt.scatter(dataset.values[:, 2], prediction, c=prediction)
plt.title("Temperatures and Clusters")

plt.figure()
plt.scatter(dataset.values[:, 0], prediction, c=prediction)
plt.title("Humidity and Clusters")

plt.figure()
plt.scatter(dataset.values[:, 3], prediction, c=prediction)
plt.title("Weathers and Clusters")
plt.show()
