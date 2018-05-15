#!/usr/bin/env python

from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer, LabelEncoder, MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv("SFWeather clean.csv", usecols=range(1, 7))
print(str(dataset.head()))

enc = LabelEncoder()
enc.fit(dataset.Weather)
dataset.Weather = enc.transform(dataset.Weather)

dataset = dataset.dropna()

features = ['Humidity', 'Pressure', 'Temperature', 'WindDirection', 'WindSpeed']
train_ds = dataset[features]

normalized_ds = MinMaxScaler().fit_transform(train_ds)

prediction = KMeans(n_clusters=8).fit_predict(normalized_ds)

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

dataset = pd.read_csv("SFWeather.csv", usecols=range(1, 7))
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











