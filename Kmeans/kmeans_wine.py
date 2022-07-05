import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import pandas as pd
import random

wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns = wine.feature_names)
wine_target = pd.DataFrame(wine.target)

wine_data = np.array(wine_df[["od280/od315_of_diluted_wines", "proline"]])
target = np.array(wine_target)

plt.scatter(wine_data[:,0], wine_data[:,1], c = target)
plt.show()

def Metric(k, data, c_center):

    min_index = []
    metrics = []

    for x in data:
        for j in range(k):
            metric = np.sqrt(((x[0] - c_center[j, 0]) ** 2) + ((x[1] - c_center[j, 1]) ** 2))
            metrics.append(metric)

        min_index.append(metrics.index(min(metrics)))
        metrics = []
    
    return min_index

def Kmeans(k, data):

    cluster_center = np.array([data[random.randint(0, len(data))] for i in range(k)])

    first_cluster_center = cluster_center

    plt.scatter(data[:,0], data[:,1])
    plt.scatter(cluster_center[:,0], cluster_center[:,1], c = "red")
    plt.show()
    
    cluster_label = Metric(k, data, cluster_center)

    while True:
        center_flag = cluster_center.copy()

        for y in range(k):
            cluster_data = np.array([a for (a, b) in zip(data, cluster_label) if b == y])
            cluster_center[y] = np.array([sum(cluster_data[:,0]) / len(cluster_data), sum(cluster_data[:,1]) / len(cluster_data)])
        
        if np.all(center_flag == cluster_center):
            return first_cluster_center, cluster_label, cluster_center
        
        cluster_label = Metric(k, data, cluster_center)

first_cluster = []

first, label, center = Kmeans(3, wine_data)
plt.scatter(wine_data[:,0], wine_data[:,1], c = label)
plt.scatter(center[:,0], center[:,1], c = "red")
plt.show()