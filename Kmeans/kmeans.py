import numpy as np
import matplotlib.pyplot as plt
import random

mean = np.array([0, 0])
cov = np.array([[4, 2], [2, 4]])

data1 = np.random.multivariate_normal(mean, cov, 100)
data1 = np.concatenate([data1, np.array([(j[0] + 17, j[1]) for j in data1]), np.array([i + 17 for i in data1])])

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
            return cluster_label, cluster_center
        
        cluster_label = Metric(k, data, cluster_center)

label, center = Kmeans(3, data1)

plt.scatter(data1[:,0], data1[:,1], c = label)
plt.scatter(center[:,0], center[:,1], c = "red")
plt.show()