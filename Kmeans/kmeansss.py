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

# plt.scatter(wine_data[:,0], wine_data[:,1], c = target)
# plt.show()

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
        
    # cluster_center = np.array([data[random.randint(0, len(data) - 1)] for i in range(k)])
    cluster_center = np.array([
        [3.116, 1225],
        [3.041, 432],
        [1.669, 609]
    ])
    first_cluster_center = cluster_center

    plt.scatter(data[:,0], data[:,1])
    plt.scatter(cluster_center[:,0], cluster_center[:,1], c = "red")
    plt.savefig("first_cls.png")
    plt.show()
    
    cluster_label = Metric(k, data, cluster_center)

    while True:
        center_flag = cluster_center.copy()

        for y in range(k):
            cluster_data = np.array([a for (a, b) in zip(data, cluster_label) if b == y])
            if cluster_data.size != 0:
                cluster_center[y] = np.array([sum(cluster_data[:,0]) / len(cluster_data), sum(cluster_data[:,1]) / len(cluster_data)])

        if np.all(center_flag == cluster_center):
            return first_cluster_center, cluster_label, cluster_center
        
        # plt.scatter(data[:,0], data[:,1], c = cluster_label)
        # plt.scatter(cluster_center[:,0], cluster_center[:,1], c = "red")
        # plt.show()
        
        cluster_label = Metric(k, data, cluster_center)

first_cluster = []

for i in range(1):
    first, label, center = Kmeans(3, wine_data)
    first = first.tolist()
    first = sorted(first, key = lambda x : x[1])
    first_cluster.append(first)
    plt.scatter(wine_data[:,0], wine_data[:,1], c = label)
    plt.scatter(center[:,0], center[:,1], c = "red")
    plt.savefig("define_center.png")
    plt.show()

first_cluster = np.array(first_cluster)
first_cluster_1 = np.array(first_cluster[:, [True, False, False]])
first_cluster_2 = np.array(first_cluster[:, [False, True, False]])
first_cluster_3 = np.array(first_cluster[:, [False, False, True]])

c1_x = np.ravel(first_cluster_1[:, :, 0])
c1_y = np.ravel(first_cluster_1[:, :, 1])
c2_x = np.ravel(first_cluster_2[:, :, 0])
c2_y = np.ravel(first_cluster_2[:, :, 1])
c3_x = np.ravel(first_cluster_3[:, :, 0])
c3_y = np.ravel(first_cluster_3[:, :, 1])

# plt.scatter(c1_x, c1_y, c = 'red', s = 75)
# plt.scatter(c2_x, c2_y, c = 'blue', s = 75)
# plt.scatter(c3_x, c3_y, c = 'green', s = 75)
# plt.scatter(wine_data[:,0], wine_data[:,1], c = target, s = 15)
# # plt.savefig("1000kmeams.png")
# plt.show()