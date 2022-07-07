import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def Loss(lb, Z):
    loss = 0
    for (t, z) in zip(lb, Z):
        loss += t * math.log(z) + (1 - t) * math.log(1 - z)
        
    return loss[0] * (-1)

def Logistic_Regression(max_iter, data, labels):

    w = np.random.rand(len(data[0])).reshape([3, 1])

    l = len(data)
    z = np.empty((l, 1))
    z_1 = np.empty((l, 1))

    loss_list = []
    p = 1
    p_list = []

    for _ in range(max_iter):
        for i, l in enumerate(labels):
            a = np.dot(w.T, data[i])
            z[i, 0] = sigmoid(a)
            z_1[i, 0] = z[i, 0] * (1 - z[i, 0])
            p *= (z[i, 0] ** l) * ((1 - z[i, 0]) ** (1 - l))
        
        p_list.append(p)
        p = 1

        zz = np.ravel(z_1)
        R = np.diag(zz)
        
        H_inv = np.linalg.inv(data.T @ R @ data)
        w = w -  (H_inv @ (data.T @ (z - labels)))

        loss = Loss(labels, z)
        print("loss :", loss)
        loss_list.append(loss)

    return p_list, loss_list

if __name__ == "__main__":
    mean = np.array([0, 0])
    cov = np.array([[0.03, 0.02], [0.02, 0.05]])

    data1 = np.random.multivariate_normal(mean, cov, 250)
    data1 = np.insert(data1, 2, 0, axis = 1)
    data2 = np.array([(j[0] + 0.5, j[1]) for j in data1])
    data2 = np.insert(data2, 2, 1, axis = 1)
    data = np.concatenate([data1, data2])
    data = np.insert(data, 0, 1, axis = 1)

    plt.scatter(data[:,1], data[:,2], c = data[:,3])
    plt.show()

    np.random.shuffle(data)

    label = data[:,3].reshape(1, -1).T
    data = np.delete(data, obj = 3, axis = 1)

    P_max, LR_loss = Logistic_Regression(12, data, label)

    plt.plot(LR_loss)
    plt.show()
    plt.clf()
    plt.plot(P_max)
    plt.show()