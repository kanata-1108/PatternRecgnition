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

    return p_list, loss_list, z, w

def Accuracy(return_label, true_label):
    count = 0
    for re_label, tr_label in zip(return_label, true_label):
        re_label = np.round(re_label)
        if re_label == tr_label:
            count += 1
    
    accuracyscore = (count / len(true_label)) * 100

    return accuracyscore

def Boundary(xy_data, param_w):
    x = np.linspace(min(xy_data[:,1]), max(xy_data[:,1]), 100)
    y = ((param_w[0] + param_w[1] * x) / param_w[2]) * (-1)

    return x, y

if __name__ == "__main__":
    mean = np.array([0, 0])
    cov = np.array([[0.03, 0.02], [0.02, 0.05]])

    data1 = np.random.multivariate_normal(mean, cov, 250)
    data1 = np.insert(data1, 2, 0, axis = 1)
    data2 = np.array([(j[0] + 0.5, j[1]) for j in data1])
    data2 = np.insert(data2, 2, 1, axis = 1)
    data = np.concatenate([data1, data2])
    data = np.insert(data, 0, 1, axis = 1)

    np.random.shuffle(data)

    label = data[:,3].reshape(1, -1).T
    data = np.delete(data, obj = 3, axis = 1)

    plt.scatter(data[:,1], data[:,2], c = label[:,0])
    plt.xlim(min(data[:,1]) - 0.04, max(data[:,1]) + 0.04)
    plt.ylim(min(data[:,2]) - 0.04, max(data[:,2]) + 0.04)
    plt.show()
    plt.clf()

    P_max, LR_loss, result_label, result_w = Logistic_Regression(12, data, label)

    accuracy = Accuracy(result_label, label)
    print("Accuracy :", accuracy)

    boundary_x, boundary_y = Boundary(data, result_w)

    plt.scatter(data[:,1], data[:,2], c = label[:,0])
    plt.plot(boundary_x, boundary_y, c = "red")
    plt.xlim(min(data[:,1]) - 0.04, max(data[:,1]) + 0.04)
    plt.ylim(min(data[:,2]) - 0.04, max(data[:,2]) + 0.04)
    plt.show()

    plt.clf()
    plt.plot(LR_loss)
    plt.show()

    plt.clf()
    plt.plot(P_max)
    plt.show()