import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 正規化を行う関数
def Normalization(data):
    for j in range(2):
        min_data = min(data[:,j + 1])
        max_data = max(data[:,j + 1])

        # 使ってる正規化方法はmin-max正規化
        for k in range(len(data)):
            data[k, j + 1] = (data[k, j + 1] - min_data) / (max_data - min_data)
    
    return data

def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def Loss(lb, Z):
    loss = 0
    for (t, z) in zip(lb, Z):
        loss += t * math.log(z) + (1 - t) * math.log(1 - z)
        
    return loss[0] * (-1)

def Logistic_Regression(max_iter, data, labels):

    w = np.random.rand(len(data[0])).reshape([3, 1])

    first_boundary_x, first_boundary_y = Boundary(data, w)

    plt.scatter(data[:,1], data[:,2], c = label[:,0])
    plt.plot(first_boundary_x, first_boundary_y, c = "red")
    plt.xlim(min(data[:,1]), max(data[:,1]))
    plt.ylim(min(data[:,2]) - 1, max(data[:,2]) + 1)
    plt.show()
    plt.clf()

    l = len(data)
    z = np.empty((l, 1))
    z_1 = np.empty((l, 1))

    loss_list = []
    p = 1
    p_list = []

    for _ in range(max_iter):
        for i, l in enumerate(labels):
            a = np.dot(w.T, data[i])
            z[i, 0] = Sigmoid(a)
            z_1[i, 0] = z[i, 0] * (1 - z[i, 0])
            p *= (z[i, 0] ** l) * ((1 - z[i, 0]) ** (1 - l))

        zz = np.ravel(z_1)
        R = np.diag(zz)
        
        H_inv = np.linalg.inv(data.T @ R @ data)
        w = w -  (H_inv @ (data.T @ (z - labels)))

        loss = Loss(labels, z)
        print("loss :{} likelihood :{}".format(loss, p[0]))
        loss_list.append(loss)
        p_list.append(p)
        
        p = 1

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
    
    # irisデータの読み込み
    iris = load_iris()

    # irisデータのデータ部分の取り出し
    iris_data = iris.data

    # 今回は3種類のうちVersicolorとVersinicaの2種類を判別（初めの50個まではSetonaのデータなのでそれ以降を抽出）
    # がく片の長さ、がく片の幅、花びらの長さ、花びらの幅の4つの特徴量からがく片の幅、花びらの長さを選択
    data = iris_data[50:, [1, 2]]
    data = np.insert(data, 0, 1, axis = 1)

    # 値が大きいとシグモイド関数がオーバーフローしてしまうので0から1の値になるように正規化
    data = Normalization(data)

    # 正解ラベルの抽出
    label = iris.target[50:].reshape(1, -1).T

    # 正解ラベルが1と2なので0と1になるように置換
    label = np.where(label == 1, 0, 1)

    P_max, LR_loss, result_label, result_w = Logistic_Regression(10, data, label)

    accuracy = Accuracy(result_label, label)
    print("Accuracy :", accuracy)

    boundary_x, boundary_y = Boundary(data, result_w)

    plt.scatter(data[:,1], data[:,2], c = label[:,0])
    plt.plot(boundary_x, boundary_y, c = "red")
    plt.xlim(min(data[:,1]), max(data[:,1]))
    plt.ylim(min(data[:,2]) - 1, max(data[:,2]) + 1)
    plt.show()

    plt.clf()
    plt.plot(LR_loss)
    plt.show()

    plt.clf()
    plt.plot(P_max)
    plt.show()