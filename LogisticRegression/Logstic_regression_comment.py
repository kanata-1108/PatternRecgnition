import numpy as np
import math
import matplotlib.pyplot as plt

# シグモイド間数の宣言
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# 誤差関数の定義
# 引数：データのラベル, シグモイド関数の返り値
def Loss(lb, Z):
    loss = 0
    for (t, z) in zip(lb, Z):
        loss += t * math.log(z) + (1 - t) * math.log(1 - z)
        
    return loss[0] * (-1)

def Logistic_Regression(max_iter, data, labels):

    # 重み(更新パラメータ)の初期化
    w = np.random.rand(len(data[0])).reshape([3, 1])

    # zとz(1 - z)の空配列の作成
    l = len(data)
    z = np.empty((l, 1))
    z_1 = np.empty((l, 1))

    # 更新ごとの誤差を格納するリストの作成
    loss_list = []

    # 最大化したい尤度の初期化
    p = 1

    # 更新後との尤度を格納するリストの作成
    p_list = []

    # max_iter回更新を行う
    for _ in range(max_iter):
        for i, l in enumerate(labels):

            # データと重みの積を求める
            a = np.dot(w.T, data[i])

            # 求めたaをシグモイド関数に渡す
            z[i, 0] = sigmoid(a)

            # z(1 - z)を求める
            z_1[i, 0] = z[i, 0] * (1 - z[i, 0])

            # 尤度の計算
            p *= (z[i, 0] ** l) * ((1 - z[i, 0]) ** (1 - l))
        
        # パラメータの更新ごとの尤度をリストに挿入
        p_list.append(p)

        # 尤度の初期化
        p = 1
        
        # 10行1列の二次元行列なので1次元にする
        # 次に使うnp.diagは一次元配列を受け取ったら二次元の対角行列を返すので一次元化する必要がある。
        zz = np.ravel(z_1)

        # 求めたz(1 - z)を対角成分とする対角行列の作成
        R = np.diag(zz)
        
        # ヘッセ行列の逆行列の計算
        H_inv = np.linalg.inv(data.T @ R @ data)

        # 重みパラメータの更新
        w = w - (H_inv @ (data.T @ (z - labels)))

        # 誤差の計算
        loss = Loss(labels, z)

        # 更新ごとの誤差の表示
        print("loss :", loss)

        # パラメータの更新ごとの誤差をリストに挿入
        loss_list.append(loss)

    return p_list, loss_list, z, w

# 正解率を求める関数　引数：ロジスティック回帰後に得られた各データのラベル(確率), 本来のラベル
def Accuracy(return_label, true_label):
    count = 0
    for re_label, tr_label in zip(return_label, true_label):

        # ロジスティック回帰後のラベルは確率なので四捨五入して0か1に四捨五入する
        re_label = np.round(re_label)

        # 回帰後のラベルと本来のラベルが等しかったらcountに1加算する
        if re_label == tr_label:
            count += 1
    
    # 正解したデータ数を全データ数で割って百分率で返す
    accuracyscore = (count / len(true_label)) * 100

    return accuracyscore

# 決定境界を求める関数
def Boundary(xy_data, param_w):

    # データのx軸の一番小さい値からy軸の一番大きい値の範囲で100個データを生成
    x = np.linspace(min(xy_data[:,1]), max(xy_data[:,1]), 100)

    # ロジスティック回帰後に得られた重みと上で生成したデータからy座標を算出
    y = ((param_w[0] + param_w[1] * x) / param_w[2]) * (-1)

    return x, y

if __name__ == "__main__":
    # 平均と分散共分散行列の定義
    mean = np.array([0, 0])
    cov = np.array([[0.03, 0.02], [0.02, 0.05]])

    # データの作成 [x座標, y座標, ラベル(0 or 1)]
    data1 = np.random.multivariate_normal(mean, cov, 250)
    data1 = np.insert(data1, 2, 0, axis = 1)
    data2 = np.array([(j[0] + 0.5, j[1]) for j in data1])
    data2 = np.insert(data2, 2, 1, axis = 1)
    data = np.concatenate([data1, data2])
    data = np.insert(data, 0, 1, axis = 1)

    # 作成したデータをシャッフルする
    np.random.shuffle(data)

    # ラベルを抽出して、dataからラベルの列を削除
    label = data[:,3].reshape(1, -1).T
    data = np.delete(data, obj = 3, axis = 1)

    plt.scatter(data[:,1], data[:,2], c = label[:,0])

    # 表示するx軸とy軸の上限下限を定義
    plt.xlim(min(data[:,1]) - 0.04, max(data[:,1]) + 0.04)
    plt.ylim(min(data[:,2]) - 0.04, max(data[:,2]) + 0.04)
    plt.show()
    plt.clf()

    P_max , LR_loss, result_label, result_w = Logistic_Regression(12, data, label)

    # 正解率の算出と表示
    accuracy = Accuracy(result_label, label)
    print("Accuracy :", accuracy)

    # 決定境界の座標の算出
    boundary_x, boundary_y = Boundary(data, result_w)

    # 元のデータ上に決定境界を表示
    plt.scatter(data[:,1], data[:,2], c = label[:,0])
    plt.plot(boundary_x, boundary_y, c = "red")
    plt.xlim(min(data[:,1]) - 0.04, max(data[:,1]) + 0.04)
    plt.ylim(min(data[:,2]) - 0.04, max(data[:,2]) + 0.04)
    plt.show()
    plt.clf()

    # 更新ごとの誤差の推移グラフの表示
    plt.plot(LR_loss)
    plt.show()
    plt.clf()

    # 更新後との尤度グラフの表示
    plt.plot(P_max)
    plt.show()