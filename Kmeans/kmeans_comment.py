import numpy as np
import matplotlib.pyplot as plt
import random

# 平均と分散共分散行列の定義
mean = np.array([0, 0])
cov = np.array([[2, 4], [4, 2]])

# 平均と分散共分散行列からデータ点を100個生成
data1 = np.random.multivariate_normal(mean, cov, 100)
# 生成したデータのx座標を13ずらしたデータとx,y座標を13ずらしたデータを生成して一つに結合
data1 = np.concatenate([data1, np.array([(j[0] + 13, j[1]) for j in data1]), np.array([i + 13 for i in data1])])

# データとの距離を求める関数。引数：k, 生成したデータ, クラスタ中心
def Metric(k, data, c_center):

    min_index = []
    metrics = []

    # データ一つと各クラスタとの距離(ユークリッド距離)を計算
    for x in data:
        # 各クラスタ中心と一つのデータ点との距離を計算するから回数はk回
        for j in range(k):
            # ユークリッド距離の計算
            metric = np.sqrt(((x[0] - c_center[j, 0]) ** 2) + ((x[1] - c_center[j, 1]) ** 2))
            # 各クラスタとの求めた距離をリストに格納
            metrics.append(metric)
        
        # 3つのクラスタのうち一番距離が近いクラスタの番号をリストに格納
        min_index.append(metrics.index(min(metrics)))
        metrics = []
    
    # 全てのデータ点において一番近いクラスタ番号が格納されたリストを返り値
    return min_index

def Kmeans(k, data):

    # data1からランダムな3箇所の座標を抽出(クラスタ中心の初期値)
    cluster_center = np.array([data[random.randint(0, len(data))] for i in range(k)])

    # 初期位置を表示
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(cluster_center[:,0], cluster_center[:,1], c = "red")
    plt.show()
    
    # 初期位置からデータ点までの距離を計算
    cluster_label = Metric(k, data, cluster_center)

    while True:
       # 更新前のクラスタ中心を代入 
       center_flag = cluster_center.copy()

       for y in range(k):
           # 各クラスタに所属するデータを配列として代入
           cluster_data = np.array([a for (a, b) in zip(data, cluster_label) if b == y])

           # 各クラスタに所属するデータから平均を求めて新しいクラスタ中心の決定
           cluster_center[y] = np.array([sum(cluster_data[:,0]) / len(cluster_data), sum(cluster_data[:,1]) / len(cluster_data)])
       
       # 更新前と更新後のクラスタ中心の座標が変わらなければWhile文を終了
       if np.all(center_flag == cluster_center):
           return cluster_label, cluster_center

       # 更新したクラスタ中心と元データから距離を計算
       cluster_label = Metric(k, data, cluster_center)
    
label, center = Kmeans(3, data1)

plt.scatter(data1[:,0], data1[:,1], c = label)
plt.scatter(center[:,0], center[:,1], c = "red")
plt.show()