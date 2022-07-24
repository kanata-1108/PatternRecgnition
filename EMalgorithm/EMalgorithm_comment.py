from curses.ascii import NAK
import numpy as np
import matplotlib.pyplot as plt

# 負担率を求める関数
def Resp(k, data, mean, cov, pi):
    new_resp = np.empty((len(data), k))
    resp_den = 0

    for point in range(len(data)):
        for pi_i in range(k):
            # 負担率の分子の算出
            den_N = np.exp((-0.5) * (data[point] - mean[pi_i]).T @ np.linalg.inv(cov[pi_i]) @ (data[point] - mean[pi_i])) / np.sqrt(((2 * np.pi) ** 2) * np.linalg.det(cov[pi_i]))
            resp_den += pi[pi_i] * den_N
        
        for pi_k in range(k):
            # 負担率の算出
            new_resp[point, pi_k] = (pi[pi_k] * (np.exp((-0.5) * (data[point] - mean[pi_k]).T @ np.linalg.inv(cov[pi_k]) @ (data[point] - mean[pi_k])) / np.sqrt(((2 * np.pi) ** 2) * np.linalg.det(cov[pi_k])))) / resp_den
        
        resp_den = 0
    
    return new_resp

# 平均
def Mean(k, data, resp):
    new_mean = np.empty((3, 2))
    tmp1 = 0
    tmp2 = 0

    for i in range(k):
        for j in range(len(data)):
            # Nkを算出（分母）
            tmp1 += resp[j, i]
            # 分子の算出
            tmp2 += resp[j, i] * data[j]
        
        # 平均の算出
        new_mean[i] = tmp2 / tmp1
        tmp1 = 0
        tmp2 = 0

    return new_mean

# 分散
def Cov(k, data, resp, mean):
    new_cov = np.empty((3, 2, 2))
    tmp1 = 0
    tmp2 = 0

    for i in range(k):

        # (2,)の配列から(2,1)へ変換。これをしないと転置ができない
        cov_mean = mean[i].reshape(1, -1)
        for j in range(len(data)):
            cov_data = data[j].reshape(1, -1)

            # Nkの算出（分母）
            tmp1 += resp[j, i]

            # tmp2とtmp3で分子の算出
            tmp3 = ((cov_data - cov_mean).T @ (cov_data - cov_mean))
            tmp2 += resp[j, i] * tmp3
        
        # 分散・共分散行列の算出
        new_cov[i] = tmp2 / tmp1
        tmp1 = 0
        tmp2 = 0

    return new_cov

# 混合係数
def Pi(k, data, resp):
    new_pi = np.empty(3)
    tmp1 = 0

    for i in range(k):
        for j in range(len(data)):

            # Nkの算出
            tmp1 += resp[j, i]

        # 混合係数の算出
        new_pi[i] = tmp1 / len(data)
        tmp1 = 0

    return new_pi

# 尤度
def Likelihood(k, data, mean, cov, pi):
    tmp = 0
    likelihood = 0

    for i in range(len(data)):
        for j in range(k):

            # 正規分布Nの算出
            N = np.exp((-0.5) * (data[i] - mean[j]).T @ np.linalg.inv(cov[j]) @ (data[i] - mean[j])) / np.sqrt(((2 * np.pi) ** 2) * np.linalg.det(cov[j]))
            tmp += pi[j] * N
        
        # 対数尤度の算出
        likelihood += np.log(tmp)
        tmp = 0
    
    return likelihood

# EMアルゴリズム
def EMalgorithum(max_iter, k, data, mean, cov, pi):
    likelihood_list = []
    cluster_label = []

    # 尤度の初期値を算出
    first_likelihood = Likelihood(k, data, mean, cov, pi)
    likelihood_list.append(first_likelihood)
    print(first_likelihood)

    # 指定した回数(max_iter)EMアルゴリズムを行う
    for _ in range(max_iter):
        resp = Resp(k, data, mean, cov, pi)
        mean = Mean(k, data, resp)
        cov = Cov(k, data, resp, mean)
        pi = Pi(k, data, resp)

        # 更新したパラメータで尤度を算出
        after_likelihood = Likelihood(k, data, mean, cov, pi)
        likelihood_list.append(after_likelihood)

        print("対数尤度 :", after_likelihood)
    
    # 最終的に得られた各データの負担率のうち一番大きい値のインデックスをリストに格納
    # 格納したリストが推定したデータのラベルになる
    for l in resp:
        max_index = np.argmax(l)
        cluster_label.append(max_index)
    
    return likelihood_list, cluster_label

if __name__ == "__main__":

    # 生成するデータの平均と分散共分散行列の定義
    data_mean = np.array([0, 0])
    data_cov = np.array([[4, 1], [1, 3]])

    # データの生成
    sample_data1 = np.random.multivariate_normal(data_mean, data_cov, 100)
    sample_data2 = np.array([(j[0] + 10, j[1]) for j in sample_data1])
    sample_data3 = np.array([i + 10 for i in sample_data1])

    # データのラベル付け
    sample_data1 = np.insert(sample_data1, 2, 0, axis = 1)
    sample_data2 = np.insert(sample_data2, 2, 1, axis = 1)
    sample_data3 = np.insert(sample_data3, 2, 2, axis = 1)

    # 3つのデータの結合
    sample_data = np.concatenate([sample_data1, sample_data2, sample_data3])

    # データのシャッフル
    np.random.shuffle(sample_data)

    # シャッフルしたデータのラベルの抽出
    label = sample_data[:,2].reshape(1, -1).T

    # データからラベル列の削除
    sample_data = np.delete(sample_data, obj = 2, axis = 1)

    # 平均、分散共分散行列、混合係数の初期化
    rand_mean = np.random.randint(3, 10, (3, 2))
    rand_cov = np.array([np.identity(2) * 5 for _ in range(3)])
    rand_pi = np.random.random_sample(3)

    # クラスタ数（分布数）
    k = 3

    # EMアルゴリズムを実行し、尤度のリストと推定したラベルを取得
    lkhood, result_label = EMalgorithum(20, k, sample_data, rand_mean, rand_cov, rand_pi)

    # 元データの分布の描画
    plt.scatter(sample_data[:,0], sample_data[:,1], c = label)
    plt.show()
    plt.clf()

    # 推定後の分布の描画
    plt.scatter(sample_data[:,0], sample_data[:,1], c = result_label)
    plt.show()
    plt.clf()

    # 尤度の推移のグラフ
    plt.plot(range(0, 31), lkhood)
    plt.show()
