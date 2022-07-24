import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import pandas as pd

# 正規化を行う関数
def Normalization(data):
    for j in range(2):
        min_data = min(data[:,j])
        max_data = max(data[:,j])

        # 使ってる正規化方法はmin-max正規化
        for k in range(len(data)):
            data[k, j] = (data[k, j] - min_data) / (max_data - min_data)
    
    return data

def Resp(k, data, mean, cov, pi):
    new_resp = np.empty((len(data), k))
    resp_den = 0

    for point in range(len(data)):
        for pi_i in range(k):
            den_N = np.exp((-0.5) * (data[point] - mean[pi_i]).T @ np.linalg.inv(cov[pi_i]) @ (data[point] - mean[pi_i])) / np.sqrt(((2 * np.pi) ** 2) * np.linalg.det(cov[pi_i]))
            resp_den += pi[pi_i] * den_N
        
        for pi_k in range(k):
            new_resp[point, pi_k] = (pi[pi_k] * (np.exp((-0.5) * (data[point] - mean[pi_k]).T @ np.linalg.inv(cov[pi_k]) @ (data[point] - mean[pi_k])) / np.sqrt(((2 * np.pi) ** 2) * np.linalg.det(cov[pi_k])))) / resp_den
        
        resp_den = 0
    
    return new_resp

def Mean(k, data, resp):
    new_mean = np.empty((3, 2))
    tmp1 = 0
    tmp2 = 0

    for i in range(k):
        for j in range(len(data)):
            tmp1 += resp[j, i]
            tmp2 += resp[j, i] * data[j]
        
        new_mean[i] = tmp2 / tmp1
        tmp1 = 0
        tmp2 = 0

    return new_mean

def Cov(k, data, resp, mean):
    new_cov = np.empty((3, 2, 2))
    tmp1 = 0
    tmp2 = 0

    for i in range(k):
        cov_mean = mean[i].reshape(1, -1)
        for j in range(len(data)):
            cov_data = data[j].reshape(1, -1)
            tmp1 += resp[j, i]
            tmp3 = ((cov_data - cov_mean).T @ (cov_data - cov_mean))
            tmp2 += resp[j, i] * tmp3
        
        new_cov[i] = tmp2 / tmp1
        tmp1 = 0
        tmp2 = 0

    return new_cov

def Pi(k, data, resp):
    new_pi = np.empty(3)
    tmp1 = 0

    for i in range(k):
        for j in range(len(data)):
            tmp1 += resp[j, i]
    
        new_pi[i] = tmp1 / len(data)
        tmp1 = 0

    return new_pi

def Likelihood(k, data, mean, cov, pi):
    tmp = 0
    likelihood = 0

    for i in range(len(data)):
        for j in range(k):
            N = np.exp((-0.5) * (data[i] - mean[j]).T @ np.linalg.inv(cov[j]) @ (data[i] - mean[j])) / np.sqrt(((2 * np.pi) ** 2) * np.linalg.det(cov[j]))
            tmp += pi[j] * N
        
        likelihood += np.log(tmp)
        tmp = 0
    
    return likelihood

def EMalgorithum(max_iter, k, data, mean, cov, pi):
    likelihood_list = []
    cluster_label = []

    first_likelihood = Likelihood(k, data, mean, cov, pi)
    likelihood_list.append(first_likelihood)
    print("log likelihood :", first_likelihood)

    for _ in range(max_iter):
        resp = Resp(k, data, mean, cov, pi)
        mean = Mean(k, data, resp)
        cov = Cov(k, data, resp, mean)
        pi = Pi(k, data, resp)
        after_likelihood = Likelihood(k, data, mean, cov, pi)
        likelihood_list.append(after_likelihood)

        print("log likelihood :", after_likelihood)
    
    for l in resp:
        max_index = np.argmax(l)
        cluster_label.append(max_index)
    
    return likelihood_list, cluster_label, mean, cov, pi

if __name__ == "__main__":
    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns = wine.feature_names)
    wine_target = pd.DataFrame(wine.target)

    wine_data = np.array(wine_df[["od280/od315_of_diluted_wines", "proline"]])
    wine_data = Normalization(wine_data)
    target = np.array(wine_target)

    rand_mean = np.random.rand(3, 2)
    rand_cov = np.array([np.identity(2) * 0.05 for _ in range(3)])
    rand_pi = np.random.random_sample(3)

    k = 3

    lkhood, result_label, result_mean, result_cov, result_pi = EMalgorithum(40, k, wine_data, rand_mean, rand_cov, rand_pi)

    plt.scatter(wine_data[:,0], wine_data[:,1], c = target)
    plt.show()
    plt.clf()

    plt.scatter(wine_data[:,0], wine_data[:,1], c = result_label)
    plt.show()
    plt.clf()

    plt.plot(range(0, 41), lkhood)
    plt.show()
