#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/17 14:46
# @Author  : gao

import os
import sys
sys.path.append(os.path.abspath('.'))
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from cal_clusters_num import cal_clusters_num
import copy
from sklearn import datasets

# 用散点图展示聚类效果
def plot_cluster(X, labels, t, path_data):
    plt.rcParams['savefig.dpi'] = 150       # 图片像素
    plt.rcParams['figure.dpi'] = 100        # 分辨率
    fig = plt.figure(1, figsize=(6, 4))
    ax = Axes3D(fig, rect=[0, 0, .90, 1], elev=48, azim=134)
    if type(X) is not np.ndarray:
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], c=labels.astype(np.float), edgecolor='k')
        ax.set_xlim3d(min(X.iloc[:, 0])-0.01, max(X.iloc[:, 0])+0.01)
        ax.set_ylim3d(min(X.iloc[:, 1])-0.01, max(X.iloc[:, 1])+0.01)
        ax.set_zlim3d(min(X.iloc[:, 2])-0.01, max(X.iloc[:, 2])+0.01)
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
        ax.set_zlabel(X.columns[2])
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels.astype(np.float), edgecolor='k')
        ax.set_xlim3d(min(X[:, 0])-0.01, max(X[:, 0])+0.01)
        ax.set_ylim3d(min(X[:, 1])-0.01, max(X[:, 1])+0.01)
        ax.set_zlim3d(min(X[:, 2])-0.01, max(X[:, 2])+0.01)
        ax.set_xlabel('first column')
        ax.set_ylabel('second column')
        ax.set_zlabel('third column')
    sil = metrics.silhouette_score(X, labels, metric='euclidean')
    har = metrics.calinski_harabaz_score(X, labels)
    plt.title('K = %s, clusters, silhouette coef = %.03f, calinski_harabaz_score coef = %.03f' % (t, sil, har))
    #ax.set_title('kmeans_clusters')
    print('K = %s, clusters, silhouette coef = %.03f, calinski_harabaz_score coef = %.03f' % (t, sil, har))
    ax.dist = 12
    plt.savefig((path_data + '/output/clusters_{n1}_{n2}_{n3}_{t}_{sil2}.png').format(n1=X.columns[0],
                                                                                      n2=X.columns[1],
                                                                                      n3=X.columns[2],
                                                                                      t=t, sil2=sil), dpi=100)
    #plt.show()
    plt.clf()

# dbscan聚类
def dbscan_clusters(X, y):
    np.random.seed(5)
    if y == 'y':
        scaler = MinMaxScaler()
        scaler.fit(X.astype(float))
        X = scaler.transform(X)
    dbs = DBSCAN(eps=0.5,  # 邻域半径
           min_samples=5,  # 最小样本点数，MinPts
           metric='euclidean',
           metric_params=None,
           algorithm='auto',  # 'auto','ball_tree','kd_tree','brute',4个可选的参数 寻找最近邻点的算法，例如直接密度可达的点
           leaf_size=10,  # balltree,cdtree的参数
           p=None,
           n_jobs=2)
    dbs.fit(X)
    return X, dbs

# 列名抽取n个的不同组合
def Cnr(lst, n):
    result = []
    tmp = [0] * n
    length = len(lst)
    def next_num(li=0, ni=0):
        if ni == n:
            result.append(copy.copy(tmp))
            return
        for lj in range(li, length):
            tmp[ni] = lst[lj]
            next_num(lj+1, ni+1)
    next_num()
    return result

# 加载数据
def load_file():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data)
    y = pd.DataFrame(iris.target)
    ddata = pd.merge(X, y, left_index=True, right_index=True, how='left')
    ddata.columns = ['sepal_len', 'sepal_width', 'petal_len', 'petal_width', 'class']
    return ddata

# 不同组合聚类
def do_main():
    path_data = os.path.abspath('.')
    ddata = load_file()
    dd = [each for each in list(ddata.columns) if each not in ['class']]
    lis = Cnr(dd, 3)
    for jj in lis:
        print(jj)
        X, est = dbscan_clusters(ddata[jj], 'n')
        if len(set(est.labels_)) != 1:
            plot_cluster(X, est.labels_, len(set(est.labels_)), path_data)
        else:
            pass


if __name__ == '__main__':
    do_main()