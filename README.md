
#### DBSCAN聚类算法
---------------------------
DBSCAN（Density-Based Spatial Clustering of Applications with Noise，具有噪声的基于密度的聚类方法）是一种基于密度的空间聚类算法。 该算法将具有足够密度的区域划分为簇，并在具有噪声的空间数据库中发现任意形状的簇，它将簇定义为密度相连的点的最大集合。



*class sklearn.cluster.DBSCAN(eps=0.5, *, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)*

[https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)



DBSCAN算法是一种基于密度的聚类算法，其优势是聚类的时候不需要预先指定簇的个数，其会根据设定的参数确定最终的聚类个数及形态，DBSCAN算法将数据点分为三类：

 - 核心点：在半径Eps内含有超过MinPts数目的点。
 - 边界点：在半径Eps内点的数量小于MinPts,但是落在核心点的邻域内的点。
 - 噪音点：既不是核心点也不是边界点的点。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201118124352769.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTIzMzE1Nw==,size_16,color_FFFFFF,t_70#pic_center)


#### 聚类主要步骤

随机取距离半径eps=2，最小数量minpts=3为例

 1. 对每个点计算其邻域eps=2内点的集合
 2. 集合内点的个数超过minpts=3的点为核心点，查看剩余点是否在核心点的领域内，若在则为边界点，否则为噪声点
 3. 将距离不超过eps=2的点相互连接，构成一个簇，注意核心点领域内的点也要被加入到这个簇中

<br>

#### 两个参数的选择

从聚类的过程看，选取的距离半径和簇内最小数量两个参数对结果影响很大，主要有两个方法可以尝试下：

 1. 通过找到所有点中距离最大的进行分割确定距离半径
 2. 距离半径和最小数量组合寻优