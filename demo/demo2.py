# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     :2024/6/17 0:27
# @AUTHOR   :Jun
# 聚类分析脚本

import pandas as pd
import numpy as np
import numpy.linalg as nlg
import math
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



columns = ['factor1',
           'factor2',
           'factor3',
           'factor4',
           'factor5',
           'factor6',
           'factor7'
           ]
dim_num = 7
K = 8

factor = pd.read_excel(r'../resource/data_0618.xlsx',sheet_name='消除量纲_因子分析_因子数7')
factor_pro = factor.dropna()
factor_arr = factor_pro.iloc[:,1:].values
data = pd.read_excel(r'../resource/data_0618.xlsx',sheet_name='data')
data_pro = data.loc[:,factor_pro.iloc[:,0].tolist()].dropna()
data_arr = data_pro.values
#test = pd.read_excel(r'../resource/data_0618.xlsx',sheet_name='data_r')
#test_pro = test.loc[130:,factor_pro.iloc[:,0].tolist()].dropna()
#test_arr = test_pro.values
# 聚类分析
## https://blog.csdn.net/qq_38614074/article/details/137456095
data_cluster = np.dot(data_arr, factor_arr)
df_cluster = pd.DataFrame(data = data_cluster, columns = columns ,index=np.array(data.iloc[:,0]))
# 碎石图，K的选取
I = range(2,20)
meandistortions = []
silhouettes=[]
calinski_harabasz =[]
daviesbouldin = []
for i in I:
    kmeans = KMeans(n_clusters=i,random_state=0)
    #kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=3000, tol=0.00001)
    kmeans.fit(data_cluster)
    meandistortions.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(data_cluster,kmeans.labels_))
    calinski_harabasz.append(metrics.calinski_harabasz_score(data_cluster, kmeans.labels_))
    daviesbouldin.append(metrics.davies_bouldin_score(data_cluster, kmeans.labels_))
plt.plot(I,meandistortions,'bx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('Selecting K-means clusters')
plt.show()
plt.plot(I,silhouettes,'bx-')
plt.xlabel('k')
plt.ylabel('silhouettes')
plt.show()
plt.plot(I,calinski_harabasz,'bx-')
plt.xlabel('k')
plt.ylabel('calinski_harabasz_score')
plt.show()
plt.plot(I,daviesbouldin,'bx-')
plt.xlabel('k')
plt.ylabel('davies_bouldin_score')
plt.show()

#kmeans = KMeans(n_clusters = K, init = 'k-means++', max_iter=3000, tol=0.00001)
kmeans = KMeans(n_clusters = K, random_state =0)
y_cluster = kmeans.fit_predict(data_cluster)
score_cluster_centers = MinMaxScaler(feature_range=(1,5)).fit_transform(kmeans.cluster_centers_)
# 测试集数据
#test_cluster = np.dot(test_arr, factor_arr)
#df_test = pd.DataFrame(data = test_cluster, columns = columns ,index=np.array(test.iloc[130:,0]))
#y_test = kmeans.predict(test_cluster)
# 绘制雷达图
angles = [n/dim_num * 2 * np.pi for n in range(dim_num)]
angles += angles[:1]
ax = plt.subplot(111,polar = True)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1],columns)
ax.set_rlabel_position(0)
for i in range(len(score_cluster_centers)):
    score = score_cluster_centers[i].tolist()
    score += score[:1]
    ax.plot(angles,score,linewidth=1,linestyle='solid')
    ax.fill(angles,score,alpha = 0.1)
plt.show()
# 各类统计数据
ar, num = np.unique(y_cluster, return_counts = True)
plt.grid(ls="--",alpha=0.5)
plt.bar(ar,num)
plt.show()
print("\n聚类分析SSE(误差平方和)：\n", kmeans.inertia_)
print("\n聚类分析轮廓系数：\n", silhouette_score(data_cluster,kmeans.labels_))
print("\n聚类分析样本轮廓系数：\n", silhouette_samples(data_cluster,kmeans.labels_))
print("\n聚类分析CH：\n", metrics.calinski_harabasz_score(data_cluster, kmeans.labels_))
print("\n聚类分析DB：\n", metrics.davies_bouldin_score(data_cluster, kmeans.labels_))
# 聚类结果输出
filepath = r'D:\file_jun\管培大作业\oc\聚类\分类结果表_py输出.xlsx'
with pd.ExcelWriter(filepath) as writer:
    for i in range(K):
        df = df_cluster.iloc[np.where(kmeans.labels_==i)[0],:]
        #df = df_test.iloc[np.where(y_test == i)[0], :]
        df.to_excel(writer, sheet_name= 'group'+str(i))
