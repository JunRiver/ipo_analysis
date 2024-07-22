# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     :2024/4/24 20:40
# @AUTHOR   :Jun

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

# https://www.163.com/dy/article/IFADOA920531D9VR.html

data = pd.read_excel(r'../resource/data_sel.xlsx',sheet_name='主成分分析数据集（规模164）')
# 按标签选择 loc[]   按位置选择 iloc[]
# 删除空值  dropna()
colums_names = ['证券代码'
    ,'新股发行数量（股）'
    ,'报告期每股净资产（元）'
    ,'报告期总资产（元）'
    ,'报告期营业收入（元）'
    ,'报告期每股收益EPS'
    ,'报告期净资产收益率（%）'
    ,'总资产净利率（%）'
    ,'资产负债率（%）'
    ,'流动比率（%）'
    ,'存货周转率（次）'
    ,'应收账款周转率（次）'
    #,'总资产周转率（次）'
    ,'三年营业收入复合增长率（%）'
    ,'研发支出总额占营业收入比例(%)'
    ,'首发时所属行业市盈率（倍）']
data_pro = data.loc[:129,colums_names].dropna()
# 预处理
#data_pro['新股发行数量（股）'] = data_pro['新股发行数量（股）'].map(lambda x: math.log(x))
#data_pro['报告期总资产（元）'] = data_pro['报告期总资产（元）'].map(lambda x: math.log(x))
#data_pro['报告期营业收入（元）'] = data_pro['报告期营业收入（元）'].map(lambda x: math.log(x))

# 相关性矩阵
data_corr = data_pro.corr(numeric_only=True)
#correlation_matrix = sm.graphics.plot_corr(data_pro.corr(numeric_only=True),)

# kmo和bartlett检验
## https://blog.csdn.net/kmmel/article/details/130144052
data_arr =data_pro.iloc[:,1:].values
kmo_all, kmo_model = calculate_kmo(data_arr)
print("KMO值:", kmo_model)
chi_square_value, p_value = calculate_bartlett_sphericity(data_arr)
print("Bartlett球形度检验的卡方值：", chi_square_value)
print("Bartlett球形度检验的P值:", p_value)

# 主成分分析
## https://blog.csdn.net/qq_41081716/article/details/103332472
fa = FactorAnalyzer(rotation=None, n_factors=15, method='principal')
fa.fit(data_arr)
fa_15_sd = fa.get_factor_variance()
fa_15_df = pd.DataFrame({'特征值': fa_15_sd[0], '方差贡献率': fa_15_sd[1], '方差累计贡献率': fa_15_sd[2]})
print("\n 初始特征值")
print("\n",fa_15_df)

# 公因子数设为7个，重新拟合
fa_7 = FactorAnalyzer(rotation=None, n_factors=7, method='principal')
fa_7.fit(data_arr)
fa_co_df=pd.DataFrame({'变量名': np.array(colums_names[1:]), '初始公因子方差': fa.get_communalities(), '提取后公因子方差（公因子提取度）': fa_7.get_communalities()})
fa_7_sd = fa_7.get_factor_variance()
fa_7_df = pd.DataFrame({'特征值': fa_7_sd[0], '方差贡献率': fa_7_sd[1], '方差累计贡献率': fa_7_sd[2]})
print("\n 提取载荷平方和")
print("\n",fa_7_df)
print("\n",fa_co_df)
print("\n因子载荷矩阵:\n", fa_7.loadings_)

# 使用最大方差法旋转因子载荷矩阵
fa_7_rotate = FactorAnalyzer(rotation = 'varimax', n_factors = 7, method = 'principal')
fa_7_rotate.fit(data_arr)
fa_7_rotate_sd = fa_7.get_factor_variance()
fa_7_rotate_df = pd.DataFrame({'特征值': fa_7_rotate_sd[0], '方差贡献率': fa_7_rotate_sd[1], '方差累计贡献率': fa_7_rotate_sd[2]})
print("\n 旋转载荷平方和")
print("\n",fa_7_rotate_df)
rotate_loadings = pd.DataFrame(data = fa_7_rotate.loadings_, columns = ['factor1', 'factor2', 'factor3', 'factor4', 'factor5','factor6', 'factor7'],index=np.array(colums_names[1:]))
print("\n旋转后的因子载荷矩阵:\n",rotate_loadings)

# 因子得分（回归方法）（系数矩阵的逆乘以因子载荷矩阵）
X1 = np.mat(data_corr)
X1 = nlg.inv(X1)

# B=(R-1)*A  15*5
factor_score = np.dot(X1, fa_7_rotate.loadings_)
factor_score = pd.DataFrame(factor_score)
factor_score.columns = ['factor1', 'factor2', 'factor3', 'factor4', 'factor5','factor6', 'factor7']
factor_score.index = data_corr.columns
print("\n因子得分：\n", factor_score)


# 聚类分析
## https://blog.csdn.net/qq_38614074/article/details/137456095
# 归一化处理
std_scaler = StandardScaler()
std_scaler.fit(data_arr)
data_arr_std = std_scaler.transform(data_arr)
data_cluster = np.dot(data_arr_std, factor_score)
df_cluster = pd.DataFrame(data = data_cluster, columns = ['factor1', 'factor2', 'factor3', 'factor4', 'factor5', 'factor6', 'factor7'],index=np.array(data_pro.iloc[:,0]))
#std_scaler = StandardScaler()
#std_scaler.fit(data_cluster)
#data_cluster = std_scaler.transform(data_cluster)
# 碎石图，K的选取
K = range(2,20)
meandistortions = []
silhouettes=[]
calinski_harabasz =[]
daviesbouldin = []
for k in K:
    kmeans = KMeans(n_clusters=k,random_state =0)
    #kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=3000, tol=0.00001)
    kmeans.fit(data_cluster)
    meandistortions.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(data_cluster,kmeans.labels_))
    calinski_harabasz.append(metrics.calinski_harabasz_score(data_cluster, kmeans.labels_))
    daviesbouldin.append(metrics.davies_bouldin_score(data_cluster, kmeans.labels_))
plt.plot(K,meandistortions,'bx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('Selecting K-means clusters')
plt.show()
plt.plot(K,silhouettes,'bx-')
plt.xlabel('k')
plt.ylabel('silhouettes')
plt.show()
plt.plot(K,calinski_harabasz,'bx-')
plt.xlabel('k')
plt.ylabel('calinski_harabasz_score')
plt.show()
plt.plot(K,daviesbouldin,'bx-')
plt.xlabel('k')
plt.ylabel('davies_bouldin_score')
plt.show()
K = 8
#kmeans = KMeans(n_clusters = K, init = 'k-means++', max_iter=3000, tol=0.00001)
kmeans = KMeans(n_clusters = K, random_state =0)
y_cluster = kmeans.fit_predict(data_cluster)
score_cluster_centers = MinMaxScaler(feature_range=(0,5)).fit_transform(kmeans.cluster_centers_)
# print(kmeans.cluster_centers_)
# print(score_cluster_centers)
# 绘制雷达图
dim_num = 7
radar_labels = ['factor1', 'factor2', 'factor3', 'factor4', 'factor5','factor6', 'factor7']
angles = [n/dim_num * 2 * np.pi for n in range(dim_num)]
angles += angles[:1]
ax = plt.subplot(111,polar = True)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1],radar_labels)
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
        df.to_excel(writer, sheet_name= 'group'+str(i))

'''
dim_num = 7
radians = np.linspace(0,2*np.pi,dim_num,endpoint=False)
radians = np.concatenate((radians, [radians[0]]))
score_a = np.concatenate((score_cluster_centers[0], [score_cluster_centers[0][0]]))
score_b = np.concatenate((score_cluster_centers[1], [score_cluster_centers[1][0]]))
score_c = np.concatenate((score_cluster_centers[2], [score_cluster_centers[2][0]]))
score_d = np.concatenate((score_cluster_centers[3], [score_cluster_centers[3][0]]))
plt.polar(radians,score_a, score_b, score_c,score_d)
radar_labels = ['factor1', 'factor2', 'factor3', 'factor4', 'factor5','factor6', 'factor7']
radar_labels = np.concatenate((radar_labels,[radar_labels[0]]))
angles = radians *180/np.pi
plt.thetagrids(angles,labels=radar_labels)
tables = plt.fill(radians,score_a,score_b,score_c,score_d,alpha=0.25)
plt.show()
'''

'''

# F=XB  27*15 15*5=  27 5
fa_t_score = np.dot(np.mat(data_arr), np.mat(factor_score))
print("\n因子得分：\n", pd.DataFrame(fa_t_score))


# 回归模型建立：逐步回归分析
## https://blog.csdn.net/qq_41780234/article/details/135674535
# 回归分析需要因变量，ipo定价？
# 初始化模型，包含常数项
X = np.dot(np.mat(data_arr),fa_7_rotate.loadings_)
X = sm.add_constant(factor_score)
# 各个股票的ipo定价
y= data.loc[:129,['首发价格（元）']].dropna()
model = sm.OLS(y, X).fit()

# 打印初始模型的摘要
print("初始模型:")
print(model.summary())

# 逐步回归分析
while True:
    # 获取当前模型的最大p值
    max_pvalue = model.pvalues[1:].idxmax()
    max_pvalue_value = model.pvalues[1:].max()

    # 如果最大p值大于阈值（例如，0.05），则去除该特征
    if max_pvalue_value > 0.05 and max_pvalue != 'const':
        X = X.drop(max_pvalue, axis=1)
        model = sm.OLS(y, X).fit()
        print(f"去除特征 '{max_pvalue}', 当前模型:")
        print(model.summary())
    else:
        break

# 打印最终逐步回归分析的结果
print("最终模型:")
print(model.summary())
'''

