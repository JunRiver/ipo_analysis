# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     :2024/4/24 20:40
# @AUTHOR   :Jun

import pandas as pd
import numpy as np
import numpy.linalg as nlg
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

# https://www.163.com/dy/article/IFADOA920531D9VR.html

data = pd.read_excel(r'../resource/data.xlsx',sheet_name='万得')
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
    ,'总资产周转率（次）'
    ,'三年营业收入复合增长率（%）'
    ,'研发支出总额占营业收入比例(%)'
    ,'首发时所属行业市盈率（倍）']
data_pro = data.loc[:,colums_names].dropna()

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
print("\n",fa_15_df)

# 公因子数设为5个，重新拟合
fa_5 = FactorAnalyzer(rotation=None, n_factors=5, method='principal')
fa_5.fit(data_arr)
print("\n公因子提取度:\n", fa_5.get_communalities())
print("\n因子载荷矩阵:\n", fa_5.loadings_)

# 使用最大方差法旋转因子载荷矩阵
fa_5_rotate = FactorAnalyzer(rotation = 'varimax', n_factors = 5, method = 'principal')
fa_5_rotate.fit(data_arr)
print("\n旋转后的因子载荷矩阵:\n",fa_5_rotate.loadings_)

# 因子得分（回归方法）（系数矩阵的逆乘以因子载荷矩阵）
X1 = np.mat(data_corr)
X1 = nlg.inv(X1)

# B=(R-1)*A  15*5
factor_score = np.dot(X1, fa_5_rotate.loadings_)
factor_score = pd.DataFrame(factor_score)
factor_score.columns = ['factor1', 'factor2', 'factor3', 'factor4', 'factor5']
factor_score.index = data_corr.columns
print("\n因子得分：\n", factor_score)

# F=XB  27*15 15*5=  27 5
fa_t_score = np.dot(np.mat(data_arr), np.mat(factor_score))
print("\n应试者的五个因子得分：\n", pd.DataFrame(fa_t_score))

# 回归模型建立：逐步回归分析
## https://blog.csdn.net/qq_41780234/article/details/135674535
# 回归分析需要因变量，ipo定价？
# 初始化模型，包含常数项
X = sm.add_constant(factor_score)
# 各个股票的ipo定价
y= X
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

