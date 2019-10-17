# python 3.7.2 @hehaoran
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
from scipy.optimize import leastsq,fmin,minimize
from datetime import datetime

from dateutil.parser import parse
import tushare as ts
from prettytable import PrettyTable

# *******************Chapter 4********************
# 基于MLE构建GARCH(1,1)
# 导入数据
data4 = pd.read_excel('/Users/hhr/Desktop/Projects/pydata/var_chapter4_data.xlsx')

#计算log return
data4['log return'] = np.log(data4.Close / data4.Close.shift(1))

# 删除NaN值
data4 = data4.dropna()
data4 = data4.reset_index(drop=True)

# 设置initial values
, alpha, beta = 5e-6, 0.1, 0.85

# 形式1: 定义最大似然函数
def negative_log_likelihood(params,rt):
    omega,alpha,beta = params
    sigma2 = np.ones(len(rt))
    sigma2[0] = np.var(rt)
    for i in range(1,len(rt)):
        sigma2[i] = omega + alpha * rt[i - 1] ** 2 + beta * sigma2[i - 1]
    log_likelihood = (np.log(2 * np.pi) + np.log(sigma2) + rt ** 2/sigma2).sum()/2
    return log_likelihood

# 形式2: 定义最大似然函数omega = var(rt) * (1 - alpha - beta)
def negative_log_likelihood(params, rt):
    alpha, beta = params
    sigma2 = np.ones(len(rt))
    sigma2[0] = np.var(rt)
    for i in range(1, len(rt)):
        sigma2[i] = np.var(rt) * (1 - alpha - beta) + alpha * rt[i - 1] ** 2 + beta * sigma2[i - 1]
    log_likelihood = (np.log(2 * np.pi) + np.log(sigma2) + rt ** 2/sigma2).sum()/2
    return log_likelihood


# 参数估计结果
result = fmin(negative_log_likelihood, x0=[omega,alpha, beta], args=(
    data4['log return'].values,), ftol=1e-7)
persistence = result[1] + result[2]

# 基于MLE构建NGARCH(1,1)
# 设置initial values
omega, alpha, theta, beta = 5e-6, 0.07, 0.5, 0.85

# 定义最大似然函数
def negative_log_likelihood(params,rt):
    omega, alpha, theta, beta = params
    sigma2 = np.ones(len(rt))
    sigma2[0] = np.var(rt)
    for i in range(1, len(rt)):
        sigma2[i] = omega + alpha * (rt[i - 1] - theta * sigma2[i - 1] ** 0.5) ** 2 + beta * sigma2[i - 1]
    log_likelihood = (np.log(2 * np.pi) + np.log(sigma2) + rt ** 2 / sigma2).sum()/2
    return log_likelihood


# 参数估计结果
result = fmin(negative_log_likelihood, x0=[omega, alpha, theta, beta], args=(
    data4['log return'].values,),ftol=1e-7)
persistence = result[1] * (1 + result[2] ** 2) + result[3]

# 计算LR test
LR = 2 * (7915.17 - 7848.74) #分别为两个模型的最大似然值

# autocorrelations for lag 1 through 100
data4['log return2'] = data4['log return'].map(lambda x: x**2)

# 计算sigma2
n = len(data4) # 样本数
sigma2 = np.ones(n)
sigma2[0] = np.var(data4['log return'].values)
for i in range(1, n):
    sigma2[i] = result[0] + result[1] * (
        data4['log return'][i - 1] - result[2] * sigma2[i - 1] ** 0.5) ** 2 + result[3] * sigma2[i - 1]
data4['sigma2'] = sigma2

data4['r2/s2'] = data4['log return2'] / data4['sigma2']
# 计算自相关系数
data4 = data4.set_index('Date')
autocorr_r2s2_result = {}
for i in range(1, 101):
    autocorr_r2s2_result[i] = get_auto_corr(data4['r2/s2'].values, i)

# 作图4.2
fig,ax = plt.subplots(figsize=(10,8))

autocorr_r2s2_result = pd.Series(autocorr_r2s2_result)
autocorr_result = pd.concat([autocorr_r2_result, autocorr_r2s2_result],axis=1)

autocorr_result.rename({0: 'Autocorrelation of squared returns',
                        1: 'Autocorrelation of squared shocks(NGARCH)'},axis=1,inplace=True)

autocorr_result.plot()
plt.axhline(y=-0.05,linestyle='--',color='black')
plt.axhline(y=0.05, linestyle='--', color='black')
plt.show()

# 基于MLE和VIX指数(explanatory variable),构建LE-GARCH(1,1)
# 设置initial values
omega, alpha, theta, beta, gamma = 5e-6,0.04,2,0.5,0.07

# 定义最大似然函数
def negative_log_likelihood(params, rt, VIX):
    omega, alpha, theta, beta ,gamma= params
    sigma2 = np.ones(len(rt))
    sigma2[0] = np.var(rt)
    for i in range(1, len(rt)):
        sigma2[i] = omega + alpha * (
            rt[i - 1] - theta * sigma2[i - 1] ** 0.5) ** 2 + beta * sigma2[i - 1] + (gamma * VIX[i - 1] ** 2) / 252
    log_likelihood = (np.log(2 * np.pi) + np.log(sigma2) + rt ** 2 / sigma2).sum()/2
    return log_likelihood

# 参数估计结果
result = fmin(negative_log_likelihood, x0=[omega, alpha, theta, beta, gamma], args=(
    data4['log return'].values,data4['VIX'].values), ftol=1e-7)
persistence = result[1] * (1 + result[2] ** 2) + result[3]