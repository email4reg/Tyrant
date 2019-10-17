# python 3.7.2 @hehaoran
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import random
from scipy.optimize import leastsq, fmin, minimize
from scipy.stats import percentileofscore, chi2, t, f, norm, genpareto
from scipy.special import gamma



# *******************Chapter 7********************
# Exercise 1
# 导入数据
da1 = pd.read_excel('C:/Users/hehaoran/Desktop/Projects/pydata/var_chapter7_data.xlsx')
da1 = da1.set_index('Date')

# 计算收益率
da1['SP500 Return'] = np.log(da1.SP500_Close / da1.SP500_Close.shift(1))
da1['US10FT Return'] = np.log(da1.US10FT_Close / da1.US10FT_Close.shift(1))

# 作图
fig,axes = plt.subplots(2,1,figsize=(10,10))

axes[0].plot(da1['SP500 Return'])
axes[1].plot(da1['US10FT Return'],color='black')
axes[0].set_ylabel('SP500 Log Return')
axes[1].set_ylabel('US10FT Log Return')
plt.show()

# Exercise 2
# Compute the unconditional covariance and the correlation for the two assets.
# 导入数据
da2 =da1.copy()
da2[['SP500 Return','US10FT Return']].cov() # 
da2[['SP500 Return','US10FT Return']].corr()

# Exercise 3
# Calculate the unconditional 1-day,1% VaR for a portfolio and individually,Use the normal distribution
# 数据
da3 = da1.copy()


# 1.分别计算VaR
var_SP500 = da3['SP500 Return'].var() # 计算SP500方差
var_US10FT = da3['US10FT Return'].var() # 计算US10FT方差
VaR_SP500 = - var_SP500 ** 0.5 * - norm.isf(0.01) * np.sqrt(0.5) # SP500 1% VaR
VaR_US10FT = - var_US10FT ** 0.5 * - norm.isf(0.01) * np.sqrt(0.5) # US10FT 1% VaR

# 2.计算portfolio VaR
cov_PF = da3[['SP500 Return','US10FT Return']].cov()
var_PF = 0.5 ** 2 * var_SP500 + 0.5 ** 2 * var_US10FT + 2 * 0.5 * 0.5 * cov_PF.values[1,0]
VaR_PF = - var_PF ** 0.5 * - norm.isf(0.01) # portfolio VaR

# Exercise 4
# Estimate NGARCH(1,1) and standardize each return

# 基于MLE构建NGARCH(1,1)
# step1:定义最大似然函数
def negative_log_likelihood(params,rt):
    omega, alpha, theta, beta = params
    sigma2 = np.ones(len(rt))
    sigma2[0] = np.var(rt)
    for i in range(1, len(rt)):
        sigma2[i] = omega + alpha * (rt[i - 1] - theta * sigma2[i - 1] ** 0.5) ** 2 + beta * sigma2[i - 1]
    log_likelihood = (np.log(2 * np.pi) + np.log(sigma2) + rt ** 2 / sigma2).sum()/2
    return log_likelihood

# step 2:参数估计结果
param1 = fmin(negative_log_likelihood, x0=[1.5e-6, 0.05, 1.25, 0.8], args=(
    da3['SP500 Return'][1:].values,),ftol=1e-7) # SP500 Return

param2 = fmin(negative_log_likelihood, x0=[5e-6, 0.03, 0.00, 0.97], args=(
    da3['US10FT Return'][1:].values,),ftol=1e-7) # US10FT Return

# step3
# 1.计算SP500的conditional Variance
r_SP500_data = da3['SP500 Return'][1:].copy()
sigma2_SP500 = np.ones(len(r_SP500_data))
sigma2_SP500[0] = np.var(r_SP500_data)
for i in range(1, len(r_SP500_data)):
    sigma2_SP500[i] = param1[0] + param1[1] * (r_SP500_data[i - 1] - param1[2] * sigma2_SP500[i - 1]
        ** 0.5) ** 2 + param1[3] * sigma2_SP500[i - 1]

da3['SP500 Variance'] = np.insert(sigma2_SP500,0,np.nan)

# 2.计算US10FT的conditional Variance
r_US10FT_data = da3['US10FT Return'][1:].copy()
sigma2_US10FT = np.ones(len(r_US10FT_data))
sigma2_US10FT[0] = np.var(r_US10FT_data)
for i in range(1, len(r_US10FT_data)):
    sigma2_US10FT[i] = param2[0] + param2[1] * (r_US10FT_data[i - 1] - param2[2] * sigma2_US10FT[i - 1]
        ** 0.5) ** 2 + param2[3] * sigma2_US10FT[i - 1]

da3['US10FT Variance'] = np.insert(sigma2_US10FT,0,np.nan)

# step 4:Standardize each return
da3['Standardized SP500 Return'] = da3['SP500 Return'] / da3['SP500 Variance'] ** 0.5
da3['Standardized US10FT Return'] = da3['US10FT Return'] / da3['US10FT Variance'] ** 0.5

# Exercise 5-6
# Use QMLE to estimate λ in the exponential smoother version of the dynamic conditional correlation (DCC) model
da5 = pd.read_excel(
    '/Users/hhr/Desktop/Projects/pydata/var_chapter7_data5.xlsx') # 基于NGARCH的标准化的收益率
da5 = da5.set_index('Date')

# 构建DCC-RM模型
# 定义最大似然函数
def negative_log_likelihood(p,rt1,rt2):
    """
    :param rt1,rt1:为标准化的收益率
    """
    nu = p
    n_sample = len(rt1)
    # 定义标准化收益的方差矩阵
    qt11 = np.ones(n_sample)
    qt12 = np.ones(n_sample)
    qt22 = np.ones(n_sample)
    # 初始值
    qt11[0] = 1
    qt12[0] = np.sum(rt1 * rt2) / n_sample
    qt22[0] = 1
    # 生成条件方差值
    for i in range(1,len(rt1)):
        qt11[i] = (1 - nu) * rt1[i - 1] ** 2 + nu * qt11[i - 1]
        qt12[i] = (1 - nu) * rt1[i - 1] * rt2[i - 1] + nu * qt12[i - 1]
        qt22[i] = (1 - nu) * rt2[i - 1] ** 2 + nu * qt22[i - 1]
    rho12 = qt12 / np.sqrt(qt11 * qt22)
    return 0.5 * np.sum(np.log(1 - rho12 ** 2) + (rt1 ** 2 + rt2 ** 2 - 2 * rho12 * rt1 * rt2) / (1 - rho12 ** 2))


# 估计DCC-RM模型参数
param1 = fmin(negative_log_likelihood, x0=[0.94], args=(
    da5['SP500_Return'].values, da5['US10FT_Return'].values), ftol=1e-7)

# 生成条件方差
def calc_con_corr_RM(rt1,rt2,nu):
    n_sample = len(rt1)
    # 定义标准化收益的方差矩阵
    qt11 = np.ones(n_sample)
    qt12 = np.ones(n_sample)
    qt22 = np.ones(n_sample)
    # 初始值
    qt11[0] = 1
    qt12[0] = np.sum(rt1 * rt2) / n_sample
    qt22[0] = 1
    for i in range(1, len(rt1)):
        qt11[i] = (1 - nu) * rt1[i - 1] ** 2 + nu * qt11[i - 1]
        qt12[i] = (1 - nu) * rt1[i - 1] * rt2[i - 1] + nu * qt12[i - 1]
        qt22[i] = (1 - nu) * rt2[i - 1] ** 2 + nu * qt22[i - 1]
    return qt12 / np.sqrt(qt11 * qt22)


# 调用函数
con_corr_t1 = calc_con_corr_RM(
    da5['SP500_Return'].values, da5['US10FT_Return'].values, param1)
# 转变成Series
con_corr_t1 = pd.Series(data=con_corr_t1, index=da5.index)
# 作图
plt.plot(con_corr_t1)
plt.xlabel('Return Date')
plt.ylabel('Conditional Correlation')
plt.show()

# 构建DCC-GARCH模型
# 定义最大似然函数
def negative_log_likelihood(params, rt1, rt2):
    """
    :param rt1,rt1:为标准化的收益率
    """
    alpha,beta = params
    n_sample = len(rt1)
    # 定义标准化收益的方差矩阵
    qt11 = np.ones(n_sample)
    qt12 = np.ones(n_sample)
    qt22 = np.ones(n_sample)
    # 初始值
    qt11[0] = 1
    qt12[0] = np.sum(rt1 * rt2) / n_sample
    qt22[0] = 1
    uncon_rho12 = np.sum(rt1 * rt2) / n_sample
    for i in range(1, len(rt1)):
        qt11[i] = 1 + alpha * (rt1[i - 1] ** 2 - 1) + beta * (qt11[i - 1] - 1)
        qt12[i] = uncon_rho12 + alpha * (rt1[i - 1] * rt2[i - 1] - uncon_rho12) + beta * (qt12[i - 1] - uncon_rho12)
        qt22[i] = 1 + alpha * (rt2[i - 1] ** 2 - 1) + beta * (qt22[i - 1] - 1)
    rho12 = qt12 / np.sqrt(qt11 * qt22)
    return 0.5 * np.sum(np.log(1 - rho12 ** 2) + (rt1 ** 2 + rt2 ** 2 - 2 * rho12 * rt1 * rt2) / (1 - rho12 ** 2))


# 估计DCC-GARCH模型参数
param2 = fmin(negative_log_likelihood, x0=[0.05,0.90], args=(
    da5['SP500_Return'].values, da5['US10FT_Return'].values), ftol=1e-7)


# 生成条件方差
def calc_con_corr_GARCH(rt1,rt2,alpha,beta):
    n_sample = len(rt1)
    # 定义标准化收益的方差矩阵
    qt11 = np.ones(n_sample)
    qt12 = np.ones(n_sample)
    qt22 = np.ones(n_sample)
    # 初始值
    qt11[0] = 1
    qt12[0] = np.sum(rt1 * rt2) / n_sample
    qt22[0] = 1
    uncon_rho12 = np.sum(rt1 * rt2) / n_sample
    # 生成条件方差值
    for i in range(1, len(rt1)):
        qt11[i] = 1 + alpha * (rt1[i - 1] ** 2 - 1) + beta * (qt11[i - 1] - 1)
        qt12[i] = uncon_rho12 + alpha * (rt1[i - 1] * rt2[i - 1] - uncon_rho12) + beta * (qt12[i - 1] - uncon_rho12)
        qt22[i] = 1 + alpha * (rt2[i - 1] ** 2 - 1) + beta * (qt22[i - 1] - 1)
    return qt12 / np.sqrt(qt11 * qt22)


# 调用函数
con_corr_t2 = calc_con_corr_GARCH(
    da5['SP500_Return'].values, da5['US10FT_Return'].values,param2[0],param2[1])
# 转变成Series
con_corr_t2 = pd.Series(data=con_corr_t2,index=da5.index)
# 作图
plt.plot(con_corr_t2)
plt.show()
# 合并两张图
with plt.style.context('seaborn-paper'):
    fig,axes = plt.subplots(2,1,figsize=(8,10))
    axes[0].plot(con_corr_t1, color='b')
    axes[0].set_xlabel('Return Date')
    axes[0].set_ylabel('Conditional Correlation')
    axes[1].plot(con_corr_t2, color='black')
    axes[1].set_xlabel('Return Date')
    axes[1].set_ylabel('Conditional Correlation')
plt.show()

# 计算1-day, 1% Value-at-Risk(见p154-155)





