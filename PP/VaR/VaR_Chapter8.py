# python 3.7.3 @hehaoran
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import random
from scipy.optimize import leastsq, fmin, minimize
from scipy.stats import percentileofscore, chi2, t, f, norm, genpareto
from scipy.special import gamma

# Exercise 1 and 3
# Construct the 10-day,1% VaR using FHS,RiskMetrics and NGARCH(1,1) (MC simulations)
# 导入数据
da1 = pd.read_excel(
    '/Users/hhr/Desktop/Projects/pydata/var_chapter8_data.xlsx')
da1 = da1.set_index('Date')

# 计算收益率
da1['Log Return'] = np.log(da1.Close / da1.Close.shift(1))

# ***************构建FHS-NGARCH-MC****************
# step1: 构建NGARCH并标准化收益率
# 模型估计
def negative_log_likelihood(params,rt):
    omega, alpha, theta, beta = params
    sigma2 = np.ones(len(rt))
    sigma2[0] = np.var(rt)
    for i in range(1, len(rt)):
        sigma2[i] = omega + alpha * (rt[i - 1] - theta * sigma2[i - 1] ** 0.5) ** 2 + beta * sigma2[i - 1]
    log_likelihood = (np.log(2 * np.pi) + np.log(sigma2) + rt ** 2 / sigma2).sum()/2
    return log_likelihood


param1 = fmin(negative_log_likelihood, x0=[1.5e-6, 0.05, 1.25, 0.8], args=(
    da1['Log Return'][1:].values,),ftol=1e-7) # 参数估计结果

# 计算conditional Variance
r_SP500_data = da1['Log Return'][1:].copy()
sigma2_SP500 = np.ones(len(r_SP500_data))
sigma2_SP500[0] = np.var(r_SP500_data)
for i in range(1, len(r_SP500_data)):
    sigma2_SP500[i] = param1[0] + param1[1] * (r_SP500_data[i - 1] - param1[2] * sigma2_SP500[i - 1]
                                               ** 0.5) ** 2 + param1[3] * sigma2_SP500[i - 1]

da1['SP500 Variance(NGARCH)'] = np.insert(sigma2_SP500, 0, np.nan)


# 标准化收益率
da1['Standardized Return'] = da1['Log Return'] / \
    da1['SP500 Variance(NGARCH)'] ** 0.5

# step2: 随机生成 MC,MC=10000,k-day为10;生成uniform random(m),m为样本数,即生成(10000,10)的1～m的随机数.
np.random.seed(999)  # 设置随机种子
rnd_HFS = np.random.randint(1, len(da1), size=(10000, 10))

# step3: 根据NGARCH模型计算k-day的收益率和条件方差
k_return = []
k_sigma2 = []
for k in range(10):
        std_r = np.array([da1['Standardized Return'][index]
                          for index in rnd_HFS[:, k]])
        if k == 0:
            sigma2_last = da1['SP500 Variance(NGARCH)'][-1]  # 最后一日标准方差
            k_return.append(std_r * sigma2_last ** 0.5)
            k_sigma2.append(param1[0] + param1[1] * (k_return[-1] - param1[2]
                                                     * sigma2_last ** 0.5) ** 2 + param1[3] * sigma2_last)
        else:
            k_return.append(std_r * k_sigma2[-1] ** 0.5)
            k_sigma2.append(param1[0] + param1[1] * (k_return[-1] - param1[2]
                                                     * k_sigma2[-1] ** 0.5) ** 2 + param1[3] * k_sigma2[-1])


k_returns = np.array(k_return).T  # 合并结果

# STEP4: 根据MC=10000个结果,运用percentile找出1% VaR
var_FHS = np.percentile(k_returns.sum(axis=1), 0.01 *
                        100, interpolation='linear')

# STEP 5: 计算1% ES
es_FHS = np.sort(k_returns.sum(axis=1))[:100].mean()

# *******************假定k-day是正态分布,构建NGARCH-MC*******************
# STEP1: 构建NGARCH并标准化收益率(同上)
# STEP2: 随机生成 MC,MC=10000,k-day为10;生成normal random(0,1).
np.random.seed(999)  # 设置随机种子
rnd_N = np.random.randn(10000, 10)

# STEP3: 根据NGARCH模型计算k-day的收益率和条件方差
k_return = []
k_sigma2 = []
for k in range(10):
        rnd_r = rnd_N[:, k]
        if k == 0:
            sigma2_last = da1['SP500 Variance(NGARCH)'][-1]  # 最后一日标准方差
            k_return.append(rnd_r * sigma2_last ** 0.5)
            k_sigma2.append(param1[0] + param1[1] * (k_return[-1] - param1[2]
                                                     * sigma2_last ** 0.5) ** 2 + param1[3] * sigma2_last)
        else:
            k_return.append(rnd_r * k_sigma2[-1] ** 0.5)
            k_sigma2.append(param1[0] + param1[1] * (k_return[-1] - param1[2]
                                                     * k_sigma2[-1] ** 0.5) ** 2 + param1[3] * k_sigma2[-1])

k_returns = np.array(k_return).T  # 合并结果

# STEP4: 根据MC=10000个结果,运用percentile找出1% VaR
var_MC = np.percentile(k_returns.sum(axis=1), 0.01 *
                       100, interpolation='linear')

# STEP 5: 计算1% ES
es_MC = np.sort(k_returns.sum(axis=1))[:100].mean()

# ************RM scaling the daily VaRs (although it is incorrect)***********
# 计算conditional Variance
nu = 0.94
r_SP500_data = da1['Log Return'][1:].copy()
sigma2_SP500 = np.ones(len(r_SP500_data))
sigma2_SP500[0] = np.var(r_SP500_data)
for i in range(1, len(r_SP500_data)):
    sigma2_SP500[i] = (1 - nu) * r_SP500_data[i - 1] ** 2 + \
        nu * sigma2_SP500[i - 1]

da1['SP500 Variance(RM)'] = np.insert(sigma2_SP500, 0, np.nan)

# 计算1% VaR和ES
sigma2_last = da1['SP500 Variance(RM)'][-1]  # RM最后一日标准方差
var_RM = - np.sqrt(10) * sigma2_last ** 0.5 * norm.isf(0.01)

es_RM = np.sqrt(10) * sigma2_last ** 0.5 * norm.pdf(norm.isf(0.01)) / 0.01

# Exercise 2(略)

# Exercise 4
# the correlation forecasts estumated in Chapter 7 using 100000 MC

# Part 1: Constant Correlations using Multivariate Monte Carlo Simulation
# 导入数据(标准化的收益率和条件方差)
da4 = pd.read_excel(
    '/Users/hhr/Desktop/Projects/pydata/var_chapter8_data4.xlsx')
da4 = da4.set_index('Date')

# STEP1: draw a vector of uncorrelated random normal variables
np.random.seed(999)
uncorr_z = np.random.multivariate_normal(mean=(0,0),cov=[[1,0],[0,1]],size=(10000,10))

# STEP2: use the matrix square root to correlate the random variables
# calculate unconditional variation
n_sample = len(da4)
# uncon_rho12 = np.sum(da4['Standardized Return SP500'].values[1:] * da4['Standardized Return US10FT'].values[1:]) / n_sample
uncon_rho12 = da4['Log Return SP500'].corr(da4['Log Return US10FT']) # 与上面的值不一样

# calculate matrix square root
gamma_square = np.array([[1, 0], [uncon_rho12, np.sqrt(1 - uncon_rho12 ** 2)]])

# correlate the random variables
corr_z = []
for i in range(10):
    for j in range(10000):
        corr_z.append(np.dot(gamma_square, uncorr_z[j, i, :]))
corr_z = np.array(corr_z)

# STEP3: update the variances
con_sigma2_SP500_last = da4['Conditional Variance SP500'][-1]  # SP500最后一日方差
con_sigma2_US10FT_last = da4['Conditional Variance US10FT'][-1]  # US10FT最后一日方差

return_SP500_last = da4['Log Return SP500'][-1]  # SP500最后一日收益率
return_US10FT_last = da4['Log Return US10FT'][-1]  # US10FT最后一日收益率

# 根据前面RM或者GARCH模型计算day+1的方差(t+1),这里采用NGARCH(1,1)
param2 = fmin(negative_log_likelihood, x0=[1.5e-6, 0.05, 1.25, 0.8], args=(
    da4['Log Return US10FT'][1:].values,), ftol=1e-7)  # 参数估计结果

def getNGARCH(rt,sigma2,*params):
    return params[0] + params[1] * (rt - params[2] * sigma2 ** 0.5) ** 2 +params[3] * sigma2

sigma2_SP500_d1 = getNGARCH(return_SP500_last, con_sigma2_SP500_last, *param1)
sigma2_US10FT_d1 = getNGARCH(return_US10FT_last, con_sigma2_US10FT_last, *param2)

# STEP4: update the return
# update the variances
return_k = []
sigma2_k = [np.array([sigma2_SP500_d1, sigma2_US10FT_d1])]
for k in range(10):
    if k == 0:
        return_k.append(np.dot(corr_z[:10000],np.diag(sigma2_k[-1]))) # t+1收益率;shape:10000*2
        sigma2_k.append(np.array([getNGARCH(return_k[-1][:, 0], sigma2_k[-1][0], *param1), 
                                getNGARCH(return_k[-1][:, 1], sigma2_k[-1][1],*param2)])) # t+2方差;shape=2*10000
    else:
        # Loop through STEP3,STEP4
        return_k.append(corr_z[10000 * k: 10000 * (k + 1)] * sigma2_k[-1].T) # t+k收益率；shape=10000*2
        sigma2_k.append(np.array([getNGARCH(return_k[-1][:, 0], sigma2_k[-1][:,0][0], *param1),
                                getNGARCH(return_k[-1][:, 1], sigma2_k[-1][:,1][1], *param2)]))  # t+2方差;shape=2*10000

# STEP5 : 根据MC=10000个结果,运用percentile找出1% VaR(书中没有计算,不确定是否正确)
# calculate 1% VaR and ES,portfolio weights = 0.5
return_PF = np.array([np.sum(return_k[i], axis=1) * 0.5 for i in range(10)])
var_cons_corr = - np.percentile(return_PF.sum(axis=0), 0.01 * 100, interpolation='linear')

# Part 2: Dynamic Correlations using Multivariate Monte Carlo Simulation







