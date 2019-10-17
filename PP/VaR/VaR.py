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

# ******************本模块包含以下内容**************
# 一、实现RiskMatrics(RM)和GARCH(NGARCH)
# 二、实现DCC-RM、DCC-GARCH
# 三、对未来k-day，运用Monte Carlo Simulation，实现1，2
# 四、计算VaR和ES(包括一个资产和投资组合)


# GARCH模型


def negative_log_likelihood_GARCH(params,rt):
    """定义GARCH模型最大似然函数
    """
    omega,alpha,beta = params
    sigma2 = np.ones(len(rt))
    sigma2[0] = np.var(rt)
    for i in range(1,len(rt)):
        sigma2[i] = omega + alpha * rt[i - 1] ** 2 + beta * sigma2[i - 1]
    log_likelihood = (np.log(2 * np.pi) + np.log(sigma2) + rt ** 2 / sigma2).sum() / 2
    return log_likelihood


def getGARCH(rt,*params):
    """计算NGARCH条件方差
    ：param last_sigma2:上一期的方差,初始方差t0为unconditional variate
    """
    con_sigma2 = np.ones(len(rt))
    con_sigma2[0] = rt.var()
    for i in range(1,len(rt)):
        con_sigma2[i] = params[0] + params[1] * rt[i - 1] ** 2 + params[2] * con_sigma2[i - 1]
    return con_sigma2


def negative_log_likelihood_NGARCH(params, rt):
    """定义NGARCH模型最大似然函数
    """
    omega, alpha, theta, beta = params
    sigma2 = np.ones(len(rt))
    sigma2[0] = np.var(rt)
    for i in range(1, len(rt)):
        sigma2[i] = omega + alpha * (
            rt[i - 1] - theta * sigma2[i - 1] ** 0.5) ** 2 + beta * sigma2[i - 1]
    log_likelihood = (np.log(2 * np.pi) + np.log(sigma2) + rt ** 2 / sigma2).sum() / 2
    return log_likelihood


def getNGARCH(rt,*params):
    """计算NGARCH条件方差
    ：param last_sigma2:上一期的方差,初始方差t0为unconditional variate
    """
    con_sigma2 = np.ones(len(rt))
    con_sigma2[0] = rt.var()
    for i in range(1,len(rt)):
        con_sigma2[i] = params[0] + params[1] * (rt[i - 1] - params[2] * con_sigma2[i - 1] ** 0.5) ** 2 + params[3] * con_sigma2[i - 1]
    return con_sigma2


def negative_log_likelihood_DCC_RM(p,rt1,rt2):
    """定义DCC-RM最大似然函数
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


def negative_log_likelihood_DCC_GARCH(params, rt1, rt2):
    """定义DCC-GARCH最大似然函数
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


def getDCC_GARCH(rt1, rt2, alpha, beta, *last_qt):
    """预测相关系数
    : param *params:为估计的DCC-GARCH模型参数
    : param **last_qt:为最后一日的q11,q12,q22
    : return:相关系数和qt11,qt12,qt22
    """
    qt11 = 1 + alpha * (rt1 ** 2 - 1) + beta * (last_qt[0] - 1)
    qt12 = uncon_rho12 + alpha * (rt1 * rt2 - uncon_rho12) + beta * (last_qt[1] - uncon_rho12)
    qt22 = 1 + alpha * (rt2 ** 2 - 1) + beta * (last_qt[2] - 1)
    return np.array([qt12 / np.sqrt(qt11 * qt22), qt11, qt12, qt22])

# Pro 1:实现 Monte Carlo Simulation的DCC-GARCH，并计算投资组合的在未来k-day的VaR和ES
# Dynamic Correlations using Multivariate Monte Carlo Simulation
# 导入数据
da1 = pd.read_excel('/Users/hhr/Desktop/Projects/pydata/var_chapter8_data.xlsx')
da1 = da1.set_index('Date')

# 计算收益率
da1['SP500 Return'] = np.log(da1.SP500_Close / da1.SP500_Close.shift(1))
da1['US10FT Return'] = np.log(da1.US10FT_Close / da1.US10FT_Close.shift(1))

# STEP1: 估计GARCH模型,并进行标准化处理
params_GARCH_SP500 = fmin(negative_log_likelihood_GARCH, x0=[5e-6, 0.1, 0.85], args=(
    da1['SP500 Return'][1:].values,),ftol=1e-7) # SP500 Return

params_GARCH_US10FT = fmin(negative_log_likelihood_GARCH, x0=[5e-6, 0.1, 0.85], args=(
    da1['US10FT Return'][1:].values,),ftol=1e-7) # US10FT Return


# 计算条件方差
da1['Conditional Variance SP500'] = np.insert(getGARCH(da1['SP500 Return'][1:],*params_GARCH_SP500),0,np.nan)
da1['Conditional Variance US10FT'] = np.insert(getGARCH(da1['US10FT Return'][1:],*params_GARCH_US10FT),0,np.nan)

# 标准化收益率
da1['Standardized SP500 Return'] = da1['SP500 Return'] / da1['Conditional Variance SP500'] ** 0.5
da1['Standardized US10FT Return'] = da1['US10FT Return'] / da1['Conditional Variance US10FT'] ** 0.5

# STEP2: 估计DCC-GARCH模型
params_DCC_GARCH = fmin(negative_log_likelihood_DCC_GARCH, x0=[0.05,0.90], args=(
    da1['Standardized SP500 Return'][1:].values, da1['Standardized US10FT Return'][1:].values), ftol=1e-7)

# calculate dynamic conditional variation and correlations
# 定义矩阵
n_sample = len(da1) - 1
qt11 = np.ones(n_sample)
qt12 = np.ones(n_sample)
qt22 = np.ones(n_sample)
# 初始值
rt1 = da1['Standardized SP500 Return'][1:].copy()
rt2 = da1['Standardized US10FT Return'][1:].copy()
qt11[0] = 1
qt12[0] = uncon_rho12 = np.sum(rt1 * rt2) / n_sample
qt22[0] = 1
for i in range(1, n_sample):
    qt11[i] = 1 + params_DCC_GARCH[0] * (rt1[i - 1] ** 2 - 1) + params_DCC_GARCH[1] * (qt11[i - 1] - 1)
    qt12[i] = uncon_rho12 + params_DCC_GARCH[0] * (rt1[i - 1] * rt2[i - 1] - uncon_rho12) + params_DCC_GARCH[1] * (qt12[i - 1] - uncon_rho12)
    qt22[i] = 1 + params_DCC_GARCH[0] * (rt2[i - 1] ** 2 - 1) + params_DCC_GARCH[1] * (qt22[i - 1] - 1)


da1['qt11'] = np.insert(qt11, 0, np.nan)
da1['qt12'] = np.insert(qt12, 0, np.nan)
da1['qt22'] = np.insert(qt22, 0, np.nan)
da1['Conditional Correlation(SP500-US10FT)'] = np.insert(qt12 /
                                                         np.sqrt(qt11 * qt22), 0, np.nan)


# STEP3: calculate dynamic matrix square root using DCC-GARCH
# draw a vector of uncorrelated random normal variables
np.random.seed(999)
uncorr_z = np.random.multivariate_normal(
    mean=(0, 0), cov=[[1, 0], [0, 1]], size=(10000, 11)) # 预测10天实际上需要11组随机数


# 计算未来k天的相关系数
result_DCC_GARCH = []
gamma_square = []
corr_z = []
# k为MC模拟的次数
for k in range(11):
        if k == 0:
            # 最后一天的数值
            rt0 = [da1['Standardized SP500 Return'][-1],da1['Standardized US10FT Return'][-1]]
            qt0 = [qt11[-1], qt12[-1], qt22[-1]]
            # 初始化(输入最后一天的rt1,rt2,qt11,qt12,qt22)
            result1 = getDCC_GARCH(rt0[0], rt0[1], params_DCC_GARCH[0], params_DCC_GARCH[1], *qt0)
            # calculate dynamic matrix square root and correlate the random variables
            gamma_square1 = np.array([[1, 0], [result1[0], np.sqrt(1 - result1[0] ** 2)]])
            for i in range(10000):
                corr_z.append(np.dot(gamma_square1, uncorr_z[i, k, :])) # 列表,长度为10000,每个单元为2*array
        else:
        # 预测第k天的相关系数
            for i in range(10000):
                if k == 1:
                    qt = result1[1:]
                else:
                    qt = result_DCC_GARCH[10000 * (k - 2) + i][1:]
                index = 10000 * (k - 1) + i  # 位置索引
                result_DCC_GARCH.append(getDCC_GARCH(
                    corr_z[index][0], corr_z[index][1], params_DCC_GARCH[0], params_DCC_GARCH[1], *qt))
                # calculate dynamic matrix square root and correlate the random variables
                con_rho = result_DCC_GARCH[index][0]
                gamma_square.append(np.array([[1, 0], [con_rho, np.sqrt(1 - con_rho ** 2)]]))
                corr_z.append(np.dot(gamma_square[index], uncorr_z[i, k, :]))

results_DCC_GARCH = np.array(result_DCC_GARCH) # MC模拟结果

# 计算k-day的相关系数(平均值)
corr_rho_k = []
for k in range(10):
    corr_rho_k.append(results_DCC_GARCH[:, 0][:10000 * (k + 1)].mean())


# STEP 4: 根据GARCH模型计算k-day的收益率和条件方差
# 初始设置
k_return = []
k_sigma2 = []
params_GARCH = np.array([params_GARCH_SP500, params_GARCH_US10FT])
corr_z = np.array(corr_z)  # correlate the random variables
for k in range(10):
        if k == 0:
            # 最后一天收益率和方差
            rt_last = np.array([da1['SP500 Return'][-1],da1['US10FT Return'][-1]])
            sigma2_last = np.array([da1['Conditional Variance SP500'][-1],da1['Conditional Variance US10FT'][-1]])
            # 预测第一天的方差和MC模拟第一天的收益率
            sigma2_d1 = params_GARCH[:,0] + params_GARCH[:,1] * rt_last ** 2 + params_GARCH[:,2] * sigma2_last
            for i in range(10000):
                k_return.append(sigma2_d1 ** 0.5 * corr_z[10000 * k + i])
        else:
            # 预测第k天的方差和MC模拟第k+1天的收益率
            for i in range(10000):
                if k == 1:
                    k_sigma2.append(params_GARCH[:, 0] + params_GARCH[:, 1] 
                                    * k_return[10000 * (k - 1) + i] ** 2 + params_GARCH[:, 2] * sigma2_d1)
                    k_return.append(k_sigma2[-1] ** 0.5 * corr_z[10000 * k + i])
                else:
                    k_sigma2.append(params_GARCH[:, 0] + params_GARCH[:, 1] * k_return[10000 * (k - 1) + i]
                                    ** 2 + params_GARCH[:, 2] * k_sigma2[10000 * (k - 2) + i])
                    k_return.append(k_sigma2[-1] ** 0.5 * corr_z[10000 * k + i])

# 整理结果


# STEP4: 根据MC=10000个结果,运用percentile找出1% VaR
var_SP500_MC = np.percentile(k_returns.sum(axis=1), 0.01 *
                        100, interpolation='linear')

# STEP 5: 根据MC=10000个结果,运用percentile找出1% VaR


# STEP 6: 计算1% ES



