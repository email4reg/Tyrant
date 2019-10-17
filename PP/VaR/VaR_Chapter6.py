# python 3.7.2 @hehaoran
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
from scipy.optimize import leastsq, fmin, minimize
from scipy.stats import percentileofscore, chi2, t, f, norm, genpareto
from scipy.special import gamma


# *******************Chapter 6********************
# Q-Q图
# 导入数据
da1 = pd.read_excel('C:/Users/hehaoran/Desktop/Projects/pydata/var_chapter6_data.xlsx')
da1 = da1.set_index('Date')
da1['Return'] = np.log(da1.Close / da1.Close.shift(1))

# 计算 Normalized Return
n_sample = len(da1) - 1
r_mean = da1.Return.mean()  # 均值
r_std_dev = da1.Return.var() ** 0.5  # 标准差
r_skewness = da1.Return[1:].skew()  # 偏度
r_kurtosis = da1.Return[1:].kurt()  # 峰度

da1['Normalized Return'] = da1.Return / r_std_dev

# sample quantiles
da1.sort_values(by='Normalized Return', inplace=True)  # 排序
da1['Ranking'] = [i for i in range(1, len(da1) + 1)]  # 排序

# theoretical quantiles
da1['Normal Quantile'] = [- norm.isf((i - 0.5) / n_sample)
                          for i in da1['Ranking']]

# 画图
plt.scatter(da1['Normal Quantile'], da1['Normalized Return'], s=6)

ax = plt.gca()  # get current axis 获得坐标轴对象
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')  # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

# 设置字体
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 11,
         }

plt.xlabel('Normal Quantile', font1)
plt.ylabel('Normalized Return', font1)
plt.title('QQ Plot of S & P 500 Returns', font1)
plt.show()

# Exercises 2-3(略)

# Exercises 4
# 基于QMLE构建NGARCH-t模型(见p128-131)
# MLE假设是正态分布,如果不是正态分布,那么就是QMLE(方法是一样的)
da4 = da1.copy()


# step1:估计NGARCH模型参数(p74-75)
def negative_log_likelihood(params, rt):
    omega, alpha, theta, beta = params
    sigma2 = np.ones(len(rt))
    sigma2[0] = np.var(rt)
    for i in range(1, len(rt)):
        sigma2[i] = omega + alpha * \
            (rt[i - 1] - theta * sigma2[i - 1]
             ** 0.5) ** 2 + beta * sigma2[i - 1]
    log_likelihood = (np.log(2 * np.pi) + np.log(sigma2) +
                      rt ** 2 / sigma2).sum()/2
    return log_likelihood


# 估计NGARCH模型参数
params = fmin(negative_log_likelihood, x0=[5e-6, 0.07, 0.5, 0.85], args=(
    da4['Return'][1:].values,), ftol=1e-7)

# 计算sigma2
r_data = da4['Return'][1:].copy()
sigma2 = np.ones(len(r_data))
sigma2[0] = np.var(r_data)
for i in range(1, len(r_data)):
    sigma2[i] = params[0] + params[1] * (r_data[i - 1] - params[2] * sigma2[i - 1]
                                         ** 0.5) ** 2 + params[3] * sigma2[i - 1]

da4 = da4.iloc[1:, :].copy()
da4['Sigma2'] = sigma2
n_sample = len(da4)

# step2:估计d
def negative_log_likelihood(p, rt, sigma2):
    d = p
    log_likelihood = - n_sample * ((np.log(gamma((d + 1) / 2)) - np.log(gamma(d / 2)) - np.log(np.pi) /
                             2 - np.log(d - 2) / 2)) + 0.5 * np.sum((1 + d) * np.log(1 + (rt / sigma2 ** 0.5) ** 2 / (d - 2)))
    return log_likelihood


# 估计结果
df = fmin(negative_log_likelihood, x0=[10], args=(
    da4['Return'].values, da4['Sigma2'].values), ftol=1e-7)  # d估计值为11.35

# 标准化
da4['Normalized Return'] = da4.Return / da4.Sigma2 ** 0.5

# sample quantiles
da4.sort_values(by='Normalized Return', inplace=True)  # 排序
da4['Ranking'] = [i for i in range(1, len(da4) + 1)]  # 排序

# theoretical quantiles
t_value = {}
for i in da4['Ranking']:
    p = (i - 0.5) / n_sample
    if p <= 0.5:
        t_value[i] = - np.abs(t(df=df).isf(p)) * np.sqrt((df - 2) / df)
    else:
        t_value[i] = np.abs(t(df=df).isf(1 - p)) * np.sqrt((df - 2) / df)
da4['t(d)-distribution Quantile'] = [t_value[i][0] for i in t_value]

# 画图
plt.scatter(da4['t(d)-distribution Quantile'], da4['Normalized Return'], s=6)
ax = plt.gca()  # get current axis 获得坐标轴对象
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')  # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

# 设置字体
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 11,
         }

plt.xlabel('t(d)-distribution Quantile', font1)
plt.ylabel('Normalized Return', font1)
plt.title('QQ Plot of S & P 500 NGARCH(1, 1) Shocks Against the Standardized t-distribution', font1)
plt.show()

# Exercises 5
# Estimate the EVT model on the standardized portfolio returns using the Hill estimator.
# 导入数据
da5 = pd.read_excel('C:/Users/hehaoran/Desktop/Projects/pydata/var_chapter6_data5.xlsx')
da5 = da5.set_index('Date')
# Hill Estimator(见p139-140)
n_sample = len(da5) - 1
u = da5['SortedStandardized Return'][51]
Tu = 50
xi = np.sum(np.log(da5['ABS(Standardized Return)'][1:51].values / np.abs(u))) / Tu
c = (Tu / n_sample) * np.abs(u) ** (1 / xi)
p = 0.01

# Calculate the 0.01% VaRs by following models: normal, t(d), EVT, and Cornish-Fisher.
# VaR-EVT
da5['VaR_EVT'] = - da5['st'] * u * (p / (Tu / n_sample)) ** - xi
# VaR-Normal
da5['VaR-Normal'] = - da5['st'] * - norm.isf(p)
# VaR-t(d)
df = 11.26 # 前面的估计结果
da5['VaR-t(d)'] = da5['st'] * np.sqrt((df - 2) / df) * t.isf(p,df)
# VaR-CF
def inverse_Cornish_Fisher(r_skewness,r_kurtosis,p):
    """Cornish-Fisher反函数
    """
    n_value = - norm.isf(p)
    return n_value + r_skewness / 6 * (n_value ** 2 - 1) + r_kurtosis / 24 * (n_value ** 3 - 3 * n_value) \
        - r_skewness ** 2 / 36 * (2 * n_value ** 3 - 5 * n_value)
# 计算偏度和峰度
r_skewness = da5['Standardized Return'].skew()
r_kurtosis = da5['Standardized Return'].kurt()
da5['VaR-CF'] = - da5['st'] * inverse_Cornish_Fisher(r_skewness,r_kurtosis,0.01)

# 对每年进行分组
grouped = da5[['VaR_EVT','VaR-Normal','VaR-t(d)','VaR-CF']].groupby(lambda x: x.year)
var_year = dict(list(grouped))
# 作图
var_year[2001].plot()
plt.show()

# Exercises 6 (见p141)
# Construct the QQ plot using the EVT distribution for the 50 largest losses. 
da6 = da5.copy()
q_EVT = [- np.abs(u) * (((i - 0.5) / n_sample) / (
    Tu / n_sample)) ** - xi for i in da6['Rank'][1:]]
q_EVT.insert(0,np.nan)
da6['EVT Left Tail Quantile'] = q_EVT

# 作图
plt.scatter(da6['EVT Left Tail Quantile'][1:51],da6['SortedStandardized Return'][1:51])
plt.show()

# Exercises 7
# calculate the 1-day, 1% VaRs, in 2010
# 导入数据
da2010 = da1[da1.index.year == 2010] # 选取2010年的样本
da_251 = da1[da1.index.year == 2009][-251:] # 选取2009年最后251-day
da7 = pd.concat([da_251,da2010]) # 合并两个数据

# Risk-Metrics
def var_rm(data, init_var, t=1, l=0.94):
    """RM VaR方法
    :param data: an array,需要包含前t天的收益率值
    :param t: 未来的天数,默认为1-day
    :param init_var: 初始的方差值
    :return : VaR值(array)
    """
    con_var = [init_var]
    for i in range(1,len(data)):
        con_var.append((1 - l) * (data[i - 1] ** 2) + l * con_var[-1])
    return - np.sqrt(t) * np.array(list(map(lambda x: np.sqrt(x), con_var[1:]))) * (-2.33)

init_var = da_251['Return'].var()   # 计算初始值
var_RiskM = var_rm(da7['Return'][250:].values,init_var=init_var) # 计算VaR-RM

# NGARCH(1,1)-t(d),根据exercise 5,d=11.26
var_NGARCH_t = da5[da5.index.year == 2010]['VaR-t(d)']

# Historical Simulation
def var_hs(data, n=250, t=1, p=0.01):
    """HS VaR方法
    :param data: an array
    """
    var_value = []
    for i in range(len(data) - n):
        var_value.append(np.percentile(data[i:n+i],p * 100,interpolation='linear'))
    return - np.sqrt(t) * np.array(var_value)

var_HS = var_hs(da7['Return'].values,n=251) # 计算VaR-RM,与书里不一致

# Filtered Historical Simulation (略)

# Exercises 7-8
# Use the asymmetric t distributio



