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


print(ts.__version__)

# Elements of Financial Risk Management
"""
VaR: 我有p%的把握明天的损失率(收益率)不会大于VaR;
ES: (1-p)%糟糕的状况发生之后的加权平均损失;
这里的p需要理解成置信度,与书里的p相反,书里实际上是显著性水平a(写的是p,需要区分)
Notes: Spectral Risk Measure
"""
# *******************Chapter 1**************
data = pd.read_excel('/Users/hhr/Desktop/Projects/pydata/var_chapter1_data.xlsx')
data = data.set_index('Date')

# 计算log return
logreturn = [np.nan]
for i in range(1,len(data)):
    logreturn.append(math.log((data.iloc[i, :] / data.iloc[i - 1, :]).values[0]))
data['log return'] = logreturn

# correlation of daily S&P 500 returns with returns lagged from 1 to 100 days


def get_auto_corr(timeSeries,k):
    """
    Notes: 输入时间序列timeSeries,滞后阶数k
    """
    l = len(timeSeries)
    # 取出要计算的两个数组
    timeSeries1 = timeSeries[0:l-k]
    timeSeries2 = timeSeries[k:]
    timeSeries_mean = timeSeries.mean()
    timeSeries_var = np.array([i**2 for i in timeSeries-timeSeries_mean]).sum()
    auto_corr = 0
    for i in range(l-k):
        temp = (timeSeries1[i]-timeSeries_mean)*(timeSeries2[i]-timeSeries_mean)/timeSeries_var
        auto_corr = auto_corr + temp
    return auto_corr

auto_corr_result = {}
for i in range(1,101):
    auto_corr_result[i] = get_auto_corr(data['log return'][1:, ].values, i)

auto_corr_result = pd.Series(auto_corr_result)
# 作图1
auto_corr_result.plot()
plt.show()


# 1-day,1% VaR using RiskMetrics in S&P 500 portfolio January 1,2001– December 31,2010.
# 计算conditional varirance


def var_rm(data, t=1, is_merged=False,l=0.94):
    """RM VaR方法
    :param t: 未来的天数,默认为1-day
    :return : VaR值(Series)
    """
    con_var = []
    for i in range(len(data) + 1):
        if i == 0:
            con_var.append(np.nan)
        elif i == 1:
            con_var.append(0)
        else:
            con_var.append((1 - l) * (data['Return'][i - 1] ** 2) + l * con_var[-1])
    var_value = - math.sqrt(t) * np.array(list(map(lambda x: math.sqrt(x), con_var[:-1]))) * (-2.33)
    if not is_merged:
        return pd.Series(data=var_value,index=data.index)
    else:
        data['%d-day RM VaR' % t] = var_value


# 作图2.1
fig,ax = plt.subplots(figsize=(10,6))
data['1% VaR'].plot(ax=ax)
plt.show()

# *******************Chapter 2********************
# 导入数据
data2 = pd.read_excel('/Users/hhr/Desktop/Projects/pydata/var_chapter2_data.xlsx')
data2 = data2.set_index('Date')
data2 = data2.iloc[1:,:]

# 计算前m天的权重(m=250)
def calc_weight(m,g=0.99):
    """
    m:样本量
    g:公式参数,默认为0.99
    """
    weights = dict()
    for i in range(1,m+1):
        weights[i] = (g**(i - 1) * (1 - g)) / (1 - g**m)
    return weights

# 取前m天的样本,并匹配权重后升序排列
# 1987年10月交易日共22天(day)
def sampling_m(data,day,m):
    """
    :param day: 估计的天数
    :param m: 样本量
    :param data: 样本
    """
    # samples_day = []
    cum_weights = []  # samples_day,cum_weights = [[]] * 2
    for i in range(day):
        sample = data.iloc[(-(m + day) + i):(-day + i), :].copy() # 选取样本
        # reverse()没有返回值,而且是一个实例方法,reversed()是一个BIF,返回一个反转的迭代器;[::-1]顺序相反操作
        weight = [i for i in calc_weight(m).values()]
        sample['weight'] = [i for i in reversed(weight)]
        sample.sort_values(by='Return', inplace=True)  # 根据Return升序排列
        for index in range(len(sample)):
            # 计算累计权重
            if index == (len(sample) - 1):
                cum_weights.append(sample.iloc[index, -1])
            else:
                cum_weights.append(sum(sample.iloc[index:, -1]))
        sample['cum.weight'] = cum_weights
        # samples_day.append(sample)
        cum_weights.clear()
        yield sample # 变成一个迭代器
    # return samples_day


def var_whs(data,day,m,p=0.01):
    """
    计算VaR
    p: 显著性水平
    """
    var_value = []
    for each in sampling_m(data, day, m):
        var_value.append(-each[each['cum.weight'] >= 1 - p].iloc[-1, 1]) # 这里取收益率的负数就是VaR值(to long position)
    date = data.index[-day:]
    sns.lineplot(x=date, y=var_value)
    sns.lineplot(x=date, y=-data.Return.iloc[-day:])
    plt.xlabel('Loss Date')
    plt.ylabel('1% VaR')
    plt.title(
        'WHS 1% VaR and Daily Losses from Long S&P 500 Position /tOctober 1987.',
        fontsize=9)
    plt.show()


# HS VaR方法
def var_hs(data, n=250, is_merged=False, t=1, p=0.01):
    """
    :param :is_merged:是否将结果合并到数据中
    """
    var_value = [np.nan] * (n + 1)
    for i in range(1,len(data) - n):
        var_value.append(data['Return'][i:n+i].quantile(p))
    if not is_merged:
        return - math.sqrt(t) * pd.Series(data=var_value, index=data.index)
    else:
        data['%d-day HS VaR' % t] = - math.sqrt(t) * np.array(var_value)


# Question 2.3 & 2.4
# 导入数据2
da2 = pd.read_excel('/Users/hhr/Desktop/Projects/pydata/var_chapter2_data2.xlsx')
da2 = da2.set_index('Date')


# 作图2.2
fig, ax = plt.subplots(figsize=(10, 8))
# RM VaR值
t1 = var_rm(da2,t=10)
t1[(datetime(2008, 7, 1) <= t1.index) & (t1.index <= datetime(2009, 12, 31))].plot(ax=ax)

# HS VaR值
t2 = var_hs(data=da2, t=10)
t2[(datetime(2008, 7, 1) <= t2.index) & (t2.index <=
                                                datetime(2009, 12, 31))].plot(ax=ax)
plt.ylabel('1% 10-day VaR')
plt.legend(['Risk Metrics','HS'])
plt.title(
    '10-day, 1 % VaR from Historical Simulation and RiskMetrics During the July 1, 2008 - Dec 30, 2009',
    fontsize=9)
plt.show()

# 计算每日(P/L)from RM and HS VaR
# 计算RM and HS Va
data = da2.copy()
pl_value= [np.nan]
# 计算全样本的VaR值
# method ='RM'
var_rm(data=data,t=10,is_merged=True)
# method = 'HS'
var_hs(data=data,t=10,n=250,is_merged=True)
# 选取实验的期间的样本
sample = data[(parse('2008-06-30') <= data.index) &
              (data.index <= parse('2009-12-31'))].copy()
# 计算最大投资额position
sample['position HS'] = 100000 / sample['10-day HS VaR']
# 计算日P/L
for i in range(len(sample) - 1):
    pl_value.append(((sample['S&P500'][i + 1] / sample['S&P500'][i]) - 1)
                    * sample['position HS'][i + 1])
sample['P/L HS VaR'] = pl_value
sample = sample.iloc[1:, :].copy() # 去掉第一行
# 计算累计P/L
sample['Cumulative P/L with HS'] = sample['P/L HS VaR'].cumsum()

# 作图2.3
sample.iloc[:, -2:].plot()
plt.show()

# *******************Chapter 3********************
# 创建输出表格
tb = PrettyTable()

# 导入数据
data3 = pd.read_excel('/Users/hhr/Desktop/Projects/pydata/var_chapter3_data.xlsx')

# 转变为层次索引
# 注:names表示这一层索引的名称,levels是名称,labels表示位置
class1 = ['sample1', 'sample1','sample2', 'sample2', 'sample3', 'sample3', 'sample4', 'sample4']
class2 = data3.iloc[0,:].values
mulindex = pd.MultiIndex.from_arrays(
    [class1, class2], names=['samples','variable'])
data3.columns = mulindex
data3 = data3.iloc[1:,].applymap(lambda x: float(x)).copy()


# 时间序列计算基本描述性统计和相关关系、协方差
pd.set_option('precision', 2)  # 设置精度

# 计算基本统计量
moments = data3.describe().loc[['mean','std'],:]
data3.corr()

# 作图3.1
sns.lmplot(x='x', y='y', data=data3['sample1'])
plt.show()

# 随机漫步模型
da2 = pd.read_excel('/Users/hhr/Desktop/Projects/pydata/var_chapter3_data2.xlsx')

# 定义拟合的函数
def univ_reg(p,x):
    a,b = p
    return a*x + b

# 定义func
def error(p,x,y):
    return univ_reg(p,x) - y

results = []
for col in range(1,101):
    xi = da2.iloc[1:,col - 1].values
    yi = da2.iloc[0:-1,col - 1].values
    results.append(leastsq(error,x0=[0,0],args=(xi,yi))[0])
params = pd.DataFrame(results,columns=['a','b'])
sns.distplot(params['a'])
plt.show()

# 利用最大似然估计,构建MA(1)
da3 = pd.read_excel('/Users/hhr/Desktop/Projects/pydata/var_chapter3_data3.xlsx')

# 设置初始参数
theta0 = da3.mean().values[0]
theta1 = 0
sigma2 = da3.var().values[0]

# 定义最大似然函数
def negative_log_likelihood(params, rt):
    theta0,theta1,sigma2 = params
    epsiliont = np.ones(len(rt))
    epsiliont[0] = 0
    for i in range(1,len(rt)):
        epsiliont[i] = rt[i] - theta0 - theta1 * epsiliont[i - 1]
    log_likelihood = (np.log(2 * np.pi) + np.log(sigma2) + epsiliont ** 2/sigma2).sum()/2
    return log_likelihood


# 参数估计结果
result = fmin(negative_log_likelihood, x0=[theta0, theta1, sigma2], args=(da3.values,),ftol=1e-7)

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
omega, alpha, beta = 5e-6, 0.1, 0.85

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

# 获取股票数据
TOKEN = '4e28d8b91ed71e2c5c3e1d917fd81eeed3b70063265344bb173006e5'
pro = ts.pro_api(token=TOKEN)
data = pro.index_daily(ts_code='000001.SH',
                       fields='trade_date,open,close,pre_close,high,low,change,pct_chg,amount')

# 计算每日收益率
