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
