# python 3.7.3 @hehaoran
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import leastsq, fmin, minimize
from scipy.stats import norm,chi2, t, f, multivariate_normal, genpareto,stats
from scipy.special import gamma
from pycopula.copula import ArchimedeanCopula,StudentCopula,GaussianCopula
from pycopula import simulation

from statsmodels.sandbox.distributions import copula
from ambhas.copula import Copula
# *******************Chapter 9********************

# Exercise 1: Threshold correlation for S&P 500 versus 10-year treasury bond returns.
# 导入数据
da1 = pd.read_excel(
    '/Users/hhr/Desktop/Projects/pydata/var_chapter8_data.xlsx')
da1 = da1.set_index('Date')

# 计算收益率
da1['SP500 Return'] = np.log(da1.SP500_Close / da1.SP500_Close.shift(1))
da1['US10FT Return'] = np.log(da1.US10FT_Close / da1.US10FT_Close.shift(1))

# 30,000 randomly generated Bivariate Normal numbers
np.random.seed(999)
uncorr_z = np.random.multivariate_normal(
    mean=(0, 0), cov=[[1, 0], [0, 1]], size=(30000, 1))


# STEP 1: compute S&P and US10FT Threshold
def qrange(start,stop,step=1):
    """定义一个生成器
    """
    for i in range(start,stop,step):
        yield round(i * 0.01,2)

# compute S&P and US10FT Threshold
threshold_corr = {}
return_data = da1[['SP500 Return', 'US10FT Return']][1:].values
for q in qrange(1,100):
    # 计算百分数
    emp_q = np.percentile(return_data, q * 100, axis=0,interpolation='linear')  # 2*1数组
    if q <= 0.5:
        threshold_return = return_data[(return_data[:, 0] <= emp_q[0]) & (return_data[:, 1] <= emp_q[1])]
    else:
        threshold_return = return_data[(return_data[:, 0] > emp_q[0]) & (return_data[:, 1] > emp_q[1])]
    # 若样本必须大于20才能计算相关系数
    if len(threshold_return) > 20:
        threshold_corr[q] = np.corrcoef(threshold_return[:, 0], threshold_return[:,1])[0,1]

threshold_corr = pd.Series(threshold_corr)

# STEP 2:compute threshold correlations using Bivariate Normal numbers
norm_corr = {}
norm_return = uncorr_z[:,0]
for q in qrange(1, 100):
    # 计算百分数
    emp_q = np.percentile(norm_return, q * 100, axis=0,interpolation='linear')  # 2*1数组
    if q <= 0.5:
        threshold_return = norm_return[(norm_return[:, 0] <= emp_q[0]) & (norm_return[:, 1] <= emp_q[1])]
    else:
        threshold_return = norm_return[(norm_return[:, 0] > emp_q[0]) & (norm_return[:, 1] > emp_q[1])]
    # 若样本必须大于20才能计算相关系数
    if len(threshold_return) > 20:
        norm_corr[q] = np.corrcoef(threshold_return[:, 0], threshold_return[:, 1])[0, 1]

norm_corr = pd.Series(norm_corr)


# STEP 3:整理结果,并作图(只选取0.15-0.85区间)
corr_emp_norm = pd.concat([threshold_corr, norm_corr], axis=1, join='inner', keys=[
                          'Empirical Threshold Correlation', 'Threshold Correlation Implied by Bivariate Normal'])
corr_emp_norm.plot()
plt.show()

# Exercise 2: Simulated threshold correlations from bivariate normal distributions with various linear correlations.
# STEP 1:
# rho = -0.3,0,0.3,0.6,0.9 # 相关系数
# cov = [[[1,i],[i,1]] for i in rho] # 方差矩阵
np.random.seed(999)
uncorr_z = np.random.multivariate_normal(mean=(0, 0), cov=[[1,0],[0,1]], size=(30000, 1))

# 30,000 randomly generated Bivariate Normal numbers using different rho
corr_i = []
for rho in [-0.3, 0, 0.3, 0.6, 0.9]:
    for i in range(30000):
        # calculate dynamic matrix square root and correlate the random variables
        gamma_square=np.array([[1, 0], [rho, np.sqrt(1 - rho ** 2)]])
        corr_i.append(np.dot(gamma_square, uncorr_z[i,0]))

# STEP 2:compute threshold correlations using Bivariate Normal numbers
norm_corr = {}
norm_corr_result = []
corr_z = [np.array(corr_i[30000 * i:30000 * (i + 1)]) for i in range(5)]

for each in corr_z:
    for q in qrange(1, 100):
        # 计算百分数
        emp_q = np.percentile(each, q * 100, axis=0,interpolation='linear')  # 2*1数组
        if q <= 0.5:
            threshold_return = each[(each[:,0] <= emp_q[0]) & (each[:,1] <= emp_q[1])]
        else:
            threshold_return = each[(each[:,0] > emp_q[0]) & (each[:,1] > emp_q[1])]
        # 若样本必须大于20才能计算相关系数
        if len(threshold_return) > 20:
            norm_corr[q] = np.corrcoef(threshold_return[:, 0], threshold_return[:, 1])[0, 1]
    norm_corr_result.append(pd.Series(norm_corr))

# STEP 3:整理结果,并作图
rho_corr_result = pd.concat(norm_corr_result,axis=1,join='inner',keys=['rho=-0.3','rho=0','rho=0.3','rho=0.6','rho=0.9'])
rho_corr_result.plot()
plt.show()

# Exercise 3: Estimate a normal copula model on the S&P500 and 10-year bond return data.
# STEP 1: Estimate the d parameter for each asset first.
# loading dataset
n_sample = len(da1) - 1
da3 = da1.copy()
# compute time-varying variance using RiskMetric model
def getRM(rt,nu=0.94):
    """get dynamic conditional variation
    """
    # 设置初始值
    sigma2 = np.ones(len(rt))
    sigma2[0] = np.var(rt)
    # nu = 0.94
    for i in range(1,len(rt)):
        sigma2[i] = (1 - nu) * rt[i - 1] ** 2 + nu * sigma2[i - 1]
    return sigma2

da3['Conditional Variance SP500(RM)'] = np.insert(getRM(da3['SP500 Return'][1:].values),0,np.nan)
da3['Conditional Variance US10FT(RM)'] = np.insert(getRM(da3['US10FT Return'][1:].values), 0, np.nan)

# Estimate the d parameter
def negative_log_likelihood(p, rt, sigma2):
    """
    The log likelihood function of estimating the d parameter
    """
    d = p
    log_likelihood = - n_sample * ((np.log(gamma((d + 1) / 2)) - np.log(gamma(d / 2)) - np.log(np.pi) /
                                    2 - np.log(d - 2) / 2)) + 0.5 * np.sum((1 + d) * np.log(1 + (rt / sigma2 ** 0.5) ** 2 / (d - 2)))
    return log_likelihood

# the results(the t distribution of the asset shock,not the asset return self)
df_SP500 = fmin(negative_log_likelihood, x0=[10], args=(
    da3['SP500 Return'][1:].values, da3['Conditional Variance SP500(RM)'][1:].values), ftol=1e-7)  # d估计值为8.30
df_US10FT = fmin(negative_log_likelihood, x0=[10], args=(
    da3['US10FT Return'][1:].values, da3['Conditional Variance US10FT(RM)'][1:].values), ftol=1e-7)  # d估计值为14.42

# STEP 2: Estimate a normal copula model(p205-206)
# standardizing the each assets
da3['Standardized SP500 Return'] = da3['SP500 Return'] / da3['Conditional Variance SP500(RM)'] ** 0.5
da3['Standardized US10FT Return'] = da3['US10FT Return'] / da3['Conditional Variance US10FT(RM)'] ** 0.5

# standardize the t-distribution of the asset shock and compute u1 and u2
da3['u1_SP500'] = [t(df=8.30).cdf(i * np.sqrt(8.30/(8.30-2))) for i in da3['Standardized SP500 Return']]
da3['u1_US10FT'] = [t(df=14.42).cdf(i * np.sqrt(14.42/(14.42-2))) for i in da3['Standardized US10FT Return']]

# the inverted u1 and u2 of the invert of CDF
da3['inv_u1_SP500'] = [norm.ppf(i) for i in da3['u1_SP500']]
da3['inv_u1_US10FT'] = [norm.ppf(i) for i in da3['u1_US10FT']]


# The global log-likelihood of normal copula to maximize
def log_likelihood_norm_copula(p,rt1,rt2):
    """the log likelihood function of normal copula:
    :params rt1,rt2: the inverted u1 and u2
    """
    rho = p
    log_likelihood = - 0.5 * n_sample * np.log(1 - rho ** 2) - np.sum(
        (rt1 ** 2 + rt2 ** 2 - 2 * rho * rt1 * rt2) / (2 * (1 - rho ** 2)) + (rt1 ** 2 + rt2 ** 2) / 2)
    return log_likelihood


# estimate the rho parameter
init_rho = da3['inv_u1_SP500'].corr(da3['inv_u1_US10FT'])
copula_rho = fmin(lambda p,rt1,rt2: -log_likelihood_norm_copula(p,rt1,rt2), x0=[init_rho], args=(
    da3['inv_u1_SP500'][1:].values,da3['inv_u1_US10FT'][1:].values),ftol=1e-7) # rho=-0.28

# Exercise 4: Simulate 10,000 sets of returns from the normal copula and compute the 1% VaR and ES from the model.
# STEP 1: get shock from Rt(standarize the return)
# STEP 2: estimate a density model for each asset
# STEP 3: estimate the parameters in the copula model(STEP1,2,3,based on above)
# STEP 4: simulate the probabilities(u1,u2,..)from the copula model.

# ArchimedeanCopula(error)
archimedean = ArchimedeanCopula(family="gumbel", dim=2)  # a copula type
data = da3.iloc[:, -2:][1:].values.copy()
archimedean.fit(data, method="cmle")
print(archimedean)

# StudentCopula(error)
student = StudentCopula(dim=2) # a copula type

# GaussianCopula(error)
gussian = GaussianCopula(dim=2)  # a copula type

# set the "mle" parameters using dictionary
paramX1 = {'df': 8.30}  # Hyper-parameters of Gamma
paramX2 = {'df': 14.42}  # Hyper-parameters of Exp
hyperParams = [paramX1,paramX2]
params_mle = dict(marginals=[t, t], hyper_param=hyperParams)
gussian.fit(da3.iloc[:, -2:][1:].values,method='mle', **params_mle)  # make an error
print(gussian)

# ambhas
foo = Copula(data[:,0], data[:,1], 'frank')
u, v = foo.generate_uv(n=10000)

# 10,000 randomly generated Bivariate Normal numbers
np.random.seed(999)
uncorr_z = np.random.multivariate_normal(
    mean=(0, 0), cov=[[1, 0], [0, 1]], size=(10000, 1))

# calculate matrix square root
rho_star = -0.19
gamma_square = np.array([[1, 0], [rho_star, np.sqrt(1 - rho_star ** 2)]])

# correlate the random variables
corr_z = []
for j in range(10000):
    corr_z.append(np.dot(gamma_square, uncorr_z[j, 0, :]))
corr_z = np.array(corr_z)

# simulate the probabilities(u1,u2,..)from the copula model.
u1 = np.array([norm.cdf(i) for i in corr_z[:, 0]])
u2 = np.array([norm.cdf(i) for i in corr_z[:, 1]])

# create shocks from the copula probabilities using the marginal inverse
z1 = np.array([np.sqrt((8.30 - 2) / 8.30) * t(8.30).ppf(i) for i in u1])
z2 = np.array([np.sqrt((14.42 - 2) / 14.42) * t(14.42).ppf(i) for i in u2])

# create returns from shocks using the dynamic volatility models
sigma2_SP500 = 0.00004
sigma2_US10FT = 0.000051
r1 = np.sqrt(sigma2_SP500) * z1
r2 = np.sqrt(sigma2_US10FT) * z2

r_PF = 0.5 * r1 + 0.5 * r2 # w1,w2=0.5

# based on MC=10000 we can use np.percentile to compute 1% VaR
var_MC = - np.percentile(r_PF, 0.01 * 100, interpolation='linear')

# compute 1% ES
es_MC = - np.sort(r_PF)[:100].mean()
