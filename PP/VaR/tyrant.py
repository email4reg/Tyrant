# python 3.7.2 @hehaoran
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.parser import parse
from scipy.optimize import leastsq, fmin, minimize
from scipy.stats import norm, chi2, t, f, multivariate_normal, genpareto, stats
from datetime import datetime
import networkx

# pdf: Probability Density Function
# cdf: Cumulative Distribution Function
# sf: Survival Function(1-CDF)
# ppf: Percent Point Function(Inverse of CDF)
# isf: Inverse Survival Function(Inverse of SF)

def get_autocorr(x, k):
    """compute the autocorrelation(k)
    :param x: array_like
    :param k: the lag order
    """
    l = len(x)
    x1 = x[0:l-k]
    x2 = x[k:]
    # compute mean and variance
    _mean = np.mean(x)
    _var = np.var(x) * l
    autocorr = 0
    for i in range(l-k):
        _corr = (x1[i] - _mean) * (x2[i] - _mean) / _var
        autocorr = autocorr + _corr
    return autocorr


def get_var_rm(x, t=1, l=0.94,p=0.01):
    """compute VaR using RiskMetrics
    H0: return are normally distributed with zero mean and standard deviation(t+1)
    :param x: array_like
    :param t: the next t days
    :param l: the ratio of RiskMetrics
    """
    con_var = np.ones(len(x))
    con_var[0] = np.var(x)
    for i in range(1,len(x)):
        con_var[i] = (1 - l) * (x[i - 1] ** 2) + l * con_var[i - 1]
    return  - np.sqrt(t) * np.sqrt(con_var[:-1]) * norm.isf(1 - p)


def getNGARCH(rt, *params):
    """计算NGARCH条件方差
    ：param last_sigma2:上一期的方差,初始方差t0为unconditional variate
    """
    con_sigma2 = np.ones(len(rt))
    con_sigma2[0] = rt.var()
    for i in range(1, len(rt)):
        con_sigma2[i] = params[0] + params[1] * \
            (rt[i - 1] - params[2] * con_sigma2[i - 1]
             ** 0.5) ** 2 + params[3] * con_sigma2[i - 1]
    return con_sigma2


def getDCC_GARCH(rt1, rt2, alpha, beta, *last_qt):
    """预测相关系数
    : param *params:为估计的DCC-GARCH模型参数
    : param **last_qt:为最后一日的q11,q12,q22
    : return:相关系数和qt11,qt12,qt22
    """
    n_sample = len(rt1)
    uncon_rho12 = np.sum(rt1 * rt2) / n_sample
    qt11 = 1 + alpha * (rt1 ** 2 - 1) + beta * (last_qt[0] - 1)
    qt12 = uncon_rho12 + alpha * \
        (rt1 * rt2 - uncon_rho12) + beta * (last_qt[1] - uncon_rho12)
    qt22 = 1 + alpha * (rt2 ** 2 - 1) + beta * (last_qt[2] - 1)
    return np.array([qt12 / np.sqrt(qt11 * qt22), qt11, qt12, qt22])


