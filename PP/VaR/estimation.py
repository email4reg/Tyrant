# python 3.7.2 @hehaoran
import numpy as np
from scipy.special import gamma
from scipy.optimize import leastsq, fmin, minimize
from scipy.stats import norm, chi2, t, f, multivariate_normal, genpareto, stats

def negative_log_likelihood_GARCH(params, rt):
    """定义GARCH模型最大似然函数
    """
    omega, alpha, beta = params
    sigma2 = np.ones(len(rt))
    sigma2[0] = np.var(rt)
    for i in range(1, len(rt)):
        sigma2[i] = omega + alpha * rt[i - 1] ** 2 + beta * sigma2[i - 1]
    log_likelihood = (np.log(2 * np.pi) + np.log(sigma2) +
                      rt ** 2 / sigma2).sum() / 2
    return log_likelihood


def negative_log_likelihood_NGARCH(params, rt):
    """定义NGARCH模型最大似然函数
    """
    omega, alpha, theta, beta = params
    sigma2 = np.ones(len(rt))
    sigma2[0] = np.var(rt)
    for i in range(1, len(rt)):
        sigma2[i] = omega + alpha * (
            rt[i - 1] - theta * sigma2[i - 1] ** 0.5) ** 2 + beta * sigma2[i - 1]
    log_likelihood = (np.log(2 * np.pi) + np.log(sigma2) +
                      rt ** 2 / sigma2).sum() / 2
    return log_likelihood


# Estimate the d parameter
def negative_log_likelihood(p, rt, sigma2):
    """
    The log likelihood function of estimating the d parameter
    """
    d = p
    n = len(rt)
    log_likelihood = - n * ((np.log(gamma((d + 1) / 2)) - np.log(gamma(d / 2)) - np.log(np.pi) /
                             2 - np.log(d - 2) / 2)) + 0.5 * np.sum((1 + d) * np.log(1 + (rt / sigma2 ** 0.5) ** 2 / (d - 2)))
    return log_likelihood


def negative_log_likelihood_DCC_RM(p, rt1, rt2):
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
    for i in range(1, len(rt1)):
        qt11[i] = (1 - nu) * rt1[i - 1] ** 2 + nu * qt11[i - 1]
        qt12[i] = (1 - nu) * rt1[i - 1] * rt2[i - 1] + nu * qt12[i - 1]
        qt22[i] = (1 - nu) * rt2[i - 1] ** 2 + nu * qt22[i - 1]
    rho12 = qt12 / np.sqrt(qt11 * qt22)
    return 0.5 * np.sum(np.log(1 - rho12 ** 2) + (rt1 ** 2 + rt2 ** 2 - 2 * rho12 * rt1 * rt2) / (1 - rho12 ** 2))


def negative_log_likelihood_DCC_GARCH(params, rt1, rt2):
    """定义DCC-GARCH最大似然函数
    :param rt1,rt1:为标准化的收益率
    """
    alpha, beta = params
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
        qt12[i] = uncon_rho12 + alpha * \
            (rt1[i - 1] * rt2[i - 1] - uncon_rho12) + \
            beta * (qt12[i - 1] - uncon_rho12)
        qt22[i] = 1 + alpha * (rt2[i - 1] ** 2 - 1) + beta * (qt22[i - 1] - 1)
    rho12 = qt12 / np.sqrt(qt11 * qt22)
    return 0.5 * np.sum(np.log(1 - rho12 ** 2) + (rt1 ** 2 + rt2 ** 2 - 2 * rho12 * rt1 * rt2) / (1 - rho12 ** 2))

