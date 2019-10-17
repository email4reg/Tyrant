# R 3.6.1 @hehaoran
# -*- coding: utf-8 -*-
# load the packages
library("copula")
library("stats")
# set random seed
set.seed(999)

## P1: construct and generate(10000) a bivariate distribution
# define a loglikelihood function of GARCH
loglikelihood.GARCH <- function(params, rt) # loglikelihood function
{
    sigma2 <- c(rep(1,length(rt)));
    sigma2[1] <- var(rt);
    for (i in 2:length(rt))
    {
        sigma2[i] <- params[1] + params[2] * rt[i - 1] ** 2 + params[3] * sigma2[i - 1];
    }
    loglikelihood <- -sum(log(2 * pi) + log(sigma2) + rt ** 2 / sigma2) / 2;
    # return negative loglikelihood
    return(-loglikelihood);
}

# define a function of getting conditional variance(GARCH)
getGARCH <- function(params,rt)
{
   sigma2 <- c(rep(1, length(rt)));
    sigma2[1] <- var(rt);
    for (i in 2:length(rt))
    {
        sigma2[i] <- params[1] + params[2] * rt[i - 1] ** 2 + params[3] * sigma2[i - 1];
    }
    return(sigma2)
}

# define a loglikelihood function of NGARCH
loglikelihood.NGARCH <- function(params, rt) 
{
    sigma2 <- c(rep(1, length(rt)));
    sigma2[1] <- var(rt);
    for (i in 2:length(rt)) 
    {
        sigma2[i] <- params[1] + params[2] * (rt[i - 1] - params[3] * sigma2[i - 1] ** 0.5) ** 2 + params[4] * sigma2[i - 1];
    }
    loglikelihood <- -sum(log(2 * pi) + log(sigma2) + rt ** 2 / sigma2) / 2;
    # return negative loglikelihood
    return(-loglikelihood);
}

# define a function of getting conditional variance(NGARCH)
getNGARCH <- function(params, rt) 
{
    sigma2 <- c(rep(1, length(rt)));
    sigma2[1] <- var(rt);
    for (i in 2:length(rt)) 
    {
        sigma2[i] <- params[1] + params[2] * (rt[i - 1] - params[3] * sigma2[i - 1] ** 0.5) ** 2 + params[4] * sigma2[i - 1];
    }
    return(sigma2)
}

# PART 1:
# load the var dataset
da1 <- read.csv('/Users/hhr/Desktop/Tyrant/RP/Rdata/var_data.csv')

# optimize the loglikelihood function
# estimate the GARCH model
SP500.params <- optim(c(5e-6, 0.07, 0.5,0.85), loglikelihood.NGARCH, rt = da1$SP500.Return)$par
US10FT.params <- optim(c(5e-6, 0.07,0.5, 0.85), loglikelihood.NGARCH, rt = da1$US10FT.Return)$par

# compute conditional variance
da1$SP500.sigma2 <- getNGARCH(SP500.params, da1$SP500.Return)
da1$US10FT.sigma2 <- getNGARCH(US10FT.params, da1$US10FT.Return)

# standardize the return
da1$SP500.zt <- da1$SP500.Return / da1$SP500.sigma2 ** 0.5
da1$US10FT.zt <- da1$US10FT.Return / da1$US10FT.sigma2 ** 0.5

# estimate the d parameter
# define a loglikelihood function of d parameter
loglikelihood.t <- function(d, zt) {
    n <- length(zt);
    loglikelihood <- n * (log(gamma((d + 1) / 2)) - log(gamma(d / 2)) - log(pi) / 2 - log(d - 2) / 2
    ) - 0.5 * sum((1 + d) * log(1 + zt ** 2 / (d - 2)));
    return(loglikelihood);
}

SP500.d <- optimize(loglikelihood.t, c(2, 99), zt = da1$SP500.zt, maximum = T, tol = 1e-7)$maximum # d=9.02(-3502.402)
US10FT.d <- optimize(loglikelihood.t, c(2, 99), zt = da1$US10FT.zt, maximum = T, tol = 1e-7)$maximum # d=14.48(-3526.24)

# PART 2:
# compute u1 and u2
u1 <- pt(q=da1$SP500.zt * sqrt(9.02/(9.02 - 2)),df = 9.02)
u2 <- pt(q = da1$US10FT.zt * sqrt(14.48/(14.48 - 2)), df = 14.48)
u <- matrix(c(u1,u2),ncol=2)

# estimate the Copula parameter rho
loglikelihood.normCopula <- function(rho, u1, u2) # the normal Copula loglikelihood
{
    n <- length(u1);
    loglikelihood <- -0.5 * n * log(1 - rho**2) - sum((
        qnorm(u1)**2 + qnorm(u2)**2 - 2 * rho * qnorm(u1) * qnorm(u2))/(2 * (1 - rho**2)) + (qnorm(u1)**2 + qnorm(u2)**2) / 2);
    return(loglikelihood);
}

# rho <- optimize(loglikelihood.normCopula,c(-1,1),u1=u1,u2=u2,maximum = T,tol = 1e-7)(?)

# compute the normal copula parameter rho(the same to above but the loglikelihood is different)**
ncop.rho <- fitCopula(normalCopula(dim = 2, dispstr = "un"), u, method = "ml") # rho=-0.2995

# PART 3:
# compute VaR and ES
# construct a bivariate distribution whose marginals(e.g. t distribution)
mc.nc <- mvdc(normalCopula(-0.2995), c('t', 't'), list(9.02, 14.48)) # rCopula
# generate 10000 samples and visualize
mc.z <- rMvdc(10000, mc.nc)
# compute the return
mc.r <- mc.z * sqrt(c(0.00004,0.000051))
# compute the portfolio return
mc.PF <- mc.r[,1] * 0.5 + mc.r[,2] * 0.5
# end result
var.t1 <- quantile(mc.PF,0.01)

