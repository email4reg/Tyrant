# R 3.6.1 @hehaoran
# -*- coding: utf-8 -*-
# load the packages
library("SparseM")
library("quantreg") # 依赖于SparseM包
library("stochvol") # 实现SV模型

# test 1
data(engel)
fit <- rq(engel$foodexp~engel$income,tau=0.5,method="br")

# test 2
# generate randomal data
sim <- svsim(500, mu = -10, phi = 0.99, sigma = 0.2)
# sampling
draws <- svsample(sim$y, draws = 4000, burnin = 100, priormu = c(-10, 1),priorphi = c(20, 1.2), priorsigma = 0.2)
# predict
fore <- predict(draws, 20)
summary(fore)
plot(draws, forecast = fore)
