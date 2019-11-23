# python 3.7.4 @hehaoran
import numpy as np
import pandas as pd
import random
import tensorly as tl
from hmmlearn import hmm  # Hidden Markov Model
import pymc3 as pm # MCMC
from tensorly.decomposition import parafac


# example 1 : Hidden Markov Model
# question 1
states = ["box1", "box2", "box3"]
n_states = len(states)

observations = ["red", "white"]
n_observations = len(observations)

initial_prob = np.array([0.2, 0.4, 0.4])

trans_prob = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])

emission_prob = np.array([
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_ = initial_prob
model.transmat_ = trans_prob
model.emissionprob_ = emission_prob

seen = np.array([[0, 1, 0]]).T
logprob, box = model.decode(seen, algorithm="viterbi")
print(logprob)

# question 2

x2 = np.array([[0, 1, 0, 1], [0, 0, 0, 1], [1, 0, 1, 1]])
model2 = hmm.MultinomialHMM(n_components=n_states,n_iter=20,tol=0.01)
model2.fit(x2)

print(model2.startprob_)
print(model2.transmat_)
print(model2.emissionprob_)
print(model2.score(x2))

# question 3: MCMC---->SV model

returns = pd.read_csv(pm.get_data('SP500.csv'), parse_dates=True, index_col=0)

with pm.Model() as sp500_model:
    nu = pm.Exponential('nu', 1/10., testval=5.)
    sigma = pm.Exponential('sigma', 1/0.02, testval=.1)

    s = pm.GaussianRandomWalk('s', sigma=sigma, shape=len(returns))
    volatility_process = pm.Deterministic('volatility_process', pm.math.exp(-2*s)**0.5)

    r = pm.StudentT('r', nu=nu, sigma=volatility_process,observed=returns['change'])


with sp500_model:
    trace = pm.sample(1000)

pm.traceplot(trace, varnames=['nu', 'sigma'])

# question 4: Markov Chain
# probabilistic matrix decomposition
transfer_a1 = np.array([[0, 2/3, 1/3],
                        [2/9, 0, 7/9],
                        [5/8, 3/8, 0]])

transfer_a2 = np.array([[0, 4/5, 1/5],
                        [1/3, 0, 2/3],
                        [4/9, 5/9, 0]])

transfer_h1 = np.array([[0, 2/7, 5/7],
                        [10/13, 0, 3/13],
                        [5/12, 7/12, 0]])

transfer_h2 = np.array([[0, 3/7, 4/7],
                        [8/13, 0, 5/13],
                        [1/4, 3/4, 0]])
for i in range(25):
    u = np.dot(v0, transfer_a)
    v = np.dot(u0, transfer_h)
    tol = np.linalg.norm(u - u0) + np.linalg.norm(v - v0)
    v0 = v
    u0 = u
    print(i, '\n', np.array([v0, u0]))

u = np.array([u0]).T
v = np.array([v0])
m = np.dot(u, v)  # 等于1，正确

# probabilistic tensor decomposition
u0 = np.array([[0.2, 0.1, 0.7]])
v0 = np.array([[0.3, 0.2, 0.5]])
w0 = np.array([[0.3, 0.7]])

a = np.array([[0, 2/5, 1/5 , 0, 8/25, 2/25],
                [1/9, 0, 7/18, 1/6, 0, 1/3],
                [5/17, 3/17, 0, 4/17, 5/17, 0]])

a = np.array([[[0, 2/5, 1/5], [1/9, 0, 7/18], [5/17, 3/17, 0]],
                [[0, 8/25, 2/25], [1/6, 0, 1/3], [4/17, 5/17, 0]]])

h = np.array([[0, 1/7, 5/14, 0, 3/14, 2/7],
                [5/13, 0, 3/26, 4/13, 0, 5/26],
                [1/4, 7/20, 0, 1/10, 3/10, 0]])

r = np.array([[0, 5/16, 5/32, 1/16, 0, 7/32, 5/32, 3/32, 0],
                [0, 2/7, 1/14, 3/28, 0, 3/14, 1/7, 5/28, 0]])


# Tensor Decomposition
x = tl.tensor(np.arange(24).reshape(3,4,2).astype(float))
core, factors = parafac(x, rank=3)
print(factors)



