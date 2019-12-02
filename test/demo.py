# python 3.7.4 @hehaoran
# data
import numpy as np
import pandas as pd

# model
import tensorly as tl
# from hmmlearn import hmm  # Hidden Markov Model
# import pymc3 as pm # MCMC
from tensorly.decomposition import parafac

# figure plot
import networkx as nx
import matplotlib.pyplot as plt

### 1
### Tensor Decomposition
x = tl.tensor(np.arange(24).reshape(3,4,2).astype(float))
core, factors = parafac(x, rank=3)
print(factors)

### 2
### financial network
G = nx.Graph()

G.add_edge(11,12)
G.add_edges_from([(1,2),(1,3)])

# plot
nx.draw(G, with_labels=True)
plt.show()



