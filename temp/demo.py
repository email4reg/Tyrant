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
from matplotlib.lines import Line2D

# pylint: disable = no-member
### 1
### Tensor Decomposition
x = tl.tensor(np.arange(24).reshape(3,4,2).astype(float))
core, factors = parafac(x, rank=3)
print(factors)
nx.classes.gra
### 2
### financial network
DG = nx.DiGraph()
G = nx.Graph()

DG.add_weighted_edges_from([(1,2,0.5),(1,7,0.1),(3,1,0.75),(1,4,0.3),(2,4,0.1),(1,5,0.2),(2,6,0.3)])
G.add_edges_from([(1,2),(1,7),(3,1),(1,4),(2,4),(1,5),(2,6)])
for i,j in zip(range(1,8),np.random.randint(100,1000,7)):
    DG.nodes[i]['assets'] = j

# 150 ~ 600
max_node = max([DG.nodes[node]['assets'] for node in DG])
min_node = min([DG.nodes[node]['assets'] for node in DG])
node_size = list(map(lambda x: 150 + (450 / (max_node - min_node)) * (x - min_node), [DG.nodes[node]['assets'] for node in DG]))

# 0.5 ~ 2
max_edge = max([DG.edges[i,j]['weight'] for i,j in DG.edges])
min_edge = min([DG.edges[i, j]['weight'] for i, j in DG.edges])
edge_color = list(map(lambda x: 0.5 + (1.5 / (max_edge - min_edge)) * (x - min_edge), [DG.edges[i,j]['weight'] for i,j in DG.edges]))

# plot
labels = {}
labels[1] = "a"
labels[2] = "b"
labels[3] = "c"
labels[4] = "d"
labels[5] = "e"
labels[6] = "f"
labels[7] = "g"
legend_elements = [Line2D([0], [0], color="#6495ED", lw=4, label='Line'),
                   Line2D([0], [0], color="#EEEE00", lw=4, label='Line'), 
                   Line2D([0], [0], color="#EE9A00", lw=4, label='Line'),
                   Line2D([0], [0], color="#EE0000", lw=4, label='Line')]

plt.text(0,-1.16,r"$\bigstar$" + ": the shocked bank", va='bottom', ha='center')
plt.legend(["unimportant(<25%)", "general(>25%)", "important(>50%)",
            "very important(>75%)"], ncol=4, handles=legend_elements, fontsize=7, loc='bottom', frameon=False)
nx.draw(DG, pos=nx.circular_layout(DG), labels=labels,node_color=['g','b','y','y','r','b','r'],edge_color=edge_color, width=2, edge_cmap=plt.cm.Greys, with_labels=True)
plt.show()
