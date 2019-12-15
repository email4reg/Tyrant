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
import seaborn as sns

### 1
### Tensor Decomposition
x = tl.tensor(np.arange(24).reshape(3,4,2).astype(float))
core, factors = parafac(x, rank=3)
print(factors)

### 2
### financial network
G = nx.Graph()
G.add_nodes_from(['Rural Commercial Bank of Zhangjiagang',
             'Zhangjiagang Rural Commercial Bank', 
             'Qingdao Rural Commercial Bank',
             'Wuxi Rural Commercial Bank',
             'Xi an Bank',
             'Chongqing Rural Commercial Bank',
             'Changshu Rural Commercial Bank',
             'China Industrial Bank',
             'Zijin Rural Commercial Bank',
             'China Construction Bank',
             'Suzhou Rural Commercial Bank'])

G.add_edge('Rural Commercial Bank of Zhangjiagang', 'Xi an Bank')

DG = nx.DiGraph()

DG.add_weighted_edges_from([(1,2,0.5),(3,1,0.75),(1,4,0.3),(2,4,0.1),(1,5,0.2),(2,6,0.3)])
for i,j in zip(range(1,7),np.random.randint(100,1000,6)):
    DG.nodes[i]['assets'] = j


# 150 ~ 600
max_node = max([DG.nodes[node]['assets'] for node in DG])
min_node = min([DG.nodes[node]['assets'] for node in DG])
node_size = list(map(lambda x: 150 + (450 / (max_node - min_node)) * (x - min_node), [DG.nodes[node]['assets'] for node in DG]))

# 0.5 ~ 2
max_edge = max([DG.edges[i,j]['weight'] for i,j in DG.edges])
min_edge = min([DG.edges[i, j]['weight'] for i, j in DG.edges])
edge_color = list(map(lambda x: 0.5 + (1.5 / (max_edge - min_edge)) * (x - min_edge), [DG.edges[i,j]['weight'] for i,j in DG.edges]))

draw_params = {'node_size': node_size,
               'node_color': None,
               'width': 1.5,
               'edge_color': None,
               'with_labels': True}

# plot
nx.draw(DG, pos=nx.circular_layout(DG), node_size=node_size, edge_color=edge_color, width=2, edge_cmap=plt.cm.Blues)
plt.show()

