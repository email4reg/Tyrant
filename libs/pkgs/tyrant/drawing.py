# @author:                hehaoran
# @environment: python 3.7.4
# pylint: disable = no-member
from dateutil.parser import parse
import numpy as np
import pandas as pd
import os
# for the error class <Finetwork.draw>
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# figure
import networkx as nx
from networkx.algorithms import centrality as ct
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

##
from tyrant import Data
from tyrant.debtrank import DebtRank, NonLinearDebtRank


class Finetwork():
    """
    Construct a Direct Graph based on the following parameters

    Parameters:
    ---
    `data`: tyrant.debtrank.Data
        All data required. see detail in tyrant.network.Data.
    ---
    `G`: nx.Graph or nx.DiGraph
        see detail in networkx
    ---   
    `is_remove`: bool
        Remove all edges equal to 0. Default is True.
    ---
    Return:
    ---
        network
    """
    def __init__(self, data, G=None, is_remove=True):
        assert isinstance(data, Data), "ERROR: data must be a <Data>"

        self._data = data
        self._Ad_ij = self._data.getExposures() # the interbank exposures
        assert self._data.N() == len(self._Ad_ij), "ERROR: the length of data is not equal to Ad_ij"
        
        # remove isolated banks
        self._Ad_ij = self._remove_isolated_banks()
        
        if G is None:
            # create a direct graph
            self._FN = nx.DiGraph()
            edges = [(i, j, self._Ad_ij.loc[i,j]) for i in self._Ad_ij.index for j in self._Ad_ij.columns]
            self._FN.add_weighted_edges_from(edges) # add all the weighted node to the grap
        else:
            assert isinstance(G,(nx.classes.graph.Graph,nx.classes.digraph.DiGraph)),"ERROR: G should be type networkx.Graph or networkx.DiGraph"
            self._FN = G # TODO
        
        self._nodes = self._FN.nodes()
        self._edges = self._FN.edges()

        # remove weight=0
        if is_remove:
            self._FN = self._remove_0_weight()

        self._nodes = self._FN.nodes()
        self._edges = self._FN.edges()

        # add the attribute to nodes(i.e assets)
        for i, j in zip(self._data._bank_name_i, self._data.A_i()):
            self._FN.nodes[i]['assets'] = j
        # TODO: the weight of a directed network(defult = the loan amount)

        # create a draw paramters
        attr_nodes = [self._FN.nodes[node]['assets'] for node in self._FN]
        attr_edges = [self._FN.edges[i, j]['weight'] for i, j in self._FN.edges]
        self._draw_params(attr_nodes, attr_edges)
        
    def __str__(self):
        return self._data._label_net + ' on ' + self._data._label_year
    
    def _remove_isolated_banks(self):
        for i in self._Ad_ij.index:
            if self._Ad_ij.loc[i, :].sum() == 0.0 and self._Ad_ij.loc[:, i].sum() == 0.0:
                self._Ad_ij.drop(i, axis=0, inplace=True)
                self._Ad_ij.drop(i, axis=1, inplace=True)
                print("Warning: %s was removed!" % i)
        return self._Ad_ij

    def _remove_0_weight(self):          
        weight_0 = [(u, v) for (u, v) in self._edges if not self._FN.edges[u, v]['weight']]
        self._FN.remove_edges_from(weight_0)
        return self._FN

    def _scale(self,x, y_min, y_max):
        # y_min and y_max are the values you expected
        x_min = np.min(x)
        x_max = np.max(x)
        return list(map(lambda x: y_min + ((y_max - y_min) / (x_max - x_min)) * (x - x_min), x))

    def _draw_params(self,attr_nodes,attr_edges):
        ## the size of nodes
        self._node_assets = self._scale(attr_nodes, y_min=150, y_max=600)
        ## the colour of nodes

        ## the width and colour of edges
        self._edge_weights = self._scale(attr_edges, y_min=0.01, y_max=1)
    
    def _run_centrality(self, **kwargs):
        # compute centrality
        # run DebtRank algorithm
        if kwargs['method'] == 'dr':
            dr = DebtRank(self._data, self._data.A_i())
            for _ in dr.iterator(h_i_shock=kwargs['h_i_shock'],t_max=kwargs['t_max']):
                pass
            # get the values of debtrank of nodes and rank by First-Third quantile
            self._node_centrality = dr.h_i()
        elif kwargs['method'] == 'nldr':
            nldr = NonLinearDebtRank(self._data)
            for _ in nldr.iterator(h_i_shock=kwargs['h_i_shock'],alpha=kwargs['alpha'], t_max=kwargs['t_max']):
                pass
            # get the values of debtrank of nodes and rank by First-Third quantile
            self._node_centrality = nldr.h_i()
        elif kwargs['method'] == 'dc':
            self._node_centrality = np.array([kwargs['centrality'][k] for k in kwargs['centrality']])
        elif kwargs['method'] == 'bc':
            self._node_centrality = np.array([kwargs['centrality'][k] for k in kwargs['centrality']])
        elif kwargs['method'] == 'cc':
            self._node_centrality = np.array([kwargs['centrality'][k] for k in kwargs['centrality']])
        elif kwargs['method'] == 'kc':
            self._node_centrality = np.array([kwargs['centrality'][k] for k in kwargs['centrality']])
        else:
            pass
        
        q1, q2, q3 = np.percentile(self._node_centrality, [25, 50, 75])
        # create the four kinds of colour of nodes
        nodes_color = []
        for i in self._node_centrality:
            if i < q1:
                nodes_color.append('#6495ED')
            elif i >= q1 and i < q2:
                nodes_color.append('#EEEE00')
            elif i >= q2 and i < q3:
                nodes_color.append('#EE9A00')
            else:
                nodes_color.append('#EE0000')
        return nodes_color
    
    def nodes(self):
        return list(self._nodes)
    
    def edges(self):
        return list(self._edges)
    
    def draw(self, method='', h_i_shock=None, alpha=None, t_max=100, is_savefig=False, font_size=5, node_color='b', **kwargs):
        # initial setting
        title = 'The ' + self._data._label_net + '(%s)' % self._data._label_year
        method = str(method)
        debtrank_alias = {'dr': 'debtrank','nldr': 'nonlinear debtrank'}
        centrality_alias = {
                            'dc': 'degree centrality',
                            'bc': 'betweenness centrality',
                            'cc': 'closeness centrality',
                            'kc': 'katz centrality'
                            }
        # method
        if method in debtrank_alias:
            if h_i_shock is None:
                try :
                    self._h_i_shock = self._data.h_i_shock
                except:
                    raise Exception("ERROR: the parameter 'h_i_shock' cannot be empty.", h_i_shock)
            else:
                self._h_i_shock = h_i_shock
            
            assert isinstance(self._h_i_shock,(list,np.ndarray)), "ERROR: 'h_i_shock' should be provided(i.e. <list> or <np.ndarray>)."
            assert len(self._h_i_shock) == self._data._N, "ERROR: the length of provided 'h_i_shock' is not equal to data."

            # the node labels
            self._node_labels = {}
            for i, j in zip(self._nodes, self._h_i_shock):
                assert j >= 0, "ERROR: the value of h_i_shock should in [0,1]"
                if j == 0.0:
                    self._node_labels[i] = i
                else:
                    self._node_labels[i] = i + r"$\bigstar$"
            # the method of debtrant
            if method == 'dr':
                # the legend labels
                self._legend_labels = ['debtrank < 25%', 'debtrank > 25%','debtrank > 50%','debtrank > 75%']
                # the color of nodes
                self._nodes_color = self._run_centrality(method = 'dr', h_i_shock=self._h_i_shock, t_max=t_max)
            elif method == 'nldr':
                if alpha is None:
                    alpha = 0
                    print("Warning: the paramater of 'alpha' is essential! Default = %.2f" % alpha)
                # rename figure title
                title = 'The ' + self._data._label_net + ', ' + r'$\alpha = %.2f$' % alpha + ' (%s)' % self._data._label_year 
                # the legend labels
                self._legend_labels = ['nonlinear debtrank < 25%', 'nonlinear debtrank > 25%',
                                    'nonlinear debtrank > 50%', 'nonlinear debtrank > 75%']
                # the color of nodes
                self._nodes_color = self._run_centrality(method='nldr', h_i_shock=self._h_i_shock, alpha=alpha, t_max=t_max)
            else:
                pass # TODO

            _legend_elements = [
            Line2D([0], [0], marker='o', color="#6495ED", markersize=3.5, label=self._legend_labels[0]),
            Line2D([0], [0], marker='o', color="#EEEE00", markersize=3.5, label=self._legend_labels[1]), 
            Line2D([0], [0], marker='o', color="#EE9A00", markersize=3.5, label=self._legend_labels[2]),
            Line2D([0], [0], marker='o', color="#EE0000", markersize=3.5, label=self._legend_labels[3]),
            Line2D([0], [0], marker='*', markerfacecolor="#000000", color='w', markersize=6.5, label='the initial shock')
            ]
            _ncol = 5
        
        elif method in centrality_alias:
            # the node labels
            self._node_labels = dict(zip(self._nodes, self._nodes))
            # 'dc'
            if method == 'dc':
                # dict: dictionary. see detail in centrality.
                # the legend labels
                self._legend_labels = ['degree centrality < 25%', 'degree centrality > 25%',
                                    'degree centrality > 50%', 'degree centrality > 75%']
                # the color of nodes
                self._degree_centrality = ct.degree_centrality(self._FN)
                self._nodes_color = self._run_centrality(method='dc', centrality=self._degree_centrality)
            elif method == 'bc':
                # dict: dictionary. see detail in centrality.
                # the legend labels
                self._legend_labels = ['betweenness centrality < 25%', 'betweenness centrality > 25%',
                                       'betweenness centrality > 50%', 'betweenness centrality > 75%']
                # the color of nodes
                self._betweenness_centrality = ct.betweenness_centrality(self._FN, k=kwargs['k'])
                self._nodes_color = self._run_centrality(method='bc', centrality=self._betweenness_centrality)
            elif method == 'cc':
                # dict: dictionary. see detail in centrality.
                # the legend labels
                self._legend_labels = ['closeness centrality < 25%', 'closeness centrality > 25%',
                                       'closeness centrality > 50%', 'closeness centrality > 75%']
                # the color of nodes
                self._closeness_centrality = ct.closeness_centrality(self._FN, u=kwargs['u'])
                self._nodes_color = self._run_centrality(method='cc', centrality=self._closeness_centrality)
            elif method == 'kc':
                # dict: dictionary. see detail in centrality.
                # the legend labels
                self._legend_labels = ['closeness centrality < 25%', 'closeness centrality > 25%',
                                       'closeness centrality > 50%', 'closeness centrality > 75%']
                # the color of nodes
                self._katz_centrality = ct.katz_centrality(self._FN, alpha=kwargs['u'])
                self._nodes_color = self._run_centrality(method='kc', centrality=self._katz_centrality)
            else:
                pass # TODO

            _legend_elements = [
            Line2D([0], [0], marker='o', color="#6495ED", markersize=3.5, label=self._legend_labels[0]),
            Line2D([0], [0], marker='o', color="#EEEE00", markersize=3.5, label=self._legend_labels[1]), 
            Line2D([0], [0], marker='o', color="#EE9A00", markersize=3.5, label=self._legend_labels[2]),
            Line2D([0], [0], marker='o', color="#EE0000", markersize=3.5, label=self._legend_labels[3])
            ]
            _ncol = 4
            
        else:
            # the node labels
            self._node_labels = dict(zip(self._nodes, self._nodes))
            self._nodes_color = node_color  # "#00BFFF"
            print("Warning: the color of nodes have no special meaning.")
        
        # draw
        draw_default = {'node_size': self._node_assets,
                        'node_color': self._nodes_color,
                       'edge_color': self._edge_weights,
                       'edge_cmap': plt.cm.binary,
                        'labels': self._node_labels,
                        'width': 0.8,
                        'style': 'solid',
                        'with_labels' : True
                        }
        
        # customize your nx.draw
        if 'node_size' in kwargs:
            draw_default['node_size'] = kwargs['node_size']
        if 'node_color' in kwargs:
            draw_default['node_color'] = kwargs['node_color']
        if 'edge_cmap' in kwargs:
            draw_default['edge_cmap'] = kwargs['edge_cmap']
        if 'labels' in kwargs:
            draw_default['labels'] = kwargs['labels']
        if 'style' in kwargs:
            draw_default['style'] = kwargs['style']
        if 'with_labels' in kwargs:
            draw_default['with_labels'] = kwargs['with_labels']

        draw_kwargs = draw_default

        plt.rcParams['figure.dpi'] = 160
        plt.rcParams['savefig.dpi'] = 400
        plt.title(title, fontsize = font_size + 2)
        nx.draw(self._FN, pos=nx.circular_layout(self._FN), font_size=font_size, **draw_kwargs)
        if method:
            plt.legend(handles=_legend_elements, ncol=_ncol, fontsize=font_size - 1, loc='lower center', frameon=False)
        
        if is_savefig:
            net,date = '',''
            net = net.join(self._data._label_net.split(' '))
            date = parse(self._data._label_year).strftime("%Y%m%d")
            plt.savefig(net + date + '.png', format='png', dpi=400)
            print("save to '%s'" % os.getcwd() + ' and named as %s' % (net + date) + '.png')
        else:
            plt.show()

    def getFN(self):
        return self._FN

    def save(self, path):
       nx.write_gexf(self._FN, path + ".gexf")
    
    ## Generate a series of basic stats for the network
    def stats(self):
#       include: number nodes; number edges; density; conectively       
        nNodes, nEdges = self._FN.order(), self._FN.size()
        avg_deg = float(nEdges) / nNodes
        
        # nb os strongly and weakly connected nodes
        scc = nx.number_strongly_connected_components(self._FN)
        wcc = nx.number_weakly_connected_components(self._FN)
        
        inDegree = self._FN.in_degree()
        outDegree = self._FN.out_degree()
        avgInDegree = np.mean(list(zip(*inDegree))[1])
        avgOutnDegree = np.mean(list(zip(*outDegree))[1])
        density = nx.density(self._FN)

        stats = {}
        
        stats['nbNodes'] = np.round(nNodes, 0)
        stats['nbEdges'] = np.round(nEdges, 0)
        stats['avg_deg'] = np.round(avg_deg, 2)
        stats['stronglyConnectedComponents'] = np.round(scc, 0)
        stats['weaklyConnectedComponents'] = np.round(wcc, 0)
        stats['avgInDegree'] = np.round(avgInDegree, 2)
        stats['avgOutnDegree'] = np.round(avgOutnDegree, 2)
        stats['density'] = np.round(density, 2)
        
        return pd.Series(stats,name='stats')

    def centrality(self):
#       include: degree centrality,
        self._degree_centrality = ct.degree_centrality(self._FN)
        self._betweenness_centrality = ct.betweenness_centrality(self._FN)
        self._closeness_centrality = ct.closeness_centrality(self._FN)
        self._katz_centrality = ct.katz_centrality(self._FN)

        network_centrality = list(
            self._degree_centrality, self._betweenness_centrality, self._closeness_centrality, self._katz_centrality)
        
        return pd.DataFrame(network_centrality)


if __name__ == "__main__":
    import os
    import tyrant as tt

    PATH = os.getcwd()

    # load bank data,like:
    path_bank_specific_data = PATH + '/bank_specific_data(2010, 6, 30).csv'
    data = Data(path_bank_specific_data)
    # create a financial network
    fn = tt.Finetwork(data)
    # create the initial shock, like:
    h_i_shock = tt.creating_initial_shock(data.N(), [1, 2], 0.01)
    fn.draw(method='nldr', alpha=0.01, h_i_shock=h_i_shock)
    fn.draw(method='dr', h_i_shock=h_i_shock)
    fn.draw(method='dc')
    # states
    fn.stats()
    # or add h_i_shock to data in advance
    fn.draw(method='nldr', alpha=0.01)
    fn.draw(method='dr')
    fn.draw(method='dr', is_savefig=True)
    fn.draw()
