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
    """Construct a Direct Graph based on the following parameters.

    Parameters:
    ---
    `data`: <tyrant.debtrank.Data>.
        All data required. see detail in tyrant.network.Data.
    
    `G`: <nx.Graph or nx.DiGraph>.
        see detail in networkx
       
    `is_remove`: <True>.
        Remove all edges equal to 0. Default is True.
    
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
        node_assets = [self._FN.nodes[node]['assets'] for node in self._FN]
        edge_weight = [self._FN.edges[i, j]['weight'] for i, j in self._FN.edges]
        self._draw_params(node_assets, edge_weight)
        
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

    def _draw_params(self,node_assets,edge_weight):
        ## the size of nodes
        self._node_assets = self._scale(node_assets, y_min=150, y_max=600)
        ## the colour of nodes

        ## the width and colour of edges
        self._edge_color = self._scale(edge_weight, y_min=0.01, y_max=1)
    
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
        else:
            self._node_centrality = np.array([kwargs['centrality'][k] for k in kwargs['centrality']])
        
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
        return {'centrality': self._node_centrality, 'node color': nodes_color}
    
    def nodes(self):
        return list(self._nodes)
    
    def edges(self):
        return list(self._edges)
    
    def draw(self, method='', h_i_shock=None, alpha=None, max_iter=100, is_savefig=False, font_size=5, node_color='b', seed=None, **kwargs):
        """draw financial network.

        Paramaters:
        ---
        `method`: <str>.
            the optional, the color of nodes map to the important level of bank. i.e. {'dr','nldr','dc',...}. Default = 'dr'.
        
        `h_i_shock`: <np.ndarray>. 
            the initial shock. see `tt.creating_initial_shock()`.

        `alpha`: <float>.
            optional, the parameter of Non-Linear DebtRank. Default = 0.

        `t_max`: <int>. 
            the max number of iteration. Default = 100.

        `is_savefig`: <False>. 
            optional, if True, it will be saved to the current work environment. otherwise, plt.show().

        `font_size`: <int>. 
            the size of the labels of nodes. Default = 5.  

        `node_color`: <str or RGB>.
            the color of nodes. if method is not empty, the colors reflect the importance level.  

        `**kwargs`: 
            customize your figure, see detail in networkx.draw.
        """
        # initial setting
        title = 'The ' + self._data._label_net + '(%s)' % self._data._label_year
        method = str(method)
        debtrank_alias = {'dr': 'debtrank','nldr': 'nonlinear debtrank'}
        centrality_alias = {
                            'idc': 'in-degree centrality',
                            'odc': 'out-degree centrality',
                            'dc': 'degree centrality',
                            'bc': 'betweenness centrality',
                            'cc': 'closeness(in) centrality',
                            'occ': 'out-closeness centrality',
                            'ec': 'eigenvector(in) centrality',
                            'oec': 'out-eigenvector centrality',
                            'kc': 'katz centrality',
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
            
            assert isinstance(self._h_i_shock,(list,np.ndarray)), "ERROR: the 'h_i_shock' you provided should be a list or np.ndarray."
            assert len(self._h_i_shock) == self._data._N, "ERROR: the length of 'h_i_shock' you provided is not equal to data."

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
                self._nodes_color = self._run_centrality(method = 'dr', h_i_shock=self._h_i_shock, t_max=max_iter)['node color']
            elif method == 'nldr':
                if alpha is None:
                    alpha = 0
                    print("Warning: the paramater of 'alpha' is essential! Default = %.2f" % alpha)
                # rename figure title
                title = 'The ' + self._data._label_net + ', ' + r'$\alpha = %.2f$' % alpha + ' (%s)' % self._data._label_year 
                # the legend labels
                self._legend_labels = ['nonlinear debtrank < 25%', 'nonlinear debtrank > 25%', 'nonlinear debtrank > 50%', 'nonlinear debtrank > 75%']
                # the color of nodes
                self._nodes_color = self._run_centrality(method='nldr', h_i_shock=self._h_i_shock, alpha=alpha, t_max=max_iter)['node color']
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
            if method == 'idc':
                # dict: dictionary. see detail in centrality.
                # the legend labels
                self._legend_labels = ['in-degree centrality < 25%', 'in-degree centrality > 25%',
                                       'in-degree centrality > 50%', 'in-degree centrality > 75%']
                # the color of nodes
                self._in_degree_centrality = ct.in_degree_centrality(self._FN)
                self._nodes_color = self._run_centrality(method='idc', centrality=self._in_degree_centrality)['node color']
            elif method == 'odc':
                # dict: dictionary. see detail in centrality.
                # the legend labels
                self._legend_labels = ['out-degree centrality < 25%', 'out-degree centrality > 25%',
                                       'out-degree centrality > 50%', 'out-degree centrality > 75%']
                # the color of nodes
                self._out_degree_centrality = ct.out_degree_centrality(self._FN)
                self._nodes_color = self._run_centrality(method='odc', centrality=self._out_degree_centrality)['node color']
            elif method == 'dc':
                # dict: dictionary. see detail in centrality.
                # the legend labels
                self._legend_labels = ['degree centrality < 25%', 'degree centrality > 25%',
                                       'degree centrality > 50%', 'degree centrality > 75%']
                # the color of nodes
                self._degree_centrality = ct.degree_centrality(self._FN)
                self._nodes_color = self._run_centrality(method='dc', centrality=self._degree_centrality)['node color']
            elif method == 'bc':
                # dict: dictionary. see detail in centrality.
                # the legend labels
                self._legend_labels = ['betweenness centrality < 25%', 'betweenness centrality > 25%',
                                       'betweenness centrality > 50%', 'betweenness centrality > 75%']
                # the color of nodes
                self._betweenness_centrality = ct.betweenness_centrality(self._FN, weight='weight', seed=seed)
                self._nodes_color = self._run_centrality(method='bc', centrality=self._betweenness_centrality)['node color']
            elif method == 'cc' or method == 'icc':
                # dict: dictionary. see detail in centrality.
                # the legend labels
                self._legend_labels = ['in-closeness centrality < 25%', 'in-closeness centrality > 25%',
                                       'in-closeness centrality > 50%', 'in-closeness centrality > 75%']
                # the color of nodes
                self._in_closeness_centrality = ct.closeness_centrality(self._FN, distance='weight')
                self._nodes_color = self._run_centrality(method='cc', centrality=self._in_closeness_centrality)['node color']
            elif method == 'occ':
                # dict: dictionary. see detail in centrality.
                # the legend labels
                self._legend_labels = ['out-closeness centrality < 25%', 'out-closeness centrality > 25%',
                                       'out-closeness centrality > 50%', 'out-closeness centrality > 75%']
                # the color of nodes
                self._out_closeness_centrality = ct.closeness_centrality(self._FN.reverse(), distance='weight')
                self._nodes_color = self._run_centrality(method='occ', centrality=self._out_closeness_centrality)['node color']
            elif method == 'ec' or method == 'iec':
                # dict: dictionary. see detail in centrality.
                # the legend labels
                self._legend_labels = ['in-eigenvector centrality < 25%', 'in-eigenvector centrality > 25%',
                                       'in-eigenvector centrality > 50%', 'in-eigenvector centrality > 75%']
                # the color of nodes
                self._in_eigenvector_centrality = ct.eigenvector_centrality(self._FN, max_iter=max_iter, weight='weight')
                self._nodes_color = self._run_centrality(method='ec', centrality=self._in_eigenvector_centrality)['node color']
            elif method == 'oec':
                # dict: dictionary. see detail in centrality.
                # the legend labels
                self._legend_labels = ['out-eigenvector centrality < 25%', 'out-eigenvector centrality > 25%',
                                       'out-eigenvector centrality > 50%', 'out-eigenvector centrality > 75%']
                # the color of nodes
                self._out_eigenvector_centrality = ct.eigenvector_centrality(self._FN.reverse(), max_iter=max_iter, weight='weight')
                self._nodes_color = self._run_centrality(method='oec', centrality=self._out_eigenvector_centrality)['node color']
            elif method == 'kc': # bug
                # dict: dictionary. see detail in centrality.
                # the legend labels
                self._legend_labels = ['katz centrality < 25%', 'katz centrality > 25%',
                                       'katz centrality > 50%', 'katz centrality > 75%']
                # the color of nodes
                phi, _ = np.linalg.eig(self._Ad_ij)
                self._katz_centrality = ct.katz_centrality(self._FN, alpha=1/np.max(phi) - 0.01, weight='weight')
                self._nodes_color = self._run_centrality(method='kc', centrality=self._katz_centrality)['node color']
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
                       'edge_color': self._edge_color,
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
        num_nodes = self._FN.order()
        num_edges = self._FN.size()
        density = nx.density(self._FN)
        node_connectivity = nx.node_connectivity(self._FN)
        
        stats = {}
        
        stats['nodes'] = np.round(num_nodes, 0)
        stats['edges'] = np.round(num_edges, 0)
        stats['density'] = np.round(density, 2)
        stats['connectivity'] = np.round(node_connectivity, 0)
        
        return pd.Series(stats,name='network stats')

    def centrality(self, h_i_shock=None, alpha=0.0, rank=False, seed=123, max_iter=100, **kwargs):
#       include: degree centrality,...
        cdntrality_index = ['in-degree centrality', 'out-degree centrality', 'degree centrality', 'betweenness centrality',
                            'in-closeness centrality', 'out-closeness centrality', 'in-eigenvector centrality', 'out-eigenvector centrality', 'debtrank', 'non-linear debtrank']

        # the greater the value, the more important
        self._in_degree_centrality = ct.in_degree_centrality(self._FN)
        # reflect the enthusiasm of banks
        self._out_degree_centrality = ct.out_degree_centrality(self._FN)
        # the greater the value, the more important
        self._degree_centrality = ct.degree_centrality(self._FN)
        # the greater the value, the more important
        self._betweenness_centrality = ct.betweenness_centrality(self._FN, weight='weight',seed=seed)
        # integration
        self._in_closeness_centrality = ct.closeness_centrality(self._FN, distance='weight')
        # radiality
        self._out_closeness_centrality = ct.closeness_centrality(self._FN.reverse(), distance='weight')
        # # the greater the value, the more important, Similar to PageRank
        self._in_eigenvector_centrality = ct.eigenvector_centrality(self._FN, max_iter=max_iter, weight='weight')
        self._out_eigenvector_centrality = ct.eigenvector_centrality(self._FN.reverse(), max_iter=max_iter, weight='weight')
        # self._katz_centrality = ct.katz_centrality(self._FN, weight='weight') # bug
        
        # debtrank
        if h_i_shock is None:
            h_i_shock = self._data.h_i_shock
        
        assert isinstance(h_i_shock,(list,np.ndarray)), "ERROR: the 'h_i_shock' you provided should be a list or np.ndarray."
        assert len(h_i_shock) == self._data._N, "ERROR: the length of 'h_i_shock' you provided is not equal to data."
        
        self._debtrank = self._run_centrality(method='dr', h_i_shock=h_i_shock, t_max=max_iter)['centrality']
        self._debtrank = dict(zip(self._nodes, self._debtrank))
        self._nonlinear_debtrank = self._run_centrality(method='nldr', h_i_shock=h_i_shock, alpha=alpha, t_max=max_iter)['centrality']
        self._nonlinear_debtrank = dict(zip(self._nodes, self._nonlinear_debtrank))

        network_centrality = [self._in_degree_centrality, self._out_degree_centrality, self._degree_centrality, self._betweenness_centrality, self._in_closeness_centrality,
                              self._out_closeness_centrality, self._in_eigenvector_centrality, self._in_eigenvector_centrality, self._debtrank, self._nonlinear_debtrank]
        df = pd.DataFrame(network_centrality).T
        df.columns = cdntrality_index

        if rank:
            df = df.rank(method='min',ascending=False)
        
        return df


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
    fn.centrality(alpha=0.05)
    # or add h_i_shock to data in advance
    fn.draw(method='nldr', alpha=0.01)
    fn.draw(method='dr')
    fn.draw(method='dr', is_savefig=True)
    fn.draw()
