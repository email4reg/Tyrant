# @author:                hehaoran
# @environment: python 3.7.4
import numpy as np
import pandas as pd
import time

#
from tqdm import tqdm
from dateutil.parser import parse

# 
from rpy2.robjects.packages import importr
from rpy2.robjects import r as r
from rpy2.robjects import FloatVector

# 
import sys
from operator import xor
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # for the error class <Finetwork.draw>

# figure
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Manipulation of Folders
import os
import errno

# pylint: disable = no-member

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def creating_initial_shock(n, node_num=None, shock=None):
    message = ["ERROR:The initial shock should in [0,1]","ERROR:The index of nodes should in [0,n - 1]"]

    h_i_shock = np.zeros(n, dtype=float)
    nodes_shock = {}

    if isinstance(node_num, int) and isinstance(shock, (int, float)):
        assert node_num <= n - 1 and node_num >= 0, message[1] 
        assert shock <= 1 and shock >= 0, message[0]
        nodes_shock[node_num] = shock
    elif isinstance(node_num,list) and isinstance(shock,(int,float)):
        assert shock <= 1 and shock >= 0, message[0]
        for i in node_num:
            assert i <= n - 1 and i >= 0, message[1]
        shock = [shock for _ in range(len(node_num))]
        nodes_shock = dict(zip(node_num, shock))
    elif isinstance(node_num, list) and isinstance(shock,list):
        assert len(node_num) == len(shock), "ERROR: the length of nodes and shocks should be equal"
        for i in node_num:
            assert i <= n - 1 and i >= 0, message[1]
        for j in shock:
            assert j <= 1 and j >= 0, message[0]
        nodes_shock = dict(zip(node_num,shock))
    else:
        raise Exception("ERROR: node_num and shock should be a int, float or list", TypeError)
    
    k = list(nodes_shock.keys())
    v = list(nodes_shock.values())

    for key, val in zip(k, v):
        h_i_shock[key] = val
    
    return h_i_shock


class DebtRank():
    """
     @param graph: the graph
     @param SD: set of nodes for which the measure must be calculated
     @param h: scalar (double) in [0,1] with the initial level of distress (equal for all nodes). 1 is default
     @param maxIter: maximum number of iterations
     @param relevance: dictionnary with absolute economic relevance of each node (could be Makt cap or other)
    """
    def __init__(self, data, relevance):
        assert isinstance(data, Data)
        self._data = data
        self._N = self._data.N()
        self._W_ji = self._data.Lambda_ij()
        self._relevance = relevance

        self._h_i = np.zeros(self._N, dtype=np.double)
        self._state = np.zeros(self._N, dtype=np.str)
    
    def iterator(self, h_i_shock, t_max, check_stationarity=True):
#       print h_i_shock
#       0.1
#       0.2
#       ...
        if check_stationarity:
            self._stationarity = False
            self._last_h_i = np.zeros(self._N, dtype=np.double)

        # Check h_i_shock
        h_i_shock = np.array(h_i_shock, dtype=np.double)
        assert len(h_i_shock) == self._N

        self._t_max = int(t_max)
        assert self._t_max >= 0

        # STEP t = 0
        self._t = 0
        self._h_i[:] = 0.0  # h_i(t=0)=0

        yield self._t

        # STEP t = 1

        self._t += 1
        self._h_i[:] = h_i_shock
        self._state[:] = ['U' if not i else 'D' for i in h_i_shock]
        yield self._t

        # STEP t = 2,3....
        while self._t < t_max:

            self._t += 1
            _D_i = [i[0] for i in np.argwhere(self._state == 'D')]

            # self._h_i += self._W_ji.dot(self._h_i)
            for i in range(self._N):
                for j in _D_i:
                    self._h_i[i] += self._h_i[j] * self._W_ji[j, i]
            self._h_i = np.minimum(1.0, self._h_i)
                              
            # Update the statex
            for i in range(self._N):
                if self._state[i] == 'D' or self._state[i] == 'I':
                    self._state[i] = 'I'
                else:
                    if self._h_i[i] > 0 and self._state[i] != 'I':
                        self._state[i] = 'D'
                    else:
                        self._state[i] = 'U'
            
            yield self._t

            if check_stationarity:
                if np.allclose(self._h_i, self._last_h_i):
                    self._stationarity = True
                    continue
                self._last_h_i[:] = self._h_i

    def h_i(self):
        return self._h_i.copy()

    def mean_h(self):
        return self._h_i.mean()

    def std_h(self):
        return self._h_i.std()

    def max_h(self):
        return self._h_i.max()

    def min_h(self):
        return self._h_i.min()

    def num_defaulted(self):
        return (self._h_i >= 1.0).sum()

    def num_active(self):
        return self._N - self.num_defaulted()

    def num_stressed(self):
        return ((self._h_i > 0.0) & (self._h_i < 1.0)).sum()

    def state(self):
        return self._state
    
    # cumulative last distress
    def R_i(self):
        return self._h_i * self._relevance / self._relevance.sum()

    # cumulative initial distress
    def R0(self,h_i_shock):
        return h_i_shock * self._relevance / self._relevance.sum()

    # excluding the initial distress
    def R(self, h_i_shock):
        return np.sum(self.R_i() - self.R0(h_i_shock))


class Data:
    """
    Loads the bank-specific data for the banks. 
    In essence, data provides all quantities. 
    like A_i, L_i, A_ij, L_ij, IB_A_i, EX_A_i, Lambda_ij, etc., 
    where IB means Inter-Bank and EX means External. 
    All these quantities correspond to the time t=0; 
    i.e. immediately before the shock which occurs at time t=1.

    Parameters
    -----------
    filein_Lambda_ij : <str>
        The file name and path of the file containing the inter-bank assets.
    filein_bank_specific_data : <str>
        The file name and path of the file containing the bank-specific data about banks.
    R_ij: None, <float> or <np.ndarray((N,N)):double>
        1) defines how much the value of the asset A_ij(0) worths after bank j defaulted. R_ij := rho*A_ij(0), rho in [0,1]
        2) If None, then the matrix R_ij is set to zero. 
           If <float>, then all entries of the matrix R_ij are set to it. 
           Otherwise R_ij is initialized as the provided matrix.
    checks : <bool>
        If True, a battery of checks is being run during object initialization.
    clipneg : <bool>
        If True, negative values are set to zero.
    year : <str>
        A label that can be provided or not, about the nature of the data. In this case, the year.
    p: <str>
        defined as the number of reconstructed edges divided by the number of possible edges (N(N − 1)) 
    net: <str>
         1)A label that can be provided or not, about the nature of the data. In this case, the net sample id.
    """

    def __init__(self, filein_bank_specific_data, h_i_shock=None, R_ij=None, checks=True, clipneg=True, year='', p='', net='', r_seed=123):

        # the network label
        self._label_year = parse(year).strftime("%Y-%m-%d")
        self._label_p=str(p)
        self._label_net=str(net)
        #
        self._filein_bank_specific_data=str(filein_bank_specific_data)
        # The initial shock to the banks
        self.h_i_shock = h_i_shock

        ## create bank_specific_data
#       bank_id   total_assets    equity         inter_bank_assets   inter_bank_liabilities  bank_name
#         1       2527465000.0    95685000.0      159769000.0          137316000.0         "HSBC Holdings Plc"
#         2       2888526820.49   64294758.6352   96239646.83          260571978.577        "BNP Paribas"
        df_bank_specific_data = pd.read_csv(filein_bank_specific_data)
        self._A_i = df_bank_specific_data['total_assets'].values
        self._E_i = df_bank_specific_data['equity'].values
        self._IB_A_i = df_bank_specific_data['inter_bank_assets'].values
        self._IB_L_i = df_bank_specific_data['inter_bank_liabilities'].values
        self._bank_name_i = list(df_bank_specific_data['bank_name'].values)

        if clipneg:
            self._A_i.clip(0.,None)
            self._E_i.clip(0.,None)
            self._IB_A_i.clip(0.,None)
            self._IB_L_i.clip(0.,None)

        self._N = len(self._A_i)

        ## create inner_bank_exposure via above filein_bank_specific_data
#       source  target  exposure
#         1       2       18804300.1765828
#         1       3       593429.0704464162
#         1       4       7180905.941936611
        nrm = importr("NetworkRiskMeasures")
        self._A_ij = np.zeros((self._N, self._N), dtype=np.double)

        r('set.seed(%s)' % r_seed)
        print('Estimating the exposure matrix....')
        md_mat = nrm.matrix_estimation(FloatVector(self._IB_A_i), FloatVector(self._IB_L_i), method='md', verbose='F')
        print('Finished!')
        self._df_edges = self._df2exposures(md_mat)

        # df_edges = pd.read_csv(filein_A_ij)
        for _,i,j,w in self._df_edges[1].itertuples():
            ii = int(i - 1) # here is a bug, must convert to <int>
            jj = int(j - 1)
            assert ii >= 0 and ii < self._N
            assert jj >= 0 and jj < self._N
            if clipneg:
                w=max(0.,w)
            else:
                assert w > 0
            self._A_ij[ii,jj] = w

#        print '# Creating R_ij...'
        if R_ij is None:
            self._R_ij = np.zeros( (self._N,self._N) , dtype=np.double )
        else:
            try:
                rho=float(R_ij)
                self._R_ij = np.ndarray( (self._N,self._N) , dtype=np.double )
                self._R_ij[:,:]=rho
            except:
                self._R_ij = np.ndarray( R_ij , dtype=np.double )
                assert self._R_ij.shape == ( self._N, self._N )

#        print '# Creating Lambda_ij...'

        self._Lambda_ij=np.zeros( (self._N,self._N) , dtype=np.double )
        for i in range(self._N):
            for j in range(self._N):

                if clipneg:
                    if self._E_i[i] > 0.0 and self._A_ij[i,j] > 0.0:
                        tmp = self._A_ij[i,j]*(1.0-self._R_ij[i,j])/self._E_i[i]
                    else:
                        tmp = 0.0
                    assert tmp >= 0.0
                else:
                    tmp = self._A_ij[i,j]*(1.0-self._R_ij[i,j])/self._E_i[i]

                self._Lambda_ij[i,j] = tmp

        if checks:
            for i in range(self._N):
                assert self.IB_A_i()[i] == self.A_ij()[i, :].sum(), "ERROR: the inter_bank_assets should be equal to the sum of lending to others"
                assert self.IB_L_i()[i] == self.L_ij()[i, :].sum(), "ERROR: the inter_bank_liabilities should be equal to the sum of loaning from others"

    def _df2exposures(self,r_mat):
        df = pd.DataFrame(np.array(list(r_mat)).reshape(self._N, self._N).T, columns=self._bank_name_i, index=self._bank_name_i)
        inner_bank_exposure = [(i + 1, j + 1, df.values[i, j]) for i in range(self._N) for j in range(self._N) if i != j]
        return df, pd.DataFrame(np.array(inner_bank_exposure), columns=['source', 'target', 'exposure'])
    
    def label_year(self):
        return self._label_year

    def label_p(self):
        return self._label_p
    def label_net(self):
        return self._label_net

    def filein_A_ij(self):
        return self._filein_A_ij

    def filein_bank_specific_data(self):
        return self._filein_bank_specific_data

    def get_h_i_shock(self):
        return self.h_i_shock
    
    def getExposures(self,data='wide'):
        if data == 'wide':
            return self._df_edges[0].copy()
        else:
            return self._df_edges[1].copy()

    ###

    def N(self):
        return self._N

    def A_ij(self):
        return self._A_ij.copy()

    def A_i(self):
        return self._A_i.copy()

    def E_i(self):
        return self._E_i.copy()

    def IB_A_i(self):
        return self._IB_A_i.copy()

    def IB_L_i(self):
        return self._IB_L_i.copy()

    def bank_name_i(self):
        return self._bank_name_i.copy()

    ###

    def Lambda_ij(self):
        return self._Lambda_ij.copy()

    def L_ij(self):
        return np.transpose(self._A_ij.copy())

    def L_i(self):
        return self._E_i - self._A_i

    def IB_E_i(self):
        return self._IB_A_i - self._IB_L_i

    def EX_A_i(self):
        return self._A_i - self._IB_A_i

    def EX_E_i(self):
        return self._E_i - self.IB_E_i()
    
    def EX_L_i(self):
        return self.L_i() - self._IB_L_i


class NonLinearDebtRank:
    """
    This implements a non-linear DebtRank instance.

    Parameters
    -----------
    data : Data
        The data needed by the non-linear-DebtRank which is necessary to compute different quantities.
    h_i_shock : ndarray of floats
        The initial shock to the banks, i.e. h_i(t=1). Notice, it is assumed that h_i(0)=0 for all i (see comment C1 above).
    alpha : float
        The interpolation parameter; alpha = 0.0 corresponds to linear-DebtRank, and alpha = <big number> to "Furfine".
    """

    def __init__(self, data):

        assert isinstance(data, Data)
        self._data = data
        self._N = self._data.N()
        self._Lambda_ij = self._data.Lambda_ij()

        self._h_i = np.zeros(self._N, dtype=np.double)
        # This represents p_i(t-1)
        self._p_i_past = np.zeros(self._N, dtype=np.double)
        # This represents p_i(t)
        self._p_i_pres = np.zeros(self._N, dtype=np.double)

        self._isolated_banks = None
        self._num_isolated = len(self.isolated_banks())

        self._stationarity = None

    def isolated_banks(self):
        """Returns a list with the indexes of the isolated banks.

        A Bank is considered to be isolated if it does not have nor ingoing inputs and not ongoing inputs.
        """
        if self._isolated_banks is None:
            self._isolated_banks = []
            for i in range(self._N):
                if self._Lambda_ij[i, :].sum() == 0.0 and self._Lambda_ij[:, i].sum() == 0.0:
                    self._isolated_banks.append(i)
        return self._isolated_banks

    def num_isolated(self):
        return self._num_isolated

    def default_i(self, i):
        return self._h_i[i] > 1.0

    def _compute_p_i(self, h_i):
        """It takes h_i(t) and computes p_i(t)"""
        return np.minimum(1.0, h_i * np.exp(self._alpha * (h_i - 1.0)))

    def iterator(self, h_i_shock, alpha, t_max, zeroize_isolated_banks=True, check_stationarity=True):
        """This creates an iterator of the system dynamics.

        """

        if check_stationarity:
            self._stationarity = False
            self._last_h_i = np.zeros(self._N, dtype=np.double)

        # Check h_i_shock
        h_i_shock = np.array(h_i_shock, dtype=np.double)
        assert len(h_i_shock) == self._N

        self._alpha = float(alpha)
        assert self._alpha >= 0.0

        self._t_max = int(t_max)
        assert self._t_max >= 0

        # STEP t=0

        self._t = 0

        self._h_i[:] = 0.0  # h_i(t=0)=0

        # self._p_i_past[:] = 0.0                          # p_i(t=-1)
        # self._p_i_pres[:] = self._compute_p_i(h_i_shock)  # p_i(t=0)

        yield self._t

        # STEP t=1

        self._t += 1  # t -> t+1

        self._h_i[:] = h_i_shock  # h_i(t=1)=h_i_shock
        if zeroize_isolated_banks:
            for i in self.isolated_banks():
                self._h_i[i] = 0.0

        self._p_i_past[:] = 0.0                          # p_i(t=0)
        self._p_i_pres[:] = self._compute_p_i(h_i_shock)  # p_i(t=1)

        yield self._t
        #
        if check_stationarity:
            if np.allclose(self._h_i, self._last_h_i):
                self._stationarity = True
                return
            self._last_h_i[:] = self._h_i

        # STEPS t=2,3,...,t_max

        # Previously, there was a minor bug here. It used to stops at t_max+1 instead of t_max.
        while self._t < self._t_max:

            self._t += 1  # t -> t+1

            # Compute h_i(t+1)
            #
#            self._h_i += self._Lambda_ij.dot( self._p_i_pres - self._p_i_past )
            dp = self._p_i_pres - self._p_i_past
#            if ( dp < 0.0 ).sum() > 0:
#            print '########## t',self._t
#            print '########## h_i',self._h_i
#            print '########## PAST',self._p_i_past
#            print '########## PRES',self._p_i_pres
#            print '########## DIFF',dp
#            assert False
            #
            self._h_i += self._Lambda_ij.dot(dp)
            #
            self._h_i = np.minimum(1.0, self._h_i)

            self._p_i_past[:] = self._p_i_pres               # p_i(t)
            self._p_i_pres[:] = self._compute_p_i(self._h_i)  # p_i(t+1)

            yield self._t
            #
            if check_stationarity:
                if np.allclose(self._h_i, self._last_h_i):
                    self._stationarity = True
                    return
                self._last_h_i[:] = self._h_i

    def h_i(self):
        return self._h_i.copy()

    def mean_h(self):
        return self._h_i.mean()

    def std_h(self):
        return self._h_i.std()

    def max_h(self):
        return self._h_i.max()

    def min_h(self):
        return self._h_i.min()

    def num_defaulted(self):
        return (self._h_i >= 1.0).sum()

    def num_active(self):
        return self._N - self.num_defaulted()

    def num_stressed(self):
        return ((self._h_i > 0.0) & (self._h_i < 1.0)).sum()

    def H_i(self):
        """This returns the total relative equity loss at time t, defined as

            H(t) = sum_i H_i(t)

        where H_i(t) is the relative equity loss of bank i at time t, and is defined as

            H_i(t) = h_i(t) * E_i(0)/( sum_i E_i(0) )
        """
        _E_i = self._data.E_i()
        sum_E_i = _E_i.sum()
        _H_i = self._h_i * _E_i / sum_E_i
        return _H_i

    def H(self):
        return self.H_i().sum()

    def stationarity(self):
        return self._stationarity


class InitialShockGenerator:
    """
    This is a Shock Generator, i.e. a generator for h_i_ini = h_i(1).

    Parameters
    ----------
    data : <Data>
        The shock generator can use the loaded data to compute the shocks.
    filein_h_shock : None or <str>
        If provided it loads h_i_ini from file "filein_h_shock".
    filein_x_shock : None or <str>
        If provided it loads x_i from file "filein_x_shock", from where h_i_ini is computed according to the formula h_i(1) := x_i * A_i^{E}(0) / E_i(0). the devaluation of external assets
    p_i_shock : None or <float>
        If provided, a number in [0,1]. It is the probability for each bank to be shocked.(np.random.random() < self._p_i_shock)
    h_shock : None or <float>
        If provided, a number in [0,1], used to set h_i(1)=h_shock whenever i is a shocked bank according to the use of "p_i_shock".
    x_shock : None or <float>
        If provided, a number in [0,1], used to set h_i(1)= x_shock * A_i^{E}(0) / E_i(0) whenever i is a shocked bank according to the use of "p_i_shock".
    """
    def __init__(self,data,filein_h_shock=None,filein_x_shock=None,p_i_shock=None,h_shock=None,x_shock=None):

        assert isinstance(data,Data)
        self._data=data

        self._loaded_h_i=None
        self._loaded_x_i=None
        self._p_i_shock=None
        self._h_shock=None
        self._x_shock=None
        self._N = self._data.N()
        self._sample_h_i=np.zeros(self._N, dtype=np.double)
        self._sample_x_i=np.zeros(self._N, dtype=np.double)

        self._EX_A_i_div_E_i = np.zeros(self._N, dtype=np.double)
        EX_A_i = self._data.EX_A_i()
        E_i = self._data.E_i()
        for i in range(len(self._EX_A_i_div_E_i)):
            if E_i[i] <= 0.0:
                self._EX_A_i_div_E_i[i]=0.0
            else:
                self._EX_A_i_div_E_i[i]=EX_A_i[i]/E_i[i]
            assert self._EX_A_i_div_E_i[i] >= 0.0

        if xor(filein_h_shock is None, filein_x_shock is None):
            if filein_h_shock is not None:
                self._loaded_h_i = self._load_vector(filein_h_shock)
            else:
                self._loaded_x_i = self._load_vector(filein_x_shock)
                self._loaded_h_i = self._x_i_shock_2_h_i_shock(self._loaded_x_i)
        else:
            self._p_i_shock=float(p_i_shock)
            assert self._p_i_shock >= 0.0 and self._p_i_shock <= 1.0, 'ERROR: p_i_shock should be in [0,1].'
            assert xor(h_shock is None, x_shock is None), 'ERROR: either h_shock, or either x_shock should be provided.'
            try:
                self._h_shock=float(h_shock)
            except:
                self._x_shock=float(x_shock)

    def N(self):
        return self._N

    def _load_vector(self,filein,check=True):
        v = []
#       print filein_h_shock or filein_x_h_shock
#       0.1
#       0.2
#       0.3
        with open(filein,'r') as fh:
            for line in fh.readlines():
                val = float(line.replace('\n',''))
                if check:
                    assert v >= 0.0 and v <= 1.0, 'ERROR: the loaded vector entries are out of bound; i.e. not in [0,1].'
                v.append(val)
        assert len(v) == self._N, 'ERROR: len(loaded_vector) != N'
        return np.array(v,dtype=np.double)

    def _x_i_shock_2_h_i_shock(self,x_i_shock):
        assert len(x_i_shock) == self._N
        h_i_shock = x_i_shock * self._EX_A_i_div_E_i #abs( self._data.EX_A_i() / self._data.E_i() )
        assert len(h_i_shock[h_i_shock < 0.0]) == 0
        h_i_shock = np.minimum(1.0, h_i_shock)    
        return h_i_shock

    def sample_h_i_shock(self):
        if self._loaded_h_i is not None:
            self._sample_h_i[:] = self._loaded_h_i
        elif self._h_shock is not None:
            self._sample_h_i[:] = 0.0
            for i in range(self._N):
                if np.random.random() < self._p_i_shock:
                    self._sample_h_i[i]=self._h_shock
        elif self._x_shock is not None:
            self._sample_x_i[:]=0.0
            for i in range(self._N):
                if np.random.random() < self._p_i_shock:
                    self._sample_x_i[i]=self._x_shock
            self._sample_h_i[:]=self._x_i_shock_2_h_i_shock(self._sample_x_i)
        else:
            assert False, 'ERROR: there is no way to generate the sample h_i_shock.'
        return self._sample_h_i

class SmartExperimenter:
    """ 
    Smart Experimenter

    Parameters
    ----------
    rho: <str> or <float>
        The value in [0,1], and the same as R_ij on <Data>.
    p_shock: <str> or <float>
        The probability for each bank to be shocked, the same as p_i_shock on <InitialShockGenerator>.
    x_shock: <str> or <float>
        a factor of the devaluation of external assets. See <InitialShockGenerator> in detail
    alpha: <str> or <float>
        the parameter of the non-linear DebtRank algorithm. See <NonLinearDebtRank> in detail
    num_samples: <int>
        the number of sample_h_i_shock
    """

    def __init__(self,data,filein_parameter_space,t_max,num_samples,baseoutdir,seed=None):
        self._data=data
        self._filein_parameter_space=str(filein_parameter_space)
        self._t_max=int(t_max)
        self._num_samples=int(num_samples)
        self._baseoutdir=str(baseoutdir)
        self._seed=seed
        assert isinstance(self._data,Data)
        assert self._num_samples > 0
        assert self._t_max >= 0

        self._parameter_space=[]
        with open(self._filein_parameter_space,'r') as fh:
            for line in fh.readlines():
                if '#' in line:
                    continue
                # 1.rho 2.p_shock 3.x_shock 4.alpha
                cols=line.split()
                rho,p_shock,x_shock,alpha=cols
                rho=float(rho)
                p_shock=float(p_shock)
                x_shock=float(x_shock)
                alpha=float(alpha)
                assert rho >= 0.0
                assert p_shock > 0.0 and p_shock <= 1.0
                assert x_shock > 0.0
                assert alpha >= 0.0
                self._parameter_space.append( (rho,p_shock,x_shock,alpha) )
    
    def rho_ith_experiment(self,i):
        return self._parameter_space[i][0]
    def p_shock_ith_experiment(self,i):
        return self._parameter_space[i][1]
    def x_shock_ith_experiment(self,i):
        return self._parameter_space[i][2]
    def alpha_ith_experiment(self,i):
        return self._parameter_space[i][3]
    def num_experiments(self):
        return len(self._parameter_space)
    def run_ith_experiment(self,i):
        """Runs the i-th experiment on the paramter list."""

        year=self._data.label_year()
        p=self._data.label_p()
        net=self._data.label_net()

        rho,p_shock,x_shock,alpha = self._parameter_space[i]
        t_max = self._t_max

        Lshock="x_shock"
        Vshock=str(x_shock)
        #
        OUTDIR=self._baseoutdir+"/year"+str(year)+"/p"+str(p)+"/p_i_shock"+str(p_shock)+"/"+Lshock+Vshock+"/"
        make_sure_path_exists(OUTDIR)
        #
        execname='nonlinear_debt_rank_v4'
        fileout=OUTDIR+execname+"_p"+str(p)+"_year"+str(year)+"_net"+str(net)+"_rho"+str(rho)+"_alpha"+str(alpha)+"_p_i_shock"+str(p_shock)+"_h_shockNone"+"_x_shock"+str(x_shock)+".txt"
        with open(fileout,'w') as fhw:

            print('# year',year,file=fhw)
            print('# net',net,file=fhw)
            print('# p',p,file=fhw)
            print('# rho',rho,file=fhw)
            print('# p_i_shock',p_shock,file=fhw)
            print('# h_shock',None,file=fhw)
            print('# x_shock',x_shock,file=fhw)
            print('# alpha',alpha,file=fhw)
            print('# filein_A_ij',self._data.filein_A_ij(),file=fhw)
            print('# filein_bank_specific_data', self._data.filein_bank_specific_data(),file=fhw)
            print('# filein_parameter_space',self._filein_parameter_space,file=fhw)
            print('# num_samples',self._num_samples,file=fhw)
            print('# seed',self._seed,file=fhw)

            print('# Creating the Initial Shock Generator...',file=fhw)
            isg = InitialShockGenerator(self._data,p_i_shock=p_shock,x_shock=x_shock) # Here: p_shock = p_i_shock

            print('# Creating the Non-Linear Debt-Rank...',file=fhw)
            nldr=NonLinearDebtRank(self._data)

            print('# nldr.isolated_banks()',nldr.isolated_banks(),file=fhw)
            print('# nldr.num_isolated()',nldr.num_isolated(),file=fhw)

            for sample in range(1,self._num_samples+1):

                print(file=fhw)
                print(file=fhw)
                print('# sample',sample,file=fhw)

                sample_h_i_shock=isg.sample_h_i_shock()

                print('# 1.t 2.num_active 3.num_stressed 4.num_defaulted 5.min_h 6.mean_h 7.max_h 8.H',file=fhw)

                # Lets speed up. We do this by checking if we are already in the stationary state.
                for t in nldr.iterator(sample_h_i_shock, alpha, t_max):
                    #print >>,fhw,t, nldr.num_active(), nldr.num_stressed(), nldr.num_defaulted(), nldr.min_h(), nldr.mean_h(), nldr.max_h(), nldr.H()
                    A=nldr.num_active() 
                    S=nldr.num_stressed()
                    D=nldr.num_defaulted()
                    min_h=nldr.min_h()                 
                    mean_h=nldr.mean_h()                 
                    max_h=nldr.max_h()                 
                    H=nldr.H() 
                    print(t,A,S,D,min_h,mean_h,max_h,H,file=fhw)

                # Now, if we are here is because, either t=t_max, or either because we have reached the stationary point.
                if nldr.stationarity() is not None:
                    if not nldr.stationarity() and t == t_max:
                        print('# WARNING: NONSTATIONARITY',file=fhw)
                # Thus, we complete the simulation just printing, as it is              
                print('# tstar',t,file=fhw)
                while t < t_max:
                    t+=1
                    print(t,A,S,D,min_h,mean_h,max_h,H,file=fhw)


class Finetwork():
    """
    Construct a Direct Graph based on the following parameters

    @param Ad_ij: <DataFrame>
        DataFrame with columns and rowns indexed by node names. Position (i,j) contains the impact of node j over node i
    @param staticAttributes: 
        dictionary where the keyas are node identifiers and the values are list containing the values for a list of static attributes
    @param attributeNames: 
        list containing the names (as strng) of the various attributes from the dictionary staticAttributes
    """

    def __init__(self, data, G=None, is_remove=True):
        assert isinstance(data, Data), "ERROR: data must be a <Data>"

        self._data = data
        self._Ad_ij = self._data.getExposures() # the interbank exposures
        assert self._data.N() == len(self._Ad_ij), "ERROR: the length of data is not equal to Ad_ij"
        
        # remove isolated banks
        self._Ad_ij = self._remove_isolated_banks()
        
        # create a direct graph
        self._FN = nx.DiGraph()
        edges = [(i, j, self._Ad_ij.loc[i,j]) for i in self._Ad_ij.index for j in self._Ad_ij.columns]

        self._FN.add_weighted_edges_from(edges) # add all the weighted node to the grap
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

        # create a draw paramters
        attr_nodes = [self._FN.nodes[node]['assets'] for node in self._FN]
        attr_edges = [self._FN.edges[i, j]['weight'] for i, j in self._FN.edges]
        self._draw_params(attr_nodes, attr_edges)
        
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
    
    def _run_debtrank(self, h_i_shock, t_max):
        # run DebtRank algorithm
        dr = DebtRank(self._data, self._data.A_i())
        for _ in dr.iterator(h_i_shock=h_i_shock, t_max=t_max):
            pass
        # get the values of debtrank of nodes and rank by First-Third quantile
        self._node_debtrank = dr.h_i()
        q1, q2, q3 = np.percentile(self._node_debtrank, [25, 50, 75])
        # create the four kinds of colour of nodes
        nodes_color = []
        for i in self._node_debtrank:
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
    
    def draw(self, font_size=5, width=0.8, node_color='#6495ED', method='dr', h_i_shock=None, t_max=100, is_savefig=False, **kwargs):

        if h_i_shock is None:
            try :
                self._h_i_shock = self._data.get_h_i_shock()
            except:
                print("ERROR: the parameter 'h_i_shock' cannot be empty")
        else:
            self._h_i_shock = h_i_shock
        
        assert isinstance(self._h_i_shock,(list,np.ndarray)), "ERROR: 'h_i_shock' should be provided(i.e. <list> or <np.ndarray>)"

        method_alias = {'dr':'debtrank','nldr':'nonlinear debtrank'}

        if str(method) in method_alias:
            # the legend labels of method
            self.__legend_labels = ['debtrank < 25%', 'debtrank > 25%','debtrank > 50%','debtrank > 75%']
            # the colors of nodes
            self._nodes_color = self._run_debtrank(h_i_shock=self._h_i_shock, t_max=t_max)
            # the labels of nodes
            self._nodes_label = {}
            for i, j in zip(self._nodes, self._h_i_shock):
                assert j >= 0,"ERROR: the value of h_i_shock should in [0,1]"
                if j == 0.0:
                    self._nodes_label[i] = i 
                else:
                    self._nodes_label[i] = i + r"$\bigstar$"
        elif method == 'nldr':
            self.__legend_labels = ['nonlinear debtrank < 25%', 'nonlinear debtrank > 25%','nonlinear debtrank > 50%','nonlinear debtrank > 75%']
            pass # TODO
        else:
            self._nodes_color = node_color
        
        draw_default = {'node_size': self._node_assets,
                        'node_color': self._nodes_color,
                       'edge_color': self._edge_weights,
                       'edge_cmap': plt.cm.binary,
                        'labels': self._nodes_label,
                        'style': 'solid',
                        'with_labels' : True
                        }
        if not kwargs:
            kwargs = draw_default
        plt.rcParams['figure.dpi'] = 160
        plt.rcParams['savefig.dpi'] = 400
        legend_elements = [
                Line2D([0], [0], marker='o', color="#6495ED", markersize=5, label=self.__legend_labels[0]),
                Line2D([0], [0], marker='o', color="#EEEE00", markersize=5, label=self.__legend_labels[1]), 
                Line2D([0], [0], marker='o', color="#EE9A00", markersize=5, label=self.__legend_labels[2]),
                Line2D([0], [0], marker='o', color="#EE0000", markersize=5, label=self.__legend_labels[3]),
                Line2D([0], [0], marker='*', markerfacecolor="#000000", color='w',markersize=9, label='the shocked bank')
                ]
        plt.title('The ' + self._data._label_net + '(%s)' % self._data._label_year, fontsize = font_size+2)
        nx.draw(self._FN, pos=nx.circular_layout(self._FN), font_size=font_size, width=width, **kwargs)
        plt.legend(handles=legend_elements, fontsize=font_size, loc='best', frameon=False)
        if is_savefig:
            net,date = '',''
            net = net.join(self._data._label_net.split(' '))
            date = parse(self._data._label_year).strftime("%Y%m%d")
            plt.savefig(net + date, format='png', dpi=400)
        plt.show()

    def getFN(self):
        return self._FN

    def save(self, path):
       nx.write_gexf(self._FN, path + ".gexf")
    
    ## Generate a series of basic stats for the network
    def stats(self):
        
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
