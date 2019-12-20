# @author:                hehaoran
# @environment: python 3.7.4
# pylint: disable = no-member
from dateutil.parser import parse
from tqdm import tqdm  # show progress bar
import numpy as np
import pandas as pd

#
from rpy2.robjects.packages import importr
from rpy2.robjects import r as r
from rpy2.robjects import FloatVector
nrm = importr("NetworkRiskMeasures")


def creating_initial_shock(n, node_num, shock):
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


class Data:
    """
    Loads the bank-specific data for the banks. 
    In essence, data provides all quantities. 
    like A_i, L_i, A_ij, L_ij, IB_A_i, EX_A_i, Lambda_ij, etc., 
    where IB means Inter-Bank and EX means External. 
    All these quantities correspond to the time t=0; 
    i.e. immediately before the shock which occurs at time t=1.
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


if __name__ == "__main__":
    pass
