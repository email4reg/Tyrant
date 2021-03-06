# @author:                hehaoran
# @environment: python 3.7.4
import numpy as np
import os
import errno
from operator import xor

## 
from tyrant.network import Data

# pylint: disable = no-member

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


class OriginalDebtRank():
    """
    Construct a original DebtRank

    Parameters:
    ---
    `data`: <Data, optional>
    >All data required. see tyrant.network.Data.
    
    `relevance`: <array-like, optional>
    >Absolute economic relevance of each node (could be Market cap or other)
    note:
    ---
    see Battiston S, Puliga M, Kaushik R, Tasca P, Caldarelli G. 
    DebtRank: too central to fail? Financial networks, the FED and systemic risk[J]. 
    Scientific Reports, 2012, 2(8): 541.
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
                    break
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


class NonLinearDebtRank:
    """
    This implements a non-linear DebtRank instance.

    Parameters
    -----------
    `data` : <Data>.
        The data needed by the non-linear-DebtRank which is necessary to compute different quantities.  
     note:
    ---
    see Bardoscia M, Caccioli F, Perotti J I, Vivaldo G, Caldarelli G. 
    Distress Propagation in Complex Networks: The Case of Non-Linear DebtRank[J]. 
    PLOS ONE, 2016, 11(10): e0163825.
    """
    def __init__(self, data):

        assert isinstance(data, Data)
        self._data = data
        self._N = self._data.N()
        self._Lambda_ij = self._data.Lambda_ij()
        self._EX_A_i = self._data.EX_A_i()

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

    def iterator(self, h_i_shock, alpha, t_max=100, zeroize_isolated_banks=True, check_stationarity=True, verbose=False):
        """This creates an iterator of the system dynamics.

        Parameters:
        ---
        `h_i_shock` : <array-like>.
            The initial shock to the banks, i.e. h_i(t=1). Notice, it is assumed that h_i(0)=0 for all i (see comment C1 above).  
        
        `alpha` : <float>.
            The interpolation parameter; alpha = 0.0 corresponds to linear-DebtRank, and alpha = <big number> to "Furfine".
        
        `t_max`: <int>.
            the max number of iteration. Default = 100.
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
            self._last_h_i[:] = self._h_i

        # STEPS t=2,3,...,t_max
        # Previously, there was a minor bug here. It used to stops at t_max+1 instead of t_max.
        while self._t < self._t_max:

            self._t += 1  # t -> t+1

            dp = self._p_i_pres - self._p_i_past

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
                    if verbose:
                        print("Warning: stationarity t = %s and please ignore it" % self._t)
                    break
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
    
    # calculate Systemically important financial institutions(SIFIs)
    # calculate loss matrix
    def Loss_matrix(self, t):
        _temp_a = np.linalg.inv(np.identity(self._N) - self._Lambda_ij)
        _temp_b = np.identity(self._N) - pow(self._Lambda_ij, t)
        return np.dot(_temp_a, _temp_b)

    def L_i(self, t, x_shock):
        self._L_i = np.zeros(self._N)
        self._loss_matrix = self.Loss_matrix(t)
        for i in range(self._N):
            self._L_i[i] = x_shock * self._EX_A_i[i] * np.sum(self._loss_matrix[i,:])
        return self._L_i / np.sum(self._L_i)

    # calculate systemically vulnerable financial institutions(SVFIs)
    def H(self):
        """
        the systematic vulnerability
        ---
        This returns the total relative equity loss at time t, defined as

            H(t) = sum_i H_i(t)

        where H_i(t) is the relative equity loss of bank i at time t, and is defined as

            H_i(t) = h_i(t) * E_i(0)/( sum_i E_i(0) )
        """
        _E_i = self._data.E_i()
        sum_E_i = _E_i.sum()
        _H_i = self._h_i * _E_i / sum_E_i
        return np.sum(_H_i)

    def H_i(self):
        """
        the systematic vulnerability of bank i.
        """
        sum_h_i = self._h_i.sum()
        _H_i = self._h_i / sum_h_i
        return _H_i

    def stationarity(self):
        return self._stationarity


class InitialShockGenerator:
    """
    This is a Shock Generator, i.e. a generator for h_i_ini = h_i(1).

    Parameters
    ----------
    `data` : <Data>
        The shock generator can use the loaded data to compute the shocks.

    `filein_h_shock` : None or <str>
        If provided it loads h_i_ini from file "filein_h_shock".
    
    `filein_x_shock` : None or <str>
        If provided it loads x_i from file "filein_x_shock", from where h_i_ini is computed according to the formula h_i(1) := x_i * A_i^{E}(0) / E_i(0). the devaluation of external assets
    
    `p_i_shock` : None or <float>
        If provided, a number in [0,1]. It is the probability for each bank to be shocked.(np.random.random() < self._p_i_shock)
    
    `h_shock` : None or <float>
        If provided, a number in [0,1], used to set h_i(1)=h_shock whenever i is a shocked bank according to the use of "p_i_shock".
    
    `x_shock` : None or <float>
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
#       print filein_h_shock or filein_x_h_shock
#       0.1
#       0.2
#       0.3
        v = []
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
    p_i_shock: <str> or <float>
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
                # 1.rho 2.p_i_shock 3.x_shock 4.alpha
                cols=line.split()
                rho,p_i_shock,x_shock,alpha=cols
                rho=float(rho)
                p_i_shock=float(p_i_shock)
                x_shock=float(x_shock)
                alpha=float(alpha)
                assert rho >= 0.0
                assert p_i_shock > 0.0 and p_i_shock <= 1.0
                assert x_shock > 0.0
                assert alpha >= 0.0
                self._parameter_space.append( (rho,p_i_shock,x_shock,alpha) )
    
    def rho_ith_experiment(self,i):
        return self._parameter_space[i][0]
    def p_i_shock_ith_experiment(self,i):
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

        rho,p_i_shock,x_shock,alpha = self._parameter_space[i]
        t_max = self._t_max

        Lshock="x_shock"
        Vshock=str(x_shock)
        #
        OUTDIR=self._baseoutdir+"/year"+str(year)+"/p"+str(p)+"/p_i_shock"+str(p_i_shock)+"/"+Lshock+Vshock+"/"
        make_sure_path_exists(OUTDIR)
        #
        execname='nonlinear_debt_rank_v4'
        fileout=OUTDIR+execname+"_p"+str(p)+"_year"+str(year)+"_net"+str(net)+"_rho"+str(rho)+"_alpha"+str(alpha)+"_p_i_shock"+str(p_i_shock)+"_h_shockNone"+"_x_shock"+str(x_shock)+".txt"
        with open(fileout,'w') as fhw:

            print('# year',year,file=fhw)
            print('# net',net,file=fhw)
            print('# p',p,file=fhw)
            print('# rho',rho,file=fhw)
            print('# p_i_shock',p_i_shock,file=fhw)
            print('# h_shock',None,file=fhw)
            print('# x_shock',x_shock,file=fhw)
            print('# alpha',alpha,file=fhw)
            print('# filein_A_ij',self._data.filein_A_ij(),file=fhw)
            print('# filein_bank_specific_data', self._data.filein_bank_specific_data(),file=fhw)
            print('# filein_parameter_space',self._filein_parameter_space,file=fhw)
            print('# num_samples',self._num_samples,file=fhw)
            print('# seed',self._seed,file=fhw)

            print('# Creating the Initial Shock Generator...',file=fhw)
            isg = InitialShockGenerator(self._data,p_i_shock=p_i_shock,x_shock=x_shock)

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


if __name__ == "__main__":
    import os
    import tyrant as tt
    from tyrant.debtrank import DebtRank, NonLinearDebtRank
    
    PATH = os.getcwd()

    # load bank data,like:
    path_bank_specific_data = PATH + '/bank_specific_data(2010, 6, 30).csv'
    data = Data(path_bank_specific_data)
    # debtrank or nonlinear debtrank
    h_i_shock = tt.creating_initial_shock(data.N(),[1,2],0.05)
    dr = DebtRank(data)
    # iteration,if t_max = 100.
    for _ in dr.iterator(h_i_shock,t_max=100):
        pass
    # result
    h = dr.h_i()
    dr.num_active()
    dr.num_defaulted()
