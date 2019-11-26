# pylint: disable = no-member
import sys
import numpy as np
# import networkx as nx
#import scipy
#import scipy.special 
import pandas as pd
from operator import xor

# Manipulation of Folders
import os
import errno
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

class Data:
    """Loads the bank-specific data for the banks. 
    In essence, data provides all quantities like A_i, L_i, A_ij, L_ij, IB_A_i, EX_A_i, Lambda_ij, etc., 
    where IB means Inter-Bank and EX means External. 
    All these quantities correspond to the time t=0; 
    .e. immediately before the shock which occurs at time t=1.
    """
    def __init__(self,filein_A_ij,filein_bank_specific_data,R_ij=None,checks=True,clipneg=True,year='',p='',net=''):

        self._label_year=str(year)
        self._label_p=str(p)
        self._label_net=str(net)
        #
        self._filein_A_ij=str(filein_A_ij)
        self._filein_bank_specific_data=str(filein_bank_specific_data)

#        print '# Loading bank-specific data from',filein_bank_specific_data

#bank_id total_assets    equity  inter_bank_assets       inter_bank_liabilities  bank_name
#1       2527465000.0    95685000.0      159769000.0     137316000.0     "HSBC Holdings Plc"
#2       2888526820.49   64294758.6352   96239646.83     260571978.577   "BNP Paribas"

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

#        print '# Loading inter-bank assets from',filein_A_ij

#source  target  exposure
#1       2       18804300.1765828
#1       3       593429.0704464162
#1       4       7180905.941936611
#1       5       13568931.097857257

        self._A_ij = np.zeros( (self._N,self._N) , dtype=np.double )
        df_edges = pd.read_csv(filein_A_ij)
        for _,i,j,w in df_edges.itertuples():
            ii = i - 1
            jj = j - 1
            assert ii >= 0 and ii < self._N
            assert jj >= 0 and jj < self._N
            if clipneg:
                w=max(0.,w)
            else:
                assert w > 0.
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
               
#        print '# Running checks...' 

        if checks:

            def tol(x,y,rtol=0.05):
                tot=abs(x)+abs(y)
                if tot==0.0:
                    return True
                rdif=2.0*abs(x-y)/tot
                if not rdif < rtol:
                    print("# WARNING")
                    print("# rdif",rdif)
                    print("# x",x)
                    print('# y',y)
#                    assert False

#            print '# Checking consistenct checks; Network quantities should be equivalent to array quantities...'            

            _np_A_ij = self.A_ij()
            _np_L_ij = self.L_ij()
            _np_IB_A_i = self.IB_A_i()
            _np_IB_L_i = self.IB_L_i()
            for i in range(self._N):
                tol( _np_IB_A_i[i], _np_A_ij[i,:].sum() )
                tol( _np_IB_L_i[i], _np_L_ij[i,:].sum() )

#        print '# NLDR has been created...'

    ###

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
    """This implements a non-linear DebtRank instance.

    Parameters
    ----------
        data : <Data>
            The data needed by the non-linear-DebtRank which is necessary to compute different quantities.
        h_i_shock : <np.array:double>
            The initial shock to the banks, i.e. h_i(t=1). Notice, it is assumed that h_i(0)=0 for all i (see comment C1 above).
        alpha : <float>
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
            self._old_h_i = np.zeros(self._N, dtype=np.double)

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

        self._p_i_past[:] = 0.0                          # p_i(t=-1)
        self._p_i_pres[:] = self._compute_p_i(h_i_shock)  # p_i(t=0)

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
            if np.allclose(self._h_i, self._old_h_i):
                self._stationarity = True
                return
            self._old_h_i[:] = self._h_i

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
                if np.allclose(self._h_i, self._old_h_i):
                    self._stationarity = True
                    return
                self._old_h_i[:] = self._h_i

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
    """This is a Shock Generator, i.e. a generator for h_i_ini = h_i(1).

    Parameters
    ----------
    data : <Data>
        The shock generator can use the loaded data to compute the shocks.
    filein_h_shock : None or <str>
        If provided it loads h_i_ini from file "filein_h_shock".
    filein_x_shock : None or <str>
        If provided it loads x_i from file "filein_x_shock", from where h_i_ini is computed according to the formula h_i(1) := x_i * A_i^{E}(0) / E_i(0) .
    p_i_shock : None or <float>
        If provided, a number in [0,1]. It is the probability for each bank to be shocked.
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
        self._sample_h_i=np.zeros( self._N , dtype=np.double )
        self._sample_x_i=np.zeros( self._N , dtype=np.double )

        self._EX_A_i_div_E_i=np.zeros( self._N , dtype=np.double )
        EX_A_i=self._data.EX_A_i()
        E_i=self._data.E_i()
        for i in range(len(self._EX_A_i_div_E_i)):
            if E_i[i] <= 0.0:
                self._EX_A_i_div_E_i[i]=0.0
            else:
                self._EX_A_i_div_E_i[i]=EX_A_i[i]/E_i[i]
            assert self._EX_A_i_div_E_i[i] >= 0.0

        if xor( filein_h_shock is None, filein_x_shock is None ):
            if filein_h_shock is not None:
                self._loaded_h_i = self._load_vector(filein_h_shock)
            else:
                self._loaded_x_i = self._load_vector(filein_x_shock)
                self._loaded_h_i = self._x_i_shock_2_h_i_shock(self._loaded_x_i)
        else:
            self._p_i_shock=float(p_i_shock)
            assert self._p_i_shock >= 0.0 and self._p_i_shock <= 1.0, 'ERROR: p_i_shock should be in [0,1].'
            assert xor( h_shock is None, x_shock is None ), 'ERROR: either h_shock, or either x_shock should be provided.'
            try:
                self._h_shock=float(h_shock)
            except:
                self._x_shock=float(x_shock)

    def N(self):
        return self._N

    def _load_vector(self,filein,check=True):
        v = []
        with open(filein,'r') as fh:
            for line in fh.readlines():
                val = float(line.replace('\n',''))
                if check:
                    assert v >= 0.0 and v <= 1.0, 'ERROR: the loaded vector entries are out of bound; i.e. not in [0,1].'
                v.append( val )
        assert len(v) == self._N, 'ERROR: len(loaded_vector) != N'
        return np.array(v,dtype=np.double)

    def _x_i_shock_2_h_i_shock(self,x_i_shock):
        assert len(x_i_shock) == self._N
        h_i_shock = x_i_shock * self._EX_A_i_div_E_i #abs( self._data.EX_A_i() / self._data.E_i() )
        assert len( h_i_shock[  h_i_shock < 0.0 ] ) == 0
        h_i_shock = np.minimum( 1.0 , h_i_shock )    
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
    """ TODO """

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

        # Create OUTPUT... stream
        #
        # if [ "${h_shock}" = "None" ]; then
        #     Lshock=x_shock
        #     Vshock=${x_shock}
        # else
        #     Lshock=h_shock
        #     Vshock=${h_shock}
        # fi
        # DIROUT=DATA_OUT_2/year${year}/p${p}/p_i_shock${p_i_shock}/${Lshock}${Vshock}/
        # mkdir -p ${DIROUT}
        # fileout=${DIROUT}/${execname}_p${p}_year${year}_net${net}_rho${rho}_alpha${alpha}_p_i_shock${p_i_shock}_h_shock${h_shock}_x_shock${x_shock}.dat
        
        Lshock="x_shock"
        Vshock=str(x_shock)
        #
        OUTDIR=self._baseoutdir+"/year"+str(year)+"/p"+str(p)+"/p_i_shock"+str(p_shock)+"/"+Lshock+Vshock+"/"
        make_sure_path_exists(OUTDIR)
        #
        execname='nonlinear_debt_rank_v4'
        fileout=OUTDIR+execname+"_p"+str(p)+"_year"+str(year)+"_net"+str(net)+"_rho"+str(rho)+"_alpha"+str(alpha)+"_p_i_shock"+str(p_shock)+"_h_shockNone"+"_x_shock"+str(x_shock)+".dat"
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
            isg = InitialShockGenerator(self._data,p_i_shock=p_shock,x_shock=x_shock)

            print('# Creating the Non-Linear Debt-Rank...',file=fhw)
            nldr=NonLinearDebtRank(self._data)

            print('# nldr.isolated_banks()',nldr.isolated_banks(),file=fhw)
            print('# nldr.num_isolated()',nldr.num_isolated(),file=fhw)

            for sample in range(1,self._num_samples+1):

                print(file=fhw)
                print(file=fhw)
                print('# sample',sample,file=fhw)

                sample_h_i_shock=isg.sample_h_i_shock()

                dynamics=False
                if dynamics:
                    print('TODO')
                    assert False ##'TODO'
                    #print >>fhw,'# 1.t 2.h_1,h_2,...,h_N'
                else:
                    print('# 1.t 2.num_active 3.num_stressed 4.num_defaulted 5.min_h 6.mean_h 7.max_h 8.H',file=fhw)

                # Lets speed up. We do this by checking if we are already in the stationary state.
                for t in nldr.iterator(sample_h_i_shock,alpha,t_max):
                    if dynamics:
                        print('TODO')
                        assert False # 'TODO'
                        #print >>fhw,t,
                        #for h in list(nldr.h_i()):
                        #    print >>fhw,h,
                        #print
                    else:
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
                    if not nldr.stationarity() and t==t_max:
                        print('# WARNING: NONSTATIONARITY',file=fhw)
                # Thus, we complete the simulation just printing, as it is              
                print('# tstar',t,file=fhw)
                while t < t_max:
                    t+=1
                    print(t,A,S,D,min_h,mean_h,max_h,H,file=fhw)

    def __iter__(self):
        print('TODO')


if __name__ == "__main__":
    pass




