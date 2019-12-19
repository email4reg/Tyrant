# Year End Project
# program:                nonlinear-DebtRank
# @author:                hehaoran
# @Description: Define the class FinancialNetwork and the utility functions to calculate the Debt Rank
# @environment: anaconda3:python 3.7.4
# pylint: disable = no-member

# system
import os
import glob  # 文件操作相关模块，用它可以查找符合自己目的的文件

## load the library
import numpy as np
import pandas as pd

# plot
import networkx as nx
import matplotlib.pyplot as plt

from rpy2.robjects.packages import importr
from rpy2.robjects import r as r
from rpy2.robjects import FloatVector
import rpy2.robjects as robjects

# local packages
import tyrant as tt
from tyrant import debtrank as dr

def GetFile(fname):
    df = pd.read_csv(fname)
    df['Filein'] = fname
    return df.set_index(['Filein'])

# path
PATH = os.getcwd()

# Notes:
filein_bank_specific_data = [GetFile(fname).set_index('end_date') for fname in glob.glob(PATH + '/bank_specific_data*.csv')]

file_path = robjects.StrVector(glob.glob(PATH + 'bank_specific_data*.csv'))
filein_bank_specific_rdata = [r['read.csv'](file_path[i]) for i in range(len(file_path))]
# print(filein_bank_specific_data[0][6])
time = [path.split('(')[1].split(')')[0] for path in file_path]

# 1 Finding central, important or systemic nodes on the network(DebtRank and non-Linear DebtRank)

## loading bank data
### example 1:
path_bank_specific_data = '/Users/hehaoran/Desktop/bankdata/bank_specific_data(2010, 6, 30).csv'
h_i_shock = tt.creating_initial_shock(14,[1,2],0.05)
bank_data = tt.Data(filein_bank_specific_data=path_bank_specific_data, h_i_shock=h_i_shock,
                    checks=False, year='2010-06-30', p='0.05', net='Interbank Network')

### example 2:
path_bank_specific_data = '/Users/hehaoran/Desktop/bankdata/bank_specific_data(2010, 9, 30).csv'
bank_data = tt.Data(filein_bank_specific_data=path_bank_specific_data, 
                    checks=False, year='2010-09-30', p='0.05', net='Interbank Network')

## experiment
seed = 123
filein_parameterspace = '/Users/hehaoran/Desktop/parameter_space.txt'
#               print filein_parameterspace

#               0.1 0.2 0.3 0.4
#               0.2 0.3 0.4 0.5
#               ...
t_max = 100
numsamples = bank_data.N()
baseoutdir = '/Users/hehaoran/Desktop/nldr_se_results'
se = dr.SmartExperimenter(bank_data, filein_parameterspace, t_max,
                    numsamples, baseoutdir, seed=seed)

for i in range(se.num_experiments()):
        se.run_ith_experiment(i)

fn = tt.Finetwork(bank_data)

h_i_shock = tt.creating_initial_shock(bank_data.N(), [1, 2], 0.01)
fn.draw(method='nldr',alpha=0.01,h_i_shock=h_i_shock)
fn.draw(method='dr', h_i_shock=h_i_shock)

fn.draw(method='nldr',alpha=0.01)
fn.draw(method='dr')
fn.draw(method='dr',is_savefig=True)
fn.draw()

fn.stats()

