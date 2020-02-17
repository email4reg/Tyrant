# Year End Project
# program:                nonlinear-DebtRank
# @author:                hehaoran
# @Description: Define the class FinancialNetwork and the utility functions to calculate the Debt Rank
# @environment: anaconda3:python 3.7.4
# pylint: disable = no-member
## load the library
import numpy as np
import pandas as pd
#
import os
import glob
from dateutil.parser import parse
from tqdm import tqdm
#
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
#
import tyrant as tt
from tyrant import debtrank as dr

def GetFile(fname):
    df = pd.read_csv(fname)
    df['Filein'] = fname
    return df.set_index(['Filein'])

# path
PATH = os.getcwd()

# data
banks_data = [GetFile(fname).set_index('end_date') for fname in glob.glob(PATH + '/res/bank_specific_data*.csv')]
file_path = glob.glob(PATH + '/res/bank_specific_data*.csv')

# 1.Evolution of Interbank debt and network
# file_path = robjects.StrVector(glob.glob(PATH + '/res/bank_specific_data*.csv'))
# filein_bank_specific_rdata = [r['read.csv'](file_path[i]) for i in range(len(file_path))]

## experiment 1: total Interbank debt
total_interbank_data = []
for each in banks_data:
    tmp = {}
    tmp['date'] = each.index[0]
    tmp['total interbank assets'] = each['inter_bank_assets'].sum() / 1e6
    tmp['total interbank liabilities'] = each['inter_bank_liabilities'].sum() / 1e6
    tmp['error'] = tmp['total interbank assets'] - tmp['total interbank liabilities']
    total_interbank_data.append(tmp)

total_interbank_data = pd.DataFrame(total_interbank_data).set_index('date')
total_interbank_data.index = [parse(i) for i in total_interbank_data.index]
total_interbank_data.sort_index(inplace=True)

## plot
sns.set_style('white')
fig, ax = plt.subplots(figsize=(9, 7))
sns.lineplot(data=total_interbank_data[['total interbank assets', 'total interbank liabilities','error']],ax=ax)
plt.tick_params(axis='both', tick1On=True)
plt.ylabel('million')
plt.legend(loc='best')
plt.show()

## network features
network_properties_dict = {}
pbar = tqdm(file_path)
for path in pbar:
    date = path.split('(')[1].split(')')[0]
    bank_data = tt.Data(path,checks=False,year=date)
    fn = tt.Finetwork(bank_data)
    network_properties_dict[date] = fn.stats()
    pbar.set_description("Processing %s" % path.split('a')[6])

network_properties = pd.DataFrame(network_properties_dict).T
network_properties.index = [parse(i) for i in network_properties.index]
network_properties.sort_index(inplace=True)

## plot
fig,ax = plt.subplots(1,2,figsize=(12,7))

ax[0].plot(network_properties['nodes'], '-', label='nodes')
ax[0].plot(network_properties['connectivity'], '-.', label='connectivity')
ax[0].tick_params(axis='x', tick1On=True)
ax[0].legend(loc='upper left')

ax[1].plot(network_properties['density'], '-', label='density')
ax[1].plot(network_properties['assortativity'], '--', label='assortativity')
ax[1].legend(loc='upper left')

ax1 = plt.twinx(ax=ax[0])
ax1.plot(network_properties['edges'], ls='--', color='r', label='edges')
ax1.legend(loc='upper right')

plt.show()


# 2 Finding central, important or systemic nodes on the network(DebtRank and non-Linear DebtRank)
## experiment 2
shock = np.arange(1, 10) / 10
alphas = [0, 1, 2, 4]
res = []
file_path.sort()
file_path = [file_path[i] for i in range(0, len(file_path), 4)]
del file_path[-1]
pbar = tqdm(file_path)
for path in pbar:
    date = parse(path.split('(')[1].split(')')[0]).strftime("%Y-%m-%d")
    data = tt.Data(filein_bank_specific_data=path, checks=True, year=date)
    for s in shock:
        N = data.N()
        h_i_shock = tt.creating_initial_shock(N, 'all', s)
        nldr = dr.NonLinearDebtRank(data)
        for alpha in alphas:
            for _ in nldr.iterator(h_i_shock=h_i_shock, alpha=alpha):
                pass
            res.append(dict(date=date, alpha=alpha,shock=s, H=nldr.H()))
        pbar.set_description("Processing %s" % path.split('a')[6])

dfs = pd.DataFrame(res)
grouped = dfs.groupby('alpha')
results = []
for _, df in grouped:
    df = df.set_index(['date','shock']).copy()
    df.sort_index(inplace=True)
    df = df.reset_index().set_index('date')
    results.append(df)

date = list({i for i in results[0].index})
date.sort()

# X,Y,Z
x = np.array(list(map(lambda x: parse(x).strftime("%Y"), np.array(date))),dtype='int')
y = shock
z = np.zeros((9,12))
df = results[3]
for i in range(9):
    for j in range(12):
        z[i,j] = df.H.values[i + j * 9]

X,Y = np.meshgrid(x,y)
Z = z

# figure
fig = plt.figure(figsize=(9,6))
ax = Axes3D(fig)

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
ax.set_xlabel('year')
ax.set_ylabel('the initial shock')
ax.set_zlabel('total relative equity loss(H)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title(r'$\alpha = %.2f$' % alphas[3])
plt.show()

## experiment 3
path_bank_specific_data = [
                           PATH + '/res/bank_specific_data(2007, 12, 31).csv',
                           PATH + '/res/bank_specific_data(2008, 12, 31).csv',
                           PATH + '/res/bank_specific_data(2009, 12, 31).csv',
                           PATH + '/res/bank_specific_data(2010, 12, 31).csv']
dates = [parse(path.split('(')[1].split(')')[0]).strftime("%Y-%m-%d") for path in path_bank_specific_data]

data = tt.Data(filein_bank_specific_data=path_bank_specific_data[3],checks=True, year=dates[3])
fn = tt.Finetwork(data)
N = data.N()
h_i_shock = tt.creating_initial_shock(N, 'all', 0.1)

fn.draw(method='nldr',alpha=4, h_i_shock=h_i_shock)
fn.draw(method='lp',x_shock=0.1, t=10)
