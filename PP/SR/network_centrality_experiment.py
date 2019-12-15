# Year End Project
# program:                nonlinear-DebtRank
# @author:                hehaoran
# @Description: Define the class FinancialNetwork and the utility functions to calculate the Debt Rank
# @environment: anaconda3:python 3.7.4

## load the library
import numpy as np
import pandas as pd

# time
from dateutil.parser import parse
from datetime import datetime
# import tensorly as tl

# 
import googletrans as gg
import tushare as ts

# system
import sys
import os
import glob  # 文件操作相关模块，用它可以查找符合自己目的的文件

# plot
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# R
# r['install.packages']("NetworkRiskMeasures")
import rpy2.robjects as robjects
from rpy2.robjects import r as r
from rpy2.robjects.packages import importr
nrm = importr("NetworkRiskMeasures")
igraph = importr("igraph")

# local library
from tyrant import debtrank as dr

def GetFile(fname):
    df = pd.read_csv(fname)
    df['Filein'] = fname
    return df.set_index(['Filein'])

# initial setup
PATH = os.getcwd()

ts.set_token("4e28d8b91ed71e2c5c3e1d917fd81eeed3b70063265344bb173006e5")
pro = ts.pro_api()

## 1 getting bank-specific data for the banks
#bank_id total_assets    equity  inter_bank_assets       inter_bank_liabilities  bank_name
#1       2527465000.0    95685000.0      159769000.0     137316000.0     "HSBC Holdings Plc"
#2       2888526820.49   64294758.6352   96239646.83     260571978.577   "BNP Paribas"
data1 = pro.stock_basic(exchange='', list_status='L',
                       fields='ts_code,symbol,name,area,industry,list_date')
bank_ts_code = data1[data1["industry"] == "银行" ].copy()
bank_ts_code['name'] = [gg.Translator().translate(name).text for name in bank_ts_code['name']]
bank_ts_code['name'].loc[1277] = 'Rural Commercial Bank of Zhangjiagang'
bank_ts_code['name'].loc[1247] = 'Zhangjiagang Rural Commercial Bank'
bank_ts_code['name'].loc[1386] = 'Qingdao Rural Commercial Bank'
bank_ts_code['name'].loc[2945] = 'Wuxi Rural Commercial Bank'
bank_ts_code['name'].loc[2950] = 'Xi an Bank'
bank_ts_code['name'].loc[3016] = 'Chongqing Rural Commercial Bank'
bank_ts_code['name'].loc[3033] = 'Changshu Rural Commercial Bank'
bank_ts_code['name'].loc[3041] = 'China Industrial Bank'
bank_ts_code['name'].loc[3141] = 'Zijin Rural Commercial Bank'
bank_ts_code['name'].loc[3165] = 'China Construction Bank'
bank_ts_code['name'].loc[3357] = 'Suzhou Rural Commercial Bank'
# bank_basic_SSE = pro.stock_company(
    # exchange='SSE', fields='ts_code,exchange,reg_capital,setup_date,province,employees,business_scope')  # SSE上交所
# bank_basic_SZSE = pro.stock_company(
    # exchange='SZSE', fields='ts_code,exchange,reg_capital,setup_date,province,employees,business_scope')  # SZSE深交所

# loanto_oth_bank_fi : loaning to other banks
# loan_oth_bank : borrowing from other banks

bank_balancesheet = []
for code in bank_ts_code["ts_code"]:
    bank_balancesheet.append(pro.balancesheet(ts_code=code, start_data='20060101', end_data='20181231',
                             fields='ts_code,end_date,report_type,comp_type,total_assets,total_liab,loanto_oth_bank_fi,loan_oth_bank'))
bank_specific_data = pd.concat(bank_balancesheet, axis=0)

bank_specific_name_data = pd.merge(bank_specific_data, bank_ts_code[[
                              'ts_code', 'name', 'list_date']], how='outer', on='ts_code') # merge name and list_date
bank_specific_name_data['equity'] = bank_specific_name_data['total_assets'] - \
    bank_specific_name_data['total_liab']

# Preprocessing
bank_specific_name_data.rename(
    { 
    'loanto_oth_bank_fi': 'inter_bank_assets', 
    'loan_oth_bank': 'inter_bank_liabilities',
    'name':'bank_name'
    }, axis=1,inplace=True)
# bank_specific_name_data.to_excel('/Users/hehaoran/Desktop/Tyrant/PP/bank_specific_data.xlsx')

bank_specific_name_data['end_date'] = [
    parse(str(date)) for date in bank_specific_name_data['end_date']]
bank_specific_name_data['list_date'] = [
    parse(str(date)) for date in bank_specific_name_data['list_date']]

#bank_specific_name_data = bank_specific_name_data.loc[bank_specific_name_data['list_date'] <= datetime( # IPO before 2008.
    # ss2008, 1, 1)]
bank_specific_name_data = bank_specific_name_data[(bank_specific_name_data['end_date'] >= datetime(
    2007, 1, 1)) & (bank_specific_name_data['end_date'] > bank_specific_name_data['list_date'])]

# dropping the value of 'comp_type' and 'total_assets' is null
bank_specific_name_data = bank_specific_name_data[bank_specific_name_data['comp_type'].isnull() == False]
bank_specific_name_data = bank_specific_name_data[bank_specific_name_data['total_assets'].isnull() == False]
# fill NA value with 0.
bank_specific_name_data = bank_specific_name_data.fillna(0)

bank_specific_name_data.drop_duplicates(subset=['ts_code', 'report_type', 'comp_type',
                                                'inter_bank_assets', 'total_assets', 'inter_bank_liabilities',
                                                'total_liab', 'bank_name', 'list_date', 'equity'], keep='first', inplace=True)

bank_specific_name_data = bank_specific_name_data.set_index('end_date')
grouped_report_date = bank_specific_name_data.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day])

# bank_specific_date = [each[1] for each in grouped_report_date] # grouped object == 'tuple' so [1]
# count_bank_date = np.array([each.shape[0] for each in bank_specific_date]) # count the number of banks in every period

for name,each in grouped_report_date:
    each = each.drop_duplicates(subset='bank_name', keep='first').copy()
    each.to_csv(
        '/Users/hehaoran/Desktop/bank_specific_data/bank_specific_data%s.csv' % str(name))


## 2 getting the inter-bank assets(maximum entropy (Upper, 2004) and minimum density estimation (Anand et al, 2015))
#source  target  exposure
#1       2       18804300.1765828
#1       3       593429.0704464162
#1       4       7180905.941936611
#1       5       13568931.097857257

rscript_A_ij = """

    data <- read.csv('/Users/hehaoran/Desktop/bankdata/bank_specific_data(2010, 6, 30).csv')

    set.seed(123)
    md_mat = matrix_estimation(data$inter_bank_assets,data$inter_bank_liabilities,method="md",verbose=FALSE)
    rownames(md_mat) <- colnames(md_mat) <- data$bank_name

    return(md_mat)
"""
print(r(rscript_A_ij))

# if the axis=0,1 of a matrix are same,then just .T, else replacement
bank_mat_md20100630 = pd.DataFrame(np.array(list(r.md_mat)).reshape(14, 14).T, columns=list(
    r['row.names'](r.md_mat)), index=list(r['row.names'](r.md_mat)))
# bank_mat_md20100630.to_csv(
#     '/Users/hehaoran/Desktop/bank_lambda_(2010, 6, 30).csv')

bank_count = bank_mat_md20100630.shape[0]
inner_bank_exposure = []
for i in range(bank_count):
    for j in range(bank_count):
        if i != j:
            inner_bank_exposure.append((i + 1, j + 1, bank_mat_md20100630.values[i, j]))
inner_bank_exposure = pd.DataFrame(np.array(inner_bank_exposure), columns=['source', 'target', 'exposure'])
inner_bank_exposure.set_index('source',inplace=True)
# inner_bank_exposure.to_csv(
#    '/Users/hehaoran/Desktop/Tyrant/data/inner_bank_exposure.csv')

## 3.1 Finding central, important or systemic nodes on the network(just one)
# TODO: md_mat:directed and the weight(defult = the loan amount)

rscript_network_stat = """

    library(ggplot2)
    library(ggnetwork)
    library(igraph)

    gmd <- graph_from_adjacency_matrix(md_mat, weighted = T)

    d <- igraph::degree(gmd)
    bw <- igraph::betweenness(gmd)
    cn <- igraph::closeness(gmd)
    eigen <- igraph::eigen_centrality(gmd)$vector
    alpha <- igraph::alpha_centrality(gmd, alpha = 0.5)

    imps <- impact_susceptibility(exposures = gmd, buffer = data2$equity)
    impd <- impact_diffusion(exposures = gmd, buffer = data2$equity, weights = data2$total_assets)$total

"""
r(rscript_network_stat)

rscript_contagion = """

    contdr <- contagion(exposures = md_mat, buffer = data2$equity, weights = data2$total_assets, shock = "all", method = "debtrank", verbose = F)
    contdr_summary <- summary(contdr)
    debtrank <- contdr_summary$summary_table$additional_stress
"""
r(rscript_contagion)


network_center = pd.DataFrame([list(r.d), list(r.bw), list(r.cn), list(r.eigen), list(r.alpha), list(r.imps), list(r.impd), list(r.debtrank)], index=[
                              'degree', 'betweenness', 'closeness', 'eigen_centrality', 'alpha_centrality', 'impact_susceptibility', 'impact_diffusion', 'DebtRank'], columns=r['row.names'](r.md_mat)).T
# network_center.to_excel('/Users/hehaoran/Desktop/data/network_center.xls')

# 3.2 Finding central, important or systemic nodes on the network(ALL)
filein_bank_specific_data = [GetFile(fname).set_index('end_date') for fname in glob.glob(
    '/Users/hehaoran/Desktop/bank_specific_data/bank_specific_data*.csv')]
# bank_specific_datas = pd.concat(filein_bank_specific_data,axis=0)
# bank_specific_datas.to_csv(
#   '/Users/hehaoran/Desktop/Tyrant/data/bank_specific_datas.csv')

file_path = robjects.StrVector(glob.glob("/Users/hehaoran/Desktop/bank_specific_data/bank_specific_data*.csv"))

filein_bank_specific_rdata = [r['read.csv'](file_path[i]) for i in range(len(file_path))]
# print(filein_bank_specific_data[0][6])

network_center = []
time = [path.split('(')[1].split(')')[0] for path in file_path]

def exposures2df(r_mat, t, x, index):
    """
    @param r_mat: the r Matrix from the r method
    @param t: the period of samples
    @param x: the number of samples
    @param index: bank name
    """
    mat_exposure = np.array(list(r_mat)).reshape(x, x).T
    df = pd.DataFrame(mat_exposure, columns=index, index=index)
    df['date'] = parse(t)
    return df


for t,i in zip(time, range(len(time))):
    print("the iterator is estimating....>>>%s"%parse(t))
    n = len(filein_bank_specific_data[i])
    bank_name = filein_bank_specific_data[i].bank_name.values
    r('set.seed(123)')
    #1 estimate exposure matrix
    mat_exposure = nrm.matrix_estimation(
        filein_bank_specific_rdata[i][4], filein_bank_specific_rdata[i][6], method='md', verbose='F')
    #2 about graph
    df = exposures2df(mat_exposure, t, n, bank_name)
    gmd = igraph.graph_from_adjacency_matrix(mat_exposure, weighted='T')

    df['degree'] = list(igraph.degree(gmd))
    df['betweenness'] = list(igraph.betweenness(gmd))
    df['closeness'] = list(igraph.closeness(gmd))
    df['eigenvector centrality'] = list(igraph.eigen_centrality(gmd)[0])
    df['alpha centrality'] = list(igraph.alpha_centrality(gmd, alpha=0.5))

    buffer = robjects.FactorVector(filein_bank_specific_data[i].equity.values)
    buffer = r['as.numeric'](buffer)
    weights = robjects.FactorVector(filein_bank_specific_data[i].total_assets.values)
    weights = r['as.numeric'](weights)
    df['imps'] = list(nrm.impact_susceptibility(exposures=gmd, buffer=buffer))
    df['impd'] = list(nrm.impact_diffusion(exposures=gmd, buffer=buffer, weights=weights)[3])

    # debtrank
    # contdr = nrm.contagion(exposures=mat_exposure, buffer=buffer,weights=weights, shock="all", method="debtrank", verbose='F')
    # contdr_summary = r['summary'](contdr)
    # df['debtrank_ori'] = contdr_summary[1].rx2('original_stress')
    # df['debtrank_add'] = contdr_summary[1].rx2('additional_stress')
    # df['debtrank_defaults'] = contdr_summary[1].rx2('additional_defaults')

    network_center.append(df)
    print("finished....>>>%s/%s"%(i + 1,len(time)))
else:
    print('Please call network_center!')

network_center[0][['degree', 'betweenness','closeness', 'eigenvector centrality', 'alpha centrality', 'imps', 'impd']].rank(
    axis=0, ascending=False, method='min').applymap(lambda x: int(x)).sort_index(by='degree')

## 3.3 Finding central, important or systemic nodes on the network(DebtRank and non-Linear DebtRank)

# loading bank data
path_bank_specific_data = '/Users/hehaoran/Desktop/bankdata/bank_specific_data(2010, 6, 30).csv'
bank_data = nldr.Data(filein_bank_specific_data=path_bank_specific_data,checks=False, year='20100630', p='0.05', net='ANY')

# experiment
seed = 123
filein_parameterspace = '/Users/hehaoran/Desktop/parameter_space.txt'
#               print filein_parameterspace

#               0.1 0.2 0.3 0.4
#               0.2 0.3 0.4 0.5
#               ...
t_max = 100
numsamples = bank_data.N()
baseoutdir = '/Users/hehaoran/Desktop/nldr_se_results'
se = nldr.SmartExperimenter(bank_data, filein_parameterspace, t_max,
                    numsamples, baseoutdir, seed=seed)

for i in range(se.num_experiments()):
        se.run_ith_experiment(i)

fn = nldr.Finetwork(bank_data)
x = bank_data.getExposures()
fn.stats()
fn.draw()
