# Year End Project
# program:                nonlinear-DebtRank
# @author:                hehaoran
# @Description: Define the class FinancialNetwork and the utility functions to calculate the Debt Rank
# @environment: anaconda3:python 3.7.4


# File Structure:
#       1. Libraries and parameters
#       2. Definition of the class FinancialNetwork and its methods
#       3. Utility funciotns

## loading library
import numpy as np
import pandas as pd

# time
from dateutil.parser import parse
from datetime import datetime

#
import googletrans as gg
import networkx as nx
import tushare as ts
# import tensorly as tl

# R
# pylint: disable=no-member
# r['install.packages']("NetworkRiskMeasures")
import rpy2.robjects as robjects
from rpy2.robjects import r as r
from rpy2.robjects.packages import importr
nrm = importr("NetworkRiskMeasures")


# initial setup
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
bank_specific_name_data = bank_specific_name_data.loc[bank_specific_name_data['end_date'] >= datetime(  # report after 2007.
    2007, 1, 1)]

# dropping the value of 'comp_type' and 'total_assets' is null
bank_specific_name_data = bank_specific_name_data[bank_specific_name_data['comp_type'].isnull() == False]
bank_specific_name_data = bank_specific_name_data[bank_specific_name_data['total_assets'].isnull() == False]
# fill NA value with 0.
bank_specific_name_data = bank_specific_name_data.fillna(0)

bank_specific_name_data.drop_duplicates(subset=['ts_code', 'report_type', 'comp_type',
                                                'inter_bank_assets', 'total_assets', 'inter_bank_liabilities',
                                                'total_liab', 'bank_name', 'list_date', 'equity'], keep='first', inplace=True)

bank_specific_name_data = bank_specific_name_data.set_index('end_date')
grouped_report_date = bank_specific_name_data.groupby([
    lambda x: x.year, lambda x: x.month, lambda x: x.day])

# bank_specific_date = [each[1] for each in grouped_report_date] # grouped object == 'tuple' so [1]
# count_bank_date = np.array([each.shape[0] for each in bank_specific_date]) # count the number of banks in every period

for name,each in grouped_report_date:
    each.to_csv(
        '/Users/hehaoran/Desktop/data/bank_specific_date_{}.csv'.format(str(name)))


## 2 getting the inter-bank assets(maximum entropy (Upper, 2004) and minimum density estimation (Anand et al, 2015))
#source  target  exposure
#1       2       18804300.1765828
#1       3       593429.0704464162
#1       4       7180905.941936611
#1       5       13568931.097857257

rscript_A_ij = """

    set.seed(123)

    data2 <- read.csv('/Users/hehaoran/Desktop/data/bank_specific_date_(2010, 6, 30).csv')
    md_mat = matrix_estimation(data2$inter_bank_assets, data2$inter_bank_liabilities, method="md",verbose = F)
    rownames(md_mat) <- colnames(md_mat) <- data2$bank_name

    return(md_mat)
"""
r(rscript_A_ij)

# if the axis=0,1 of a matrix are same,then just .T, else replacement
# bank_mat_md20100630 = pd.DataFrame(np.array(list(r.md_mat)).reshape(19, 19).T, columns=list(
    # r['row.names'](r.md_mat)), index=list(r['row.names'](r.md_mat)))
# bank_mat_md20100630.to_excel(
    # '/Users/hehaoran/Desktop/data/bank_lambda_(2010, 6, 30).xls')

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
    
    print(matrix(d, bw, cn, eigen, alpha, ncol=5))
"""
r(rscript_network_stat)

