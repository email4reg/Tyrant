# Year End Project
# program:                nonlinear-DebtRank
# @author:                hehaoran
# @Description: Define the class FinancialNetwork and the utility functions to calculate the Debt Rank
# @environment: anaconda3:python 3.7.4

# File Structure:
#       1. Libraries and parameters
#       2. Definition of the class FinancialNetwork and its methods
#       3. Utility funciotns

import numpy as np
import pandas as pd

# Third party Library
import tensorly as tl
import tushare as ts


# TODO
ts.set_token("4e28d8b91ed71e2c5c3e1d917fd81eeed3b70063265344bb173006e5")
pro = ts.pro_api()

df = pro.balancesheet(ts_code='600000.SH', start_date='20100101', end_date='20181231',
                      fields='ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,cap_rese')

# SETP 1:
# creating bank-specific data
#        print '# Loading bank-specific data from',filein_bank_specific_data

#bank_id total_assets    equity  inter_bank_assets       inter_bank_liabilities  bank_name
#1       2527465000.0    95685000.0      159769000.0     137316000.0     "HSBC Holdings Plc"
#2       2888526820.49   64294758.6352   96239646.83     260571978.577   "BNP Paribas"

#        print '# Loading inter-bank assets from',filein_A_ij
#source  target  exposure
#1       2       18804300.1765828
#1       3       593429.0704464162
#1       4       7180905.941936611
#1       5       13568931.097857257
df = pd.DataFrame(data=np.array([[1, 1, 1, 1], [2, 3, 4, 5], [188, 593, 718, 135]]).T, columns=["source", "target", "exposure"])


with open("/Users/hehaoran/Desktop/systemic-risk-estimation/Equity.txt",'w') as eq:
    print(file=eq)
