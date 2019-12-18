# Year End Project
# program:                creating the specific data of banks
# @author:                hehaoran
# @Description: creating the specific data of banks via tushare
# @environment: anaconda3:python 3.7.4

## load the library
import numpy as np
import pandas as pd
import tushare as ts

#
import googletrans as gg
import os

# time
from dateutil.parser import parse
from datetime import datetime


def GetFile(fname):
    df = pd.read_csv(fname)
    df['Filein'] = fname
    return df.set_index(['Filein'])


PATH = os.getcwd()
## set your token of tushare,like:
ts.set_token("4e28d8b91ed71e2c5c3e1d917fd81eeed3b70063265344bb173006e4")
pro = ts.pro_api()

# 1 getting bank-specific data of banks, like:
#bank_id total_assets    equity  inter_bank_assets       inter_bank_liabilities  bank_name
#1       2527465000.0    95685000.0      159769000.0     137316000.0     "HSBC Holdings Plc"
#2       2888526820.49   64294758.6352   96239646.83     260571978.577   "BNP Paribas"

## get the names of banks
data1 = pro.stock_basic(exchange='', list_status='L',
                        fields='ts_code,symbol,name,area,industry,list_date')
bank_ts_code = data1[data1["industry"] == "银行"].copy()
bank_ts_code['name'] = [gg.Translator().translate(name).text for name in bank_ts_code['name']]

### rename
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

## getting the balancesheet of banks
### notes:
"""
loanto_oth_bank_fi : loaning to other banks
loan_oth_bank : borrowing from other banks
"""
bank_balancesheet = []
for code in bank_ts_code["ts_code"]:
    bank_balancesheet.append(pro.balancesheet(ts_code=code, start_data='20060101', end_data='20181231',
                             fields='ts_code,end_date,report_type,comp_type,total_assets,total_liab,loanto_oth_bank_fi,loan_oth_bank'))
bank_specific_data = pd.concat(bank_balancesheet, axis=0)

bank_specific_name_data = pd.merge(bank_specific_data, bank_ts_code[[
                              'ts_code', 'name', 'list_date']], how='outer', on='ts_code') # merge name and list_date
bank_specific_name_data['equity'] = bank_specific_name_data['total_assets'] - bank_specific_name_data['total_liab']

## Preprocessing
### rename
bank_specific_name_data.rename(
    { 
    'loanto_oth_bank_fi': 'inter_bank_assets', 
    'loan_oth_bank': 'inter_bank_liabilities',
    'name':'bank_name'
    }, axis=1,inplace=True)

### specified scope
bank_specific_name_data['end_date'] = [
    parse(str(date)) for date in bank_specific_name_data['end_date']]
bank_specific_name_data['list_date'] = [
    parse(str(date)) for date in bank_specific_name_data['list_date']]

#bank_specific_name_data = bank_specific_name_data.loc[bank_specific_name_data['list_date'] <= datetime( # IPO before 2008.
# ss2008, 1, 1)]
bank_specific_name_data = bank_specific_name_data[(bank_specific_name_data['end_date'] >= datetime(
    2007, 1, 1)) & (bank_specific_name_data['end_date'] > bank_specific_name_data['list_date'])]

### dropping the value of 'comp_type' and 'total_assets' is null
bank_specific_name_data = bank_specific_name_data[bank_specific_name_data['comp_type'].isnull() == False]
bank_specific_name_data = bank_specific_name_data[bank_specific_name_data['total_assets'].isnull() == False]
### fill NA value with 0.
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
    each.to_csv(PATH + '/bank_specific_data%s.csv' % str(name))

