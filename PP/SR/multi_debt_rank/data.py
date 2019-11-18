# python 3.7.4 @hehaoran
import numpy as np
import pandas as pd

import tensorly as tl
import tushare as ts


ts.set_token("4e28d8b91ed71e2c5c3e1d917fd81eeed3b70063265344bb173006e5")
pro = ts.pro_api()

df = pro.balancesheet(ts_code='600000.SH', start_date='20100101', end_date='20181231',
                      fields='ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,cap_rese')


