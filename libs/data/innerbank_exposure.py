# Year End Project
# program:                creating the interbank exposure
# @author:                hehaoran
# @Description: creating the specific data of banks via bank_specific_data and rpy2
# @environment: anaconda3:python 3.7.4

## load the library
import numpy as np
import pandas as pd

#
import os

# R
import rpy2.robjects as robjects
from rpy2.robjects import r as r
from rpy2.robjects.packages import importr
nrm = importr("NetworkRiskMeasures")

PATH = os.getcwd()

# getting the interbank assets(maximum entropy (Upper, 2004) and minimum density estimation (Anand et al, 2015)), like:
#source  target  exposure
#1       2       18804300.1765828
#1       3       593429.0704464162
#1       4       7180905.941936611
#1       5       13568931.097857257

path_bank_specific_data = PATH + '/bank_specific_data(2010, 6, 30).csv'
rscript_A_ij = """

    data <- read.csv(%r)

    set.seed(123)
    md_mat = matrix_estimation(data$inter_bank_assets,data$inter_bank_liabilities,method="md",verbose=FALSE)
    rownames(md_mat) <- colnames(md_mat) <- data$bank_name

    return(md_mat)
""" % path_bank_specific_data

print(r(rscript_A_ij))

# if the axis=0,1 of a matrix are same,then just .T, else replacement
bank_mat_md20100630 = pd.DataFrame(np.array(list(r.md_mat)).reshape(14, 14).T, columns=list(
    r['row.names'](r.md_mat)), index=list(r['row.names'](r.md_mat)))
bank_mat_md20100630.to_csv(PATH + '/bank_lambda_(2010, 6, 30).csv')

bank_count = bank_mat_md20100630.shape[0]
inner_bank_exposure = []
for i in range(bank_count):
    for j in range(bank_count):
        if i != j:
            inner_bank_exposure.append(
                (i + 1, j + 1, bank_mat_md20100630.values[i, j]))
inner_bank_exposure = pd.DataFrame(np.array(inner_bank_exposure), columns=[
                                   'source', 'target', 'exposure'])
inner_bank_exposure.set_index('source', inplace=True)
inner_bank_exposure.to_csv(PATH + '/inner_bank_exposure.csv')
