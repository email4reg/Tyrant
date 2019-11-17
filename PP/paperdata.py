# python 3.7.4 @hehaoran
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.parser import parse
import glob  # 文件操作相关模块，用它可以查找符合自己目的的文件
import math
from scipy.stats import chi2, t, f, norm
import tushare as ts # 获取股票信息

# test
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split  # 选取训练集和测试集
from sklearn.svm import l1_min_c
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

# Part 1: 整理数据
# step1: 统计上市和退市公司数量，计算公司生存时间，得到数据da1
# 加载2006-2017年所有上市公司
data1 = pd.read_excel(
    '/Users/hhr/Desktop/Projects/pydata/stk20062018.xls', skiprows=(1, 2))

data1['Symbol'].duplicated().any()  # 检查有没有重复项
data1 = data1.loc[data1['StatusID'].values !=
                  'P0813', :]  # 查找发行失败的公司，并剔除，not也可以
data1 = data1.set_index('Symbol')
data1.index = data1.index.map(lambda x: str(x))
data1[['ListedDate', 'DelistedDate']] = data1[['ListedDate', 'DelistedDate']
                                              ].applymap(lambda x: parse(x) if pd.isnull(x) == False else None)
data1.head()

# 统计上市和退市公司的数量
data1 = data1.reset_index().set_index('ListedDate')
grouped_listed = data1.groupby(lambda x: x.year).size()  # 2006-2017年上市公司数量

data1 = data1.loc[data1.DelistedDate.isnull() == 0]
data1 = data1.reset_index().set_index('DelistedDate')
grouped_delisted = data1.groupby(lambda x: x.year).size()  # 2006-2017年退市公司数量

listed_delisted = pd.concat([grouped_listed, grouped_delisted], axis=1).fillna(
    0).applymap(lambda x: int(x))
listed_delisted.columns = [
    'The number of IPO firms per year', 'The number of delisted firms per year']

# 计算公司生存期数据
data1['duration'] = data1.DelistedDate - data1.ListedDate  # 生成存续期数据da
data1['duration'] = data1.duration.fillna(0)  # 若没有退市则记为0
# da1 = data1[data1.ListedDate >= pd.datetime(2015, 1, 1)] # 选取样本区间为上市时间：2015以后
# 退市公司的生存时间分布
# data1.duration.map(lambda x: int(x.days))
time = data1[data1.duration.dt.days != 0]['duration'].dt.days

# 作图: 上市与退市公司趋势和生存时间分布图
plt.rcParams['font.family'] = ['Arial Unicode MS']

sns.set_style('ticks')
fig, axes = plt.subplots(1, 2, figsize=(11, 6))

sns.lineplot(data=listed_delisted.iloc[:-1, ], linewidth=2, ax=axes[0])  # 很明显，2016为拐点
sns.distplot(time.values, ax=axes[1])
sns.despine()

axes[0].set_title('The Number of New Listing and Delisting Firms')
axes[1].set_title('The Distribution of the Survival Time of Listed Firms')
axes[0].legend(fontsize=8)
# axes[0].grid(axis='both')
# axes[1].grid(axis='both')

for i, each in enumerate(time.describe().index, start=1):
    axes[1].text(3000, 0.0012 - i*0.5e-4, str(each) +
                 ': %d' % time.describe().loc[each], fontsize=9)
plt.show()

# step2: 与data1合并得到da2
# 加载wind里面的数据，包括了2015-2017年股东户数、所有权结构、流通股数
data2 = pd.read_excel('/Users/hhr/Desktop/Projects/pydata/stkwind.xls')
data2 = data2.iloc[1:, :].drop(['ShortName', 'CompangyName'], axis=1)
data2.Symbol = data2.Symbol.map(lambda x: x.strip('.OC'))
data2 = data2.iloc[:, :-6]  # 暂时不考虑每股股利
data2 = data2.drop_duplicates(subset='Symbol', keep='last').set_index('Symbol')
data2.head()

# 注册资本,来自公司基本信息表
data_size = pd.read_excel('/Users/hhr/Desktop/Projects/pydata/stkindex.xls')
data_size = data_size.iloc[1:, ]  # 去除第一行
data_size.columns = data_size.columns.map(lambda x: x.title())  # 大小写转化
data_size = data_size.set_index('Symbol')
data_size.index = data_size.index.map(lambda x: x.replace(' ', ''))

data2 = pd.concat([data2, data_size.Registeredcapital], axis=1, join='inner')
da2 = pd.merge(data1, data2, left_index=True, right_index=True, how='inner')
da2.rename({'Registeredcapital': 'Size',
            'Num_sholder2015': 'NumSholder2015',
            'Num_sholder2016': 'NumSholder2016',
            'Num_sholder2017': 'NumSholder2017'}, axis=1, inplace=True)
# step3: 对交易数据预处理、计算流动性和波动性指标


def GetFile(fname):
    '''定义文件读取方法'''
    df = pd.read_excel(fname, skiprows=(1, 2))
    df['File'] = fname
    return df.set_index(['File'])


def loadtrans():
    '''加载2015-2017年所有日交易数据'''
    transrecords = list()
    filename = ['stktrans20152016', 'stktrans2017']
    for each in filename:
        trans = [GetFile(fname) for fname in glob.glob(
            '/Users/hhr/Desktop/Projects/pydata/%s/SQS_Quotation*.xls' % each)]
        transrecords.append(trans)
        print('loaded: %s' % each)
    tran1 = pd.concat(transrecords[0])
    tran2 = pd.concat(transrecords[1])
    return pd.concat([tran1, tran2])


data3 = loadtrans()

data3 = data3.set_index('Symbol').drop('MktLayerCode', axis=1)
data3.index = data3.index.map(lambda x: str(x))
data3.head()

# 交易数据预处理


def preprocessing():
    '''剔除交易量为0或空的交易记录;
    交易类型编码为空的也剔除;
    交易日期转为datetime;
    '''
    for each in ['ClosePrice', 'PreClosePrice', 'Volume', 'TurnoverRate1']:
        global data3
        data3 = data3[data3[each].isnull() == 0]
        data3 = data3[data3[each] != 0]
        print(data3.shape[0])
    data3['TradingDate'] = data3.TradingDate.map(lambda x: parse(x))
    return data3[data3.TransferTypecode.isnull() == 0]


data3 = preprocessing()

# 计算2015-2917年公司换手率、波动性以及 Amihud指数
data3 = data3.reset_index().set_index('TradingDate')
# data3['return_rate'] = (
#     data3.ClosePrice - data3.PreClosePrice) / data3.PreClosePrice  # 计算日收益率

data3['return_rate'] = (data3.ClosePrice / data3.PreClosePrice).map(lambda x: math.log(x))
data3['Amihud'] = [math.sqrt(i) for i in np.divide(abs(
    data3.return_rate.values) * 10**6, data3.Volume.values)]  # 计算Amihud指数

grouped_rate = data3.groupby(['Symbol', lambda x: x.year])  # 按公司、年份分组
turnover = grouped_rate['TurnoverRate1'].mean()
turnover = turnover.unstack()  # 堆叠(serirs-stack)，unstack()不要堆叠,即表格
turnover.columns = ['TurnoverRate2015', 'TurnoverRate2016', 'TurnoverRate2017']

Amihud = grouped_rate['Amihud'].mean().unstack()  # 计算2015-2017年日均Amihud指数
Amihud.columns = ['Amihud2015', 'Amihud2016', 'Amihud2017']

volume = grouped_rate['Volume'].mean().unstack()  # 计算2015-2017年日均交易量
volume.columns = ['Volume2015', 'Volume2016', 'Volume2017']

price = grouped_rate['ClosePrice'].mean().unstack()  # 计算2015-2017年日均Amihud指数
price = price.applymap(lambda x: math.log(x))
price.columns = ['LnPrice2015', 'LnPrice2016', 'LnPrice2017']

volatility = grouped_rate['return_rate'].apply(lambda x: math.log(
    x.var()**0.5 + 1) if len(x) > 1 else 0).unstack()  # 计算波动性，只有一笔交易的记为0
volatility.columns = ['volatility2015', 'volatility2016', 'volatility2017']

# 接上，生成2015-2017年交易方式变量: 若公司在该年未变化交易方式则编码为1个，否则为2个
TransPattern = grouped_rate['TransferTypecode'].unique().unstack()
TransPattern.columns = ['Transpattern2015',
                        'Transpattern2016', 'Transpattern2017']

# 合并上述结果:turnover、Amihud、volatility、TransPattern,得到da3
for index, each in enumerate([turnover, Amihud, volatility, TransPattern, volume, price], start=1):
    da2 = pd.merge(da2, each, left_index=True, right_index=True, how='inner')
    print('merged: %d' % index)
da3 = da2

# step4:计算资产负债率、公司业绩、市场可见度、市场占有率、信息不对称等指标
# 加载2015-2017年公司的财务数据


def loadstatements():
    balancereturns = list()
    filename = ['stkbalance20152017',
                'stkreturn20152017', 'stkcashflow20152017']
    for each in filename:
        chunks = [GetFile(fname) for fname in glob.glob(
            '/Users/hhr/Desktop/Projects/pydata/%s/STK_SQS*.xls' % each)]
        balancereturns.append(chunks)
        print('loaded: %s' % each)
    return balancereturns


data4 = loadstatements()

balances = pd.concat(data4[0]).set_index('Symbol')  # 2015-2017年资产负债表
returns = pd.concat(data4[1]).set_index('Symbol')  # 2015-2017年利润表
cashflows = pd.concat(data4[2]).set_index('Symbol')  # 2015-2017年现金流量表

# 财务数据预处理: 确定报表类型、重命名、统一截止日期，再合并为新的data4
dicts = {'A001218': 'invisible_asset',
         'A001': 'asset',
         'A002': 'liability',
         'A003': 'owner_equity',
         'B001101': 'income',
         'B001209': 'sale_enpense',
         'B002': 'clean_profit',
         'B003': 'Eps',
         'C001': 'opnetcash',
         'C002006': 'capital_expense'
         }


def preprocessing2(params=['balances', 'returns', 'cashflows']):
    '''处理财务报表'''
    statements = list()
    fnames = [balances, returns, cashflows]
    for index, fname in enumerate(fnames):
        fname.rename(dicts, axis=1, inplace=True)
        fname = fname[fname.StateTypeCode ==
                      'A'].drop('StateTypeCode', axis=1)
        fname.EndDate = fname.EndDate.map(lambda x: parse(x))
        fname = fname.reset_index().set_index('EndDate')
        fname = fname[fname.index.month == 12].reset_index().set_index(
            ['Symbol', 'EndDate']).unstack('EndDate')
        statements.append(fname)
        print('processed: %s' % params[index])
    return pd.concat(statements, axis=1, join='inner')


data4 = preprocessing2()

# def modindex():
#     '''重命名列索引'''
#     statementindex = list()
#     for i in range(10):
#         for j in range(3):
#             statementindex.append(
#                 data4.columns.levels[0][i] + str(data4.columns.levels[1][j].year))
#     return statementindex


def genrateindex():
    '''generate列索引'''
    statementindex = list()
    for each in dicts.values():
        for year in range(2015, 2018):
            statementindex.append(each + str(year))
    return statementindex


data4.columns = genrateindex()

# 加载2015-2017年公司的公司分类数据
markets = [GetFile(fname) for fname in glob.glob(
    '/Users/hhr/Desktop/Projects/pydata/stkmarket20152017/SQS_IndClassific*.xls')]
data_market = pd.concat(markets)
data_market = data_market.drop('StatsDate', axis=1)
data_market.ShortName = [i.replace(' ', '')
                         for i in data_market.ShortName]  # 删除字符串中的所有空格
data_market = data_market.set_index('Symbol')
data_market = data_market.reset_index().drop_duplicates(
    subset='Symbol', keep='first').set_index('Symbol')
data_market.head()

# 合并data_market到data4
data4 = pd.concat([data4, data_market.Level1Code], axis=1, join='inner')
# 合并data4和da3，得到da4
data4.index = data4.index.map(lambda x: str(x))
da4 = pd.merge(da3, data4, left_index=True, right_index=True)

# 计算各指标
# # 市场占有率
# da4['MarketRecognition2015'] = da4.sale_enpense2015 / da4.income2015
# da4['MarketRecognition2016'] = da4.sale_enpense2016 / da4.income2016
# da4['MarketRecognition2017'] = da4.sale_enpense2017 / da4.income2017
# 市场占有率
grouped_market = da4.groupby('Level1Code')
da4[['MarketShares2015', 'MarketShares2016', 'MarketShares2017']] = \
    grouped_market[['income2015', 'income2016', 'income2017']].apply(
        lambda x: x / x.sum())  # 计算份额(数值偏小)
# 信息不对称
da4['Intangibility2015'] = da4.invisible_asset2015 / da4.asset2015
da4['Intangibility2016'] = da4.invisible_asset2016 / da4.asset2016
da4['Intangibility2017'] = da4.invisible_asset2017 / da4.asset2017
# 自由现金流
da4['FCF2015'] = da4.opnetcash2015 / da4.capital_expense2015
da4['FCF2016'] = da4.opnetcash2016 / da4.capital_expense2016
da4['FCF2017'] = da4.opnetcash2017 / da4.capital_expense2017
# 公司业绩
da4['Roe2015'] = da4.clean_profit2015 / da4.owner_equity2015
da4['Roe2016'] = da4.clean_profit2016 / da4.owner_equity2016
da4['Roe2017'] = da4.clean_profit2017 / da4.owner_equity2017
# 每股净资产
for each in ['Cir_stock2015', 'Cir_stock2016', 'Cir_stock2017']:
    da4 = da4[da4[each] != 0]
    print(da4.shape)

da4['Naps2015'] = da4.owner_equity2015 / da4.Cir_stock2015
da4['Naps2016'] = da4.owner_equity2016 / da4.Cir_stock2016
da4['Naps2017'] = da4.owner_equity2017 / da4.Cir_stock2017
# 资产负债率
da4['Level2015'] = da4.liability2015 / da4.asset2015
da4['Level2016'] = da4.liability2016 / da4.asset2016
da4['Level2017'] = da4.liability2017 / da4.asset2017

# 加入做市商数量
# data_maker = pd.read_excel(
#     '/Users/hhr/Desktop/Projects/pydata/marketmaker.xls')
# data_maker = data_maker.set_index('Symbol')
# data_maker.index = data_maker.index.map(lambda x: x.strip('.OC'))
# data_maker = data_maker.drop('ShortName',axis=1)
# data_maker.index = data_maker.index.map(lambda x: int(x))

# da5 = pd.concat([da4,data_maker],axis=1,join='inner')

# 导出最终的数据表格，删除了含有inf的样本
# 剔除金融类和发行失败的公司
da4 = da4[(da4.StatusID != 'P0805') & (
    da4.Level1Code != 16)].copy()  # 2783个样本
da4.to_excel('/Users/hhr/Desktop/Projects/pydata/otcdata_best_maker.xls')


# Part 2: 选取指标，并进行描述性分析
# 加载数据
data = pd.read_excel('/Users/hhr/Desktop/Projects/pydata/otcdata_best.xls')

# 转变为层次索引
class1 = [each.split('2')[0] for each in data.columns]
class2 = data.columns
mulindex = pd.MultiIndex.from_arrays([class1, class2])
data.columns = mulindex
# 选取指标
da = data[['Symbol', 'ShortName', 'ListedDate',
           'DelistedDate', 'Amihud', 'LnPrice',
           'TurnoverRate', 'Volume', 'Intangibility',
           'Top1', 'Roe', 'Eps','FCF',
           'volatility', 'Cir_stock',
           'Naps', 'NumSholder', 'Level',
           'Transpattern','StatusID']].copy()
# 对Size、NumSholder取自然对数;避免链式赋值,即[][]
da['Size'] = da['Size'].applymap(lambda x: math.log(x))
da['NumSholder'] = da['NumSholder'].applymap(lambda x: math.log(x))

# setp 1
# 描述性分析和退市与正常公司样本之间的t检验(注:剔除了FCF无效的样本,31个)
da1 = da[['Symbol', 'MarketRecognition', 'MarketShares', 'Amihud', 'TurnoverRate', 'AsyInformation',
          'Top1', 'FCF', 'Roe', 'Eps', 'volatility', 'Naps', 'Size', 'NumSholder', 'Level', 'StatusID']].copy()
da1.columns = da1.columns.droplevel()  # 删除level=0轴,上下，左右(0,1)
da1 = da1.set_index('Symbol')

# 分组处理,获得退市和上市两个独立样本
grouped1 = dict(list(da1.groupby('StatusID')))
label1 = grouped1['P0801']
label2 = grouped1['P0802']

# 统计两个样本的样本量、均值、标准差
controlfirms = label1.describe().T[['count', 'mean', 'std']]
delistingfirms = label2.describe().T[['count', 'mean', 'std']]
stats = pd.merge(delistingfirms, controlfirms, left_index=True,
                 right_index=True, suffixes=('_delisting', '_control'))

# 修改显示格式
stats = stats.round(3)
stats[['count_delisting', 'count_control']
      ] = stats[['count_delisting', 'count_control']].applymap(lambda x: int(x))


def ftest(fvalue, df1, df2, p=0.05):
    '''双侧F检验'''
    fl = f.isf(1-p/2, dfn=df1, dfd=df2)
    fr = f.isf(p/2, dfn=df1, dfd=df2)
    if fvalue < fl or fvalue > fr:
        return 0
    else:
        return 1


def t2test(stats):
    '''两个独立样本的均值T检验'''
    results = dict()
    for each in stats.index:
        mean1 = stats.loc[each]['mean_delisting']
        mean2 = stats.loc[each]['mean_control']
        var1 = stats.loc[each]['std_delisting']**2
        var2 = stats.loc[each]['std_control']**2
        num1 = stats.loc[each]['count_delisting']
        num2 = stats.loc[each]['count_control']
        f_value = var1 / var2
        df1 = num1-1
        df2 = num2-1
        if ftest(f_value, df1=df1, df2=df2) == 0:
            k = (var1/num1) / ((var1/num1) + (var2/num2))
            df = (k**2/df1 + (1-k)**2/df2)**-1
            t_value = (mean1-mean2) / ((var1/num1)+(var2/num2))**0.5
            p_valuet = (1-t.cdf(abs(t_value), df=df))*2
        else:
            df = num1+num2-2
            var = ((num1-1)*var1 + (num2-1)*var2) / (num1+num2-2)
            t_value = (mean1 - mean2) / ((var/num1) + (var/num2))**0.5
            p_valuet = (1-t.cdf(abs(t_value), df=df))*2
        results[each] = dict(df=df, t=t_value, p=p_valuet)
    return pd.DataFrame(results).T


tstats = t2test(stats).round(4)

# 合并上述结果
tstats_des = pd.concat([stats, tstats], axis=1, join='inner')
tstats_des.to_excel('/Users/hhr/Desktop/Projects/pydata/temp.xls')

# step 2
# 运用PSM匹配样本
da2 = da[['Symbol', 'ListedDate', 'DelistedDate', 
        'Transpattern', 'Amihud', 'TurnoverRate', 
        'Volume','Intangibility', 'Top1', 'Roe', 
        'Eps', 'volatility', 'Naps', 'Cir_stock',
        'NumSholder', 'Level', 'StatusID']].copy()
da2.columns = da2.columns.droplevel()
da2.head()

# 选取2015年6月之前上市的公司
da2 = da2[(da2.ListedDate.dt.year < 2016) & (da2.ListedDate.dt.month <= 6)].copy()
# 选取2016年6月之前上市的公司
da2 = da2[(da2.ListedDate.dt.year < 2017) & (da2.ListedDate.dt.month <= 6)].copy()

# 剔除2015和2016年Transpattern都为空的样本
da2 = da2[~da2.Transpattern2015.isnull() | ~da2.Transpattern2016.isnull()].copy()
da2[['Transpattern2015', 'Transpattern2016', 'Transpattern2017']] = da2[
    ['Transpattern2015', 'Transpattern2016', 'Transpattern2017']].fillna(-1)
da2 = da2.set_index('Symbol').reset_index().copy()

# 剔除2016和2017年Transpattern都为空的样本
da2 = da2[~da2.Transpattern2016.isnull() | ~da2.Transpattern2017.isnull()].copy()
da2[['Transpattern2015', 'Transpattern2016', 'Transpattern2017']] = da2[
    ['Transpattern2015', 'Transpattern2016', 'Transpattern2017']].fillna(-1)
da2 = da2.set_index('Symbol').reset_index().copy()

# 生成交易方式虚拟变量
def pattern():
    results = dict()
    for index, (i, j) in enumerate(zip(da2.Transpattern2016.values, da2.Transpattern2017.values)):
        if i == -1:
            if isinstance(j, str):
                if 'S9002' in j:
                    results[index] = 1
                else:
                    results[index] = 0
        elif isinstance(i, str):
            if 'S9002' in i:
                results[index] = -1
            else:
                if j == -1:
                    results[index] = 0
                elif 'S9002' in j:
                    results[index] = 1
                else:
                    results[index] = 0
    return pd.Series(results)

da2['pattern'] = pattern()
da2.drop(['Transpattern2015', 'Transpattern2016', 'ListedDate', 'DelistedDate',
          'Transpattern2017'], axis=1, inplace=True)
da2 = da2.loc[:, ~da2.columns.str.contains('2015')].copy()

# 2015-2016控制组和实验组
control_data2 = da2[da2.pattern == 0].copy()  # 201:34=235 / 675
exp_data2 = da2[da2.pattern == 1].copy()  # 68:12=80 / 64

# 处理缺漏值
missing_values = ['TurnoverRate2017', 'TurnoverRate2016', 'volatility2017', 'volatility2016']
control_data2[missing_values] = control_data2[missing_values].fillna(0)

control_data2 = control_data2[(control_data2['Amihud2017'] != 0) & (
    control_data2['Amihud2016'] != 0)]
control_data2['Amihud2017'].fillna(
    control_data2['Amihud2017'].max() + control_data2['Amihud2017'].var()**0.5, inplace = True)  # 取平均值
control_data2['Amihud2016'].fillna(
    control_data2['Amihud2016'].max() + control_data2['Amihud2016'].var()**0.5, inplace = True)  # 取平均值

for each in control_data2.columns:
    control_data2 = control_data2[control_data2[each].isnull() == 0]
    print(control_data2.shape[0])

exp_data2[missing_values] = exp_data2[missing_values].fillna(0)

exp_data2 = exp_data2[(exp_data2['Amihud2017'] != 0) & (
    exp_data2['Amihud2016'] != 0)]
exp_data2['Amihud2017'].fillna(
    exp_data2['Amihud2017'].max() + exp_data2['Amihud2017'].var()**0.5, inplace=True)  # 取平均值
exp_data2['Amihud2016'].fillna(
    exp_data2['Amihud2016'].max() + exp_data2['Amihud2016'].var()**0.5, inplace=True)  # 取平均值

for each in exp_data2.columns:
    exp_data2 = exp_data2[exp_data2[each].isnull() == 0]
    print(exp_data2.shape[0])

# 导出2015-2016数据
control_data2.to_excel('/Users/hhr/Desktop/Projects/pydata/control_data2_nofillna.xls') # 105 / 
exp_data2.to_excel('/Users/hhr/Desktop/Projects/pydata/exp_data2_nofillna.xls') # 46 / 


# 导入2015-2016年的控制组和实验组

control_data1 = pd.read_excel(
    '/Users/hhr/Desktop/Projects/pydata/control_data1_nofillna.xls')  # 105 /
exp_data1 = pd.read_excel(
    '/Users/hhr/Desktop/Projects/pydata/exp_data1_nofillna.xls')  # 46 /
da201516 = pd.concat([control_data1, exp_data1])

control_data2 = pd.read_excel(
    '/Users/hhr/Desktop/Projects/pydata/control_data2_nofillna.xls')  # 105 /
exp_data2 = pd.read_excel(
    '/Users/hhr/Desktop/Projects/pydata/exp_data2_nofillna.xls')  # 46 /
da201617 = pd.concat([control_data2, exp_data2])

# 2015年-2016年
da2015 = da201516.loc[:, da201516.columns.str.contains('2015')].copy()  # 取包含‘2015’的所有变量
da2015.columns = [each[:-4] for each in da2015.columns]
da2015['time'] = 2015
da2015['unit'] = da201516['Symbol']
da2015['tr'] = da201516.pattern

da2016 = da201516.loc[:, da201516.columns.str.contains('2016')].copy()  # 取包含‘2016’的所有变量
da2016.columns = [each[:-4] for each in da2016.columns]
da2016['time'] = 2016
da2016['unit'] = da201516['Symbol']
da2016['tr'] = da201516.pattern

# 2016年-2017年
da2016 = da201617.loc[:, da201617.columns.str.contains('2016')].copy()  # 取包含‘2016’的所有变量
da2016.columns = [each[:-4] for each in da2016.columns]
da2016['time'] = 2016
da2016['unit'] = da201617['Symbol']
da2016['tr'] = da201617.pattern

da2017 = da201617.loc[:, da201617.columns.str.contains('2017')].copy()  # 取包含‘2016’的所有变量
da2017.columns = [each[:-4] for each in da2017.columns]
da2017['time'] = 2017
da2017['unit'] = da201617['Symbol']
da2017['tr'] = da201617.pattern

data_did = pd.concat([da2016, da2017], axis=0, ignore_index=True)
data_did.to_excel('/Users/hhr/Desktop/Projects/pydata/did201617_nofillna.xls')

# pannel logit STATA
da.columns = da.columns.droplevel()
da.head()

da3 = da[da.DelistedDate.dt.year == 2017].copy() # 101个样本
da3.drop(['Transpattern2015', 'Transpattern2016', 'Transpattern2017', 
        'ListedDate', 'DelistedDate', 'FCF2015', 'FCF2016','TurnoverRate2015', 
        'TurnoverRate2016', 'TurnoverRate2017','Size', 'ShortName'], axis=1, inplace=True)

# 只选取2015-2016年的数据
da3 = da3.loc[:, ~da3.columns.str.contains('2017')].copy()

da3.drop(['ListedDate', 'DelistedDate'], axis=1, inplace=True)
# 处理缺漏值
da3[['Amihud2015', 'Amihud2016']] = da3[['Amihud2015', 'Amihud2016']].fillna(
    da3[['Amihud2015', 'Amihud2016']].max()+1)
da3[['volatility2015', 'volatility2016']] = da3[[
    'volatility2015', 'volatility2016']].fillna(0)

for each in da3.columns:
    da3 = da3[da3[each].isnull() == 0]
    print(da3.shape[0]) # 剩余60个样本

# 将year数据变成虚拟变量形式
mixed2015 = da3.loc[:, da3.columns.str.contains('2015')].copy()
mixed2015.columns = [each[:-4] for each in mixed2015.columns]

# logit 2018年退市公司
data = pd.read_excel(
    '/Users/hhr/Desktop/Projects/pydata/logit2018delisted1.xls')
data = data.set_index('Symbol')

# 2015年数据,70个样本
da2015 = data.loc[:, data.columns.str.contains('2015')].copy()
da2015.columns = [each[:-4] for each in da2015.columns]
da2015['label'] = 0
da2015['year'] = 2015
for each in da2015.columns:
    da2015 = da2015[da2015[each].isnull() == 0]
    print(da2015.shape[0])

# 2016年数据,109个样本
da2016 = data.loc[:, data.columns.str.contains('2016')].copy()
da2016.columns = [each[:-4] for each in da2016.columns]
da2016['label'] = 0
da2016['year'] = 2016
for each in da2016.columns:
    da2016 = da2016[da2016[each].isnull() == 0]
    print(da2016.shape[0])

# 2017年数据,37个样本
da2017 = data.loc[:, data.columns.str.contains('2017')].copy()
da2017.columns = [each[:-4] for each in da2017.columns]
da2017['label'] = 1
da2017['year'] = 2017
for each in da2017.columns:
    da2017 = da2017[da2017[each].isnull() == 0]
    print(da2017.shape[0])
da = pd.concat([da2015, da2016, da2017])
da.to_excel('/Users/hhr/Desktop/Projects/pydata/logit2018delisted2.xls')


# logit 2017年退市公司
data = pd.read_excel(
    '/Users/hhr/Desktop/Projects/pydata/logit2017delisted.xls')
data = data.set_index('Symbol')

# 2015年数据,41个样本
da2015 = data.loc[:, data.columns.str.contains('2015')].copy()
da2015.columns = [each[:-4] for each in da2015.columns]
da2015['label'] = 0
da2015['year'] = 2015
for each in da2015.columns:
    da2015 = da2015[da2015[each].isnull() == 0]
    print(da2015.shape[0])

# 2016年数据,33个样本
da2016 = data.loc[:, data.columns.str.contains('2016')].copy()
da2016.columns = [each[:-4] for each in da2016.columns]
da2016['label'] = 1
da2016['year'] = 2016
for each in da2016.columns:
    da2016 = da2016[da2016[each].isnull() == 0]
    print(da2016.shape[0])

da = pd.concat([da2015, da2016])
da.to_excel('/Users/hhr/Desktop/Projects/pydata/logit2017delisted2.xls')

# cox比例风险模型
data = pd.read_excel('/Users/hhr/Desktop/Projects/pydata/cox.xls')
# 转变为层次索引
class1 = [each.split('2')[0] for each in data.columns]
class2 = data.columns
mulindex = pd.MultiIndex.from_arrays([class1, class2])
data.columns = mulindex
# 选取指标
da = data[['Symbol', 'ShortName', 'ListedDate',
           'DelistedDate', 'Size', 'duration','Amihud',
           'TurnoverRate', 'Intangibility','Volume',
           'Top1', 'Roe', 'Eps','LnPrice',
           'volatility', 'Cir_stock',
           'Naps', 'NumSholder', 'Level',
            'Transpattern']].copy()
da.columns = da.columns.droplevel()  # 删除level=0轴,上下，左右(0,1)

# 选取2016年退市公司数据
delisted2016 = da[da['DelistedDate'].dt.year == 2016]
delisted2016 = delisted2016.loc[:, ~delisted2016.columns.str.contains('20') | da.columns.str.contains(
    '2015')].copy()
delisted2016.columns = [each.split('2')[0] for each in delisted2016.columns]

# 处理缺漏值
# delisted2016['TurnoverRate'] = delisted2016['TurnoverRate'].fillna(0)
# delisted2016['volatility'] = delisted2016['volatility'].fillna(0)


# delisted2016 = delisted2016[delisted2016['Amihud'] != 0]
# delisted2016['Amihud'].fillna(delisted2016['Amihud'].max() + delisted2016['Amihud'].var()**0.5, inplace=True)  # 取平均值

delisted2016.Transpattern.fillna(-1,inplace=True)

# 生成交易方式虚拟变量
d_pattern = list()
for each in delisted2016.Transpattern.values:
    if each != -1:
        if 'S9002' in each:
            d_pattern.append(1)
        else:
            d_pattern.append(0)
    else:
        d_pattern.append(-1)
delisted2016['dp'] = d_pattern
delisted2016.drop('Transpattern',inplace=True,axis=1)

delisted2016['time'] = delisted2016['DelistedDate'] - delisted2016['ListedDate']
delisted2016['outcome'] = 1

# 选取2017年退市公司数据
delisted2017 = da[da['DelistedDate'].dt.year == 2017]
delisted2017 = delisted2017.loc[:, ~delisted2017.columns.str.contains('2017')].copy()

# t1=2015-12-31,outcome=0,time=t-listeddata
delisted201715 = delisted2017.loc[:, ~delisted2017.columns.str.contains('20') | delisted2017.columns.str.contains(
    '2015')].copy()
delisted201715.columns = [each.split('2')[0] for each in delisted201715.columns]

# 处理缺漏值
# missing_values = ['TurnoverRate', 'volatility']
# delisted201715[missing_values] = delisted201715[missing_values].fillna(0)
# delisted201715 = delisted201715[delisted201715['Amihud'] != 0]
# delisted201715['Amihud'].fillna(delisted201715['Amihud'].max() +
#                         delisted201715['Amihud'].var()**0.5, inplace=True)  # 取平均值

delisted201715.Transpattern.fillna(-1, inplace=True)

# 生成交易方式虚拟变量
d_pattern = list()
for each in delisted201715.Transpattern.values:
    if each != -1:
        if 'S9002' in each:
            d_pattern.append(1)
        else:
            d_pattern.append(0)
    else:
        d_pattern.append(-1)
delisted201715['dp'] = d_pattern
delisted201715.drop('Transpattern', inplace=True, axis=1)

delisted201715['t1'] = parse('2015-12-31')
delisted201715['time'] = delisted201715['t1'] - delisted201715['ListedDate']
delisted201715['outcome'] = 0

# 2016年,outcome=1,time=T-listeddata
delisted201716 = delisted2017.loc[:, ~delisted2017.columns.str.contains('20') | delisted2017.columns.str.contains(
    '2016')].copy()
delisted201716.columns = [each.split('2')[0] for each in delisted201716.columns]

# 处理缺漏值
# missing_values = ['TurnoverRate', 'volatility']
# delisted201716[missing_values] = delisted201716[missing_values].fillna(0)
# delisted201716 = delisted201716[delisted201716['Amihud'] != 0]
# delisted201716['Amihud'].fillna(delisted201716['Amihud'].max() +
#                                 delisted201716['Amihud'].var()**0.5, inplace=True)  # 取平均值

delisted201716.Transpattern.fillna(-1, inplace=True)

# 生成交易方式虚拟变量
d_pattern = list()
for each in delisted201716.Transpattern.values:
    if each != -1:
        if 'S9002' in each:
            d_pattern.append(1)
        else:
            d_pattern.append(0)
    else:
        d_pattern.append(-1)
delisted201716['dp'] = d_pattern
delisted201716.drop('Transpattern', inplace=True, axis=1)

delisted201716['time'] = delisted201716['DelistedDate'] - delisted201716['ListedDate']
delisted201716['outcome'] = 1

# 2018年退市公司数据
delisted2018 = da[da['DelistedDate'].dt.year == 2018]

# t1=2015-12-31,outcome=0,time=t1-listeddata
delisted201815 = delisted2018.loc[:, ~delisted2018.columns.str.contains('20') | delisted2018.columns.str.contains(
    '2015')].copy()
delisted201815.columns = [each.split('2')[0]
                          for each in delisted201815.columns]

# 处理缺漏值
# missing_values = ['TurnoverRate', 'volatility']
# delisted201815[missing_values] = delisted201815[missing_values].fillna(0)
# delisted201815 = delisted201815[delisted201815['Amihud'] != 0]
# delisted201815['Amihud'].fillna(delisted201815['Amihud'].max() +
#                                 delisted201815['Amihud'].var()**0.5, inplace=True)  # 取平均值

delisted201815.Transpattern.fillna(-1, inplace=True)

# 生成交易方式虚拟变量
d_pattern = list()
for each in delisted201815.Transpattern.values:
    if each != -1:
        if 'S9002' in each:
            d_pattern.append(1)
        else:
            d_pattern.append(0)
    else:
        d_pattern.append(-1)

delisted201815['dp'] = d_pattern
delisted201815.drop('Transpattern', inplace=True, axis=1)

delisted201815['t1'] = parse('2015-12-31')
delisted201815['time'] = delisted201815['t1'] - delisted201815['ListedDate']
delisted201815['outcome'] = 0

# t2=2016-12-31,outcome=0,time=t2-listeddata
delisted201816 = delisted2018.loc[:, ~delisted2018.columns.str.contains('20') | delisted2018.columns.str.contains(
    '2016')].copy()
delisted201816.columns = [each.split('2')[0]
                          for each in delisted201816.columns]

# 处理缺漏值
# missing_values = ['TurnoverRate', 'volatility']
# delisted201816[missing_values] = delisted201816[missing_values].fillna(0)
# delisted201816 = delisted201816[delisted201816['Amihud'] != 0]
# delisted201816['Amihud'].fillna(delisted201816['Amihud'].max() +
#                                 delisted201816['Amihud'].var()**0.5, inplace=True)  # 取平均值

delisted201816.Transpattern.fillna(-1, inplace=True)

# 生成交易方式虚拟变量
d_pattern = list()
for each in delisted201816.Transpattern.values:
    if each != -1:
        if 'S9002' in each:
            d_pattern.append(1)
        else:
            d_pattern.append(0)
    else:
        d_pattern.append(-1)
delisted201816['dp'] = d_pattern
delisted201816.drop('Transpattern', inplace=True, axis=1)

delisted201816['t2'] = parse('2016-12-31')
delisted201816['time'] = delisted201816['t2'] - delisted201816['ListedDate']
delisted201816['outcome'] = 0

# 2017年,outcome=1,time=T-listeddata
delisted201817 = delisted2018.loc[:, ~delisted2018.columns.str.contains('20') | delisted2018.columns.str.contains(
    '2017')].copy()
delisted201817.columns = [each.split('2')[0]
                          for each in delisted201817.columns]

# 处理缺漏值
# missing_values = ['TurnoverRate', 'volatility']
# delisted201817[missing_values] = delisted201817[missing_values].fillna(0)
# delisted201817 = delisted201817[delisted201817['Amihud'] != 0]
# delisted201817['Amihud'].fillna(delisted201817['Amihud'].max() +
#                                 delisted201817['Amihud'].var()**0.5, inplace=True)  # 取平均值

delisted201817.Transpattern.fillna(-1, inplace=True)

# 生成交易方式虚拟变量
d_pattern = list()
for each in delisted201817.Transpattern.values:
    if each != -1:
        if 'S9002' in each:
            d_pattern.append(1)
        else:
            d_pattern.append(0)
    else:
        d_pattern.append(-1)
delisted201817['dp'] = d_pattern
delisted201817.drop('Transpattern', inplace=True, axis=1)

delisted201817['time'] = delisted201817['DelistedDate'] - delisted201817['ListedDate']
delisted201817['outcome'] = 1

# 整理一下,剔除t1,t2的变量
delisted201715.drop('t1',axis=1,inplace=True)
delisted201815.drop('t1', axis=1, inplace=True)
delisted201816.drop('t2', axis=1, inplace=True)

# 合并数据
cox2 = pd.concat([delisted201715, delisted201716,delisted201815, delisted201816,delisted201817],
                 axis=0, ignore_index=True,sort=False)

# 剔除pattern=-1的样本
cox2 = cox2[cox2['dp'] != -1]

for each in cox2.columns:
    cox2 = cox2[cox2[each].isnull() == 0]
    print(cox2.shape[0])
cox2.to_excel('/Users/hhr/Desktop/Projects/pydata/cox2_2.xls')

# logit 
da.columns = da.columns.droplevel()

# 2017年退市公司
da2016 = da.loc[:, da.columns.str.contains('2016') | ~da.columns.str.contains('20')].copy()
delisted2017 = da2016[da2016.DelistedDate.dt.year == 2017].copy() # 2017年退市的样本
listing2017 = da2016[da2016.StatusID == 'P0801'].copy()  # 2017年还在上市的样本

data2017 = pd.concat([delisted2017, listing2017])
data2017.columns = [each.split('2')[0]
                    for each in data2017.columns]
data2017['year'] = 2016

data2017.Transpattern.fillna(-1, inplace=True)
# 生成交易方式虚拟变量
d_pattern = list()
for each in data2017.Transpattern.values:
    if each != -1:
        if 'S9002' in each:
            d_pattern.append(1)
        else:
            d_pattern.append(0)
    else:
        d_pattern.append(-1)
data2017['dp'] = d_pattern
data2017.drop(['ShortName', 'ListedDate', 'DelistedDate'],axis=1,inplace=True)
for each in data2017.columns:
    data2017 = data2017[data2017[each].isnull() == 0]
    print(data2017.shape[0])
data2017.to_excel('/Users/hhr/Desktop/Projects/pydata/delisted2017.xls')

# 2018年退市公司
da2017 = da.loc[:, da.columns.str.contains(
    '2017') | ~da.columns.str.contains('20')].copy()
delisted2018 = da2017[da2017.DelistedDate.dt.year == 2018].copy()  # 2017年退市的样本
listing2018 = da2017[da2017.StatusID == 'P0801'].copy()  # 2017年还在上市的样本

data2018 = pd.concat([delisted2018, listing2018])
data2018.columns = [each.split('2')[0]
                    for each in data2018.columns]
data2018['year'] = 2017

data2018.Transpattern.fillna(-1, inplace=True)
# 生成交易方式虚拟变量
d_pattern = list()
for each in data2018.Transpattern.values:
    if each != -1:
        if 'S9002' in each:
            d_pattern.append(1)
        else:
            d_pattern.append(0)
    else:
        d_pattern.append(-1)
data2018['dp'] = d_pattern

data2018.drop(['ShortName', 'ListedDate', 'DelistedDate'], axis=1, inplace=True)
for each in data2018.columns:
    data2018 = data2018[data2018[each].isnull() == 0]
    print(data2018.shape[0])
data2018.to_excel('/Users/hhr/Desktop/Projects/pydata/delisted2018.xls')

data = pd.concat([data2017,data2018])
data.to_excel('/Users/hhr/Desktop/Projects/pydata/delisted201718.xls')


