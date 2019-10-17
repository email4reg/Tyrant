# *******************Chapter 5********************
data5 = pd.read_excel(
    '/Users/hhr/Desktop/Projects/pydata/var_chapter5_data.xlsx')
data5 = data5.set_index('Date')

# 构建回归模型
params = leastsq(error, x0=[0, 0], args=(data5['sigma2'][1:], data5['R2'][1:]))[0]

#计算拟合优度R2
y_avg = data5['R2'][1:].mean() # 计算被解释变量的均值
e_SS = np.sum(((data5['sigma2'][1:].values * params[0] + params[1]) - y_avg) ** 2) # 回归平方和
t_SS = np.sum((data5['R2'][1:].values - y_avg) ** 2) # 总离差平方和
r_SS = np.sum((data5['R2'][1:].values - (data5['sigma2']
                  [1:].values * params[0] + params[1])) ** 2)  # 残差平方和
R2 = e_SS / t_SS

# 回归系数t检验(系数估计值与其标准差的比,标准差见李子奈p50)
n = len(data5)
df = (n - 1) - 1 - 1 # 计算自由度=样本个数-变量个数-1
x_avg = data5['sigma2'][1:].mean()
r_std_dev = np.sqrt(r_SS / df) # 用残差标准差除以自由度代理系数的方差,真实的未知
coef_std_dev = np.sqrt(
    (r_std_dev ** 2) / np.sum((data5['sigma2'][1:].values - x_avg) ** 2))  # 无偏估计的系数标准差,真实的未知

#计算常数项标准差
cons_std_dev = np.sqrt(
    r_std_dev ** 2 * np.sum(data5['sigma2'][1:].values ** 2) / n * np.sum((data5['sigma2'][1:].values - x_avg) ** 2))

# t值
coef_t_value = params[0] - 1 / coef_std_dev  # t值,双边检验,原假设为H0=1
cons_t_value = params[1] / cons_std_dev

# TODO 定义一个回归模型的类

# using the squared returns
da2 = pd.read_excel(
    '/Users/hhr/Desktop/Projects/pydata/var_chapter5_data2.xlsx')
da2 = da2.set_index('Date')

# 定义回归模型
def univ_reg(p, x):
    a, b = p
    return a*x + b

# 定义func
def error(p, x, y):
    return univ_reg(p, x) - y

# 估计参数
params = leastsq(error, x0=[0, 0], args=(da2['sigma2'][1:], da2['RPT'][1:]))[0]

# using RV instead of the squared returns(略)

# using RP instead of the squared returns
da4 = pd.read_excel(
    '/Users/hhr/Desktop/Projects/pydata/var_chapter5_data4.xlsx')
da4 = da4.set_index('Date')

# 估计 HAR模型
da4['DT'] = np.log(da4.High / da4.Low)
da4['RPt'] = da4.DT ** 2 / np.log(16) # 计算RP

# daily RP
da4['RPdt'] = da4['RPt']

# weekly RP
wt_RP = [np.nan] * 5
for i in range(1,len(da4) - 4):
    wt_RP.append(da4['RPdt'][i:i+5].sum() / 5)
da4['RPwt'] = wt_RP

# monthly RP
mt_RP = [np.nan] * 21
for i in range(1,len(da4) - 20):
    mt_RP.append(da4['RPdt'][i:i+21].sum() / 21)
da4['RPmt'] = mt_RP

# 估计参数
ln_da4 = da4[['DT','RPt','RPdt', 'RPwt', 'RPmt']].applymap(lambda x: np.log(x)) # 取对数
ln_da4 = ln_da4.iloc[21:,:]
# 定义回归模型
def reg_mod(p,x1,x2,x3):
    beta0,beta1,beta2,beta3 = p
    return beta0 + beta1 * x1 + beta2 * x2 + beta3 * x3

# 估计参数
params = leastsq(lambda p,x1,x2,x3,y: reg_mod(p,x1,x2,x3) - y, x0=[0, 0, 0, 0], args=(
    ln_da4['RPdt'][:-1], ln_da4['RPwt'][:-1], ln_da4['RPmt'][:-1], ln_da4['RPt'][1:]))[0] # 结果不对

reg1 = LinearRegression(fit_intercept=True)
reg1.fit(ln_da4[['RPdt', 'RPwt', 'RPmt']][:-1].values, ln_da4['RPt'][1:].values)
reg1.coef_
reg1.intercept_

# Use the next day’s RV(略)