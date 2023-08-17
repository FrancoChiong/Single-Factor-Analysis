import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

import backtest as bt
import statsmodels.api as sm
import matplotlib.dates as mdates
import pandas_market_calendars as mcal
import copy
plt.rcParams['font.sans-serif'] = 'KaiTi'




# 导入全A股（剔除北交所）样本池2019年以来的日度收益率
# 导入因子数据值：factor_size表示市值因子，factor_dp表示股息率因子，factor_g表示使用eps计算的盈利增长因子；factor_deltag表示通过g因子计算的盈利加速因子
fileaddress = r'C:\Users\27305\Desktop\factor'
return_matrix = pd.read_excel(fileaddress + r"\return_matrix.xlsx", sheet_name = "return_matrix", index_col = "date")
factor_size = pd.read_excel(fileaddress + r"\size.xlsx", sheet_name = "size_monthly", index_col = "date")
factor_dp = pd.read_excel(fileaddress + r"\dp2.xlsx", sheet_name = "dp", index_col = "date")
factor_g = pd.read_excel(fileaddress + r"\g.xlsx", sheet_name = "g", index_col = "date")
factor_deltag = pd.read_excel(fileaddress + r"\deltag.xlsx", sheet_name = "deltag", index_col = "date")
stock_st = pd.read_excel(fileaddress + r"\st.xlsx", index_col = "date")
nav_tar = pd.read_excel(fileaddress + r"\netvalue.xlsx", index_col = "date")
nav_tar.columns = ["fund",]
# 储存A股名称及上市日期，并对收益率和因子进行初始化
stock_name = factor_size.iloc[0] # 储存A股所有个股的名称
stock_name_dictionary = stock_name # 将A股的所有个股的名称和代码一一对应起来
stock_name_dictionary.index = [bt.Backtest.code_remix(x) for x in  stock_name_dictionary.index]
stock_listtime = factor_size.iloc[1] # 存储一个A股所有个股的上市时间列表
return_matrix.drop(return_matrix.index[:2], inplace = True) # 处理收益率矩阵
benchmark = pd.read_excel(fileaddress + r"\wind.xlsx", index_col="date")
benchmark = pd.Series(benchmark["benchmark"].values, index = benchmark.index, name = "benchmark",dtype= float) # 传入wind全A的时间序列
stock_st.drop(stock_st.index[:2], inplace = True) #对st列表进行规范化处理
print("数据载入完成")


# 单因子检验，一年换3次仓位（1月末，4月末和8月末换仓），以保持和相关财报数据同频
# 市值因子
# 此处调用了Backtest类进行回测
#初始化需要传入参数收益率矩阵，因子数值表等等
f_size_4m = bt.Backtest(return_matrix, factor_size, stock_st, stock_listtime, benchmark_nav= benchmark, trade_cost=0.003, isinitialization= True)
#将因子数值进行剔除缺失值，缩尾（默认1%缩尾），Z-score，以及分组
f_size_4m.factor_group(groupnum=5)
#进行分组回测，feature表明将因子定义为正向因子还是负向因子以产生多空收益
f_size_4m.getnav_daily(feature="negative")

# 股息因子
f_dp_4m = bt.Backtest(return_matrix, factor_dp, stock_st, stock_listtime, benchmark_nav= benchmark, trade_cost=0.003, isinitialization= True)
f_dp_4m.factor_group(groupnum=5)
f_dp_4m.getnav_daily()

# eps计算的增长因子
f_g_4m = bt.Backtest(return_matrix, factor_g, stock_st, stock_listtime, benchmark_nav= benchmark, trade_cost=0.003, isinitialization=True)
f_g_4m.factor_group(groupnum=5)
f_g_4m.getnav_daily()

#eps计算的盈利加速因子
f_deltag_4m = bt.Backtest(return_matrix, factor_deltag, stock_st, stock_listtime, benchmark_nav= benchmark, trade_cost=0.003, isinitialization= True)
f_deltag_4m.factor_group(groupnum=5)
f_deltag_4m.getnav_daily()


#按照换仓期进行滚动线性回归
# 以全A等权收益率为基准对金元顺安元启的超额收益率，关于四大因子多空收益进行初次单因子滚动回归，以确定首个暴露最为显著的单因子
# 由净值数据返回收益率数据
fund_returnrate = bt.Backtest.getreturnrate_fromnav(nav_tar["fund"], drop_head = False)
size_lsreturnrate  = bt.Backtest.getreturnrate_fromnav(f_size_4m.nav_daily_ls, drop_head = False)
dp_lsreturnrate  = bt.Backtest.getreturnrate_fromnav(f_dp_4m.nav_daily_ls, drop_head = False)
g_lsreturnrate  = bt.Backtest.getreturnrate_fromnav(f_g_4m.nav_daily_ls, drop_head = False)
deltag_lsreturnrate  = bt.Backtest.getreturnrate_fromnav(f_deltag_4m.nav_daily_ls, drop_head = False)

#对标的进行市场因子中性化,获得标的相对于市场的超额收益
alpha_ftom = bt.factorfree(fund_returnrate, benchmark[fund_returnrate.index])

#生成一个回归dataframe
rolling_regression_all = pd.concat([alpha_ftom, size_lsreturnrate, dp_lsreturnrate, g_lsreturnrate, deltag_lsreturnrate], join = "inner", axis = 1)
rolling_regression_all.columns = ["alpha_ftom", "size", "dp", "g", "deltag" ]

# 定义结果DataFrame
result_df_all = pd.DataFrame(columns=['factor_name',"constant", 'f_coeff',"constant_pvalue", 'f_pvalue',  'adj_r_squared'])

#进行4次1元线性回归
for i in np.arange(4)+1:
    X = pd.Series(rolling_regression_all[rolling_regression_all.columns[i]].values, dtype=float, index=rolling_regression_all[rolling_regression_all.columns[i]].index)
    y = pd.Series(rolling_regression_all[rolling_regression_all.columns[0]].values, dtype=float, index=rolling_regression_all[rolling_regression_all.columns[0]].index, name="fund_ex_return")
    results = bt.liner_reg(y, X)
    factor_name = rolling_regression_all.columns[i]
    result_df_all.loc[i] = [factor_name] + list(results.params) + list(results.pvalues) + [results.rsquared_adj]
result_df_all.set_index("factor_name")
print(result_df_all)

#由结果的调节R方分析可以得到，首个最为显著的因子为小市值因子
#有下图也可以看出小市值因子的多头组合确实解释了标的基金的较多超额收益。

#接下来的对标的基金的剩余alpha进行进一步的发掘，考虑到小市值因子对于标的基金的超额业绩的强解释效应，此处新的因子的股票池将被缩小到全A市场前20%的小市值股票中，以充分保留最终股票池的小市值特征。
# 实际上，接下来的因子可以看做是小市值哑变量和其他因子的复合因子。

#基于小市值的股息因子
#因为此处直接调用对象属性，在之前的方法中已被预处理，所以预处理相关参数均为False
factor_ss_dp = copy.deepcopy(f_dp_4m.factor)
factor_ss_dp = factor_ss_dp[f_size_4m.factor_groupby == 5]
f_ss_dp_4m = bt.Backtest(return_matrix, factor_ss_dp, stock_st, stock_listtime, benchmark_nav= benchmark, trade_cost=0.003,isinitialization=False)
f_ss_dp_4m.factor_group(nanew = False, wins = False, gscore = False)
f_ss_dp_4m.getnav_daily()

#基于小市值的盈利加速因子
factor_ss_deltag = f_deltag_4m.factor.loc[pd.DatetimeIndex([x for x in f_deltag_4m.factor.index if x.month in [1, 4, 8]])]
factor_ss_deltag = factor_ss_deltag[f_size_4m.factor_groupby == 5]
f_ss_deltag_4m = bt.Backtest(return_matrix, factor_ss_deltag, stock_st, stock_listtime, benchmark_nav= benchmark ,trade_cost=0.003,isinitialization=False)
f_ss_deltag_4m.factor_group(nanew = False, wins = False, gscore = False)
f_ss_deltag_4m.getnav_daily()

#基于小市值的盈利增长因子
factor_ss_g = copy.deepcopy(f_g_4m.factor)
factor_ss_g = factor_ss_g[f_size_4m.factor_groupby == 5]
f_ss_g_4m = bt.Backtest(return_matrix, factor_ss_g, stock_st, stock_listtime, benchmark_nav = benchmark, trade_cost=0.003, isinitialization=False)
f_ss_g_4m.factor_group(nanew = False, wins = False, gscore = False)
f_ss_g_4m.getnav_daily()

#回归以进行检验新因子对alpha的挖掘能力
#新的因子：小市值下的股息因子、小市值下的盈利增长因子，小市值下的盈利加速因子
#alpha：fund相对于市场的alpha，小市值组合相对于市场的alpha，和fund的剩余alpha
#residual alpha of fund = market alpha of fund - market alpha of small size portfolio。因为他们对于市场的暴露是各自不相同的，所以先各自剔除市场影响。
#个人认为这个和算因子的alpha不同，此为相对于某个策略的alpha。通过单因子因子暴露分析，我们可以发现标的基金对于小市值因子有明显的正暴露，所以我们可以去构造一个小市值组合去模拟该基金，效果我们由图可以看出，基本趋势相近，但是可以明显发现基金相对于该小市值组合还是会有正向的超额收益的（可以被净值劈叉验证）。所以我们此时进一步优化我们的模拟组合或者说优化我们的股票池时，是要再找到是否在该股票池中存在某种因子，对于，标的基金相对于我们既有策略的allpha（即为所提及的剩余alpha），有一定解释力度和显著暴露。

#得到小市值因子的超额收益
returnrate_sc = pd.Series(f_size_4m.returnrate_daily_allgroup["5"].values, dtype = float, index = f_size_4m.returnrate_daily_allgroup["5"].index)

alpha_sztom = bt.factorfree(returnrate_sc, benchmark[returnrate_sc.index])
alpha_ftom = bt.factorfree(fund_returnrate, benchmark[fund_returnrate.index])

#得到标的相对于小市值因子的剩余超额收益
alpha_residual = copy.deepcopy(alpha_ftom[alpha_sztom.index] - alpha_sztom)
alpha_residual.dropna(inplace = True)

#得到多空组合的收益率
lsreturn_deltag = bt.getreturnrate_fromnav(f_ss_deltag_4m.nav_daily_ls)
lsreturn_g = bt.getreturnrate_fromnav(f_ss_g_4m.nav_daily_ls)
lsreturn_dp = bt.getreturnrate_fromnav(f_ss_dp_4m.nav_daily_ls)

#
regression_step2 = pd.concat([alpha_residual, lsreturn_deltag,lsreturn_dp,lsreturn_g],join="inner",axis = 1)
regression_step2.columns = ["alpha_residual","deltag", "dp", "g"]
result_step2 = pd.DataFrame(columns=['factor_name',"constant", 'f_coeff',"constant_pvalue", 'f_pvalue','adj_r_squared'])

for i in np.arange(3)+1:
    X = pd.Series(regression_step2[regression_step2.columns[i]].values, dtype=float,
                  index=regression_step2[regression_step2.columns[i]].index)
    y = pd.Series(regression_step2[regression_step2.columns[0]].values, dtype=float,
                  index=regression_step2[regression_step2.columns[0]].index, name="fund_ex_return")
    results = bt.liner_reg(y, X)
    factor_name = regression_step2.columns[i]
    result_step2.loc[i] = [factor_name] + list(results.params) + list(results.pvalues) + [results.rsquared_adj]
result_step2.set_index("factor_name")


#使用小市值叠加高股息进一步解释基金的超额收益，在进一步缩池的基础之上研究对剩余超额收益仍有解释力度的因子，强调小市值，高股息特性
#基于小市值高股息的增长因子
factor_ss_hdp_g_4m = copy.deepcopy(f_g_4m.factor)
factor_ss_hdp_g_4m = factor_ss_hdp_g_4m[(f_ss_dp_4m.factor_groupby == 1)]
f_ss_hdp_g_4m = bt.Backtest(return_matrix, factor_ss_hdp_g_4m, stock_st, stock_listtime, benchmark_nav= benchmark , trade_cost=0.003, isinitialization=False)
f_ss_hdp_g_4m.factor_group(nanew = False, wins = False, gscore = False)
f_ss_hdp_g_4m.getnav_daily()


#基于小市值高股息的加速因子
factor_ss_hdp_deltag_4m = copy.deepcopy(f_deltag_4m.factor)
factor_ss_hdp_deltag_4m = factor_ss_hdp_deltag_4m[(f_ss_dp_4m.factor_groupby == 1)]
f_ss_hdp_deltag_4m = bt.Backtest(return_matrix, factor_ss_hdp_deltag_4m, stock_st, stock_listtime, benchmark_nav= benchmark , trade_cost=0.003, isinitialization=False)
f_ss_hdp_deltag_4m.factor_group(nanew = False, wins = False, gscore = False)
f_ss_hdp_deltag_4m.getnav_daily()



#重复前文回归过程在g和deltag中优选：
yyy = pd.Series(f_ss_dp_4m.returnrate_daily_allgroup["1"].values, index = f_ss_dp_4m.returnrate_daily_allgroup["1"].index, dtype = float)
XXX = benchmark.loc[yyy.index]
alpha_pszhdptom = bt.factorfree(yyy, XXX)
alpha_residual = alpha_ftom - alpha_pszhdptom

lsreturn_deltag2 = bt.getreturnrate_fromnav(f_ss_hdp_deltag_4m.nav_daily_ls)
lsreturn_g2 = bt.getreturnrate_fromnav(f_ss_hdp_g_4m.nav_daily_ls)


regression_step3 = pd.concat([alpha_residual, lsreturn_deltag2, lsreturn_g2],join="inner",axis = 1)
regression_step3.columns = ["alpha_residual","deltag",  "g"]
result_step3 = pd.DataFrame(columns=['factor_name',"constant", 'f_coeff',"constant_pvalue", 'f_pvalue','adj_r_squared'])

for i in np.arange(2)+1:
    X = pd.Series(regression_step3[regression_step3.columns[i]].values, dtype=float,
                  index=regression_step3[regression_step3.columns[i]].index)
    X = sm.add_constant(X)
    y = pd.Series(regression_step3[regression_step3.columns[0]].values, dtype=float,
                  index=regression_step3[regression_step3.columns[0]].index, name="fund_ex_return")
    model = sm.OLS(y, X)
    results = model.fit()
    factor_name = regression_step3.columns[i]
    result_step3.loc[i] = [factor_name] + list(results.params) + list(results.pvalues) + [results.rsquared_adj]

result_step3.set_index("factor_name")

##两种因子均不显著

#在股息率做第二因子走不下去时，转回g做第二因子的道路上
##高增长做第二位因子
#基于小市值高增长的股息因子
factor_ss_hg_dp_4m = copy.deepcopy(f_dp_4m.factor)
factor_ss_hg_dp_4m = factor_ss_hg_dp_4m[(f_ss_g_4m.factor_groupby == 1)]
f_ss_hg_dp_4m = bt.Backtest(return_matrix, factor_ss_hg_dp_4m, stock_st, stock_listtime, benchmark_nav= benchmark , trade_cost=0.003, isinitialization=False)
f_ss_hg_dp_4m.factor_group(nanew = False, wins = False, gscore = False)
f_ss_hg_dp_4m.getnav_daily()

#基于小市值高增长的加速因子
factor_ss_hg_deltag_4m = copy.deepcopy(f_deltag_4m.factor)
factor_ss_hg_deltag_4m = factor_ss_hg_deltag_4m[(f_ss_g_4m.factor_groupby == 1)]
f_ss_hg_deltag_4m = bt.Backtest(return_matrix, factor_ss_hg_deltag_4m, stock_st, stock_listtime, benchmark_nav= benchmark , trade_cost=0.003, isinitialization=False)
f_ss_hg_deltag_4m.factor_group(nanew = False, wins = False, gscore = False)
f_ss_hg_deltag_4m.getnav_daily()


#重复前文回归过程在股息和deltag中优选：
returnrate_ss_hg = pd.Series(f_ss_g_4m.returnrate_daily_allgroup["1"].values, index = f_ss_g_4m.returnrate_daily_allgroup["1"].index,dtype=float,name = "f_ss_hg")
alpha_pszhgtom = bt.factorfree(returnrate_ss_hg, benchmark.loc[returnrate_ss_hg.index])
alpha_residual = alpha_ftom[alpha_pszhgtom.index] - alpha_pszhgtom


lsreturn_deltag3 = bt.getreturnrate_fromnav(f_ss_hg_deltag_4m.nav_daily_ls)
lsreturn_dp3 = bt.getreturnrate_fromnav(f_ss_hg_dp_4m.nav_daily_ls)
regression_step4 = pd.concat([alpha_residual, lsreturn_deltag3, lsreturn_dp3], join="inner", axis = 1)
regression_step4.columns = ["alpha_residual","deltag",  "dp"]
result_step4 = pd.DataFrame(columns=['factor_name',"constant", 'f_coeff',"constant_pvalue", 'f_pvalue','adj_r_squared'])

for i in np.arange(2)+1:
    X = pd.Series(regression_step4[regression_step4.columns[i]].values, dtype=float,
                  index=regression_step4[regression_step4.columns[i]].index)
    y = pd.Series(regression_step4[regression_step4.columns[0]].values, dtype=float,
                  index=regression_step4[regression_step4.columns[0]].index, name="fund_ex_return")
    results = bt.liner_reg(y, X)
    factor_name = regression_step4.columns[i]
    result_step4.loc[i] = [factor_name] + list(results.params) + list(results.pvalues) + [results.rsquared_adj]
result_step4.set_index("factor_name")


#结论：股息此时应该做第三位因子
#我们的模拟组合就变为，小市值中的高增长中的高股息
f_size_4m.line_graph([f_ss_hg_dp_4m.nav_daily["1"], nav_tar["fund"]])
#因为股息做第二因子时，剩余残差在统计学上不显著，高增长做第三位因子时并不显著，这可能和研究未考虑因子暴露的时变有关。
f_size_4m.line_graph([f_ss_hg_dp_4m.nav_daily["1"], f_ss_hdp_g_4m.nav_daily["1"], nav_tar["fund"]])




#回归验证谁能成为第三因子
##高加速做第二位因子
#基于小市值高加速的股息因子
factor_ss_hdeltag_dp_4m = copy.deepcopy(f_dp_4m.factor)
factor_ss_hdeltag_dp_4m = factor_ss_hdeltag_dp_4m[(f_ss_deltag_4m.factor_groupby == 1)]
f_ss_hdeltag_dp_4m = bt.Backtest(return_matrix, factor_ss_hdeltag_dp_4m, stock_st, stock_listtime, benchmark_nav= benchmark , trade_cost=0.003, isinitialization=False)
f_ss_hdeltag_dp_4m.factor_group(nanew = False, wins = False, gscore = False)
f_ss_hdeltag_dp_4m.getnav_daily()


#基于小市值高加速的增长因子
factor_ss_hdeltag_g_4m = copy.deepcopy(f_g_4m.factor)
factor_ss_hdeltag_g_4m = factor_ss_hdeltag_g_4m[(f_ss_deltag_4m.factor_groupby == 1)]
f_ss_hdeltag_g_4m = bt.Backtest(return_matrix, factor_ss_hdeltag_g_4m, stock_st, stock_listtime, benchmark_nav= benchmark , trade_cost=0.003, isinitialization=False)
f_ss_hdeltag_g_4m.factor_group(nanew = False, wins = False, gscore = False)
f_ss_hdeltag_g_4m.getnav_daily()


#计算标的基金相对于小市值高加速组合的剩余超额收益
yyyyy = pd.Series(f_ss_deltag_4m.returnrate_daily_allgroup["1"].values, dtype = float,index = f_ss_deltag_4m.returnrate_daily_allgroup["1"].index, name = "小市值高加速")
XXXXX = pd.Series(benchmark.loc[f_ss_deltag_4m.returnrate_daily_allgroup.index].values,dtype=float, index = f_ss_deltag_4m.returnrate_daily_allgroup.index, name = "市场组合收益" )
alpha_ss_deltag = bt.factorfree(yyyyy, XXXXX)
alpha_residual3 = copy.deepcopy(alpha_ftom - alpha_ss_deltag)
alpha_residual3.dropna(inplace = True)

lsreturn_g3 = bt.getreturnrate_fromnav(f_ss_hdeltag_g_4m.nav_daily_ls)
lsreturn_dp3 = bt.getreturnrate_fromnav(f_ss_hdeltag_dp_4m.nav_daily_ls)

regression_step3= pd.concat([alpha_residual3, lsreturn_g3, lsreturn_dp3], join="inner", axis = 1)
regression_step3.columns = ["alpha_residual","g", "dp"]
result_step3 = pd.DataFrame(columns=['factor_name',"constant", 'f_coeff',"constant_pvalue", 'f_pvalue','adj_r_squared'])

for i in np.arange(2)+1:
    X = pd.Series(regression_step3[regression_step3.columns[i]].values, dtype=float,index=regression_step3[regression_step3.columns[i]].index)
    y = pd.Series(regression_step3[regression_step3.columns[0]].values, dtype=float, index=regression_step3[regression_step3.columns[0]].index, name="fund_ex_return")

    results = bt.liner_reg(y,X)
    factor_name = regression_step3.columns[i]
    result_step3.loc[i] = [factor_name] + list(results.params) + list(results.pvalues) + [results.rsquared_adj]
result_step3.set_index("factor_name")





#滚动回归表明



# 对市值因子进行滚动回归
years = np.arange(5)+2019
periods = [(2, 4), (5, 8), (9, 1)]
i = 0
for year in years:
    for period in periods:
        start_month, end_month = period
        if (year == 2023) & (start_month ==9):
            break
        if start_month == 9:
            rolling_regression_sample = rolling_regression_size.loc[pd.DatetimeIndex([x for x in rolling_regression_size.index if (((x.month >= start_month)  & (x.year == year)) | ((x.month <= end_month)  & (x.year == year+1)))])]
        else:
            rolling_regression_sample = rolling_regression_size.loc[pd.DatetimeIndex([x for x in rolling_regression_size.index if (((x.month >= start_month) & (x.month <= end_month)) & (x.year == year))])]

        X = pd.Series(rolling_regression_sample[rolling_regression_sample.columns[1]].values, dtype= float,index = rolling_regression_sample[rolling_regression_sample.columns[1]].index, name = "f_size_ls_return")
        X = sm.add_constant(X)
        y = pd.Series(rolling_regression_sample[rolling_regression_sample.columns[0]].values, dtype=float,
                  index=rolling_regression_sample[rolling_regression_sample.columns[0]].index, name="fund_ex_return")
        model = sm.OLS(y, X)
        results = model.fit()
        date = rolling_regression_sample.index[0]
        result_df_size.loc[i] = [date] + list(results.params) + list(results.pvalues) + [results.rsquared_adj]
        i += 1
result_df_size.set_index("Start Date")

# 对股息因子进行滚动回归
years = np.arange(5)+2019
periods = [(2, 4), (5, 8), (9, 1)]
i = 0
for year in years:
    for period in periods:
        start_month, end_month = period
        if (year == 2023) & (start_month ==9):
            break
        if start_month == 9:
            rolling_regression_sample = rolling_regression_dp.loc[pd.DatetimeIndex([x for x in rolling_regression_dp.index if (((x.month >= start_month)  & (x.year == year)) | ((x.month <= end_month)  & (x.year == year+1)))])]
        else:
            rolling_regression_sample = rolling_regression_dp.loc[pd.DatetimeIndex([x for x in rolling_regression_dp.index if (((x.month >= start_month) & (x.month <= end_month)) & (x.year == year))])]

        X = pd.Series(rolling_regression_sample[rolling_regression_sample.columns[1]].values, dtype= float,index = rolling_regression_sample[rolling_regression_sample.columns[1]].index, name = "f_dp_ls_return")
        X = sm.add_constant(X)
        y = pd.Series(rolling_regression_sample[rolling_regression_sample.columns[0]].values, dtype=float,
                  index=rolling_regression_sample[rolling_regression_sample.columns[0]].index, name="fund_ex_return")
        model = sm.OLS(y, X)
        results = model.fit()
        date = rolling_regression_sample.index[0]
        result_df_dp.loc[i] = [date] + list(results.params) + list(results.pvalues) + [results.rsquared_adj]
        i += 1
result_df_dp.set_index("Start Date")

# 对g因子进行滚动回归
years = np.arange(5)+2019
periods = [(2, 4), (5, 8), (9, 1)]
i = 0
for year in years:
    for period in periods:
        start_month, end_month = period
        if (year == 2023) & (start_month ==9):
            break
        if start_month == 9:
            rolling_regression_sample = rolling_regression_g.loc[pd.DatetimeIndex([x for x in rolling_regression_g.index if (((x.month >= start_month)  & (x.year == year)) | ((x.month <= end_month)  & (x.year == year+1)))])]
        else:
            rolling_regression_sample = rolling_regression_g.loc[pd.DatetimeIndex([x for x in rolling_regression_g.index if (((x.month >= start_month) & (x.month <= end_month)) & (x.year == year))])]

        X = pd.Series(rolling_regression_sample[rolling_regression_sample.columns[1]].values, dtype= float,index = rolling_regression_sample[rolling_regression_sample.columns[1]].index, name = "f_g_ls_return")
        X = sm.add_constant(X)
        y = pd.Series(rolling_regression_sample[rolling_regression_sample.columns[0]].values, dtype=float,
                  index=rolling_regression_sample[rolling_regression_sample.columns[0]].index, name="fund_ex_return")
        model = sm.OLS(y, X)
        results = model.fit()
        date = rolling_regression_sample.index[0]
        result_df_g.loc[i] = [date] + list(results.params) + list(results.pvalues) + [results.rsquared_adj]
        i +=1
result_df_g.set_index("Start Date")

# 对deltag因子进行滚动回归
years = np.arange(5)+2019
periods = [(2, 4), (5, 8), (9, 1)]
i = 0
for year in years:
    for period in periods:
        start_month, end_month = period
        if (year == 2023) & (start_month ==9):
            break
        if start_month == 9:
            rolling_regression_sample = rolling_regression_deltag.loc[pd.DatetimeIndex([x for x in rolling_regression_deltag.index if (((x.month >= start_month)  & (x.year == year)) | ((x.month <= end_month)  & (x.year == year+1)))])]
        else:
            rolling_regression_sample = rolling_regression_deltag.loc[pd.DatetimeIndex([x for x in rolling_regression_deltag.index if (((x.month >= start_month) & (x.month <= end_month)) & (x.year == year))])]

        X = pd.Series(rolling_regression_sample[rolling_regression_sample.columns[1]].values, dtype= float,index = rolling_regression_sample[rolling_regression_sample.columns[1]].index, name = "f_deltag_ls_return")
        X = sm.add_constant(X)
        y = pd.Series(rolling_regression_sample[rolling_regression_sample.columns[0]].values, dtype=float,
                  index=rolling_regression_sample[rolling_regression_sample.columns[0]].index, name="fund_ex_return")
        model = sm.OLS(y, X)
        results = model.fit()
        date = rolling_regression_sample.index[0]
        result_df_deltag.loc[i] = [date] + list(results.params) + list(results.pvalues) + [results.rsquared_adj]
        i += 1
result_df_deltag.set_index("Start Date")

adj_r_squared = pd.concat([result_df_size["adj_r_squared"], result_df_dp["adj_r_squared"], result_df_g["adj_r_squared"], result_df_deltag["adj_r_squared"]], axis = 1)

result_df_size
print(adj_r_squared)
adj_r_squared.index = f_size_4m.factor_groupby.index
adj_r_squared.to_excel(fileaddress + r"\adjj_r.xlsx")



#利用布林带进行择时分析
portfolio_nav = f_ss_hg_dp_4m.nav_daily["1"]

#Tming是一个择时类别，专门又来进行择时优化组合收益
#对Timing的交易系统进行初始化，初始权益仓位0.7， 交易成本千三，债券部分没有收益，择时的方法是使用bullin带，bullin参数仅20日股价均值，上下加减三个标准差，
portfolio_tim = bt.Timing(portfolio_nav, s_nav0 = 0.7, trade_cost = 0.003, returnrate_debt = 0.0, method = ["bullin", 20, 3])

#设置一个对应信号强度和交易仓位大小的Series
sign_to_pro = pd.Series([0.2, 0.3, 0.4, 0.50, 0, 0.2, 0.3, 0.4, 0.50], index = [1, 2, 3, 5, 0, -1, -2, -3, -5])

#利用布林带进行择时，上下轨道的验证周期为3天，中轨的验证周期为7天
portfolio_tim.bullin_timing(ext_valdays = 1, mid_valdays = 7,sign_to_pro = sign_to_pro)

#对结果的绩效进行分析
bt.Backtest.getperformance_outer(portfolio_tim.porfoliomod_anav_daily, nav_benchmark = portfolio_tim.portfolio_nav)










