import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from chinese_calendar import is_workday
import akshare as ak
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.rcParams['font.sans-serif'] = 'KaiTi'
import matplotlib.dates as mdates
from datetime import timedelta
from datetime import datetime, time
from tqdm import tqdm
import time
from dateutil.relativedelta import relativedelta
from scipy.stats.mstats import winsorize
import copy
from scipy.stats import gmean


class GetTime:
    """
    该类主要是为了生成符合中国A股的时间序列
ClassMethod：
    gettd():
        get trading day in china，用于返回依据中国节假日安排的在所输入起始与截止日期之间的工作日时间序列
        input：startdate(str)和enddate(str)
        output：ts_chn_wd(DatatimeIndex)，返回在startdate和enddate之间的一个中国工作日时间序列
    gettd_fr():
        get trading dates related to financial report analysis，考虑A股季度财报披露的截止日期生成的，在涉及财报指标的因子研究中返回换仓时间序列
        input：startdate(str)和enddate(str)
        output：ts_fr_chn(DatatimeIndex)，返回在startdate和enddate之间的一个新财报季第一个交易日的时间序列
    gettd_near():
        用于返回距离参考日期最近的一个A股交易日。
        input：ref_date(str)，how(str)。
            how表示追溯方向，默认为backward，向前追溯。输入forward或者f则向后追溯
        output：td(DatettimeStamp)
    """

    @classmethod
    def gettd(cls, startdate, enddate):
        print("将会生成规定起始时刻之间的A股交易日时间序列")
        china_stock_calendar = mcal.get_calendar('SSE')
        schedule = china_stock_calendar.schedule(start_date=datetime.strptime(startdate, "%Y%m%d"),
                                                 end_date=datetime.strptime(enddate, "%Y%m%d"))
        trading_days = pd.DatetimeIndex(schedule.index.date.tolist())
        return trading_days

    @classmethod
    def gettd_fr(cls, startdate, enddate):
        print("将会依据A股财报披露截止日期生成一个换仓时间序列")
        ts = cls.gettd(startdate, enddate)
        ts_pd = pd.DataFrame([ts, ts.month, ts.year]).T
        ts_pd.columns = ["date", "month", "year"]
        ts_pd.drop_duplicates(["month", "year"], inplace=True)
        ts_fr_chn = pd.DatetimeIndex(ts_pd["date"])
        ts_fr_chn = pd.DatetimeIndex(dt for dt in ts_fr_chn if dt.month in [5, 9, 11])
        return ts_fr_chn

    @classmethod
    def gettd_near(cls, ref_date, how="backward"):
        if isinstance(ref_date, str):
            reff_date = datetime.strptime(ref_date, "%Y%m%d")
        else:
            reff_date = ref_date
        china_stock_calendar = mcal.get_calendar('SSE')
        a = len(china_stock_calendar.valid_days(start_date=ref_date, end_date=reff_date))
        if a == 1:
            if how == "f" or how == "forward":
                ii = 1
                while not (len(china_stock_calendar.valid_days(start_date=ref_date,
                                                               end_date=reff_date + timedelta(ii))) == 2):
                    ii += 1
                td = (reff_date + timedelta(ii))
            else:
                ii = 1
                while not (len(china_stock_calendar.valid_days(start_date=ref_date,
                                                               end_date=reff_date - timedelta(ii))) == 2):
                    ii += 1
                td = (reff_date - timedelta(ii))
        else:
            if how == "f" or how == "forward":
                ii = 1
                while not (len(china_stock_calendar.valid_days(start_date=ref_date,
                                                               end_date=reff_date + timedelta(ii))) == 1):
                    ii += 1
                td = (reff_date + timedelta(ii))
            else:
                ii = 1
                while not (len(china_stock_calendar.valid_days(start_date=ref_date,
                                                               end_date=reff_date - timedelta(ii))) == 1):
                    ii += 1
                td = (reff_date - timedelta(ii))
        return td


class Stock(GetTime):
    """
        该类主要是为了生成与股票相关的时序数据
    attr：
        akcode：
            用于储存与输入进来的标准WIND股票代码相对应的akshare股票代码
            input：pd.Series，元素为str；output: pd.Series, 元素为str。目的是生成一个akshare API可以识别的代码格式
        stcode：
             用于储存输入进来的标准WIND股票代码

    methods:
        getprice_st():
            用于返回akcode对应的股票的股价时间序列
            input：startdate(str), enddate(str), adjustment(str), price_type(str). 其中startdate和enddate输入格式为%Y%m%d，enddate默认为当前日期；adjustment表示复权选项，默认使用后复权；price_type表示价格类型，默认为收盘价。
            output：pd.DataFrame. columns为akcodes；index元素为str

        getreturn_st():
            用于返回akcode对应的股票的股收益率时间序列
            input：startdate(str), enddate(str), adjustment(str), price_type(str). 其中startdate和enddate输入格式为%Y%m%d，enddate默认为当前日期；adjustment表示复权选项，默认使用后复权；price_type表示价格类型，默认为收盘价。
            output：pd.DataFrame. columns为akcodes；index元素为str

ClassMethods:
        code_remix():
            将标准的WIND股票代码转化为akshare股票代码

        code_back():
            code_remix()的逆操作

        getprice_st_outer():
            与getprice_st()这个实例方法相类似，用于返回实例属性中code对应个股在规定时间范围内的股价时间序列，此为类方法，可以用于
            input：

    """

    def __init__(self, return_matrix):
        self.return_matrix = copy.deepcopy(return_matrix)

        code_new = []
        for x in return_matrix.columns:
            if "." in x:
                a, b = x.split(".")
                code_new_tem = b.lower() + a
            else:
                code_new_tem = x
            code_new += [code_new_tem, ]
        self.akcode = pd.Series(code_new, name="code")
        self.return_matrix.columns = code_new

    def getreturn_st(self, startdate, enddate=datetime.now().strftime("%Y%m%d")):
        return_matrix_tar = self.return_matrix[self.akcode]
        return_matrix_tar_result = return_matrix_tar.loc[
                                   datetime.strptime(startdate, "%Y%m%d"): datetime.strptime(enddate, "%Y%m%d")]
        return return_matrix_tar_result

    def getsize(self, startdate, enddate=datetime.now().strftime("%Y%m%d"), adjustment="hfq", price_type="close"):
        sdt = Stock.gettd_near(startdate).strftime("%Y%m%d")

    @classmethod
    def code_remix(cls, x):
        a, b = x.split(".")
        code_new_tem = b.lower() + a
        return code_new_tem

    @classmethod
    def code_back(cls, x):
        a = x[:1]
        b = x[2:]
        code_new_tem = b.upper() + "." + a
        return code_new_tem

    @classmethod
    def getprice_st_outer(cls, akcode, startdate, enddate=datetime.now().strftime("%Y%m%d"), adjustment="hfq",
                          price_type="close"):
        allprice_close = pd.DataFrame()
        for code in akcode:
            close_data = ak.stock_zh_a_daily(symbol=code, start_date=startdate, end_date=enddate, adjust=adjustment)
            sole = pd.DataFrame(close_data[price_type].values, index=close_data["date"].values, columns=[code])
            if allprice_close.empty:
                allprice_close = sole
            else:
                allprice_close = allprice_close.join(sole, how='outer')
        return allprice_close

    @classmethod
    def getreturn_st_outer(cls, akcode, startdate, enddate=datetime.now().strftime("%Y%m%d"), adjustment="hfq",
                           price_type="close"):
        sdt = Stock.gettd_near(startdate)
        allprice_close = cls.getprice_st_outer(akcode, sdt, enddate, adjustment, price_type)
        allst_re = (allprice_close / allprice_close.shift(1) - 1)
        _ = allst_re.drop(allst_re.index[0], inplace=True)
        return allst_re


class Index(GetTime):
    """
        该类主要是为了生成与股票相关的时序数据
    attr：
        akcode：
            用于储存与输入进来的标准WIND股票代码相对应的akshare股票代码
            input：pd.Series，元素为str；output: pd.Series, 元素为str。目的是生成一个akshare API可以识别的代码格式
        stcode：
             用于储存输入进来的标准WIND股票代码

    methods:
        getprice():
            用于返回akcode对应的股票的股价时间序列
            input：startdate(str), enddate(str), adjustment(str), price_type(str). 其中startdate和enddate输入格式为%Y%m%d，enddate默认为当前日期；adjustment表示复权选项，默认使用后复权；price_type表示价格类型，默认为收盘价。
            output：pd.DataFrame. columns为akcodes；index元素为str

        getreturn():
            用于返回akcode对应的股票的股收益率时间序列
            input：startdate(str), enddate(str), adjustment(str), price_type(str). 其中startdate和enddate输入格式为%Y%m%d，enddate默认为当前日期；adjustment表示复权选项，默认使用后复权；price_type表示价格类型，默认为收盘价。
            output：pd.DataFrame. columns为akcodes；index元素为str

    classmethods:
        code_remix():
            将标准的WIND股票代码转化为akshare股票代码
        code_back():
            code_remix()的逆操作

    """

    def __init__(self, code):
        code_new = []
        for x in code.values:
            if "." in x:
                a, b = x.split(".")
                code_new_tem = b.lower() + a
            else:
                code_new_tem = x
            code_new += [code_new_tem, ]
        self.akcode = pd.Series(code_new, name="code")
        self.stcode = code

    def getprice_id(self, startdate, enddate=datetime.now().strftime("%Y%m%d"), price_type="close"):
        allprice_close = pd.DataFrame()
        for code in self.akcode:
            close_data = ak.stock_zh_index_daily(symbol=code)
            sole = pd.DataFrame(close_data[price_type].values, index=close_data["date"].values, columns=[code])
            sole.index = pd.DatetimeIndex(sole.index)
            if allprice_close.empty:
                allprice_close = sole
            else:
                allprice_close = allprice_close.join(sole, how='outer')
        allprice_close = allprice_close.loc[
                         startdate if datetime.strptime(startdate, "%Y%m%d") > allprice_close.index[0] else
                         allprice_close.index[0]: enddate if datetime.strptime(enddate, "%Y%m%d") <
                                                             allprice_close.index[-1] else allprice_close.index[-1]]
        return allprice_close

    def getreturn_id(self, startdate, enddate=datetime.now().strftime("%Y%m%d"), price_type="close"):
        sdt = Index.gettd_near(startdate)
        allprice_close = self.getprice_id(sdt, enddate, price_type)
        allst_re = (allprice_close / allprice_close.shift(1) - 1)
        _ = allst_re.drop(allst_re.index[0], inplace=True)
        return allst_re

    @classmethod
    def code_remix(cls, x):
        a, b = x.split(".")
        code_new_tem = b.lower() + a
        return code_new_tem

    @classmethod
    def code_back(cls, x):
        a = x[:1]
        b = x[2:]
        code_new_tem = b.upper() + "." + a
        return code_new_tem

    @classmethod
    def getprice_id_outer(cls, akcode, startdate, enddate=datetime.now().strftime("%Y%m%d"), price_type="close"):

        allprice_close = pd.DataFrame()
        if isinstance(akcode, str):
            akcode = [akcode, ]

        for code in akcode:
            close_data = ak.stock_zh_index_daily(symbol=code)
            sole = pd.DataFrame(close_data[price_type].values, index=close_data["date"].values, columns=[code])
            sole.index = pd.DatetimeIndex(sole.index)
            if allprice_close.empty:
                allprice_close = sole
            else:
                allprice_close = allprice_close.join(sole, how='outer')

        if not isinstance(startdate, str):
            startdate = startdate.strftime("%Y%m%d")

        allprice_close = allprice_close.loc[
                         startdate if datetime.strptime(startdate, "%Y%m%d") > allprice_close.index[0] else
                         allprice_close.index[0]: enddate if datetime.strptime(enddate, "%Y%m%d") <
                                                             allprice_close.index[-1] else allprice_close.index[-1]]

        return allprice_close

    @classmethod
    def getreturn_id_outer(cls, akcode, startdate, enddate=datetime.now().strftime("%Y%m%d"), price_type="close"):
        sdt = Stock.gettd_near(startdate)
        allprice_close = cls.getprice_id_outer(akcode, sdt, enddate, price_type)
        allst_re = (allprice_close / allprice_close.shift(1) - 1)
        _ = allst_re.drop(allst_re.index[0], inplace=True)
        return allst_re


class StrategyPerformance():
    """


    """

    @classmethod
    def getreturnrate_fromnav(cls, nav, drop_head=True):
        returnrate = copy.deepcopy(nav / nav.shift(1) - 1)
        if drop_head:
            returnrate.drop(returnrate.index[0], inplace=True)
        else:
            returnrate.iloc[0] = copy.deepcopy(nav.iloc[0] - 1)
        return returnrate

    @classmethod
    def annual_rate(cls, nav):
        delta = relativedelta(nav.index[-1], nav.index[0])
        time_span = delta.years + (delta.months / 12) + (delta.days / 365)
        annual_rate = (nav.iloc[-1] - 1) ** (1 / time_span) - 1
        return annual_rate

    @classmethod
    def std(cls, returnrate):
        std = returnrate.std(ddof=0)
        return std

    @classmethod
    def maxdrawdown(cls, nav):
        drawdown = []
        for i in np.array(range(len(nav) - 1)):
            drawdown += [(nav[i + 1:].min() - nav[i]) / nav[i], ]
        maxdrawdown = - pd.Series(drawdown).min()
        return maxdrawdown

    @classmethod
    def winrate(cls, returnrate, benchmark):
        winrate = pd.concat([returnrate, benchmark], axis=1, join="outer")
        winrate.columns = ["strategy", "benchmark"]
        winrate_ratio = (winrate["strategy"] > winrate["benchmark"]).sum() / len(winrate.index)
        return winrate_ratio


class Graph():
    def __init__(self):
        self.x_interval = 3
        self.xticks_rotation = 45
        self.fig_len = 11
        self.fig_wid = 7

    def line_graph(self, v_leftaxis, v_rightaxis=None):
        if v_rightaxis is None:
            fig = plt.figure(figsize=(self.fig_len, self.fig_wid))
            axe1 = plt.gca()
            if isinstance(v_leftaxis, pd.DataFrame):
                for column in v_leftaxis.columns:
                    axe1.plot(v_leftaxis.index, v_leftaxis[column].values, label=column)
            elif isinstance(v_leftaxis, pd.Series):
                axe1.plot(v_leftaxis.index, v_leftaxis.values, label=v_leftaxis.name)
            else:
                for v in v_leftaxis:
                    if isinstance(v, pd.DataFrame):
                        for column in v.columns:
                            axe1.plot(v.index, v[column].values, label=column)
                    else:
                        axe1.plot(v.index, v.values, label=v.name)
        else:
            fig = plt.figure(figsize=(self.fig_len, self.fig_wid))
            axe1 = plt.gca()
            if isinstance(v_leftaxis, pd.DataFrame):
                for column in v_leftaxis.columns:
                    axe1.plot(v_leftaxis.index, v_leftaxis[column].values, label=column)
            elif isinstance(v_leftaxis, pd.Series):
                axe1.plot(v_leftaxis.index, v_leftaxis.values, label=v_leftaxis.name)
            else:
                for v in v_leftaxis:
                    if isinstance(v, pd.DataFrame):
                        for column in v.columns:
                            axe1.plot(v.index, v[column], label=column)
                    else:
                        axe1.plot(v.index, v.values, label=v.name)
            axe2 = axe1.twinx()
            legend2 = axe2.legend()
            legend2.set_visible(False)
            if isinstance(v_rightaxis, pd.DataFrame):
                for column in v_rightaxis.columns:
                    axe2.plot(v_rightaxis.index, v_rightaxis[column].values, label=column)
            elif isinstance(v_rightaxis, pd.Series):
                axe2.plot(v_rightaxis.index, v_rightaxis.values, label=v_rightaxis.name)
            else:
                for v in v_rightaxis:
                    if isinstance(v, pd.DataFrame):
                        for column in v.columns:
                            axe2.plot(v.index, v[column], label=column)
                    else:
                        axe2.plot(v.index, v.values, label=v.name)
            handles, labels = axe1.get_legend_handles_labels()
            if v_rightaxis is not None:
                handles2, labels2 = axe2.get_legend_handles_labels()
                handles += handles2
                labels += labels2
            axe1.legend(handles, labels)
        plt.xticks(rotation=self.xticks_rotation)
        plt.rcParams['font.sans-serif'] = 'KaiTi'
        plt.legend()


class Factor(Graph):

    def __init__(self, factor, stock_st, stock_listtime, shafflemonth=None, isinitialization= True):
        super(Factor, self).__init__()
        # 因子矩阵初始化
        if shafflemonth is None:  # key parameter，本函数对因子分组结果进行resample，原频率为月度
            self.shafflemonth = [1, 4, 8]
        else:
            self.shafflemonth = copy.deepcopy(shafflemonth)
        if (not isinstance(factor.index[0], pd.Timestamp)) & isinitialization:
            factor.drop(factor.index[:2], inplace=True)
            factor[factor == 0] = np.nan
            factor[stock_st == 1] = np.nan  # 排除ST、*ST股票
        else:
            pass
        self.factor = copy.deepcopy(
            factor.loc[pd.DatetimeIndex([x for x in factor.index if x.month in self.shafflemonth])])
        self.factor_rank = copy.deepcopy(self.factor)
        self.factor_groupby = copy.deepcopy(self.factor)
        self.factor_longshort = copy.deepcopy(self.factor)
        self.stock_listtime = copy.deepcopy(stock_listtime)

    from datetime import datetime
    def na_newlist(self, listdate, days):
        for col in tqdm(self.factor.columns):
            for date in self.factor.index:
                delta = relativedelta(date, datetime.strptime(listdate[col], "%Y-%m-%d"))
                if 365 * delta.years + 12 * delta.months + delta.days < days:
                    self.factor[col].loc[date] = np.nan

    def winsorize(self, x, y):
        for index, row in tqdm(self.factor.iterrows()):
            lower = row.quantile(q=x)
            upper = row.quantile(q=(1 - y))
            self.factor.loc[index][self.factor.loc[index] < lower] = lower
            self.factor.loc[index][self.factor.loc[index] > upper] = upper

    def getzscore(self):
        self.factor = (self.factor.sub(self.factor.mean(axis=1, skipna=True), axis=0)).div(
            self.factor.std(axis=1, skipna=True), axis=0)

    # 此函数用于返回因子分组的最终结果，最后将会定义实例属性factor_groupby
    def getgroup(self, num):
        self.groupnumber = num
        for i in np.arange(len(self.factor.index)):
            self.factor_rank.iloc[i] = copy.deepcopy(self.factor.iloc[i].rank(ascending=False))
            self.factor_rank.iloc[i] = copy.deepcopy(self.factor_rank.iloc[i] / self.factor_rank.iloc[i].count())
        self.factor_groupby = copy.deepcopy(self.factor_rank)
        for numm in np.arange(num):
            self.factor_groupby[(numm / num < self.factor_groupby) & (self.factor_groupby < (numm + 1) / num)] = (
                        numm + 1)

    # 此函数为因子分组，包含排除上市不足100天，缩尾，分组，根据换仓期进行resample
    def factor_group(self, listdate_tonow=100, wins_left=0.01, wins_right=0.01, groupnum=5, nanew=True, wins=True,
                     gscore=True):
        if nanew:
            self.na_newlist(self.stock_listtime, listdate_tonow)
        if wins:
            self.winsorize(wins_left, wins_right)
        if gscore:
            self.getzscore()
        self.getgroup(groupnum)

    def liner_reg(self, y, x):
        xx = sm.add_constant(x)
        model = sm.OLS(y, xx)
        results = model.fit()
        return results

    def factorfree(self, y, x):
        result_tem = self.liner_reg(y, x)
        factor_free = copy.deepcopy(y - result_tem.params[0]*x)
        return factor_free



class Backtest(Stock, Index, StrategyPerformance, Factor, Graph):
    """
        该类主要是为了对策略进行回测使用
    Attr：
        1.实例化时就创建
            factor_matrix (pd.DataFrame)：
                用于储存与输入进来的因子分层结果矩阵
                input (pd.DataFrame):
            stcode (list / pd.Series)：
                 用于储存输入进来的因子分层结果矩阵中，columns中的标准WIND股票代码
            akcode  (list / pd.Series)：
                通过对stcode进行变换得到，用于储存输入进来的因子分层结果矩阵中，columns中的标准WIND股票代码对应的akshare股票代码

        2.通过 nav_daily() 函数调用创建
            returnrate_daily (pd.Series)：
                产生策略的每日收益
            nav_daily (pd.Series):
                产生策略的每日净值
            returnrate_daily_benchmark (pd.Series):
                产生基准的每日收益
            nav_daily (pd.Series):
                产生基准的每日净值

        3.通过performance()函数调用创建
            performance  (Dictionary)：
                得到记录策略一段时间内一些系列指标表现的一个字典
    Methods:
        getnav_d():
            用于返回策略对应的组合每日净值
            input：startdate(str), enddate(str), adjustment(str), price_type(str). 其中startdate和enddate输入格式为%Y%m%d，enddate默认为当前日期；adjustment表示复权选项，默认使用后复权；price_type表示价格类型，默认为收盘价。
            output：pd.DataFrame.   columns为akcodes；index元素为str
        performance():
            用于返回策略一段时间内的一系列指标
            input：startdate(str), enddate(str)，默认为全研究时间区间
            output：dictionary
    Classmethod：



    """

    def __init__(self, return_matrix, factor_matrix, stock_st, stock_listtime, benchmark_nav, trade_cost=0.003,
                 shafflemonth=None, isinitialization = True):
        super(StrategyPerformance, self).__init__(factor_matrix, stock_st, stock_listtime, shafflemonth, isinitialization)
        super(Backtest, self).__init__(return_matrix)
        super(Factor, self).__init__()
        self.benchmark_nav = copy.deepcopy(benchmark_nav)
        self.position_record = {}
        self.trade_cost = trade_cost

    def getnav_daily(self, enddate="20230809", feature="positive"):
        # 产生各分组净值
        self.returnrate_daily_allgroup = pd.Series(dtype=float)
        # 对类别进行遍历
        for ii in tqdm(np.arange(self.groupnumber) + 1):
            position_record = {}
            navs = pd.Series(["NAV"], index=[0, ])
            # 对每个持仓周期进行遍历
            for i in tqdm(np.arange(len(self.factor_groupby.index))):
                sdp = self.__class__.gettd_near(self.factor_groupby.index[i], how="f").strftime("%Y%m%d")
                if i == len(self.factor_groupby.index) - 1:
                    edp = enddate
                else:
                    edp = self.factor_groupby.index[i + 1].strftime("%Y%m%d")
                self.akcode = [self.__class__.code_remix(x) for x in
                               self.factor_groupby.iloc[i][self.factor_groupby.iloc[i] == ii].index]

                position_record[sdp] = self.akcode
                allst_re_tem = self.getreturn_st(sdp, edp)
                navs_tem = allst_re_tem.mean(axis=1)
                navs_tem.loc[datetime.strptime(sdp, "%Y%m%d")] = (1 + navs_tem.loc[
                    datetime.strptime(sdp, "%Y%m%d")]) * (1 - self.trade_cost) - 1
                navs_tem.loc[datetime.strptime(edp, "%Y%m%d")] = (1 + navs_tem.loc[
                    datetime.strptime(edp, "%Y%m%d")]) * (1 - self.trade_cost) - 1
                navs = pd.concat([navs, navs_tem])
            self.position_record[ii] = position_record
            returnrate_daily = copy.deepcopy(navs.drop(navs.index[0]))
            returnrate_daily.name = str(ii)
            if self.returnrate_daily_allgroup.empty:
                self.returnrate_daily_allgroup = returnrate_daily
            else:
                self.returnrate_daily_allgroup = pd.concat([self.returnrate_daily_allgroup, returnrate_daily], axis=1,join="outer")

        # 产生各个分组的日度净值
        self.nav_daily = (self.returnrate_daily_allgroup + 1).cumprod(axis=0)

        # 产生多空收益:
        if feature == "positive":  # 判断是否为正向因子
            long_short = self.returnrate_daily_allgroup["1"].sub(self.returnrate_daily_allgroup[str(self.groupnumber)])
        else:
            long_short = self.returnrate_daily_allgroup[str(self.groupnumber)].sub(self.returnrate_daily_allgroup["1"])
        self.nav_daily_ls = (long_short + 1).cumprod()

    def renew_benchmark_nav(self, benchmark_name, benchmark_nav):
        self.benchmark_nav = copy.deepcopy(benchmark_nav)
        self.benchmark_nav.name = benchmark_name
        print("已变更业绩基准为%s，请重新进行业绩分析" % benchmark_name)

    def getperformance(self, startdate=None, enddate=None):
        if startdate is None:
            startdate = self.nav_daily.index[0].strftime("%Y%m%d")
        if enddate is None:
            enddate = self.nav_daily.index[-1].strftime("%Y%m%d")

        std = datetime.strptime(startdate, "%Y%m%d")
        edd = datetime.strptime(enddate, "%Y%m%d")
        std_rownumber = self.nav_daily.index[self.nav_daily.index == std].idmax()
        edd_rownumber = self.nav_daily.index[self.nav_daily.index == edd].idmax()

        # 生成目标期净值和收益率
        # 注意标的基金或者策略对应的nav和returnrate是DataFrame，benchmark是Series
        nav_daily_tar = copy.deepcopy(self.nav_daily.iloc[std_rownumber: edd_rownumber + 1])
        returnrate_daily_tar = copy.deepcopy(self.returnrate_daily_allgroup.iloc[std_rownumber: edd_rownumber + 1])
        nav_daily_benchmark_tar = copy.deepcopy(self.benchmark_nav.iloc[std_rownumber: edd_rownumber + 1])
        returnrate_daily_benchmark_tar = self.getreturnrate_fromnav(nav_daily_benchmark_tar, drop_head=False)

        self.performance = pd.DataFrame(
            columns=["期间年化收益", "期间日度波动率", "期间最大回撤", "期间胜率", "期间基准年化收益",
                     "期间基准日度波动率", "期间基准最大回撤"])
        for col in nav_daily_tar.columns:
            annual_rate = self.__class__.annual_rate(nav_daily_tar[col])
            std = self.__class__.std(returnrate_daily_tar[col])
            maxdrawdown = self.__class__.maxdrawdown(nav_daily_tar[col])
            winrate_daily = self.__class__.winrate(returnrate_daily_tar[col], returnrate_daily_benchmark_tar)
            annual_rate_benchmark = self.__class__.annual_rate(nav_daily_benchmark_tar)
            std_benchmark = self.__class__.std(returnrate_daily_benchmark_tar)
            maxdrawdown_benchmark = self.__class__.maxdrawdown(nav_daily_benchmark_tar)
            self.performance.loc[col] = [annual_rate, std, maxdrawdown, winrate_daily, annual_rate_benchmark,
                                         std_benchmark, maxdrawdown_benchmark]

        print("研究标的为本次业绩分析起始时间为%s, 终止时间为%s \n" % (startdate, enddate))
        print("业绩指标如下 \n")
        print(self.performance)

    @classmethod
    def getperformance_outer(cls, nav_daily, nav_benchmark, startdate=None, enddate=None):
        if startdate is None:
            startdate = nav_daily.index[0].strftime("%Y%m%d")
        if enddate is None:
            enddate = nav_daily.index[-1].strftime("%Y%m%d")

        std = datetime.strptime(startdate, "%Y%m%d")
        edd = datetime.strptime(enddate, "%Y%m%d")
        std_rownumber = np.where(nav_daily.index == std)[0][0]
        edd_rownumber = np.where(nav_daily.index == edd)[0][0]

        # 生成目标期净值和收益率
        # 注意标的基金或者策略对应的nav和returnrate是DataFrame，benchmark是Series
        nav_daily_tar = copy.deepcopy(nav_daily.iloc[std_rownumber: edd_rownumber + 1])
        returnrate_daily_tar = cls.getreturnrate_fromnav(nav_daily_tar, drop_head=False)
        nav_daily_benchmark_tar = copy.deepcopy(nav_benchmark.iloc[std_rownumber: edd_rownumber + 1])
        returnrate_daily_benchmark_tar = cls.getreturnrate_fromnav(nav_daily_benchmark_tar, drop_head=False)

        if isinstance(nav_daily, pd.DataFrame):
            performance = pd.DataFrame(
                columns=["最新净值", "期间年化收益", "期间日度波动率", "期间最大回撤", "期间胜率", "期间基准年化收益",
                         "期间基准日度波动率", "期间基准最大回撤"])
            for col in nav_daily_tar.columns:
                annual_rate = cls.annual_rate(nav_daily_tar[col])
                std = cls.std(returnrate_daily_tar[col])
                maxdrawdown = cls.maxdrawdown(nav_daily_tar[col])
                winrate_daily = cls.winrate(returnrate_daily_tar[col], returnrate_daily_benchmark_tar)
                annual_rate_benchmark = cls.annual_rate(nav_daily_benchmark_tar)
                std_benchmark = cls.std(returnrate_daily_benchmark_tar)
                maxdrawdown_benchmark = cls.maxdrawdown(nav_daily_benchmark_tar)
                performance.loc[col] = [nav_daily_tar[col].iloc[-1], annual_rate, std, maxdrawdown, winrate_daily,
                                        nav_daily_benchmark_tar[-1], annual_rate_benchmark, std_benchmark,
                                        maxdrawdown_benchmark]
            print("研究标的为本次业绩分析起始时间为%s, 终止时间为%s \n" % (startdate, enddate))
            print("业绩指标如下 \n")
            print(performance)
            return performance
        elif isinstance(nav_daily, pd.Series):
            annual_rate = cls.annual_rate(nav_daily_tar)
            std = cls.std(returnrate_daily_tar)
            maxdrawdown = cls.maxdrawdown(nav_daily_tar)
            winrate_daily = cls.winrate(returnrate_daily_tar, returnrate_daily_benchmark_tar)
            annual_rate_benchmark = cls.annual_rate(nav_daily_benchmark_tar)
            std_benchmark = cls.std(returnrate_daily_benchmark_tar)
            maxdrawdown_benchmark = cls.maxdrawdown(nav_daily_benchmark_tar)
            winrate_daily_benchmark = np.nan
            performance_values = np.array(
                [nav_daily_tar[-1], nav_daily_benchmark_tar[-1], annual_rate, annual_rate_benchmark, std, std_benchmark,
                 maxdrawdown, maxdrawdown_benchmark, winrate_daily, winrate_daily_benchmark])
            performance = pd.DataFrame(performance_values.reshape(-1, 2), columns=["标的", "基准"],
                                       index=["最新净值", "年化收益", "日度波动率", "最大回撤", "胜率"])
            print("研究标的为本次业绩分析起始时间为%s, 终止时间为%s \n" % (startdate, enddate))
            print("业绩指标如下 \n")
            print(performance)
            return performance
        else:
            print("请检查对 nav_daily 的传参类型")


class Tech_factor():
    def __init__(self, portfolio_nav):
        self.portfolio_nav = copy.deepcopy(portfolio_nav)

    # 得到bullin tracks
    def gettech_bullin(self, lags=20, k=2):
        self.bullin_days = lags
        self.track_width = k
        portfolio_bullin_mid = copy.deepcopy(self.portfolio_nav)
        portfolio_bullin_upper = copy.deepcopy(self.portfolio_nav)
        portfolio_bullin_lower = copy.deepcopy(self.portfolio_nav)
        portfolio_bullin_std = copy.deepcopy(self.portfolio_nav)
        for i in np.arange(len(portfolio_bullin_mid.index)):
            if i < lags - 1:
                portfolio_bullin_mid.iloc[i] = np.nan
                portfolio_bullin_upper.iloc[i] = np.nan
                portfolio_bullin_lower.iloc[i] = np.nan
                portfolio_bullin_std.iloc[i] = np.nan
            else:
                portfolio_bullin_mid.iloc[i] = copy.deepcopy(self.portfolio_nav.iloc[i - lags + 1:i + 1].mean())
                portfolio_bullin_std.iloc[i] = copy.deepcopy(self.portfolio_nav.iloc[i - lags + 1: i + 1].std())
                portfolio_bullin_upper.iloc[i] = copy.deepcopy(
                    portfolio_bullin_mid.iloc[i] + k * portfolio_bullin_std.iloc[i])
                portfolio_bullin_lower.iloc[i] = copy.deepcopy(
                    portfolio_bullin_mid.iloc[i] - k * portfolio_bullin_std.iloc[i])
        self.bullin = copy.deepcopy(pd.concat(
            [self.portfolio_nav, portfolio_bullin_mid, portfolio_bullin_upper, portfolio_bullin_lower,
             portfolio_bullin_std], axis=1))
        self.bullin.columns = ["self.portfolio_nav", "portfolio_bullin_mid", "portfolio_bullin_upper",
                               "portfolio_bullin_lower", "portfolio_bullin_std"]
        self.bullin_mid = copy.deepcopy(self.bullin["portfolio_bullin_mid"])
        self.bullin_upper = copy.deepcopy(self.bullin["portfolio_bullin_upper"])
        self.bullin_lower = copy.deepcopy(self.bullin["portfolio_bullin_lower"])
        self.bullin_std = copy.deepcopy(self.bullin["portfolio_bullin_std"])


class Timing(Tech_factor, Graph):

    def __init__(self, portfolio_nav, s_nav0, trade_cost, returnrate_debt, method):
        super(Timing, self).__init__(portfolio_nav)
        super(Tech_factor, self).__init__()
        # 由净值数据生成对应的收益率
        if method is None:
            method = copy.deepcopy(["bullin", 20, 2])
        returnrate_daily = copy.deepcopy(self.portfolio_nav / self.portfolio_nav.shift(1) - 1)
        returnrate_daily.iloc[0] = copy.deepcopy(self.portfolio_nav.iloc[0] - 1)
        self.portfolio_returnrate_daily = returnrate_daily
        # 进行交易系统初始化
        self.s_nav = s_nav0  # 权益部分持仓初始值
        self.d_nav = copy.deepcopy(1 - self.s_nav)  # 固收部分持仓初始值
        self.trade_cost = trade_cost  # 交易费率设置
        self.returnrate_debt = returnrate_debt  # 固收部分收益率设置
        # 交易结果初始化
        self.c_trade = 0
        self.call_times = 0
        self.sold_times = 0
        self.s_nav_daily = []
        self.d_nav_daily = []
        self.c_trade_daily = []
        if method[0] == "bullin":
            self.gettech_bullin(lags=method[1], k=method[2])  # 生成bullin tracks
            self.mid_signal = copy.deepcopy(self.portfolio_nav - self.bullin_mid)
            self.upper_signal = copy.deepcopy(self.portfolio_nav - self.bullin_upper)
            self.lower_signal = copy.deepcopy(self.portfolio_nav - self.bullin_lower)

    def position_change(self, day, dir=0, pro=0.00):  # 交易指令设计
        if dir > 0:  # 增加股票持仓
            self.c_trade += self.d_nav * pro * self.trade_cost
            buy = self.d_nav * pro * (1 - self.trade_cost)  # 在昨天收盘时买进，买进金额为债务持仓的10%, 扣除手续费后为净买入
            self.d_nav *= (1 - pro)  # 债务持仓收缩为原先份额的90%
            self.s_nav += buy  # 权益持仓增加
            self.s_nav *= (1 + self.portfolio_returnrate_daily[day])  # 权益持仓享受今日收益
            self.d_nav *= (1 + self.returnrate_debt)  # 固收持仓享受今日收益
            self.call_times += 1  # 记录一次加仓
        elif dir < 0:  # 减少股票持仓，转入固收部分
            self.c_trade += self.s_nav * pro * self.trade_cost
            sold = self.s_nav * pro * (1 - self.trade_cost)  # 在昨天收盘时卖出，卖出金额为权益持仓的10%, 扣除手续费后为净卖出
            self.s_nav *= (1 - pro)  # 权益持仓收缩为原先份额的90%
            self.d_nav += sold  # 债务持仓增加
            self.s_nav *= (1 + self.portfolio_returnrate_daily[day])  # 权益持仓享受今日收益
            self.d_nav *= (1 + self.returnrate_debt)  # 固收持仓享受今日收益
            self.sold_times += 1  # 记录一次减仓
        else:
            self.s_nav *= (1 + self.portfolio_returnrate_daily[day])  # 权益持仓享受今日收益
            self.d_nav *= (1 + self.returnrate_debt)  # 固收持仓享受今日收益

        self.s_nav_daily += [copy.deepcopy(self.s_nav), ]  # 记录每一天权益部分nav
        self.d_nav_daily += [copy.deepcopy(self.d_nav), ]  # 记录每一天固收部分nav
        self.c_trade_daily += [copy.deepcopy(self.c_trade), ]  # 记录每一天自开始交易产生的手续费

    def gettradsig_bullin(self, ext_valdays=2, mid_valdays = 5):
        self.sigseries_bullin = pd.Series(np.zeros(len(self.portfolio_nav.index)), index=self.portfolio_nav.index,
                                          name="signal")  # 产生信号序列
        for i in tqdm(np.arange(len(self.portfolio_nav.index))):
            iii = 0
            if i < (self.bullin_days + max(ext_valdays, mid_valdays) + 1):
                pass
            else:
                # 没有争议的地方在于，当价格下行穿过上轨，并保持下降惯性，应该做空；当价格上行穿过下轨，并保持上升惯性，应该做多
                # 其他地方，对于上下轨道而言，强动力突破选择趋势跟踪，弱动力突破则均值回归，同时均值回归过程中会重现上述的强动力环境，所以做轻微的仓位管理
                if (self.upper_signal[i - ext_valdays - 1] * self.upper_signal[i - ext_valdays - 2]) < 0:  # 上轨异号
                    validation_series = self.upper_signal[i - 2*ext_valdays - 1: i]  # 验证1：股价走势
                    validation_series2 = self.bullin_std[i - 2*ext_valdays - 1: i]  # 验证2 ：波动率走势
                    if self.upper_signal[i - ext_valdays - 1] < 0:  # 发生向下突破上轨的信号
                        if validation_series.is_monotonic_decreasing:  # 向下惯性较强，确认该信号是超强减仓信号
                            iii = -5
                        else:  # 向下惯性并不强，从控制回撤的角度出发，做减仓处理，后续回涨会加仓
                            iii = -3
                    elif self.upper_signal[i - ext_valdays - 1] > 0: # 发生向上突破上轨的信号
                        if validation_series.is_monotonic_increasing & validation_series2.is_monotonic_decreasing:  # 向上惯性较强，确认该信号是超强加仓信号
                            iii = 5  # 向上惯性较强，确认该信号是超强加仓信号
                        elif ((validation_series.is_monotonic_increasing) and not (validation_series2.is_monotonic_decreasing)) or ((validation_series2.is_monotonic_decreasing) and not (validation_series.is_monotonic_increasing)):
                            iii = 3
                        else:  # 向上惯性较弱，会均值回归，减仓，但是减仓强度不应过大，因为后续如果真的回调，会下行穿过上轨，会释放更强的减仓信号
                            iii = -1
                if (self.lower_signal[i - ext_valdays - 1] * self.lower_signal[i - ext_valdays - 2]) < 0:  # 下轨异号
                    validation_series = self.lower_signal[i - 2*ext_valdays - 1: i]  # 验证思路：下轨距走大
                    validation_series2 = self.bullin_std[i - 2*ext_valdays - 1: i]
                    if self.lower_signal[i - ext_valdays - 1] > 0:  # 发生向上突破下轨的信号
                        if validation_series.is_monotonic_increasing:  # 向上惯性较强，确认该信号是超强加仓信号
                            iii = 5
                        else:  # 向上惯性较弱，确认该信号是弱强加仓信号，但后续若向下转跌，从趋势跟踪角度出发会减仓；向上看，均值回复有一定空间。因此可以适当多加仓位
                            iii = 4
                    else:  # 发生向下突破下轨的信号
                        if validation_series.is_monotonic_decreasing & validation_series2.is_monotonic_increasing:  # 向下惯性较强, 确认是较强减仓信号
                            iii = -5
                        elif ((validation_series.is_monotonic_increasing) and not (validation_series2.is_monotonic_decreasing)) or ((validation_series2.is_monotonic_decreasing) and not (validation_series.is_monotonic_increasing)):
                            iii = -3
                        else:  # 向下动力不强，从均值回复视角看，可小幅加仓，因为后续如果突破会强加仓
                            iii = 1
                if (self.mid_signal[i - mid_valdays - 1] * self.mid_signal[i - mid_valdays - 2]) < 0:  # 中轨异号
                    validation_series = self.mid_signal[i - 2*mid_valdays - 1: i]  # 无论如何生成一个验证序列
                    validation_series2 = self.bullin_std[i - 2*mid_valdays - 1: i]
                    if self.mid_signal[i - mid_valdays - 1] > 0:  # 发生向上突破中轨的信号
                        if validation_series.is_monotonic_increasing | validation_series2.is_monotonic_increasing:  # 向上惯性较强，确认该信号是较强加仓信号
                                iii = 2
                    else:  # 发生向下突破中轨的信号
                        if validation_series.is_monotonic_decreasing | validation_series2.is_monotonic_increasing:  # 向下惯性较强, 确认是较强减仓信号
                                iii = -2
            self.sigseries_bullin.iloc[i] = iii

    def bullin_timing(self, ext_valdays, mid_valdays, sign_to_pro=pd.Series([0.05, 0.1, 0.15, 0, 0.05, 0.1, 0.15], index=[1, 2, 3, 0, -1, -2, -3])):

        self.gettradsig_bullin(ext_valdays = ext_valdays, mid_valdays = mid_valdays)

        for i in np.arange(len(self.mid_signal.index)):
            if i <= (self.bullin_days + max(ext_valdays, mid_valdays) + 1):  # 信号无法计算或验证的时期不进行调仓
                self.position_change(day=i)
            else:
                self.position_change(day=i, dir=self.sigseries_bullin.iloc[i], pro=sign_to_pro.loc[self.sigseries_bullin.iloc[i]])

        self.a_nav = copy.deepcopy(self.s_nav + self.d_nav)
        self.portfoliomod_snav_daily = pd.Series(self.s_nav_daily, name="equity_nav", index=self.mid_signal.index)
        self.porfoliomod_dnav_daily = pd.Series(self.d_nav_daily, name="debt_nav", index=self.mid_signal.index)
        self.porfoliomod_anav_daily = copy.deepcopy(self.portfoliomod_snav_daily + self.porfoliomod_dnav_daily)
        self.porfoliomod_anav_daily.name = "all_nav"
        self.c_trade_daily = pd.Series(self.c_trade_daily, name="all_trade_cost", index=self.mid_signal.index)

        print("权益持仓额为%f" % self.s_nav)
        print("债务持仓额为%f" % self.d_nav)
        print("择时买入%i次" % self.call_times)
        print("择时卖出%i次" % self.sold_times)
        print("交易成本为%f" % self.c_trade)
        print("总资产净值为%f" % self.a_nav)

####以下为函数区域



def liner_reg(y, x , add_constant = True):
    xx = sm.add_constant(x)
    model = sm.OLS(y, xx)
    results = model.fit()
    return results

def factorfree(y, x):
    result_tem = liner_reg(y, x)
    factor_free = copy.deepcopy(y - result_tem.params[0] * x)
    return factor_free

def getreturnrate_fromnav(nav, drop_head=False):
    returnrate = copy.deepcopy(nav / nav.shift(1) - 1)
    if drop_head:
        returnrate.drop(returnrate.index[0], inplace=True)
    else:
        returnrate.iloc[0] = copy.deepcopy(nav.iloc[0] - 1)
    return returnrate








