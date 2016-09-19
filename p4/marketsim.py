# coding: utf-8
"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from math import *
from analysis import assess_portfolio 
from util import get_data, plot_data

class MarketsimTestCase(object):
    def __init__(self, description, group, inputs, outputs):
        self.description = description
        self.group = group 
        self.inputs = inputs 
        self.outputs = outputs 
        
marketsim_test_cases = [
    MarketsimTestCase(
        description="Orders 1",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-01.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 245 ,
            last_day_portval = 1115569.2 ,
            sharpe_ratio = 0.612340613407 ,
            avg_daily_ret = 0.00055037432146
        )
    ),
    MarketsimTestCase(
        description="Orders 2",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-02.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 245 ,
            last_day_portval = 1095003.35 ,
            sharpe_ratio = 1.01613520942 ,
            avg_daily_ret = 0.000390534819609
        )
    ),
    MarketsimTestCase(
        description="Orders 3",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-03.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 240 ,
            last_day_portval = 857616.0 ,
            sharpe_ratio = -0.759896272199 ,
            avg_daily_ret = -0.000571326189931
        )
    ),
    MarketsimTestCase(
        description="Orders 4",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-04.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 233 ,
            last_day_portval = 923545.4 ,
            sharpe_ratio = -0.266030146916 ,
            avg_daily_ret =  -0.000240200768212
        )
    ),
    MarketsimTestCase(
        description="Orders 5",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-05.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 296 ,
            last_day_portval = 1415563.0 ,
            sharpe_ratio = 2.19591520826 ,
            avg_daily_ret = 0.00121733290744
        )
    ),
    MarketsimTestCase(
        description="Orders 6",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-06.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 210 ,
            last_day_portval = 894604.3 ,
            sharpe_ratio = -1.23463930987,
            avg_daily_ret =  -0.000511281541086
        )
    ),
    MarketsimTestCase(
        description="Orders 7 (modified)",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-07-modified.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 237 ,
            last_day_portval = 1104930.8 ,
            sharpe_ratio = 2.07335994413 ,
            avg_daily_ret = 0.000428245010481
        )
    ),
    MarketsimTestCase(
        description="Orders 8 (modified)",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-08-modified.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 229 ,
            last_day_portval = 1071325.1 ,
            sharpe_ratio =  0.896734443277,
            avg_daily_ret = 0.000318004442115
        )
    ),
    MarketsimTestCase(
        description="Orders 9 (modified)",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-09-modified.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 37 ,
            last_day_portval = 1058990.0,
            sharpe_ratio = 2.54864656282 ,
            avg_daily_ret = 0.00164458341408
        )
    ),
    MarketsimTestCase(
        description="Orders 10 (modified)",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-10-modified.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 141 ,
            last_day_portval = 1070819.0,
            sharpe_ratio = 1.0145855303,
            avg_daily_ret =  0.000521814978394
        )
    ),
    MarketsimTestCase(
        description="Orders 11 - Leveraged SELL (modified)",
        group='leverage',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-11-modified.csv'),
            start_val=1000000
        ),
        outputs=dict(
            last_day_portval = 1053560.0
        )
    ),
    MarketsimTestCase(
        description="Orders 12 - Leveraged BUY (modified)",
        group='leverage',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-12-modified.csv'),
            start_val=1000000
        ),
        outputs=dict(
            last_day_portval = 1044437.0
        )
    ),
    MarketsimTestCase(
        description="Wiki leverage example #1",
        group='leverage',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-leverage-1.csv'),
            start_val=1000000
        ),
        outputs=dict(
            last_day_portval = 1050160.0
        )
    ),
    MarketsimTestCase(
        description="Wiki leverage example #2",
        group='leverage',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-leverage-2.csv'),
            start_val=1000000
        ),
        outputs=dict(
            last_day_portval = 1074650.0
        )
    ),
    MarketsimTestCase(
        description="Wiki leverage example #3",
        group='leverage',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-leverage-3.csv'),
            start_val=1000000
        ),
        outputs=dict(
            last_day_portval = 1050160.0
        )
    ),
]

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    orders_df = orders_df.sort_index()
    orderdates = orders_df.index.unique()
    start_date = orders_df.index[0] 
    end_date = orders_df.index[-1]
    symbols = orders_df['Symbol'].unique().tolist()
    dates = pd.date_range(start_date, end_date)
    prices = get_data(symbols, dates)
    symbols.append('cash')
    portvals = pd.DataFrame(data = np.zeros((prices.index.size, len(symbols))), index = prices.index, columns=symbols)
    DATES = prices.index.to_datetime()
    portvals['cash'] = start_val 
   
    for i, odate in enumerate(orderdates):
        orders = orders_df[odate:odate]
        temp = portvals[odate:odate].copy()
        longs = temp.ix[0][np.where(temp.ix[0] > 0)[0]].keys().tolist()
        if 'cash' in longs:
            longs.remove('cash')
        shorts = temp.ix[0][np.where(temp.ix[0] < 0)[0]].keys().tolist()
        if 'cash' in shorts:
            shorts.remove('cash')
        
        if len(longs) > 0:
            longsum = (prices[odate:odate][longs]*temp[longs]).sum(axis = 'columns')
        else:
            longsum = 0
        if len(shorts) > 0:
            shortsum = (prices[odate:odate][shorts]*temp[shorts]).sum(axis = 'columns')
        else:
            shortsum = 0
        lev1= (longsum + shortsum)/(longsum - shortsum + temp['cash'].values[0] ) 
        if isinstance(lev1, pd.Series):
            lev1 = lev1.ix[0]

        for j in range(orders.shape[0]):
            if orders.ix[j]['Order'] == 'BUY':
                temp[orders.ix[j]['Symbol']] += orders.ix[j]['Shares']
                temp['cash'] -= orders.ix[j]['Shares'] * prices.ix[odate][orders.ix[j]['Symbol']]
            else:
                temp[orders.ix[j]['Symbol']] -= orders.ix[j]['Shares']
                temp['cash'] += orders.ix[j]['Shares'] * prices.ix[odate][orders.ix[j]['Symbol']]
        longs = temp.ix[0][np.where(temp.ix[0] > 0)[0]].keys().tolist()
        if 'cash' in longs:
            longs.remove('cash')
        shorts = temp.ix[0][np.where(temp.ix[0] < 0)[0]].keys().tolist()
        if 'cash' in shorts:
            shorts.remove('cash')

        if len(longs) > 0:
            longsum = (prices[odate:odate][longs]*temp[longs]).sum(axis = 'columns')
        else:
            longsum = 0
        if len(shorts) > 0:
            shortsum = (prices[odate:odate][shorts]*temp[shorts]).sum(axis = 'columns')
        else:
            shortsum = 0
        lev2= (longsum + shortsum)/(longsum - shortsum + temp['cash'].values[0] ) 
        if isinstance(lev2, pd.Series):
            lev2 = lev2.ix[0]

        #if lev2 <= 2.0:
        if lev2 <= 2.0 or (lev1 > 2.0 and lev2 < lev1):
            td = np.where(DATES >= orderdates[i])[0].size
            portvals[orderdates[i]:] = np.repeat(temp.values, td, axis = 0 ) 
    
    symbols.remove('cash')
    tv = np.sum(portvals[symbols]*prices[symbols], axis = 1) + portvals['cash']
    pv = pd.DataFrame({'PortVal':tv})
    print isinstance(pv, pd.DataFrame)
    return pv 

def assess_port(port_val, sf=252.0, rfr = 0.0):
   
    rdr = port_val[1:]/port_val.values[:-1] - 1
    cr = port_val.values/port_val.values[0] 
    adr = rdr.mean()
    sddr = rdr.std()
    sr = (adr -rfr)/sddr*np.sqrt(sf)
    
    return (cr-1)[-1], adr,sddr,sr, port_val.ix[-1]
    


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders.csv"
    startv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = startv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio,_ = assess_port(portvals)
    
    start_date = portvals.index.to_datetime()[0]
    end_date = portvals.index.to_datetime()[-1]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY, _ =     assess_portfolio(sd = start_date, ed = end_date, syms = ['SPY'],allocs = [1.0],                    sv = startv)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

def runtestcases(cases):
    errors = []
    for case in cases:
        of, sv = case.inputs['orders_file'], case.inputs['start_val']
        print case.description, ':  ', of
        pv = compute_portvals(of, sv)
        num_days = len(pv.index)
        cr, adr, sddr, sr, ev = assess_port(pv)
        err = []
        if case.group == 'basic':
            print "Error in number of days",            np.abs(num_days - case.outputs['num_days'])/case.outputs['num_days'] 
            err.append(np.abs(num_days - case.outputs['num_days'])/case.outputs['num_days'])
            print "Error in last day value: ",             np.abs(ev - case.outputs['last_day_portval'])/case.outputs['last_day_portval']
            err.append(np.abs(ev - case.outputs['last_day_portval'])/case.outputs['last_day_portval'])
            print "Error in sharpe ratio",            np.abs(sr - case.outputs['sharpe_ratio'])/case.outputs['sharpe_ratio']
            err.append(np.abs(sr - case.outputs['sharpe_ratio'])/case.outputs['sharpe_ratio'])
            print "Error in average daily return",            np.abs(adr - case.outputs['avg_daily_ret'])/case.outputs['avg_daily_ret']
            err.append(np.abs(adr - case.outputs['avg_daily_ret'])/case.outputs['avg_daily_ret'])
            print '----------------------------------------------------------'
            print 
        else:
            print "Error in last day value: ",             np.abs(ev - case.outputs['last_day_portval'])/case.outputs['last_day_portval']
            print '----------------------------------------------------------'
            print
            err.append(np.abs(ev - case.outputs['last_day_portval'])/case.outputs['last_day_portval'])
        errors.append(np.array(err))
    return errors
        
    
if __name__ == "__main__":
    test_code()
    errors = runtestcases(marketsim_test_cases)
    print '=============TEST CASE ERRORS===================='
    print errors
