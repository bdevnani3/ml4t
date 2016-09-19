"""MC1-P1: Analyze a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','MSFT'], \
    allocs=[0.3,0.3,0.1,0.3], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY

    # normalize all prices
    for sym in ['SPY']+syms:
        prices_all[sym] = prices_all[sym]/prices_all[sym][0] # add code here to compute daily portfolio values

    prices = prices_all[syms]  # only portfolio symbols, no SPY
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    cr, adr, sddr, sr, dailyrets, port_val = \
    compute_portfolio_stats(prices=prices, allocs=allocs, rfr=rfr, sf=sf)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp)
    #     pass
    # Add code here to properly compute end value
    ev = sv * port_val[-1]

    return cr, adr, sddr, sr, ev


def compute_portfolio_stats(prices = [0.0, 0.0, 0.0, 0.0, 0.0],\
    allocs = [0.1,0.2,0.3,0.4], rfr = 0.0, sf = 252.0):

    # Get portfolio statistics (note: std_daily_ret = volatility)
    # Get daily portfolio value
    port_val = np.sum(prices*allocs,axis=1)
    cr = port_val[-1]/port_val[0] - 1
    daily_return = port_val[1:].values/port_val[0:-1].values-1
    adr = daily_return.mean()
    sddr = daily_return.std(ddof=1)
    sr = sf**(1/2.0) * (adr - rfr)/sddr

    return cr, adr, sddr, sr, daily_return, port_val

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2009,1,1)
    end_date = dt.datetime(2009,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'MSFT']
    allocations = [0.3, 0.3, 0.1, 0.3]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    test_code()