import numpy as np
import pandas as pd
import KNNLearner as knn
import LinRegLearner as ll
import BagLearner as bl
import datetime as dt
import matplotlib.pyplot as plt
from sklearn import linear_model
from util import get_data, plot_data

# get price data: Sine, IBM
start_date = dt.datetime(2007,12,31)
end_date = dt.datetime(2009,12,31)
symbols = ['IBM','SINE_FAST','SINE_SLOW','GOOG','AAPL']
dates = pd.date_range(start_date, end_date)
prices_all = get_data(symbols, dates)

tag = 'IBM'
pibm = prices_all[tag]

# contruct features X
def get_feature(pibm):
    indates = dates[1:]
    sma = pibm.rolling(window = 20, min_periods=0)
    bbup = sma.mean() + 2*sma.std() 
    bblow = sma.mean() - 2*sma.std() 
    bbvals = (pibm[1:] - sma.mean()[1:])/(4*sma.std()[1:])
    vtl = sma.std()[1:]/sma.mean()[1:]*8
    #mmtn1 = pibm.values[1:]/pibm.values[:-1]-1  
    #mmtn2 = pibm.values[2:]/pibm.values[:-2]-1
    #mmtn3 = pibm.values[3:]/pibm.values[:-3]-1
    #mmtn4 = pibm.values[4:]/pibm.values[:-4]-1
    mmtn5 = pibm.values[5:]/pibm.values[:-5]-1
    X = pd.DataFrame({'x0':bbvals[4:-5], 'vtl':vtl[4:-5],'x5':mmtn5[:-5]})
    return X, bbvals[4:-5]

# construct Y
def get_Y(pibm):
    Y = pibm.values[5:] 
    Y = Y[5:]/Y[:-5] - 1
    return Y

def trade_naive(pfl):
    for idx in range(pfl.shape[0]-1):
        if pfl['pred'].ix[idx] < pfl['pred'].ix[idx+1]:
            if pfl['shares'].ix[idx] <= 0:
                pfl['cash'].ix[idx:] = pfl['cash'].ix[idx] - pfl['price'].ix[idx]*(100 - pfl['shares'].ix[idx])
                pfl['shares'].ix[idx:] = 100
        if pfl['pred'].ix[idx] > pfl['pred'].ix[idx+1]:
            if pfl['shares'].ix[idx] >= 0:
                pfl['cash'].ix[idx:] = pfl['cash'].ix[idx] + pfl['price'].ix[idx]*(100 + pfl['shares'].ix[idx])
                pfl['shares'].ix[idx:] = -100
    pv = pfl['price']*pfl['shares'] + pfl['cash']
    return pv 

def trade(pfl):
    if pfl['price'].ix[0] < pfl['pema40'].ix[0] and pfl['pema40'].ix[0] < pfl['pema40'].ix[5]:
        pfl['shares'][:] = 100
        pfl['cash'][:] = pfl['cash'].ix[0] - pfl['price'].ix[0]*100
    elif pfl['price'].ix[0] > pfl['pema40'].ix[0] and pfl['pema40'].ix[0] > pfl['pema40'].ix[5]:
        pfl['shares'][:] = -100
        pfl['cash'][:] = pfl['cash'].ix[0] + pfl['price'].ix[0]*100

    sigs = pfl['price'].values - pfl['ema40'].values
    for idx in range(1, pfl.shape[0]):
        if sigs[idx]*sigs[idx-1] < 0:
            if sigs[idx] > 0 and pfl['shares'].ix[idx] <= 0:
                pfl['cash'].ix[idx:] = pfl['cash'].ix[idx] - pfl['price'].ix[idx]*(100 - pfl['shares'].ix[idx])
                pfl['shares'].ix[idx:] = 100
            if sigs[idx] < 0 and pfl['shares'].ix[idx] >= 0:
                pfl['cash'].ix[idx:] = pfl['cash'].ix[idx] + pfl['price'].ix[idx]*(100 + pfl['shares'].ix[idx])
                pfl['shares'].ix[idx:] = -100
    return pfl

def train(X, Y):
    kl = knn.KNNLearner()
    Ypred = np.zeros(Y.size)
    Ypred[:5] = Y[:5]
    for i in range(5, X.shape[0]):
        kl.addEvidence(X.values[:i], Y[:i])
        Ypred[i] = kl.query(X.values[i])[0]
    return Ypred, kl

                
    
#----------------------In-sample test-----------------------------#
X, _ = get_feature(pibm)
Y = get_Y(pibm)

Ypred, kl = train(X, Y)
# convert predicted Y back to price, in-sample backtest
ppred = pibm.values[5:-5]*(Ypred + 1)

pdiff = pd.DataFrame(index = pibm.index[10:], data = {'price':pibm.values[10:], 'pred':ppred})
plot_data(pdiff)

ppred = pd.Series(index = pibm.index[10:], data = ppred)# convert numpy array to pandas.Series
ema40 = pibm.ewm(span = 40, min_periods=0).mean()
pema40 = pd.concat((pibm[:10],ppred)).ewm(span = 40, min_periods=0).mean()
# initial portfolio
pfl = pd.DataFrame({'price':pibm[10:], 'ema40':ema40[10:], 'pema40':pema40[10:], 'shares':np.zeros(pibm.size-10), 'cash':np.ones(pibm.size-10)*10000})


# trading
pfl = trade(pfl)
pv = pfl['price']*pfl['shares'] + pfl['cash']
pspy = prices_all['SPY'][pfl.index]
pfl_vs_spy = pd.DataFrame(index = pfl.index, data = {'my_portval':pv/pv.ix[0], 'SPY':pspy/pspy.ix[0]})
plot_data(pfl_vs_spy, title = "My_Portfolio vs SPY", ylabel = "Accumulative Return")


#------------------------Out-Sample test---------------------------# 
tsd = dt.datetime(2009,12,31)
ted = dt.datetime(2011,12,31)
symbols = [tag]
dates = pd.date_range(tsd, ted)
tprices = get_data(symbols, dates)
tpibm = tprices[tag]

tX, _ = get_feature(tpibm)
# compare to the true price
tYpred = kl.query(tX.values)
tppred = tpibm.values[5:-5]*(tYpred + 1)
tppred = pd.Series(index = tpibm.index[10:], data = tppred)# convert numpy array to pandas.Series
#tppred = tpibm.values[5:-5]*(tX.values.dot(clf.coef_.T) + clf.intercept_ + 1)
tema40 = tpibm.ewm(span = 40, min_periods=0).mean()
tpema40 = pd.concat((tpibm[:10],tppred)).ewm(span = 40, min_periods=0).mean()
# initial portfolio
tpfl = pd.DataFrame({'price':pibm[10:], 'ema40':tema40[10:], 'pema40':tpema40[10:], 'shares':np.zeros(pibm.size-10), 'cash':np.ones(pibm.size-10)*10000})

tpdiff = pd.DataFrame(index = tpibm.index[10:], data = {'price':tpibm.values[10:], 'pred':tppred})
plot_data(tpdiff)

tpfl = pd.DataFrame({'price':tpibm[10:], 'pred':tppred, 'shares':np.zeros(tppred.size), 'cash':np.ones(tppred.size)*10000})

tpfl = trade(tpfl)
tpv = tpfl['price']*tpfl['shares'] + tpfl['cash']
tpspy = tprices['SPY'][tpfl.index]
tpfl_vs_tspy = pd.DataFrame(index = tpfl.index, data = {'my_portval':tpv/tpv.ix[0], 'SPY':tpspy/tpspy.ix[0]})
plot_data(tpfl_vs_tspy, title = "My_Portfolio vs SPY", ylabel = "Accumulative Return")


# For report


