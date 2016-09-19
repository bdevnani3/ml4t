import numpy as np
import pandas as pd
import KNNLearner as knn
import datetime as dt
import matplotlib.pyplot as plt
from sklearn import linear_model
from util import get_data, plot_data

# get price data: Sine, IBM
start_date = dt.datetime(2007,12,31)
end_date = dt.datetime(2009,12,31)
symbols = ['IBM','ML4T-240']
dates = pd.date_range(start_date, end_date)
prices_all = get_data(symbols, dates)

tag = 'IBM'
pibm = prices_all[tag]

# These two parameters decides what trading policies to use
method = 'ema'   # 'ema' or 'pema' 
                 # 'ema': use exponential moving average of true price(thus can't predict, but more precise)
                 # 'pema': use exponential moving average of predicted price(less precise, but can predict 5 days trend)
level = 'simple' # 'simple' or 'advance': see function definiton below
# contruct features X
def get_feature(pibm):
    indates = dates[1:]
    sma = pibm.rolling(window = 20, min_periods=0)
    bbup = sma.mean() + 2*sma.std() 
    bblow = sma.mean() - 2*sma.std() 
    bbvals = (pibm[1:] - sma.mean()[1:])/(4*sma.std()[1:])
    vtl = sma.std()[1:]/sma.mean()[1:]*8
    mmtn5 = pibm.values[5:]/pibm.values[:-5]-1
    X = pd.DataFrame({'x0':bbvals[4:-5], 'vtl':vtl[4:-5],'x5':mmtn5[:-5]})
    return X, bbvals[4:-5]

# construct Y
def get_Y(pibm):
    Y = pibm.values[5:] 
    Y = Y[5:]/Y[:-5] - 1
    return Y

def trade_naive(pfl):
    for idx in range(pfl.shape[0]-5):
        if pfl['pred'].ix[idx] < pfl['pred'].ix[idx+5]:
            if pfl['shares'].ix[idx] <= 0:
                pfl['cash'].ix[idx:] = pfl['cash'].ix[idx] - pfl['price'].ix[idx]*(100 - pfl['shares'].ix[idx])
                pfl['shares'].ix[idx:] = 100
        if pfl['pred'].ix[idx] > pfl['pred'].ix[idx+5]:
            if pfl['shares'].ix[idx] >= 0:
                pfl['cash'].ix[idx:] = pfl['cash'].ix[idx] + pfl['price'].ix[idx]*(100 + pfl['shares'].ix[idx])
                pfl['shares'].ix[idx:] = -100
    pv = pfl['price']*pfl['shares'] + pfl['cash']
    return pv 

def trade(pfl, method = 'ema', level = 'simple'): #method = 'ema' or 'pema'
    bds = []
    sds = []
    # Initial condition has to use 'pema' cuz we can't predict using 'ema'
    # same for both 'simple' and 'advance' 
    if level == "naive":
        for idx in range(pfl.shape[0]-5):
            if pfl['pred'].ix[idx] < pfl['pred'].ix[idx+5]:
                if pfl['shares'].ix[idx] <= 0:
                    pfl['cash'].ix[idx:] = pfl['cash'].ix[idx] - pfl['price'].ix[idx]*(100 - pfl['shares'].ix[idx])
                    pfl['shares'].ix[idx:] = 100
                    bds.append(idx)
            if pfl['pred'].ix[idx] > pfl['pred'].ix[idx+5]:
                if pfl['shares'].ix[idx] >= 0:
                    pfl['cash'].ix[idx:] = pfl['cash'].ix[idx] + pfl['price'].ix[idx]*(100 + pfl['shares'].ix[idx])
                    pfl['shares'].ix[idx:] = -100   
                    sds.append(idx)
        return pfl, bds, sds
    if pfl['pema40'].ix[5] > pfl['pema40'].ix[0]:
        pfl['shares'][:] = 100
        pfl['cash'][:] = pfl['cash'].ix[0] - pfl['price'].ix[0]*100
        bds.append(0)
    if pfl['pema40'].ix[5] < pfl['pema40'].ix[0]:
        pfl['shares'][:] = -100
        pfl['cash'][:] = pfl['cash'].ix[0] + pfl['price'].ix[0]*100   
        sds.append(0)
    if level == 'simple':
        for idx in range(1, pfl.shape[0]):
            if  pfl[method+'10'].ix[idx] > pfl[method+'40'].ix[idx] and pfl[method+'10'].ix[idx-1] < pfl[method+'40'].ix[idx-1] and pfl['shares'].ix[idx] <= 0 :
                pfl['cash'].ix[idx:] = pfl['cash'].ix[idx] - pfl['price'].ix[idx]*(100 - pfl['shares'].ix[idx])
                pfl['shares'].ix[idx:] = 100
                bds.append(idx)
            if  pfl[method+'10'].ix[idx] < pfl[method+'40'].ix[idx] and pfl[method+'10'].ix[idx-1] > pfl[method+'40'].ix[idx-1] and pfl['shares'].ix[idx] >= 0 :
                pfl['cash'].ix[idx:] = pfl['cash'].ix[idx] + pfl['price'].ix[idx]*(100 + pfl['shares'].ix[idx])
                pfl['shares'].ix[idx:] = -100
                sds.append(idx)
    if level == 'advance':
        for idx in range(2, pfl.shape[0]):
            if pfl['ema10'].ix[idx] > pfl['ema40'].ix[idx]: 
                if pfl['ema10'].ix[idx] < pfl['ema10'].ix[idx-1] and pfl['ema10'].ix[idx-1] > pfl['ema10'].ix[idx-2]: # strong sell, local maxima above ema40
                    if pfl['shares'].ix[idx] >= 0:
                        slope = (pfl['price'].ix[idx] - pfl['price'].ix[bds[-1]])/(idx - bds[-1])/pfl['price'].ix[bds[-1]]
                        thrsd = (pfl['ema40'].ix[idx] - pfl['ema40'].ix[bds[-1]])/(idx - bds[-1])/pfl['ema40'].ix[bds[-1]]
                        if slope > max(0, thrsd):
                            pfl['cash'].ix[idx:] = pfl['cash'].ix[idx] + pfl['price'].ix[idx]*(100 + pfl['shares'].ix[idx])
                            pfl['shares'].ix[idx:] = -100
                            sds.append(idx)
            if pfl['ema10'].ix[idx] < pfl['ema40'].ix[idx]: 
                if pfl['ema10'].ix[idx] > pfl['ema10'].ix[idx-1] and pfl['ema10'].ix[idx-1] < pfl['ema10'].ix[idx-2]: # strong buy, local minima below ema40
                    if pfl['shares'].ix[idx] <= 0:
                        slope = (pfl['price'].ix[idx] - pfl['price'].ix[sds[-1]])/(idx - sds[-1])/pfl['price'].ix[sds[-1]]
                        thrsd = (pfl['ema40'].ix[idx] - pfl['ema40'].ix[sds[-1]])/(idx - sds[-1])/pfl['ema40'].ix[sds[-1]]
                        #if slope < min(slope, thrsd):
                        pfl['cash'].ix[idx:] = pfl['cash'].ix[idx] - pfl['price'].ix[idx]*(100 - pfl['shares'].ix[idx])
                        pfl['shares'].ix[idx:] = 100
                        bds.append(idx)
    return pfl, bds, sds

def train(X, Y):
    kl = knn.KNNLearner()
    Ypred = np.zeros(Y.size)
    Ypred[:5] = Y[:5]
    for i in range(5, X.shape[0]):
        kl.addEvidence(X.values[:i], Y[:i])
        Ypred[i] = kl.query(X.values[i])[0]
    return Ypred, kl
def plot_pfl(pfl, bds, sds, method = 'ema'):
    plt.subplot(211)
    
    plt.subplot(212) 
    plt.plot(pfl.index, pfl['price'], label = 'price')
    plt.plot(pfl.index, pfl[method+'10'], label = method+'10')
    plt.plot(pfl.index, pfl[method+'40'], label = method+'40')
    plt.legend()
    for idx in bds:
        plt.axvline(pfl.index[idx], color = 'green')
    for idx in sds:
        plt.axvline(pfl.index[idx], color = 'red')
    plt.show()
    plt.clf()




   
#----------------------In-sample test-----------------------------#
X, _ = get_feature(pibm)
Y = get_Y(pibm)

Ypred, kl = train(X, Y)
# convert predicted Y back to price, in-sample backtest
ppred = pibm.values[5:-5]*(Ypred + 1)

pdiff = pd.DataFrame(index = pibm.index[10:], data = {'price':pibm.values[10:], 'pred':ppred})
plot_data(pdiff)

ppred = pd.Series(index = pibm.index[10:], data = ppred)# convert numpy array to pandas.Series
ema10 = pibm.ewm(span = 10, min_periods=0).mean()
ema40 = pibm.ewm(span = 40, min_periods=0).mean()
pema10 = pd.concat((pibm[:10],ppred)).ewm(span = 10, min_periods=0).mean()
pema40 = pd.concat((pibm[:10],ppred)).ewm(span = 40, min_periods=0).mean()
# initial portfolio
pfl = pd.DataFrame({'price':pibm[10:],'pred':ppred, 'ema40':ema40[10:], 'ema10':ema10[10:],'pema10':pema10[10:],'pema40':pema40[10:], 'shares':np.zeros(pibm.size-10), 'cash':np.ones(pibm.size-10)*10000})


# trading
pfl, bds, sds = trade(pfl, method, level)
pv = pfl['price']*pfl['shares'] + pfl['cash']
pspy = prices_all['SPY'][pfl.index]
pfl_vs_spy = pd.DataFrame(index = pfl.index, data = {'my_portval':pv/pv.ix[0], 'SPY':pspy/pspy.ix[0]})
#plot_data(pfl_vs_spy, title = "My_Portfolio vs SPY", ylabel = "Accumulative Return")
#plot_pfl(pfl, bds, sds, method)
print bds, sds



#------------------------Out-Sample test---------------------------# 
tsd = dt.datetime(2009,12,31)
ted = dt.datetime(2011,12,31)
symbols = [tag]
dates = pd.date_range(tsd, ted)
tprices = get_data(symbols, dates)
tpibm = tprices[tag]

tspy = get_data(['SPY'],dates) 

tX, _ = get_feature(tpibm)
# compare to the true price
tYpred = kl.query(tX.values)
tppred = tpibm.values[5:-5]*(tYpred + 1)
tppred = pd.Series(index = tpibm.index[10:], data = tppred)# convert numpy array to pandas.Series
#tppred = tpibm.values[5:-5]*(tX.values.dot(clf.coef_.T) + clf.intercept_ + 1)
tema10 = tpibm.ewm(span = 10, min_periods=0).mean()
tema40 = tpibm.ewm(span = 40, min_periods=0).mean()
tpema10 = pd.concat((tpibm[:10],tppred)).ewm(span = 10, min_periods=0).mean()
tpema40 = pd.concat((tpibm[:10],tppred)).ewm(span = 40, min_periods=0).mean()
# initial portfolio
tpfl = pd.DataFrame({'price':tpibm[10:],'pred':tppred, 'ema40':tema40[10:].values, 'ema10':tema10[10:].values, 'pema10':tpema10[10:].values,'pema40':tpema40[10:].values, 'shares':np.zeros(tpibm.size-10), 'cash':np.ones(tpibm.size-10)*10000})

tpdiff = pd.DataFrame(index = tpibm.index[10:], data = {'price':tpibm.values[10:], 'pred':tppred})
plot_data(tpdiff)

tpfl,tbds,tsds = trade(tpfl, method, level)
tpv = tpfl['price']*tpfl['shares'] + tpfl['cash']
tpspy = tprices['SPY'][tpfl.index]
tpfl_vs_tspy = pd.DataFrame(index = tpfl.index, data = {'my_portval':tpv/tpv.ix[0], 'SPY':tpspy/tpspy.ix[0]})
#plot_data(tpfl_vs_tspy, title = "My_Portfolio vs SPY", ylabel = "Accumulative Return")
#plot_pfl(tpfl,tbds,tsds, method)
print tbds, tsds
print len(tbds),len(tsds) 


#-----------Below are functions to make plots for report-------------------#
def rplot_pfl(pfl, bds, sds, pv, spy, method = 'ema'):
    plt.subplot(211)
    plt.plot(pfl.index, pfl['price'], label = 'price')
    plt.plot(pfl.index, pfl[method+'10'], label = method+'10')
    plt.plot(pfl.index, pfl[method+'40'], label = method+'40')
    plt.legend()
    for idx in bds:
        plt.axvline(pfl.index[idx], color = 'green')
    for idx in sds:
        plt.axvline(pfl.index[idx], color = 'red') 
        
    plt.subplot(212) 
    plt.plot(pfl.index, pv/pv.ix[0], label = 'my_portfolio')
    plt.plot(pfl.index, spy/spy.ix[0], label = 'SPY')
    plt.legend()
    for idx in bds:
        plt.axvline(pfl.index[idx], color = 'green')
    for idx in sds:
        plt.axvline(pfl.index[idx], color = 'red')
   
    plt.show()
    plt.clf()
rplot_pfl(pfl, bds, sds, pv, pspy)
rplot_pfl(tpfl, tbds, tsds, tpv, tpspy)



def rplot(tpibm, pibm, tpflr, pflr, bds, sds, tbds, tsds, spy, tspy):
    plt.plot(pflr.index, pflr['current_price'], label = 'current price')
    plt.plot(pflr.index, pflr['train_price'], label = 'train price')
    plt.plot(pflr.index, pflr['pred'], label = 'predicted price')
    plt.legend()
    plt.show()
    plt.clf()
    
    plt.subplot(211)
    plt.plot(pibm.index[10:], pflr['train_price'], label = 'price')
    plt.plot(pibm.index[10:], pflr['pred'], label = 'prediction')
    for idx in bds:
        plt.axvline(pfl.index[idx], color = 'green')
    for idx in sds:
        plt.axvline(pfl.index[idx], color = 'red')
    plt.legend()
    plt.subplot(212)
    plt.plot(pibm.index[10:], pflr['pv'], label = 'Accumulative return')
    plt.plot(pibm.index[10:], spy[pibm.index[10:]]/spy[pibm.index[10:]].ix[0], label = 'SPY')
    for idx in bds:
        plt.axvline(pfl.index[idx], color = 'green')
    for idx in sds:
        plt.axvline(pfl.index[idx], color = 'red')
    plt.legend()
    plt.show()
    plt.clf()
    
    plt.subplot(211)
    plt.plot(tpibm.index[10:], tpflr['tp'], label = 'price')
    plt.plot(tpibm.index[10:], tpflr['tpred'], label = 'prediction')
    for idx in tbds:
        plt.axvline(tpibm.index[10:][idx], color = 'green')
    for idx in tsds:
        plt.axvline(tpibm.index[10:][idx], color = 'red')
    plt.legend()
    plt.subplot(212)
    plt.plot(tpibm.index[10:], tpflr['tpv'], label = 'Accumulative return')
    plt.plot(tpibm.index[10:], tspy.values[10:]/tspy.values[10:][0], label = 'SPY')
    for idx in tbds:
        plt.axvline(tpibm.index[10:][idx], color = 'green')
    for idx in tsds:
        plt.axvline(tpibm.index[10:][idx], color = 'red')
    plt.legend()
    plt.show()
    plt.clf()
pflr = pd.DataFrame(index = pibm.index[5:-5], data = {'train_price':pibm[10:].values, 'pred':ppred.values, 'current_price': pibm[5:-5].values, 'pv':pv.values/pv.values[0]})
                            
tpflr = pd.DataFrame(index = tpibm.index[10:], data = {'tp': tpibm[10:].values, 'tpred':tppred.values,'tpv':tpv.values/tpv.values[0]})
rplot(tpibm, pibm, tpflr, pflr, bds, sds, tbds, tsds, prices_all['SPY'], tspy)

