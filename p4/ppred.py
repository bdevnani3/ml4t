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
pibm = prices_all['IBM']

start_val = 1000000  
risk_free_rate = 0.0
sample_freq = 252

# price prediction, choose learner

# pick features(X values): Momentum
mmtn1 = pibm.values[1:]/pibm.values[:-1]-1  
mmtn2 = pibm.values[2:]/pibm.values[:-2]-1
mmtn3 = pibm.values[3:]/pibm.values[:-3]-1
mmtn4 = pibm.values[4:]/pibm.values[:-4]-1
mmtn5 = pibm.values[5:]/pibm.values[:-5]-1
X = pd.DataFrame({'x1':mmtn1[4:][:-5], 'x2':mmtn2[3:][:-5], 'x3':mmtn3[2:][:-5], 'x4':mmtn4[1:][:-5], 'x5':mmtn5[:-5]})
Y = pibm.values[5:]
Y = Y[5:]/Y[:-5] - 1
# train regression learner
clf = linear_model.LinearRegression()
clf.fit(X.values, Y)
#In sample prediction
ppred = pibm.values[10:]*(X.values.dot(clf.coef_) + 1)
pdiff = pd.DataFrame({'price':pibm.values[10:], 'pred':ppred})
plot_data(pdiff)
#Out sample prediction
tsd = dt.datetime(2009,12,31)
ted = dt.datetime(2011,12,31)
symbols = ['IBM']

tdates = pd.date_range(tsd, ted)
tprices = get_data(symbols, tdates)
tpibm = tprices['IBM']

tmmtn1 = tpibm.values[1:]/tpibm.values[:-1]-1  
tmmtn2 = tpibm.values[2:]/tpibm.values[:-2]-1
tmmtn3 = tpibm.values[3:]/tpibm.values[:-3]-1
tmmtn4 = tpibm.values[4:]/tpibm.values[:-4]-1
tmmtn5 = tpibm.values[5:]/tpibm.values[:-5]-1
tX = pd.DataFrame({'x1':tmmtn1[4:][:-5], 'x2':tmmtn2[3:][:-5], 'x3':tmmtn3[2:][:-5], 'x4':tmmtn4[1:][:-5], 'x5':tmmtn5[:-5]})

tppred = tpibm.values[10:]*(tX.values.dot(clf.coef_.T) + 1)
tpdiff = pd.DataFrame({'price':tpibm.values[10:], 'pred':tppred})
plot_data(tpdiff)

# trading policy
def create_port(symbols, cash):
    pass

def buy(symbols, date, cash, portfolio):
    pass

def sell(symbols, date, cash, portfolio):
    pass

def predict(coef, prices, current_date):
    pass


# according to the past(5 days) true prices and future 5 days' predicted prices to make buy/sell decsion 
# record the order.
