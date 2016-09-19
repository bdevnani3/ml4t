"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""
import numpy as np
import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut
import time
class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose

    def get_feat(self, p):
        mmtn = pd.Series(index = p.index[1:], data = (p.values[1:]/p.values[:-1] - 1).flatten())
        accel = pd.Series(index = p.index[2:], data = mmtn[1:].values/mmtn[:-1].values - 1)

        sma20 = p.rolling(window = 20, min_periods = 0)
        sma20_mean = sma20.mean()
        sma20_std = sma20.std()
        bbup = sma20_mean + 2 * sma20_std
        bblow = sma20_mean - 2 * sma20_std
        bbvals = (p[1:] - sma20_mean[1:])/(4 * sma20_std[1:])
        vtl = sma20_std[1:]/sma20_mean[1:]
        sma20_mmtn = pd.Series(index = p.index[1:], data = (sma20_mean[1:].values/sma20_mean[:-1].values - 1).flatten())
        sma20_accel = pd.Series(index = p.index[2:], data = sma20_mmtn[1:].values/sma20_mmtn[:-1].values - 1)

        ema15 = p.ewm(span = 15, min_periods = 0)
        ema15_mean = ema15.mean()
        ema15_std = ema15.std()
        ema15_mmtn = pd.Series(index = p.index[1:], data = (ema15_mean[1:].values/ema15_mean[:-1].values - 1).flatten())
        ema15_accel = pd.Series(index = p.index[2:], data = ema15_mmtn[1:].values/ema15_mmtn[:-1].values - 1)
        ema40 = p.ewm(span = 40, min_periods = 0)
        ema40_mean = ema40.mean()
        ema40_std = ema40.std()
        ema40_mmtn = pd.Series(index = p.index[1:], data = (ema40_mean[1:].values/ema40_mean[:-1].values - 1).flatten())
        ema40_accel = pd.Series(index = p.index[2:], data = ema40_mmtn[1:].values/ema40_mmtn[:-1].values - 1)
        macd = ema15_mean - ema40_mean

        feats = pd.DataFrame({
            'price':p['IBM'][2:],\
                    'mmtn':mmtn[1:],\
                    'accel':accel,\
                    'sma20_mean':sma20_mean['IBM'][2:],\
                    'sma20_std':sma20_std['IBM'][2:],\
                    'bbup':bbup['IBM'][2:],\
                    'bblow':bblow['IBM'][2:],\
                    'bbvals':bbvals['IBM'][1:],\
                    'vtl':vtl['IBM'][1:],\
                    'sma20_mmtn':sma20_mmtn[1:],\
                    'sma20_accel': sma20_accel,\
                    'ema15_mean': ema15_mean['IBM'][2:],\
                    'ema15_std': ema15_std['IBM'][2:],\
                    'ema15_mmtn': ema15_mmtn[1:],\
                    'ema15_accel': ema15_accel,\
                    'ema40_mean':ema40_mean['IBM'][2:],\
                    'ema40_std':ema40_std['IBM'][2:],\
                    'ema40_mmtn':ema40_mmtn[1:],\
                    'ema40_accel':ema40_accel,\
                    'macd':macd['IBM'][2:]
                    })
        return feats

    def discretize(self, feats):
        x8 = (feats['mmtn'].values > 0).astype('int') # mmtn_sign 
        x7 = (np.abs(feats['mmtn'].values) >= 0.01).astype('int')#mmtn_mags  
        x6 = (feats['bbvals'].values > 0).astype('int') #bb_sign 
        x5 = (np.abs(feats['bbvals'].values) >= 0.5).astype('int') #bb_vals 
        x4 = (feats['macd'].values > 0).astype('int') #macd of today
        x3 = np.zeros(x4.size).astype('int') 
        x3[1:] = x4[:-1]#macd of yesterday
        x2 = (np.abs(feats['vtl'].values) >= 0.03).astype('int') #vtl 
        x1 = np.ones(x3.size).astype('int') # Holding: needs two bits to represnt: 0,-100; 1, 0;  2, +100.
        return pd.DataFrame(index = feats.index,\
                                    data = {'state': (x8<<8) + (x7<<7) + (x6<<6) +\
                                    (x5<<5) + (x4<<4) + (x3<<3) + (x2<<2) + x1})    


    def update_states(self, i, action):
        """
        @param      i: current index(i.e. date)
        @param action: 0 -> nothing
                       1 -> buy 100
                       2 -> buy 200
                       3 -> sell 100
                       4 -> sell 200
        """
        holding = self.pfl['state'].ix[i]&0b111 #holding: 0, -100; 1, 0; 2, 100 
        if action == 1:
            if holding <= 1:
                self.pfl['state'].ix[i:] += 1
                self.pfl['shares'].ix[i:] += 100
                self.pfl['cash'].ix[i:] -= 100*self.pfl['price'].ix[i]
        elif action == 2:
            if holding == 0:
                self.pfl['state'].ix[i:] += 2
                self.pfl['shares'].ix[i:] += 200
                self.pfl['cash'].ix[i:] -= 200*self.pfl['price'].ix[i]
            elif holding == 1:
                self.pfl['state'].ix[i:] += 1
                self.pfl['shares'].ix[i:] += 100
                self.pfl['cash'].ix[i:] -= 100*self.pfl['price'].ix[i]
        elif action == 3:
            if holding >= 1:
                self.pfl['state'].ix[i:] -= 1 
                self.pfl['shares'].ix[i:] -= 100
                self.pfl['cash'].ix[i:] += 100*self.pfl['price'].ix[i]
        elif action == 4:
            if holding == 2:
                self.pfl['state'].ix[i:] -= 2
                self.pfl['shares'].ix[i:] -= 200
                self.pfl['cash'].ix[i:] += 200*self.pfl['price'].ix[i]
            elif holding == 1:
                self.pfl['state'].ix[i:] -= 1 
                self.pfl['shares'].ix[i:] -= 100
                self.pfl['cash'].ix[i:] += 100*self.pfl['price'].ix[i]

        self.pfl['portval'].ix[i:] = self.pfl['cash'].ix[i:] +\
                self.pfl['shares'].ix[i:]*self.pfl['price'].ix[i:]
        
        

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
            sd=dt.datetime(2008,1,1), \
            ed=dt.datetime(2009,1,1), \
            sv = 10000): 
        st = time.time() # for timing 
        self.ql= ql.QLearner(num_states = 511, num_actions = 5,rar = 0.9, radr = 0.999)
        # add your code to do learning here

        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        
        maxiter = 2 
        iterst = time.time()
        for itr in range(maxiter):
            feats = self.get_feat(prices)
            S_init = self.discretize(feats)  
            idx = feats.index
            self.pfl = pd.DataFrame(index = idx, \
                               data = {'price'  :prices_all[symbol][idx].values,\
                                       'shares' :np.zeros(len(idx)),\
                                       'cash'   :sv*np.ones(len(idx)),\
                                       'portval':np.zeros(len(idx)),\
                                       'state'  :S_init['state'].values})
            state = self.pfl['state'].ix[0]
            action = self.ql.querysetstate(state)
            self.update_states(0, action)
            t = np.zeros(5)
            tt = np.zeros(4)
            for i in range(1, len(self.pfl.index) - 1):
                t[0]= time.time()
                r = (self.pfl['portval'].ix[i]/self.pfl['portval'].ix[i-1] - 1)*100
                t[1] = time.time()
                state = self.pfl['state'].ix[i]
                t[2] = time.time()
                action = self.ql.query(state, r)
                t[3] = time.time()
                self.update_states(i, action)
                t[4] = time.time()
                tt += np.diff(t)
            print "itr: ", itr, "  portval: ", self.pfl['portval'].ix[-1]/self.pfl['portval'].ix[0], " tt[0]: ", tt[0]\
                , " tt[1]: ", tt[1]  , " tt[2]: ", tt[2]  , " tt[3]: ", tt[3]  
        print "training time: ", time.time() - st, "  iteration time: ", time.time() - iterst     
        return self.pfl, self.ql, tt          

        #if self.verbose: print prices

        ## example use with new colname 
        #volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        #volume = volume_all[syms]  # only portfolio symbols
        #volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        #if self.verbose: print volume

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
            sd=dt.datetime(2009,1,1), \
            ed=dt.datetime(2010,1,1), \
            sv = 10000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        trades.values[:,:] = 0 # set them all to nothing
        trades.values[3,:] = 100 # add a BUY at the 4th date
        trades.values[5,:] = -100 # add a SELL at the 6th date 
        trades.values[6,:] = -100 # add a SELL at the 7th date 
        trades.values[8,:] = -100 # add a SELL at the 9th date
        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trades

if __name__=="__main__":
    print "One does not simply think up a strategy, here we go"
    sl = StrategyLearner()
    pfl, ql, tt, = sl.addEvidence()
    pfl.to_csv('pfl.csv')
    ut.plot_data(pfl['portval']/pfl['portval'].ix[0])
