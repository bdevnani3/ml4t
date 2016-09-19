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
        self.trained_pfl = None
        self.tested_pfl = None

    def get_feat(self, p, symbol):
        mmtn = pd.Series(index = p.index[1:], data = (p.values[1:]/p.values[:-1] - 1).flatten())
        accel = pd.Series(index = p.index[2:], data = mmtn[1:].values/mmtn[:-1].values - 1)

        #sma20 = p.rolling(window = 20, min_periods = 0)
        #sma20_mean = sma20.mean()
        sma20_mean = pd.rolling_mean(p, window=20, min_periods = 0)
        #sma20_std = sma20.std()
        sma20_std = pd.rolling_std(p, window=20, min_periods = 0)
        bbup = sma20_mean + 2 * sma20_std
        bblow = sma20_mean - 2 * sma20_std
        bbvals = (p[1:] - sma20_mean[1:])/(4 * sma20_std[1:])
        vtl = sma20_std[1:]/sma20_mean[1:]
        sma20_mmtn = pd.Series(index = p.index[1:], data = (sma20_mean[1:].values/sma20_mean[:-1].values - 1).flatten())
        sma20_accel = pd.Series(index = p.index[2:], data = sma20_mmtn[1:].values/sma20_mmtn[:-1].values - 1)

        #ema15 = p.ewm(span = 15, min_periods = 0)
        #ema15_mean = ema15.mean()
        ema15_mean = pd.ewma(p, span = 15, min_periods = 0)
        ema15_mmtn = pd.Series(index = p.index[1:], data = (ema15_mean[1:].values/ema15_mean[:-1].values - 1).flatten())
        ema15_accel = pd.Series(index = p.index[2:], data = ema15_mmtn[1:].values/ema15_mmtn[:-1].values - 1)
        #ema40 = p.ewm(span = 40, min_periods = 0)
        ema40_mean = pd.ewma(p, span = 40, min_periods = 0)
        ema40_mmtn = pd.Series(index = p.index[1:], data = (ema40_mean[1:].values/ema40_mean[:-1].values - 1).flatten())
        ema40_accel = pd.Series(index = p.index[2:], data = ema40_mmtn[1:].values/ema40_mmtn[:-1].values - 1)
        macd = ema15_mean - ema40_mean

        feats = pd.DataFrame({
            'price':p[symbol][2:],\
                    'mmtn':mmtn[1:],\
                    'accel':accel,\
                    'sma20_mean':sma20_mean[symbol][2:],\
                    'sma20_std':sma20_std[symbol][2:],\
                    'bbup':bbup[symbol][2:],\
                    'bblow':bblow[symbol][2:],\
                    'bbvals':bbvals[symbol][1:],\
                    'vtl':vtl[symbol][1:],\
                    'sma20_mmtn':sma20_mmtn[1:],\
                    'sma20_accel': sma20_accel,\
                    'ema15_mean': ema15_mean[symbol][2:],\
                    'ema15_mmtn': ema15_mmtn[1:],\
                    'ema15_accel': ema15_accel,\
                    'ema40_mean':ema40_mean[symbol][2:],\
                    'ema40_mmtn':ema40_mmtn[1:],\
                    'ema40_accel':ema40_accel,\
                    'macd':macd[symbol][2:]
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


    def update_states(self, i, idx, action):
        """
        @param      i: current index(i.e. date)
        @param action: 0 -> nothing
                       1 -> buy 100
                       2 -> buy 200
                       3 -> sell 100
                       4 -> sell 200
        """
        holding = int(self.pfl.loc[idx[i],'state']) & 0b11 #holding: 0, -100; 1, 0; 2, 100 
        #print "+++++++++++++++++++++++++++++++++"
        #print "before action"
        #print "action: ", action, "  holding: ", holding, "  shares: ", self.pfl.loc[idx[i], 'shares']
        #print "pfl: ", self.pfl.loc[[idx[i]]]
        if action == 1:
            if holding <= 1:
                #print "case 1"
                self.pfl.loc[idx[i:], 'state'] += 1
                self.pfl.loc[idx[i:],'shares'] += 100
                self.pfl.loc[idx[i:],'cash'] -= 100*self.pfl.loc[idx[i],'price']
        elif action == 2:
            if holding == 0:
                #print "case 2-1"
                self.pfl.loc[idx[i:],'state'] += 2
                self.pfl.loc[idx[i:],'shares'] += 200
                self.pfl.loc[idx[i:],'cash'] -= 200*self.pfl.loc[idx[i],'price']
            elif holding == 1:
                #print "case 2-2"
                self.pfl.loc[idx[i:],'state'] += 1
                self.pfl.loc[idx[i:],'shares'] += 100
                self.pfl.loc[idx[i:],'cash'] -= 100*self.pfl.loc[idx[i],'price']
        elif action == 3:
            if holding >= 1:
                #print "case 3"
                self.pfl.loc[idx[i:],'state'] -= 1 
                self.pfl.loc[idx[i:],'shares'] -= 100
                self.pfl.loc[idx[i:],'cash'] += 100*self.pfl.loc[idx[i],'price']
        elif action == 4:
            if holding == 2:
                #print "case 4-1"
                self.pfl.loc[idx[i:],'state'] -= 2
                self.pfl.loc[idx[i:],'shares'] -= 200
                self.pfl.loc[idx[i:],'cash'] += 200*self.pfl.loc[idx[i],'price']
            elif holding == 1:
                #print "case 4-2"
                self.pfl.loc[idx[i:],'state'] -= 1 
                self.pfl.loc[idx[i:],'shares'] -= 100
                self.pfl.loc[idx[i:],'cash'] += 100*self.pfl.loc[idx[i],'price']

        self.pfl.loc[idx[i:],'portval'] = self.pfl.loc[idx[i:],'cash'] +\
                self.pfl.loc[idx[i:],'shares']*self.pfl.loc[idx[i:],'price']


    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
            sd=dt.datetime(2008,1,1), \
            ed=dt.datetime(2009,1,1), \
            sv = 10000): 
        st = time.time() # for timing 
        self.learner= ql.QLearner(num_states = 511, num_actions = 5,rar = 0.9, radr = 0.999)
        # add your code to do learning here
        #symbol = 'ML4T-220' 
        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        
        iterst = time.time()
        
        threshold = 0.0001
        maxtime = 1000 
        dp = 1.0
        dt = 0
        itr = 0
        perfs = []
        while dp > threshold and dt < maxtime:
            feats = self.get_feat(prices, symbol)
            S_init = self.discretize(feats)  
            idx = feats.index
            self.pfl = pd.DataFrame(index = idx, \
                               data = {'price'  :prices_all[symbol][idx].values,\
                                       'shares' :np.zeros(len(idx)),\
                                       'cash'   :sv*np.ones(len(idx)),\
                                       'portval':np.zeros(len(idx)),\
                                       'state'  :S_init['state'].values})
            state = self.pfl['state'].ix[0]
            action = self.learner.querysetstate(state)
            self.update_states(0, idx, action)
            t = np.zeros(5)
            tt = np.zeros(4)
            for i in range(1, len(idx)):
                t[0]= time.time()
                r = (self.pfl.loc[idx[i],'portval']/self.pfl['portval'].ix[i-1] - 1)*100
                t[1] = time.time()
                state = int(self.pfl.loc[idx[i],'state'])
                t[2] = time.time()
                action = self.learner.query(state, r)
                t[3] = time.time()
                self.update_states(i, idx, action)
                t[4] = time.time()
                tt += np.diff(t)
            Np = self.pfl['portval'].ix[-1]/self.pfl['portval'].ix[0]
            if itr > 0:
                dp = abs(Np - perfs[-1])
            perfs.append(Np)
            itr += 1
            dt = time.time() - st
            print "itr: ", itr, "  portval: ", Np 
            #, " tt[0]: ", tt[0], " tt[1]: ", tt[1]  , " tt[2]: ", tt[2]  , " tt[3]: ", tt[3]  
            
        #print "training time: ", time.time() - st, "  iteration time: ", time.time() - iterst     
        self.trained_pfl = self.pfl
        return self.pfl, self.learner

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
            sd=dt.datetime(2009,1,1), \
            ed=dt.datetime(2010,1,1), \
            sv = 10000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        #symbol = 'ML4T-220'
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[[symbol]]
        
        feats = self.get_feat(prices, symbol)
        S_init = self.discretize(feats)
        idx = feats.index
        self.pfl = pd.DataFrame(index = idx, \
                           data = {'price'  :prices_all[symbol][idx].values,\
                                   'shares' :np.zeros(len(idx)),\
                                   'cash'   :sv*np.ones(len(idx)),\
                                   'portval':np.zeros(len(idx)),\
                                   'state'  :S_init['state'].values})
        
        state = self.pfl['state'].ix[0]
        action = self.learner.querysetstate(state)
        self.update_states(0, idx, action)
        for i in range(1, len(idx)):
            r = (self.pfl.loc[idx[i],'portval']/self.pfl['portval'].ix[i-1] - 1)*100
            state = int(self.pfl.loc[idx[i],'state'])
            action = self.learner.query(state, r)
            self.update_states(i, idx, action)
        
        self.tested_pfl = self.pfl
        T = np.zeros(len(idx) + 3)
        T[3:] = self.pfl['shares'].values
        trades = pd.DataFrame(index = prices.index, data = {'trade': np.diff(T)})

        #trades = prices_all[[symbol,]]  # only portfolio symbols
        #trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        #trades.values[:,:] = 0 # set them all to nothing
        #trades.values[3,:] = 100 # add a BUY at the 4th date
        #trades.values[5,:] = -100 # add a SELL at the 6th date 
        #trades.values[6,:] = -100 # add a SELL at the 7th date 
        #trades.values[8,:] = -100 # add a SELL at the 9th date
        #if self.verbose: print type(trades) # it better be a DataFrame!
        #if self.verbose: print trades
        #if self.verbose: print prices_all
        return trades

if __name__=="__main__":
    print "One does not simply think up a strategy, here we go"
    sl = StrategyLearner()
    #def addEvidence(self, symbol = "IBM", \
    #        sd=dt.datetime(2008,1,1), \
    #        ed=dt.datetime(2009,1,1), \
    #        sv = 10000): 
    pfl, slqlearner = sl.addEvidence(sd=dt.datetime(2007,1,1), ed=dt.datetime(2009,1,1))
    pfl.to_csv('pfl.csv')
    temp = pd.DataFrame({'price': pfl['price']/pfl['price'].ix[0], 'portval': pfl['portval']/pfl['portval'].ix[0]})
    ut.plot_data(temp)
    trades = sl.testPolicy(sd=dt.datetime(2009,1,1), ed=dt.datetime(2011,1,1))
    tpfl = sl.pfl
    tpfl.to_csv('tpfl.csv')
    temp = pd.DataFrame({'price': tpfl['price']/tpfl['price'].ix[0], 'portval': tpfl['portval']/tpfl['portval'].ix[0]})
    ut.plot_data(temp)
    
