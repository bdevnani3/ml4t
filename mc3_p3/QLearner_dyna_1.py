"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand
import time 
class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.rar = rar
        self.radr = radr
        self.gamma = gamma
        self.dyna = dyna
        #self.q = np.random.random((num_states, num_actions)) - 0.5 
        self.q = np.zeros((num_states, num_actions)) 
        self.visited = {}
        #self.model_s = np.zeros((num_states, num_actions))
        self.model_s = {(s, a):[] for s in range(num_states) for a in range(num_actions)}
        self.model_r = np.zeros((num_states, num_actions))
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        # generate a random number, with rar probability to assign a random action.
        #action = rand.randint(0, self.num_actions-1)
        action = self.q[s].argmax() 
        #if self.verbose: print "s =", s,"a =",action
        if not s in self.visited:
            self.visited[s] = set()
        self.visited[s].add(action)
        return action
    
    def get_a(self, s):
        rm = float(rand.randint(0,10000))/10000
        if rm < self.rar:
            a_prime = rand.randint(0, self.num_actions-1)
        else:
            a_prime= self.q[s].argmax() 
        return a_prime

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        s = self.s
        a = self.a
        if s not in self.visited:
            self.visited[s] = set()
        self.visited[s].add(a)

        alpha = self.alpha
        gamma = self.gamma

        self.model_s[(s,a)].append(s_prime)
        self.model_r[s,a] = (1 - alpha)*self.model_r[s, a] + alpha*r
        a_prime = self.get_a(s_prime)
        self.q[self.s,self.a] = (1 - alpha)*self.q[self.s, self.a] + alpha*(r + gamma*self.q[s_prime, a_prime])
        self.a = a_prime
        self.s = s_prime
        self.rar *= self.radr

        for i in range(self.dyna):
            if i == 0:
                print "Start Dyna"
            s = np.random.choice(list(self.visited.keys()))
            a = np.random.choice(list(self.visited[s]))
            s_next = np.random.choice(self.model_s[(s,a)])
            a_next = self.get_a(s_next)
            r = self.model_r[s,a]
            self.q[s,a] = (1 - alpha)*self.q[s, a] + alpha*(r + gamma*self.q[s_next, a_next])
            
        #if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return a_prime 

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
