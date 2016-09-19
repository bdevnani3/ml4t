
"""
A simple wrapper for KNN regression.  (c) 2015 Tucker Balch
"""

import numpy as np

class KNNLearner(object):

    def __init__(self, k = 3, verbose = False):
        self.k = k
        pass # move along, these aren't the drones you're looking for

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # slap on 1s column so linear regression finds a constant term
        # newdataX = np.ones([dataX.shape[0],dataX.shape[1]+1])
        # newdataX[:,0:dataX.shape[1]]=dataX

        # build and save the model
        #self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY)
        self.dataX = dataX
        self.dataY = dataY
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        #return (self.model_coefs[:-1] * points).sum(axis = 1) + self.model_coefs[-1]
        Y = np.zeros(points.shape[0])
        for i, X in enumerate(points):
            dist = np.sum((self.dataX - X)**2,axis = 1)
            sort_idx = dist.argsort()
	    Y[i] = self.dataY[sort_idx][:self.k].mean()
	    #Y[i] = np.average(self.dataY[sort_idx][:self.k], weights = 1.0/dist[dist.argsort()][:self.k])
        return Y
            
if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
