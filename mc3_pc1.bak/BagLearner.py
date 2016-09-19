
"""
A simple wrapper for KNN regression.  (c) 2015 Tucker Balch
"""

import numpy as np
import KNNLearner as knn
class BagLearner(object):
    
    def __init__(self, learner = knn.KNNLearner,\
            kwargs = {"k":3}, bags = 20, boost = False, verbose = False):
        self.Learner = learner # For later use
        learners = []
        if boost == False:
            for i in range(bags):
                learners.append(learner(**kwargs))
        self.learners = learners
        self.bags = bags
        self.kwargs = kwargs
        self.boost = boost
        pass # move along, these aren't the drones you're looking for

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # build and save the model
        # self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY)
        self.dataX = dataX
        self.dataY = dataY
	if self.boost == False:
	    for learner in self.learners:
		sample = np.random.randint(dataX.shape[0],size = dataX.shape[0])
		learner.addEvidence(dataX[sample],dataY[sample])
        else:
            self.betas = []
            self.predYs = []
            indices = np.array([range(self.dataX.shape[0])])
            # initialize weight
            W = np.ones(self.dataX.shape[0])
            prob = W/self.dataX.shape[0]
            bins = np.cumsum(prob)
            sample = indices[np.digitize(np.random.random(self.dataX.shape[0]),bins)] 
            learner = self.Learner(**self.kwargs)
	    learner.addEvidence(dataX[sample],dataY[sample])
            predY = learner.query(dataX)
            self.predYs.append(predY)
            L = np.abs(predY - dataY)/(np.abs(predY - dataY)).max() # loss function
            Lmean = (L*prob).sum()
            beta = Lmean/(1 - Lmean)
            self.betas.append(beta)
            self.learners.append(learner)
            W = W*np.pow(beta, 1 - Lmean)
            while Lmean < 0.5:
                prob = W/W.sum()
                bins = np.cumsum(prob)
                sample = indices[np.digitize(np.random.random(self.dataX.shape[0]),bins)] 
                learner = self.Learner(**self.kwargs)
                learner.addEvidence(dataX[sample],dataY[sample])
                predY = learner.query(dataX)
                L = np.abs(predY - dataY)/(np.abs(predY - dataY)).max()
                Lmean = (L_new*prob).sum() 
                beta = Lmean/(1-Lmean)
                W = W*np.pow(beta, 1 - Lmean)
                self.predYs.append(predY)
                self.betas.append(beta)
                self.learners.append(learner)
            self.predYs = np.array(self.predYs)    
            self.betas = np.array(self.betas)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        #return (self.model_coefs[:-1] * points).sum(axis = 1) + self.model_coefs[-1]
        #for i, X in enumerate(points):
        #    dist = np.sum((self.dataX - X)**2,axis = 1)
        #    sort_idx = dist.argsort()
        #    Y[i] = self.dataY[sort_idx][:self.k].mean()
        if self.boost == False:
            Y = np.zeros((self.bags, points.shape[0]))
            for i, learner in enumerate(self.learners):
                Y[i] = learner.query(points)
            return Y.mean(axis = 0) 
        else:
            predYs = np.zeros((self.betas.shape[0], points.shape[0]))
            Y = np.zeros(points.shape[0])
            for i, learner in enumerate(self.learners):
                predYs[i] = learner.query(points)
            bsum = 0.5*np.sum(np.log(1./self.betas))
            for j in range(points.shape[0]):
                relabel = predYs[:,j].argsort()
                beta_cumsum = np.cumsum(self.betas[relabel])
                Y[j] = predYs[:,j][relabel][np.where(beta_cumsum >= bsum)[0][0]]
            return Y




        
            
if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
