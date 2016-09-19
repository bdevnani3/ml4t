
"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import KNNLearner as knn
import BagLearner as bl 
import matplotlib.pyplot as plt

if __name__=="__main__":
    inf = open('Data/ripple.csv')
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = math.floor(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print testX.shape
    print testY.shape

    #maxK = 11
    #err = {'train':[], 'test':[]}
    #for K in range(1,maxK):
    #    learner = knn.KNNLearner(k=K, verbose = True) # create a KNNLearner 
    #    learner.addEvidence(trainX, trainY) # train it

    #    # evaluate in sample
    #    predY = learner.query(trainX) # get the predictions
    #    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    #    err['train'].append(rmse)
    #    c = np.corrcoef(predY, y=trainY)

    #    # evaluate out of sample
    #    predY = learner.query(testX) # get the predictions
    #    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    #    err['test'].append(rmse)
    #    c = np.corrcoef(predY, y=testY)
    #plt.clf()    
    #plt.plot(range(1,maxK), err['train'], label = 'train')
    #plt.plot(range(1,maxK), err['test'], label = 'test')
    #plt.xlabel('K')
    #plt.ylabel('RMSE')
    #plt.legend()
    #plt.savefig('knnof')
    #plt.show()
    #print np.argmin(err['test'])+1

    maxbag = 40 
    err = {'train':[], 'test':[]}
    for nbag in range(2,maxbag+1):
        learner = bl.BagLearner(bags = nbag)
        learner.addEvidence(trainX, trainY) # train it

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        err['train'].append(rmse)
        c = np.corrcoef(predY, y=trainY)

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        err['test'].append(rmse)
        c = np.corrcoef(predY, y=testY)
        
    plt.clf()
    plt.plot(range(2,maxbag+1), err['train'], label = 'train')
    plt.plot(range(2,maxbag+1), err['test'], label = 'test')
    plt.xlabel('#bags')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('bagof')
    print np.argmin(err['test'])+1
    
    maxK = 11
    err = {'train':[], 'test':[]}
    for K in range(1,maxK):
        learner = bl.BagLearner(kwargs={'k':K}, bags=24) # create a KNNLearner 
        learner.addEvidence(trainX, trainY) # train it

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        err['train'].append(rmse)
        c = np.corrcoef(predY, y=trainY)

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        err['test'].append(rmse)
        c = np.corrcoef(predY, y=testY)
        
    plt.clf() 
    plt.plot(range(1,maxK), err['train'], label = 'train')
    plt.plot(range(1,maxK), err['test'], label = 'test')
    plt.xlabel('K')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('knnbagof')
    plt.show()
    print np.argmin(err['test'])+1
    

        #learners = []
        #for i in range(0,10):
            #kwargs = {"k":i}
            #learners.append(lrl.LinRegLearner(**kwargs))
