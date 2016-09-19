"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import KNNLearner as knn
import BagLearner as bl
import best4linreg as b4l
import best4KNN as b4k
import os
import sys
import matplotlib
import matplotlib.pyplot as plt

font = {'size'   : 22}
matplotlib.rc('font', **font)

class LearningTestCase(object):
    def __init__(self, description = None, group = None, inputs = None, outputs = None):
        self.description = description
        self.group = group
        self.inputs = inputs
        self.outputs  = outputs

learning_test_cases = [
        LearningTestCase(
            description="Test Case 01",
            group='KNNLearner',
            inputs=dict(
                train_file=os.path.join('Data', 'ripple.csv'),
                test_file=os.path.join('Data', 'testcase01.csv')
                ),
            outputs=dict(
                rmse=6.08094194449e-17,
                corr=1.0
                )
            ),
        LearningTestCase(
            description="Test Case 01 - Noisy, 4 features",
            group='KNNLearner',
            inputs=dict(
                train_file=os.path.join('Data', 'ripple_noisy.csv'),
                test_file=os.path.join('Data', 'testcase01_noisy.csv')
                ),
            outputs=dict(
                rmse=0.568574577879,
                corr=0.61379073821
                )
            ),
        LearningTestCase(
            description="Test Case 03",
            group='KNNLearner',
            inputs=dict(
                train_file=os.path.join('Data', 'ripple.csv'),
                test_file=os.path.join('Data', 'testcase03.csv')
                ),
            outputs=dict(
                rmse=2.85197162452e-12,
                corr=1.0
                )
            ),
        LearningTestCase(
            description="Test Case 04",
            group='KNNLearner',
            inputs=dict(
                train_file=os.path.join('Data', 'best4KNN.csv'),
                test_file=os.path.join('Data', 'testcase04.csv')
                ),
            outputs=dict(
                rmse=0.00240027433274,
                corr=0.999999987071
                )
            ),
        LearningTestCase(
            description="Test Case 05",
            group='KNNLearner',
            inputs=dict(
                train_file=os.path.join('Data', 'ripple.csv'),
                test_file=os.path.join('Data', 'testcase05.csv')
                ),
            outputs=dict(
                rmse=0.359438468147,
                corr=0.838279931481
                )
            ),
        LearningTestCase(
                description="Test Case 06",
                group='KNNLearner',
                inputs=dict(
                    train_file=os.path.join('Data', '3_groups.csv'),
                    test_file=os.path.join('Data', 'testcase06.csv'),
                    kwargs={'k': 10}
                    ),
                outputs=dict(
                    rmse=35.1015004428,
                    corr=-0.208568812714
                    )
                ),
        LearningTestCase(
                description="Test Case 07",
                group='KNNLearner',
                inputs=dict(
                    train_file=os.path.join('Data', 'ripple.csv'),
                    test_file=os.path.join('Data', 'testcase07.csv')
                    ),
                outputs=dict(
                    rmse=0.918312680526,
                    corr=-0.119588294412
                    )
                ),
        LearningTestCase(
                description="Test Case 08",
                group='KNNLearner',
                inputs=dict(
                    train_file=os.path.join('Data', 'ripple.csv'),
                    test_file=os.path.join('Data', 'testcase08.csv')
                    ),
                outputs=dict(
                    rmse=0.0904271221715,
                    corr=0.988993695858
                    )
                ),
        LearningTestCase(
                description="Test Case 09",
                group='KNNLearner',
                inputs=dict(
                    train_file=os.path.join('Data', 'simple.csv'),
                    test_file=os.path.join('Data', 'testcase09.csv'),
                    kwargs={'k': 1}
                    ),
                outputs=dict(
                    rmse=0.0,
                    corr=1.0
                    )
                ),
        LearningTestCase(
                description="Test Case 10",
                group='KNNLearner',
                inputs=dict(
                    train_file=os.path.join('Data', 'ripple.csv'),
                    test_file=os.path.join('Data', 'testcase10.csv')
                    ),
                outputs=dict(
                    rmse=1.78531847475,
                    corr=-0.789236317359
                    )
                ),
        LearningTestCase(
                description="Test Case 01 - Bagging",
                group='BagLearner',
                inputs=dict(
                    train_file=os.path.join('Data', 'ripple.csv'),
                    test_file=os.path.join('Data', 'testcase01.csv')
                    ),
                outputs=dict(
                    rmse=0.102347955217,
                    corr=0.991328696155
                    )
                ),
        LearningTestCase(
                description="Test Case 02 - Bagging",
                group='BagLearner',
                inputs=dict(
                    train_file=os.path.join('Data', 'simple.csv'),
                    test_file=os.path.join('Data', 'testcase02.csv')
                    ),
                outputs=dict(
                    rmse=0.0894427191,
                    corr=0.99966144456
                    )
                ),
        LearningTestCase(
                description="Test Case 03 - Bagging",
                group='BagLearner',
                inputs=dict(
                    train_file=os.path.join('Data', 'ripple.csv'),
                    test_file=os.path.join('Data', 'testcase03.csv')
                    ),
                outputs=dict(
                    rmse=1.78531847475,
                    corr=-0.789236317359
                    )
                ),
        LearningTestCase(
                description="Test Case 04 - Bagging",
                group='BagLearner',
                inputs=dict(
                    train_file=os.path.join('Data', 'best4KNN.csv'),
                    test_file=os.path.join('Data', 'testcase04.csv')
                    ),
                outputs=dict(
                    rmse=0.0301204201819,
                    corr=0.999997608631
                    )
                ),
        LearningTestCase(
                description="Test Case 05 - Bagging",
                group='BagLearner',
                inputs=dict(
                    train_file=os.path.join('Data', 'ripple.csv'),
                    test_file=os.path.join('Data', 'testcase05.csv')
                    ),
                outputs=dict(
                    rmse=0.323579476488,
                    corr=0.867361902312
                    )
                ),
        LearningTestCase(
                description="Test Case 06 - Bagging",
                group='BagLearner',
                inputs=dict(
                    train_file=os.path.join('Data', '3_groups.csv'),
                    test_file=os.path.join('Data', 'testcase06.csv'),
                    kwargs={'kwargs': {'k': 1}, 'bags': 20, 'boost': False}
                    ),
                outputs=dict(
                    rmse=35.1014280336,
                    corr=-0.230388246034
                    )
                ),
        LearningTestCase(
                description="Test Case 07 - Bagging",
                group='BagLearner',
                inputs=dict(
                    train_file=os.path.join('Data', 'ripple.csv'),
                    test_file=os.path.join('Data', 'testcase07.csv'),
                    kwargs={'kwargs': {'k': 3}, 'bags': 20, 'boost': False}
                    ),
                outputs=dict(
                    rmse=0.912956660467,
                    corr=-0.112955082143
                    )
                ),
        LearningTestCase(
                description="Test Case 08 - Bagging",
                group='BagLearner',
                inputs=dict(
                    train_file=os.path.join('Data', 'ripple.csv'),
                    test_file=os.path.join('Data', 'testcase08.csv'),
                    kwargs={'kwargs': {'k': 3}, 'bags': 20, 'boost': False}
                    ),
                outputs=dict(
                    rmse=0.141072888643,
                    corr=0.971258408243
                    )
                ),
        LearningTestCase(
                description="Test Case 09 - Bagging",
                group='BagLearner',
                inputs=dict(
                    train_file=os.path.join('Data', 'simple.csv'),
                    test_file=os.path.join('Data', 'testcase09.csv'),
                    kwargs={'kwargs': {'k': 1}, 'bags': 20, 'boost': False}
                    ),
                outputs=dict(
                    rmse=0.0235702260396,
                    corr=0.999957755088
                    )
                ),
        LearningTestCase(
                description="Test Case 10 - Bagging, 5 bags",
                group='BagLearner',
                inputs=dict(
                    train_file=os.path.join('Data', 'ripple.csv'),
                    test_file=os.path.join('Data', 'testcase10.csv'),
                    kwargs={'kwargs': {'k': 3}, 'bags': 5, 'boost': False}
                    ),
                outputs=dict(
                    rmse=1.79642483731,
                    corr=-0.73463819703
                    )
                )
        ]

def RunTestCase(case, Boost = False):
    if case.group == 'KNNLearner':
        learner = knn.KNNLearner(k = 3, verbose = True)
    else:
        if case.inputs.has_key('kwargs'):
            if Boost:
                learner = bl.BagLearner(learner = knn.KNNLearner, kwargs = case.inputs['kwargs']['kwargs']
                        , bags = case.inputs['kwargs']['bags']
                        , boost = True, verbose = True)
            else:
                learner = bl.BagLearner(learner = knn.KNNLearner, kwargs = case.inputs['kwargs']['kwargs']
                        , bags = case.inputs['kwargs']['bags']
                        , boost = case.inputs['kwargs']['boost'], verbose = True)
        else:
            learner = bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":3}, bags = 20, boost = Boost, verbose = True)
    trainfile = open(case.inputs['train_file'])
    traindata = np.array([map(float,s.strip().split(',')) for s in trainfile.readlines()])
    trainX = traindata[:,:-1]
    trainY = traindata[:,-1]
    learner.addEvidence(trainX, trainY)

    testfile = open(case.inputs['test_file'])
    testdata = np.array([map(float,s.strip().split(',')) for s in testfile.readlines()])
    testX = testdata[:,:-1]
    testY = testdata[:,-1]

    predY = learner.query(testX)
    rmse = math.sqrt(((testY - predY)**2).sum()/testY.shape[0])
    print case.description, '  boost: ', Boost
    #print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]
    #print "Given TestCase result:"
    if case.outputs['rmse'] != 0.0 and rmse > case.outputs['rmse']:
        relerr = abs((rmse - case.outputs['rmse'])/case.outputs['rmse']) 
        print "relative error:", relerr
        if np.any(relerr > 0.1): 
            print "THIS CASE SUCKS!!!!"
    print 




if __name__=="__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '0':
            for case in learning_test_cases:
                try:
                    RunTestCase(case, Boost = False)            
                except:
                    pass
        elif sys.argv[1] == '1':
            b4lerr = {'train':{'lr':[], 'knn':[]}, 'test':{'lr':[], 'knn':[]}}
            for i in range(10):
                data = b4l.data_gen()
                trainX = data[:,:-1]
                trainY = data[:,-1]
                testdata = b4l.data_gen(num = 100, sigma = 0) 
                testX = testdata[:,:-1]
                testY = testdata[:,-1]
                
                learner = lrl.LinRegLearner()
                learner.addEvidence(trainX, trainY)
                # evaluate in sample
                predY = learner.query(trainX) # get the predictions
                rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
                b4lerr['train']['lr'].append(rmse)
                print
                print "=================LinReg================="
                print "In sample results"
                print "RMSE: ", rmse
                c = np.corrcoef(predY, y=trainY)
                print "corr: ", c[0,1]

                # evaluate out of sample
                predY = learner.query(testX) # get the predictions
                rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
                b4lerr['test']['lr'].append(rmse)
                print
                print "Out of sample results"
                print "RMSE: ", rmse
                c = np.corrcoef(predY, y=testY)
                print "corr: ", c[0,1]

                learner = knn.KNNLearner()
                learner.addEvidence(trainX, trainY)
                # evaluate in sample
                predY = learner.query(trainX) # get the predictions
                rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
                b4lerr['train']['knn'].append(rmse)
                print
                print "=================KNN================="
                print "In sample results"
                print "RMSE: ", rmse
                c = np.corrcoef(predY, y=trainY)
                print "corr: ", c[0,1]

                # evaluate out of sample
                predY = learner.query(testX) # get the predictions
                rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
                b4lerr['test']['knn'].append(rmse)
                print
                print "Out of sample results"
                print "RMSE: ", rmse
                c = np.corrcoef(predY, y=testY)
                print "corr: ", c[0,1]
            plt.clf()
            plt.figure(figsize=(21,9))
            plt.subplot(121)
            plt.plot(range(10), b4lerr['train']['lr'], label='LinReg')
            plt.plot(range(10), b4lerr['train']['knn'], label='KNN')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('$x_{th}$ training')
            plt.ylabel('training error')
            plt.subplot(122)
            plt.plot(range(10), b4lerr['test']['lr'], label='LinReg')
            plt.plot(range(10), b4lerr['test']['knn'], label='KNN')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('$x_{th}$ testing')
            plt.ylabel('testing error')
            plt.savefig('b4lerr.pdf', bbox_inches='tight')

        elif sys.argv[1] == '2':
            b4kerr = {'train':{'lr':[], 'knn':[]}, 'test':{'lr':[], 'knn':[]}}
            for i in range(10):
                data = b4k.data_gen()
                trainX = data[:,:-1]
                trainY = data[:,-1]
                testdata = b4k.data_gen(num = 100) 
                testX = testdata[:,:-1]
                testY = testdata[:,-1]
                
                learner = lrl.LinRegLearner()
                learner.addEvidence(trainX, trainY)
                # evaluate in sample
                predY = learner.query(trainX) # get the predictions
                rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
                b4kerr['train']['lr'].append(rmse)
                print
                print "=================LinReg================="
                print "In sample results"
                print "RMSE: ", rmse
                c = np.corrcoef(predY, y=trainY)
                print "corr: ", c[0,1]

                # evaluate out of sample
                predY = learner.query(testX) # get the predictions
                rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
                b4kerr['test']['lr'].append(rmse)
                print
                print "Out of sample results"
                print "RMSE: ", rmse
                c = np.corrcoef(predY, y=testY)
                print "corr: ", c[0,1]

                learner = knn.KNNLearner()
                learner.addEvidence(trainX, trainY)
                # evaluate in sample
                predY = learner.query(trainX) # get the predictions
                rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
                b4kerr['train']['knn'].append(rmse)
                print
                print "=================KNN================="
                print "In sample results"
                print "RMSE: ", rmse
                c = np.corrcoef(predY, y=trainY)
                print "corr: ", c[0,1]

                # evaluate out of sample
                predY = learner.query(testX) # get the predictions
                rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
                b4kerr['test']['knn'].append(rmse)
                print
                print "Out of sample results"
                print "RMSE: ", rmse
                c = np.corrcoef(predY, y=testY)
                print "corr: ", c[0,1]
            plt.clf()
            plt.figure(figsize=(21,9))
            plt.subplot(121)
            plt.plot(range(10), b4kerr['train']['lr'], label='LinReg')
            plt.plot(range(10), b4kerr['train']['knn'], label='KNN')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('$x_{th}$ testing')
            plt.xlabel('$x_{th}$ training')
            plt.ylabel('training error')
            plt.subplot(122)
            plt.plot(range(10), b4kerr['test']['lr'], label='LinReg')
            plt.plot(range(10), b4kerr['test']['knn'], label='KNN')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('$x_{th}$ testing')
            plt.ylabel('testing error')
            plt.savefig('b4kerr.pdf', bbox_inches='tight')

    else:
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

        # create a learner and train it
        #learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
        learner = knn.KNNLearner(verbose = True) # create a KNNLearner 
        learner.addEvidence(trainX, trainY) # train it

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print
        print "In sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=trainY)
        print "corr: ", c[0,1]

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print
        print "Out of sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0,1]

        #learners = []
        #for i in range(0,10):
            #kwargs = {"k":i}
            #learners.append(lrl.LinRegLearner(**kwargs))
