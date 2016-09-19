import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def data_gen(num = 2000):
    X1 = np.random.random((num/2,2)) * 10 
    Y1 = np.array([np.zeros(num/2)]).T
    X2 = np.random.random((num/2,2)) * 10  + np.array([10,10])
    Y2 = np.array([np.zeros(num/2)]).T + 100
    X = np.concatenate((X1,X2),axis = 0)
    Y = np.concatenate((Y1,Y2),axis = 0)
    return np.concatenate((X,Y),axis = 1)

if __name__ == '__main__':
    data = data_gen()
    print data.shape
    #np.savetxt('Data/best4knn.csv', data, delimiter=",")
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.scatter(data[:,0], data[:,1], data[:,2]) 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
