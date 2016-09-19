import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def data_gen(num = 2000, a=1.0, b=1.0, c = 10.0, sigma = 5):
    X = np.random.random((num,2)) * 20
    Y = np.array([a*X[:,0] + b*X[:,1] + c + sigma*np.random.randn(num)]).T
    return np.concatenate((X,Y),axis = 1)

if __name__ == '__main__':
    data = data_gen()
    print data.shape
    #np.savetxt('Data/best4linreg.csv', data, delimiter=",")
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], label = 'z = x + y + 10.0 + noise')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
