import numpy as np
def data_gen(num = 2000):
    X1 = np.random.random((num/2,2)) * 10 
    Y1 = np.array([np.zeros(num/2)]).T
    X2 = np.random.random((num/2,2)) * 10  + np.array([10,10])
    Y2 = np.array([np.zeros(num/2)]).T + 100
    X = np.concatenate((X1,X2),axis = 0)
    Y = np.concatenate((Y1,Y2),axis = 0)
    return np.concatenate((X,Y),axis = 1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib

    font = {'size'  : 16}
    matplotlib.rc('font',**font)


    data = data_gen()
    dataX = data[:,:-1]
    dataY = data[:,-1]
    newdataX = np.ones([dataX.shape[0],dataX.shape[1]+1])
    newdataX[:,0:dataX.shape[1]]=dataX
     
    coefs, residuals, rank, s =\
            np.linalg.lstsq(newdataX, dataY)
    print coefs
    
    a, b, c = coefs

    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.scatter(data[:,0], data[:,1], data[:,2]) 
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    
    x_surf = np.arange(-5, 25, 0.1)
    y_surf = np.arange(-5, 25, 0.1)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = a*x_surf + b*y_surf + c
    ax.plot_surface(x_surf,y_surf,z_surf, alpha = 0.45, linewidth=0, color = 'red')
    plt.show()
    plt.savefig('b4k.png', bbox_inches='tight')
