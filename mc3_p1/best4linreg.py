import numpy as np

def data_gen(num = 2000, a=1.0, b=1.0, c = 10.0, sigma = 5):
    X = np.random.random((num,2)) * 20
    Y = np.array([a*X[:,0] + b*X[:,1] + c + sigma*np.random.randn(num)]).T
    return np.concatenate((X,Y),axis = 1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib
    font = {'size'  : 22}
    matplotlib.rc('font',**font)
    data = data_gen()
    print data.shape
    #np.savetxt('Data/best4linreg.csv', data, delimiter=",")
    plt.clf()
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    plt.hold(True)
    x_surf = np.arange(-4,24,0.1)
    y_surf = np.arange(-4,24,0.1)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = x_surf + y_surf + 10.0
    ax.plot_surface(x_surf, y_surf, z_surf, alpha = 0.45, linewidth = 0, color='fuchsia')


    ax.scatter(data[:,0], data[:,1], data[:,2], label = '$z=x+y+10.0+\epsilon, \epsilon\sim 5*\mathcal{N}(0,1)$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.legend()
    plt.savefig('b4l.png', bbox_inches='tight')
    plt.show()
