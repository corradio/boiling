import numpy as np

def y_hist(X, V):
    N = 50
    Y = np.linspace(-1, 1, N)
    d = 0.1
    temperature = []
    density = []
    for y in Y:
        ix = ((y - d < X[:, 1]) & (X[:, 1] <= y + d))
        v = V[ix]
        temperature.append(np.mean(v))
        volume = (np.min([y + d, 1.0]) - np.max([y - d, -1.0])) * 2.0
        density.append(v.shape[0] / volume)
    return Y, temperature, density

def plot_simulation(r, interval=1):
    # Plot
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.gridspec as gridspec

    def update_plot(i, data, velocities, line, plt_temperature, plt_hist_y_temperature, plt_hist_y_density):
        #Y = data[i].reshape(N, 4)
        X = data[i]
        V = velocities[i]
        #line.set_data(Y[:, 0:2])
        line.set_array(V)
        line.set_cmap(cm)
        line.set_offsets(X[:, 0:2])

        plt_temperature.set_data(i, np.mean(V))

        Y, temperature, density = y_hist(X, V)
        plt_hist_y_temperature.set_data(temperature, Y)
        plt_hist_y_density.set_data(density, Y)

        return None

    data = np.array(r)    
    velocities = np.linalg.norm(data[:, :, 2:4], axis=2)

    # Important to have a square view!
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 3)

    ax = plt.subplot(gs[0:2, 0:2])
    cm = plt.get_cmap('Reds')
    l = plt.scatter(x=[], y=[], c=[], s=50, cmap=cm)
    plt.clim([velocities.min(), velocities.max()])
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.plot([-1, 1], [-1, -1], '-k')
    plt.plot([-1, 1], [1, 1], '-k')
    plt.plot([1, 1], [-1, 1], '-k')
    plt.plot([-1, -1], [-1, 1], '-k')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax1 = plt.subplot(gs[0:2, 2])
    plt.grid(True)
    plt_hist_y_temperature, = ax1.plot([], [], '-r')
    plt.xlim(0, 6)
    plt.ylim(-1, 1)
    ax1.set_xlabel('temperature', color='r')
    ax1.get_yaxis().set_visible(False)

    ax2 = ax1.twiny()
    plt_hist_y_density, = ax2.plot([], [], '-k')
    plt.xlim(0, 1000)
    plt.ylim(-1, 1)
    ax2.set_xlabel('density', color='k')
    ax2.get_yaxis().set_visible(False)

    plt.subplot(gs[2, :])
    plt.grid(True)
    plt.plot(np.mean(velocities, axis=1), '-')
    plt_temperature, = plt.plot(0, np.mean(velocities[0]), 'or')
    plt.title('System temperature')

    line_ani = animation.FuncAnimation(fig, update_plot, len(data), fargs=(data, velocities, l, plt_temperature, plt_hist_y_temperature, plt_hist_y_density),
        interval=interval, blit=False)

    #line_ani.save('demoanimation.gif', writer='imagemagick', fps=10);

    plt.show()

