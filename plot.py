import numpy as np

def plot_simulation(r, interval=1):
    # Plot
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.gridspec as gridspec

    def update_plot(i, data, line, e):
        #Y = data[i].reshape(N, 4)
        Y = data[i]
        #line.set_data(Y[:, 0:2])
        speeds = np.sum(np.power(Y[:, 2:4], 2.0), axis=1)
        line.set_array(speeds)
        line.set_offsets(Y[:, 0:2])
        ee = np.sum(np.power(Y[:, 2:4], 2.0), axis=0)
        e.set_data([i, i], ee)

        return line, e

    data = np.array(r)

    # Important to have a square view!
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 3)

    plt.subplot(gs[0:2, 1:3])
    cm = plt.cm.get_cmap('RdYlBu')
    l = plt.scatter(x=[], y=[], c=[], s=200, cmap=cm)
    plt.clim([0, 200])
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.plot([-1, 1], [-1, -1], '-k')
    plt.plot([-1, 1], [1, 1], '-k')
    plt.plot([1, 1], [-1, 1], '-k')
    plt.plot([-1, -1], [-1, 1], '-k')

    #line_ani.save('demoanimation.gif', writer='imagemagick', fps=10);

    plt.subplot(gs[0:2, 0])
    # TODO: Plot histogram of speed. This should show gradient of temperature

    plt.subplot(gs[2, :])
    energy = np.sum(np.power(data[:, :, 2:4], 2.0), axis=1)
    plt.plot(energy, '-')

    e, = plt.plot([0, 0], np.sum(np.power(data[0, :, 2:4], 2.0), axis=0), 'or')

    line_ani = animation.FuncAnimation(fig, update_plot, len(data), fargs=(data, l, e),
        interval=interval, blit=False)

    plt.show()

