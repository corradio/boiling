import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

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

def compute_aggregated_velocity_field(state, N=20):
    M = []
    space = np.linspace(-1, 1, N+2)[1:-1]
    delta = space[1] - space[0]
    for x1 in space:
        for x2 in space:
            ix = (x1 - delta <= state[:, 0]) & (state[:, 0] < x1 + delta) & (x2 - delta <= state[:, 1]) & (state[:, 1] < x2 + delta)
            # Sum of velocities squared
            M.append([x1, x2, np.sum(state[ix, 2]), np.sum(state[ix, 3])])
    return np.array(M)

def plot_simulation(r, interval=1, quiver=False):
    # Plot

    def update_plot(i, data, velocities, line, plt_temperature, plt_hist_y_temperature, plt_hist_y_density, plt_dE, quiver):
        X = data[i]
        V = velocities[i]

        if quiver:
            M = compute_aggregated_velocity_field(X)
            line.set_offsets(M[:, 0:2])
            line.set_UVC(M[:, 2], M[:, 3])
        else:
            line.set_array(V)
            line.set_cmap(cm)
            line.set_offsets(X[:, 0:2])

        plt_temperature.set_data(i, np.mean(V))
        plt_dE.set_data([i, i], [e_in[i], e_out[i]])

        Y, temperature, density = y_hist(X, V)
        plt_hist_y_temperature.set_data(temperature, Y)
        plt_hist_y_density.set_data(density, Y)

        return None

    data = np.array(r)
    velocities = np.linalg.norm(data[:, :, 2:4], axis=2)

    # Calculate energy received / dissipated
    R = 0.01
    dr = np.diff(r, axis=0)
    dv = np.diff(velocities, axis=0)
    vy_change = (r[:-1, :, 3] * r[1:, :, 3] < 0)
    # TODO: calculate sign change by multiplying things between n+1 and n
    ix_in = (r[:-1, :, 1] <= -1 + 2.0 * R) & vy_change # change of sign of vy at lower part
    ix_out = (r[:-1, :, 1] >= 1 - 2.0 * R) & vy_change # change of sign of vy at upper part
    de = np.power(dv, 2.0) # (norm of dv)^2
    e_in = np.zeros(de.shape[0])
    e_out = np.zeros(de.shape[0])
    for i in range(de.shape[0]): e_in[i] = de[i, ix_in[i, :]].sum()
    for i in range(de.shape[0]): e_out[i] = de[i, ix_out[i, :]].sum()

    from scipy.signal import convolve
    e_in = convolve(e_in, [1]*10, mode='same')
    e_out = convolve(e_out, [1]*10, mode='same')

    # Important to have a square view!
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(4, 3)

    ax = plt.subplot(gs[0:2, 0:2], aspect='equal')
    if quiver:
        M = compute_aggregated_velocity_field(data[-1, :, :])
        l = plt.quiver(M[:, 0], M[:, 1], M[:, 2], M[:, 3], scale=1000)
    else:
        cm = plt.get_cmap('Reds')
        l = plt.scatter(x=[], y=[], c=[], s=20, cmap=cm, lw=0.3)
        plt.clim([velocities[-1].min(), velocities[-1].max()])
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
    plt.xlim(0, 10)
    plt.ylim(-1, 1)
    ax1.set_xlabel('temperature', color='r')
    ax1.get_yaxis().set_visible(False)

    ax2 = ax1.twiny()
    plt_hist_y_density, = ax2.plot([], [], '-k')
    plt.xlim(0, 2000)
    plt.ylim(-1, 1)
    ax2.set_xlabel('density', color='k')
    ax2.get_yaxis().set_visible(False)

    ax = plt.subplot(gs[2, :])
    plt.grid(True)
    plt.plot(np.mean(velocities, axis=1), '-k')
    plt_temperature, = plt.plot(0, np.mean(velocities[0]), 'or')
    plt.legend(['System temperature'], loc=0)

    ax = plt.subplot(gs[3, :])
    plt.grid(True)
    plt.plot(e_in, '-r')
    plt.plot(e_out, '-b')
    plt_dE, = plt.plot([0, 0], [e_in[0], e_out[0]], 'ok')
    plt.ylim(0, 2000)
    plt.legend(['dE_in', 'dE_out'])

    line_ani = animation.FuncAnimation(fig, update_plot, len(data), fargs=(data, velocities, l, plt_temperature, plt_hist_y_temperature, plt_hist_y_density, plt_dE, quiver),
        interval=interval, blit=False)

    #line_ani.save('demoanimation.gif', writer='imagemagick', fps=10);

    # Total Energy flux in area = sum of velocities in that area

    plt.show()

