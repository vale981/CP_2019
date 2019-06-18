#!/usr/bin/env python3
"""
TODO
"""

import numpy as np
from functools import partial
import matplotlib
import matplotlib.pyplot as plt

def set_up_plot(t_max, limits):
    """Sets up the plot and its parameters.

    :returns: fig, axes (histogram, norm, mean, variance)
    """

    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    ax_hist, ax_norm, ax_mean, ax_var = axes

    for ax in axes[1:]:
        ax.set_xlabel('t')
        ax.set_xlim(0, t_max)

    ax_hist.set_title('Histogram of $P(x, t)$')
    ax_hist.set_xlabel('x')
    ax_hist.set_xlim(*limits)
    ax_hist.set_ylabel('$P(x, t)$ Normalized')
    ax_hist.set_ylim(0, 0.4)

    ax_norm.set_title('Norm of $P(x, t)$ ($R(t)/R(0)$)')
    ax_norm.set_ylabel('Norm')

    ax_mean.set_title('Mean of $P(x, t)$')
    ax_mean.set_ylabel('Mean')

    ax_var.set_title('Variance of $P(x, t)$')
    ax_var.set_ylabel('Variance')

    return fig, axes

def walk_with_snapshots(x, x_abs, D, v, dt, S, t_max):
    """Simulates the diffusion of particles in an environment with the
    diffusion constant D and a drift speed v.  At every S time steps,
    a snapshot of the system (particle positions, statistical moments)
    is being taken.

    :param x: inital ensble
    :param x_abs: position of the absorbing wall
    :param D: diff.  constant
    :param v: drift speed
    :param dt: time step
    :param S: leap size
    :param t_max: maximal time

    :returns: particle positions (array), an array with rows of:
              (time, norm, mean, variance)

    :rtype: Tuple
    """

    steps = int(t_max / (S*dt))  # step count
    R = len(x)  # population

    # initialize the snapshots and time
    snapshots = [(0, 1, 0, 0)]
    positions = [x]
    t = 0

    for _ in range(steps):

        # advance by S steps
        for _ in range(S):
            x = x + v*dt + \
                np.sqrt(2*D*dt)*np.random.normal(0, 1, len(x))
            x = x[x < x_abs]  # absorb those particles

        # update time and calculate norm
        t = t + dt*S
        norm = len(x)/R

        # take a snapshot
        positions.append(x)
        snapshots.append((t, norm, x.mean(), x.std(ddof=1)**2))

    return positions, np.array(snapshots)

def norm_dist(x, mu, v):
    """The gaussian normal distribution.

    :param x: x coordinates
    :param mu: mean
    :param v: variance
    :returns: the gaussian normal dist.
    """

    return 1/(2*np.pi*v)**(1/2) \
        *np.exp(-(x-mu)**2/(2*v))

class AnimatedLines():
    """updates lines in time steps."""
    def __init__(self, ax, *data, index=0, legend=True, absolute_index=False):
        """Creates an animated line.

        :param ax: axis
        :param index: initial index
        :param legend: wether to draw a legend
        :param absolute_index: wether to reaveal the y_data one-by-one
            (y_data[:i+1]) or to plot the full y_data[i] for each step
        :param *data: line data arguments of the form (x_data, y_data,
            kwargs for ax.plot)
        """

        self.ax = ax
        self._data = data
        self._index = index
        self._orig_lines = self.ax.lines.copy()
        self._absolute_index = absolute_index

        linestyles = list(matplotlib.lines.lineStyles.keys())
        linestyles.remove('None')  # no hidden lines

        # init lines
        self._lines = []
        for i, (x_data, y_data, kwargs) in enumerate(self._data):

            # figure out the linestyle
            linestyle = linestyles[i % len(linestyles)]
            if 'linestyle' in kwargs:
                linestyle = kwargs['linestyle']
                del kwargs['linestyle']

            # make the lines
            line = None
            if self._absolute_index:
                line = self.ax.plot(x_data, y_data[0],
                                    linestyle=linestyle, **kwargs)
            else:
                line = self.ax.plot(x_data, y_data,
                                    linestyle=linestyle, **kwargs)

            self._lines.append(line)

        if legend:
            self.ax.legend()

    def __del__(self):
        """Restores the original lines on the axis."""

        self.ax.lines = self._orig_lines

    def set_index(self, index):
        """Sets the current animation index

        :param index: the index
        """

        self._index = index
        self._update_lines()

    def _update_lines(self):
        """Updates the lines on the axis.
        """

        for i, line in enumerate(self._lines):
            x_data, y_data, _ = self._data[i]

            if self._absolute_index:
                line[0].set_xdata(x_data)
                line[0].set_ydata(y_data[self._index])

            else:
                line[0].set_xdata(x_data[:(self._index + 1)])
                line[0].set_ydata(y_data[:(self._index + 1)])

def plot_walk(positions, snapshots, theo_positions, theo_snapshots,
              bins, axes, event):
    """Dymanically plots a histogram of the propability distribution
    of the diffusion simulation as well as the theoretical distr.
    with and without an absorbing wall.  Three more plots show the
    simulated (with abs.) and theoretical (w/o.  abs.) values of the
    total propability (norm), mean and variance of the distributions.

    :param positions: the simulated particle positions
    :param snapshots: the snapshots of the statistical moments
    :param theo_positions: the analytical prop.  densities (without
        abs., with abs.)
    :param theo_snapshots: the snapshots of the stat.  moments without
        abs.
    :param bins: bin count
    :param fig:
    :param axes: the axes to draw on (hist, norm, mean, variance)
    :param event: the click event
    """

    # click handling boilerplate
    mode = event.canvas.toolbar.mode
    if not (event.button == 1 and mode == '' and (event.inaxes in axes)):
            return

    # unpack axes and snapshots
    ax_hist, ax_norm, ax_mean, ax_var = axes
    t, norms, means, variances = snapshots.T
    t_norms, t_means, t_variances = theo_snapshots.T
    limits = (theo_positions[0].min(), theo_positions[0].max())  # dirty

    # set those animations up
    norm_lines = AnimatedLines(ax_norm,
                               (t, norms,
                                 dict(label="Simulated Norm", color='blue')),
                                (t, t_norms,
                                 dict(label="Analytical Norm", color='green')))

    mean_lines = AnimatedLines(ax_mean,
                               (t, means,
                                 dict(label="Simulated Mean", color='blue')),
                                (t, t_means,
                                 dict(label="Analytical Mean", color='green')))

    variance_lines = \
        AnimatedLines(ax_var,
                      (t, variances,
                        dict(label="Simulated Variance", color='blue')),
                       (t, t_variances,
                        dict(label="Analytical Varance", color='green')))


    t_pos_line = AnimatedLines(ax_hist,
                               (theo_positions[0], theo_positions[1],
                                dict(
                                    label="Theoretical Prop. Dist. w/o Drain",
                                    color='green')),
                               (theo_positions[0], theo_positions[2],
                                dict(
                                    label="Theoretical Prop. Dist. w/ Drain",
                                    color='green')),
                               absolute_index=True)

    # gather 'em
    smart_lines = [norm_lines, mean_lines, variance_lines, t_pos_line]

    area = (limits[1] - limits[0])/(bins)  # unit area for one data point
    for i in range(len(t)):

        # clear the histogr.
        ax_hist.patches = []

        # determine the weights and plot the histogram
        weigths = np.ones_like(positions[i]) / (area*len(positions[i]))*norms[i]
        ax_hist.hist(positions[i], bins=100, range=limits,
                     weights=weigths, color='blue', density=False)

        # update the lines
        for line in smart_lines:
            line.set_index(i)

        # redraw!
        event.canvas.flush_events()
        event.canvas.draw()

def theoretical_diffusion(x0, v, t, D, limits):
    """Calculate the theor. prop. distr. with and w/o abs. wall.
    Parameters: see main
    :returns: (x, prop. dist, prop di)
    """

    x = np.linspace(*limits, 1000)[:, None]
    x_abs = limits[1]
    means = x0 + v*t
    variances = 2*D*t

    t = t[None, :]
    positions = norm_dist(x, x0 + v*t, 2*D*t)  # without absorbtion
    positions_abs = positions \
        - norm_dist(x, 2*x_abs - x0 + v*t, 2*D*t) \
        * norm_dist(x_abs, x0 + v*t, 2*D*t) \
        / norm_dist(x_abs, 2*x_abs - x0 + v*t, 2*D*t)  # with absorbtion

    snapshots = np.array([np.ones_like(means), means, variances]).T
    return (x, positions.T, positions_abs.T), snapshots

def main():
    """Set model parameters and Dispatch the main logic.  Used as
    convenience.
    """

    # Model Parameters
    D = 1  # Diffusion Constant
    v = 0.15  # Drift Speed
    R = 10000  # Relization Count
    S = 100  # leap-size
    t_max = 30  # simulation time
    dt = 0.01  # time step
    limits = (-25, 10)  # spatial limits (-25 empirical, position of
                        # the absorb. wall)
    x0 = 0  # initial particle position
    bins = 100  # bin count

    # print user guide, whether he wants it or not :)
    print(__doc__)

    fig, axes = set_up_plot(t_max, limits)

    # initialize the ensemple
    x = np.ones(R)*x0

    # simulate particle movement
    positions, snapshots = walk_with_snapshots(x, limits[1], D, v, dt, S, t_max)
    theo_positions, theo_snapshots = theoretical_diffusion(x0, v,
                                                           snapshots.T[0],
                                                           D, limits)

    # handle clicks and show plot
    on_click = partial(plot_walk, positions, snapshots,
                       theo_positions, theo_snapshots, bins,
                       axes)
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

if __name__ == '__main__':
    main()

###############################################################################
#                                  Diskussion                                 #
###############################################################################

"""
TODO
"""
