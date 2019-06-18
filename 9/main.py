#!/usr/bin/env python3
"""
TODO
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
from functools import partial
import pdb
def set_up_plot(t_max, limits):
    """Sets up the plot and its parameters.

    :returns: fig, ax
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

    ax_norm.set_title('Norm of $P(x, t)$ ($R(t)/R(0)$)')
    ax_norm.set_ylabel('Norm')

    ax_mean.set_title('Mean of $P(x, t)$')
    ax_mean.set_ylabel('Mean')

    ax_var.set_title('Variance of $P(x, t)$')
    ax_var.set_ylabel('Variance')

    return fig, axes

def walk(x, x_abs, D, v, dt, S=1):
    for _ in range(S):
        x = x + v*dt + \
            np.sqrt(2*D*dt)*np.random.normal(0, 1, len(x))
        x = x[x < x_abs]

    return x

def walk_with_snapshots(x, x_abs, D, v, dt, S, t_max):
    steps = int(t_max / (S*dt))
    R = len(x)

    snapshots = []  # [(0, 1, 0, 0)]
    positions = []  # [x]
    t = 0

    for _ in range(steps):
        x = walk(x, x_abs, D, v, dt, S)
        t = t + dt*S
        norm = len(x)/R

        positions.append(x)
        snapshots.append((t, norm, x.mean(), x.std(ddof=1)**2))

    return positions, np.array(snapshots)

def set_line_data(line, t, data, i):
    line.set_xdata(t[:i])
    line.set_ydata(data[:i])

def norm_dist(x, mu, v):
    return 1/(2*np.pi*v)**(1/2) \
        *np.exp(-(x-mu)**2/(2*v))

class AnimatedLines():
    def __init__(self, ax, *data, index=0, legend=True, absolute_index=False):
        self.ax = ax
        self._data = data
        self._index = index
        self._orig_lines = self.ax.lines.copy()
        self._absolute_index = absolute_index

        linestyles = list(matplotlib.lines.lineStyles.keys())
        linestyles.remove('None')

        # init lines
        self._lines = []
        for i, (x_data, y_data, kwargs) in enumerate(self._data):
            linestyle = linestyles[i % len(linestyles)]
            if 'linestyle' in kwargs:
                linestyle = kwargs['linestyle']
                del kwargs['linestyle']

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
        self.ax.lines = self._orig_lines

    def set_index(self, index):
        self._index = index
        self._update_lines()

    def _update_lines(self):
        for i, line in enumerate(self._lines):
            x_data, y_data, _ = self._data[i]

            if self._absolute_index:
                line[0].set_xdata(x_data)
                line[0].set_ydata(y_data[self._index])

            else:
                line[0].set_xdata(x_data[:(self._index + 1)])
                line[0].set_ydata(y_data[:(self._index + 1)])






def plot_walk(positions, snapshots, theo_positions, theo_snapshots, R,
              fig, axes, event):
    mode = event.canvas.toolbar.mode

    if not (event.button == 1 and mode == '' and (event.inaxes in axes)):
            return

    ax_hist, ax_norm, ax_mean, ax_var = axes
    t, norms, means, variances = snapshots.T
    t_norms, t_means, t_variances = theo_snapshots.T

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
                               (*theo_positions,
                                dict(
                                    label="Theoretical Prop. Dist. w/o Drain",
                                    color='green')),
                               absolute_index=True)

    smart_lines = [norm_lines, mean_lines, variance_lines, t_pos_line]


    for i in range(len(t)):
        ax_hist.patches = []
        ax_hist.hist(positions[i], bins=100,
                     weights=np.ones_like(positions[i])*norms[i]/R, color='blue')


        for line in smart_lines:
            line.set_index(i)


        event.canvas.flush_events()
        event.canvas.draw()

def theoretical_diffusion(x0, v, t, D, limits):
    x = np.linspace(-50, 50, 1000)[:, None]
    x_abs = limits[1]


    means = x0 + v*t
    variances = 2*D*t

    t = t[None, :]
    positions = stats.norm.pdf(x, x0 + v*t, 2*D*t) # \
        # - norm_dist(x, 2*x_abs - x0 + v*t, 2*D*t) \
        # * norm_dist(x_abs, x0 + v*t, 2*D*t) \
        # / norm_dist(x_abs, x_abs - x0 + v*t, 2*D*t)

    snapshots = np.array([np.ones_like(means), means, variances]).T  # same
                                                                     # format

    return (x, positions.T), snapshots

def main():
    """Set model parameters and Dispatch the main logic.  Used as
    convenience.

    :param R: realization count
    :param N: particle count
    :param dt: averaging time interval
    """

    R = 10000
    S = 100
    t_max = 30
    D = 1
    v = 0.15
    dt = 0.01
    x_min = -10
    x_abs = 10
    limits = (x_min, x_abs)
    x0 = 0

    # print user guide, whether he wants it or not :)
    print(__doc__)

    fig, axes = set_up_plot(t_max, limits)
    x = np.ones(R)*x0

    positions, snapshots = walk_with_snapshots(x, x_abs, D, v, dt, S, t_max)
    theo_positions, theo_snapshots = theoretical_diffusion(x0, v,
                                                           snapshots.T[0],
                                                           D, limits)
    on_click = partial(plot_walk, positions, snapshots,
                       theo_positions, theo_snapshots, R, fig, axes)
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
