#!/usr/bin/env python3
"""
This programm simulates the diffusion of particles in an environment
with the diffusion constant D=1, a drift speed v=0.15 and an absorbing
wall at x_abs=10 by implementing the Langevin-Equation.  At every
S=100 time steps (dt=0.01), a snapshot of the system (particle
positions, statistical moments) is being taken.  The simulation runs
unitl T=30.

One plot dynamicly displays a normalized histogram of the propabilty
density for the process as well as the analytically obtained prop.
densities with and w/o absorbing wall.

Three more plots show the simulated (with abs.) and theoretical (w/o.
abs.) values of the total propability (norm), mean and variance of the
distributions.
"""

from functools import partial
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
    """updates lines in time steps. STRIPPED"""
    def __init__(self, ax, *data):
        """Creates an animated line.

        :param ax: axis
        :param *data: line data arguments of the form (x_data, y_data, label)
        """

        self.ax = ax
        self._data = data
        self._index = 0
        self._orig_lines = self.ax.lines.copy()
        self._absolute_index = (self._data[0][0].shape !=
                                self._data[0][1].shape)

        linestyles = list(matplotlib.lines.lineStyles.keys())
        colors = list(matplotlib.colors.cnames.keys())
        linestyles.remove('None')  # no hidden lines

        # init lines
        self._lines = []
        for i, (x_data, y_data, label) in enumerate(self._data):
            # figure out the linestyle and color
            linestyle = linestyles[i % len(linestyles)]
            color = colors[(i+10) % len(colors)]

            # make the line
            line = None
            y_data = y_data[0] if self._absolute_index else y_data
            line = self.ax.plot(x_data, y_data, label=label,
                                    linestyle=linestyle, color=color)

            self._lines.append(line)

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

            # select the rigth rows
            x_data, y_data, _ = self._data[i]
            x_data, y_data = (x_data, y_data[self._index]) \
                if self._absolute_index \
                else (x_data[:(self._index + 1)], y_data[:(self._index + 1)])

            line[0].set_xdata(x_data)
            line[0].set_ydata(y_data)

def plot_walk(positions, snapshots, theo_positions, theo_snapshots,
              bins, axes, event):
    """Dymanically plots a histogram of the propability distribution
    of the diffusion simulation as well as the theoretical distr.
    with and without an absorbing wall and the norm, mean, and
    variance of the theoretical and simulated ditstr.

    :param positions: the simulated particle positions
    :param snapshots: the snapshots of the statistical moments
    :param theo_positions: the analytical prop.  densities (x, without
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
    norm_lines = AnimatedLines(ax_norm, (t, norms, "Simulated Norm"),
                               (t, t_norms, "Analytical Norm"))

    mean_lines = AnimatedLines(ax_mean, (t, means, "Simulated Mean"),
                               (t, t_means, "Analytical Mean"))

    variance_lines = \
        AnimatedLines(ax_var, (t, variances, "Simulated Variance"),
                      (t, t_variances, "Analytical Varance"))

    t_pos_line = AnimatedLines(ax_hist,
                               (theo_positions[0], theo_positions[1],
                                "Theoretical Prop. Dist. w/o Drain"),
                               (theo_positions[0], theo_positions[2],
                                "Theoretical Prop. Dist. w/ Drain",))

    smart_lines = [norm_lines, mean_lines, variance_lines, t_pos_line]
    area = (limits[1] - limits[0])/(bins)  # unit area for one data point

    for i in range(len(t)):
        # clear the histogr.
        ax_hist.patches = []

        # determine the weights and plot the histogram
        weigths = np.ones_like(positions[i])/(area*len(positions[i]))*norms[i]
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

    :param x0: inital position
    :param v: drift speed
    :param t: snapshot times step
    :param D: diff.  constant
    :param limits: the spatial constraints (x_min, x_abs)

    :returns: (x, prop. dist, prop dist with wall)
    """

    x = np.linspace(*limits, 1000)[:, None]  # eval points
    x_abs = limits[1]  # wall
    means = x0 + v*t  # means
    variances = 2*D*t  # variances

    t = t[None, :]
    t[t == 0] = 0.1  # ~delta limit
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

    # do it here, conserves lines :)
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    ax_hist, ax_norm, ax_mean, ax_var = axes

    for ax in axes[1:]:
        ax.set_xlabel('t')
        ax.set_xlim(0, t_max)

    ax_hist.set_title('$P(x, t)$')
    ax_hist.set_xlabel('x')
    ax_hist.set_xlim(*limits)
    ax_hist.set_ylabel('$P(x, t)$')
    ax_hist.set_ylim(0, 0.4)

    ax_norm.set_title('Norm of $P(x, t)$ ($R(t)/R(0)$)')
    ax_norm.set_ylabel('Norm')

    ax_mean.set_title('Mean of $P(x, t)$')
    ax_mean.set_ylabel('$\mu$')

    ax_var.set_title('Variance of $P(x, t)$')
    ax_var.set_ylabel('$\sigma^2$')

    # initialize the ensemple and simulate part. movement
    x = np.ones(R)*x0
    positions, snapshots = \
        walk_with_snapshots(x, limits[1], D, v, dt, S, t_max)
    theo_positions, theo_snapshots = \
        theoretical_diffusion(x0, v, snapshots.T[0], D, limits)

    # handle clicks and show plot
    on_click = partial(plot_walk, positions, snapshots,
                       theo_positions, theo_snapshots, bins,
                       axes)
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

if __name__ == '__main__':
    main() # 299 Lines!

"""
a) Der absorbierende Rand entfernt Teilchen aus dem Modell und
veringert somit die Norm ab ca.  t=3.8 merklich.  (Sie faellt
annaehernd linear, wie es ein konstantes v-drift bedingt) Der
mittelwert der Verteilung wandert ab dieser Zeit nicht mehr mit
v_drift nach rechts sondern kehrt ab t~=10 sogar nach links um.  Da
die Verteilung durch v_drift und den rand nur beding breitlaufen kann
steigt auch die Varianz immer weniger an (divergiert von theorie ab
t~=5).  Die Form der Warscheinlichkeitsverteilung schein sich
'anzustauen' da nur teilchem mit geringerem impuls 'ueberleben'.
Gleichzeitig bewegen sich teilchen in der Naehe der Wand nicht weit
von selbiger fort und werden durch die Driftgeschwindigkeit nach
kurzerzeit in diese hineinbewegt.  Somit geht auch das Histogram in
Wandnaehe gegen null.  Schaltet man die Absorbtion ab folgt die
Simulation sehr gut den Theoretischen zusammenthaengen.

b) Am meisten aendert sich die Vert.  bei x=x_abs als bei der
absorbierenden wand.  Das histogram folgt an anderen Orten immer noch
sehr gut der theoretichen Kurve doch liegt in der Naehe der Wand
deutlich ueber selbiger.  Tatsaechlich werden geringfuegig weniger
Teilchen Absorbiert.  Es vergeht nun mehr zeit zwischen
Geschwindigkeits- und Bewegungsrichtungsaenderung, es werden groessere
Spruenge gemacht.  Die verteilung naehert sich fuer grosse
Zeitschritte immer mehr der Verteilung ohne abfluss an.  Durch die
groesseren sprunge konnen sich teilchen nun weit genug weg von der
wand weg bewegen um erst viel spaeter absorbiert zu werden.  Es findet
weniger Selektion stat und somit wird auch die Form der Verteilung
weniger durch die Wand beeinflusst.

c) Fuer v=0.5 nimmt der Aufstaueffekt zu (maximum der Vert.  wandert
nach rechts).  Es werden mehr teilchen absorbiert, wobei sich die
abnahme Norm parabelartig abflacht.  Der mittelwert erreicht ein
plateau ab ca t=5 (x=3) und auch die Varanz bleibt ab dieser zeit
ungefaehr konstant.  Die teilchen driften nun so schnell nach rechts,
dass sich Breitlaufen und Absorbtion die Wage halten (nach links
limitiert durch den Drift, nach rechts durch die Wand + Drift).  Die
teilchen kommen in einem Teilschritt weiter nach rechts, kommen der
Wand naeher und werden sicherer absorbiert.  Die Uebereinstimmung mit
der Theoretischen Kurve ist immer noch sehr gut.
"""
