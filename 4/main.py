#!/usr/bin/env python3
"""
Analyze the dynamics in the driven and skewed two-well potential.  The
hamiltonian of the system considered is: H = p^2/2 + x^4 − x^2 + x *
[A + B*sin(omega*t)]

Where A = 0.2 (the skew of the potential) and B = 0.1 the parameter
for the drive coupling and omega = 1 the drive frequency.

The hamiltonian is giving rise to the equations of motion:
 - d/dt q = p
 - d/dt p = -4*x**3 + 2*x - A - B*np.sin(omega*t)

The system is analized for 200 drive periods (2*pi/omega) by plotting
phase space trajectories (obtained by numerically solving the
differential equations for (x, p) and plotting them:

    - as they are one the left subplot

    - and only for times n*2*pi/omega (n in the Natural Numbers) in
      the right hand side plot (stroboscope)

Contour lines of H for B = 0 represent regular orbits for the undriven
sytem.

New orbits can be added by mouse click.

A periodic trajectory is being drawn upon the programm start.
"""

import functools
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.optimize import minimize_scalar


def set_up_plots():
    """Sets up the plots and their parameters.

    :returns: fig, (trajecotry axis, stroboscope axis)
    """

    # Create two subplots (trajectory, stroboscope) sharing the x,y axis.
    fig, (ax_tr, ax_st) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax_tr.set_title("Phase Space Trajectory")
    ax_st.set_title("Stroboscopic Phase Space ")

    for ax in ax_tr, ax_st:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$p$")

        # set fixed aspect ratio
        ax.set_aspect("equal")

        # fix the limits
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

    return fig, (ax_tr, ax_st)


def hamiltonian(x, p, A):
    """The hamiltonian for B = 0

    :param x: x coord
    :param p: p coord
    :param A: model parameter
    :returns: the hamiltonian (in this Energy)
    :rtype: number
    """

    return p ** 2 / 2 + x ** 4 - x ** 2 + x * A


def draw_energy_countour(axes, A):
    """Draws the energy contour lines of the hamiltonian.  (= orbits
    without motor)

    :param axes: axes
    :param A: model parameter

    :rtype: None
    """

    # create a grid for the energy calculation
    coordinates = np.linspace(-1.2, 1.2, 1000)
    p_x, p_p = np.meshgrid(coordinates, coordinates)

    # calculate the energies from the hamiltonian
    energies = hamiltonian(p_x, p_p, A)
    levels = np.linspace(-0.4, 0.3, 10)

    # find the local energy maximum => separatrix
    separatrix = -minimize_scalar(
        lambda x: -hamiltonian(x, 0, A), bounds=[-1, 1], method="bounded"
    ).fun

    # plot the contours on both axes
    for ax in axes:
        ax.contour(p_x, p_p, energies, alpha=0.4, levels=levels)
        ax.contour(p_x, p_p, energies, alpha=0.4, levels=[separatrix])


def rhs(y, t, A, B, omega):
    """The right hand side of the equation of motion.

    :param numpy arrau y: (x, p) the generalized coordinates
    :param number t: time
    :param A: model parameter
    :param B: model parameter (Drive Force)
    :returns: time derivatives of (x, p) at time t
    :rtype: numpy array
    """

    x, p = y
    return np.array([p, -4 * x ** 3 + 2 * x - A - B * np.sin(omega * t)])


def add_trajectory(axes, periods, steps_per_period, initial_conditions, args):
    """Calculates the trajectory and stroboscopic trajecotry from the
    the gives initial conditions and draws them in their respective
    subplots.

    :param Tuple axes: the axes to draw to (trajectory, stroboscope)
    :param periods: drive periods to solve the DE for (n*2*pi/omega)
    :param steps_per_period: hown many steps to interpolate between
        the periods
    :param initial_conditions: (x0, p0) the initial conditions
    :param args: the model parameters (A, B, omega)

    :rtype: None
    """

    ax_tr, ax_st = axes
    _, _, omega = args

    # solve the DE, precise multiples of 2*pi/omega
    t_range = (
        np.arange(0, steps_per_period * periods) / steps_per_period * 2 * np.pi / omega
    )

    trajectory = odeint(rhs, initial_conditions, t_range, args=args)

    # unpack the solution
    x, p = trajectory.T

    # plot the trajecotry
    ax_tr.plot(x, p, linestyle="none", marker="o", markersize=0.5)

    # and the stroboscopic view
    ax_st.plot(
        x[::steps_per_period],
        p[::steps_per_period],
        linestyle="none",
        marker="o",
        markersize=0.8,
    )


def handle_mouse_click(axes, periods, steps_per_period, args, event):
    """Reacts to a mouse click if no toolbar mode is activated.

    Passes all arguments except `event` on to `add_trajectory`.
    :rtype: None

    """

    # only react if not zooming
    mode = event.canvas.toolbar.mode
    if event.button != 1 or mode != "" or (event.inaxes not in axes):
        return

    # plot the trajecotry
    add_trajectory(axes, periods, steps_per_period, (event.xdata, event.ydata), args)

    # refresh the canvas
    event.canvas.draw()


def handle_key_press(axes, event):
    """Handles key presses.

    :param ax_tr:
    :param ax_st:
    :param event:
    :returns:
    :rtype:

    """

    ax_tr, ax_st = axes
    if event.key == "u":
        ax_tr.lines = []
        ax_st.lines = []

    event.canvas.draw()


def format_coord(x, y, A):
    """Format the coordinate printout to show the energy.
    """
    return "x={:.5f}   p={:.5f}   H={:.5f}".format(x, y, hamiltonian(x, y, A))


def main():
    """Dispatch the main logic. Used as convenience.
    """

    # System Constants
    A = 0.2
    B = 0.1
    omega = 1

    # Solver Parameters
    periods = 200  # how many drive periods to solve for
    steps_per_period = 50

    # user guide
    print(__doc__)

    # set up figures and listeners
    fig, axes = set_up_plots()
    on_click = functools.partial(
        handle_mouse_click, axes, periods, steps_per_period, (A, B, omega)
    )

    fig.canvas.mpl_connect("button_press_event", on_click)

    on_key_press = functools.partial(handle_key_press, axes)
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    # show energy in the coordinate view
    for ax in axes:
        ax.format_coord = functools.partial(format_coord, A=A)

    # add the periodic orbit
    add_trajectory(
        axes, periods, steps_per_period, (0.231704, -0.349683), (A, B, omega)
    )

    # draw the contours
    draw_energy_countour(axes, A)

    # print parameters
    fig.text(0, 0, f"$A={A}$ $B={B}$ $\\omega={omega}$")

    plt.show()


if __name__ == "__main__":
    main()

###############################################################################
#                                  Diskussion                                 #
###############################################################################

"""
a) Da Energieerhaltung gilt (Hamiltonfunktion Zeitunabhaengig,
quadratisch in p) erwartet man regulaere Bewegung.  Diese verlaeft
dann auf geschlossenen Bahnen da:

    - potential-->oo fuer x-->+-oo

    - Energie ist erhalten => selbes `p` fuer jeweilige `x` koordinate
      => bewegung auf Kontourlinien von H

Um die Potentialminima erwartet man ellipsen (wie beim Harm.  Osz.).
Fuer kleine A, bei denen das zentrale Maximum des Dopp.  Muldenpot.
noch existiert erwartet man fuer Energien E=V_max eine Separatrix, die
die Bewegung um die einzelnen Potentialminima von der Bewgung durche
beide Minima trennt.  (Wobei A > 0 die Linke mulde breiter macht und
den Knoten der Separatrix nach rechts verschiebt).  Die ~~ellipsen
brechen auf und es entstehen schleifen, die mit zunemder Energie
glatter werden.

Im ortsraum: Schwingungen um die Minima (ungefaehr HO) fuer E < V_max,
sonst schwingung zwischen Potentialwaenden mit stoerung in den
Doppelmulden (schnell-langsam-schnell).

b) Zu erkennen sind regulaer anmutende orbits auf Schlauchartigen
Bahnen (links) bzw.  schleifen (rechts) besonders um die
Potentialminima herum (zentren verschoben, nach unten => siehe
vergleich mit Konturen), aber auch bei grossen energien.  Dazwischen
tritt chaotische Bewegung auf, die den Bereich zwischen hoeheren
energien (H=0.25 ohne Antrieb) und den minima (H~=0) auftritt.  Ab
energien von H=~0.25 treten wieder regulaere orbits auf, deren
schlaeuche (links) immer schlanker werden (der Motor ist nur kleine
Stoerung fuer hohe Energien).  Bemerkenswert ist, dass chaotische
Trajektorien auch regulaere Bereiche mit relatib hoher dichte
durchlaufen (nicht so im Stroboskopischen Bild!), aber immer wieder
austreten.  Im chaotischen Bereich gibt es kleinere Inseln
(stroboskop) oder Baender im linken Bild, wobei diese nur in den
Inseln im Stroboskop, nie aber auf anderen Regionen auf den Schlaechen
beginnen koennen (Energie ist nicht erhalten => Startpunkt spielt eine
Rolle).  Dabei treten diese inseln nur bei negativen p auf (und auch
E>V_min).  Diese inseln umgeben Zentren periodischer Bewegung, genau
so wie die Potentialminima und scheinen Resonanzen mit dem Antrieb
auszudruecken.  Auch um die Porentialminima gibt es im Stroboskopbild
geschlossene Orbits, die nicht die Minima zum Zentrum haben.

c) Diese 4 periodische trajektorie entspringt einem Inselsystem aus 4
Inseln: (x, p) = (0.232, -0.350) (gerundet).  Dieser orbit wird auch
beim Start des programms automatisch gezeichnet.

Interessanter weise bleibt dieser orbit auch fuer potentiale mit
anderen geraden Potenzen im ersten summanden in regulaeren inseln: (12,
6).

Da die x koordinate dieses orbits reltav klein ist, dominiert hier
noch der Quadratische Term des Potentials.
"""
