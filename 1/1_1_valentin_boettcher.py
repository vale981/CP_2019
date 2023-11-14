#!/usr/bin/env python3
"""
Draw the standard map of the kicked rotor interactively by specifying
the initial condifition with a mouse click on the plot.
"""

# Meine Muttersprache ist Deutsch. Ich verwende in Kommentaren
# generell die Englische Sprache um verwirrendes 'Denglisch' zu
# vermeiden :)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# use Cairo for performance
matplotlib.use("GTK3Agg", warn=False, force=True)

# NOTE: this is a stripped down version of the real thing (hence the
# remnants of oo design), which is included for your enjoyment

###############################################################################
#                                 Set Up Plot                                 #
###############################################################################

# I am doing this in the top level for clarity, as this module won't
# ever be imported anywhere.

# get them as objects, as I dont like fuzzing around with globals
# next time, i'll just use the global plot...
fig, ax = plt.subplots(1, 1)

# clear axis
ax.clear()

# set up the axis
ax.set_title("Standard Map")
ax.set_aspect("equal")

# set up limits and labels
ax.set_xlim([0, 2 * np.pi])
ax.set_xlabel(r"$\theta$")
ax.set_ylim([-np.pi, np.pi])
ax.set_ylabel(r"$p$")

###############################################################################
#                                    Logic                                    #
###############################################################################

# initialize default parameters
_K = 2.4  # Kick Parameter
_N = 1000  # iteration count
_POINT_SIZE = 0.5  # point size for the plot

# show K value in the figure
fig.text(0, 0, f"$K={_K}$")


def wrap_p(value):
    """Wraps the given value into the interval [-pi, pi).

    :param float value: the value to wrap
    :returns: a value in [-pi, pi)
    :rtype: float

    """
    return ((value + np.pi) % 2 * np.pi) - np.pi


def get_standard_map(theta_0, p_0, K, N):
    """Calculates the standard map for the given parameters.

    :param theta_0: initial theta
    :param p_0: initial p
    :param K: the K parameter
    :param N: the number of iterations
    :returns: a tuple of the (thetas, ps)
    :rtype: tuple[NumpyArray, NumpyArray]
    """

    # Initialize the arrays
    theta = np.zeros(N + 1)  # Start Point + N
    p = np.zeros(N + 1)

    # set the initial parameters
    norm_p = get_normalizer(2 * np.pi, -np.pi)

    # initial values
    theta[0] = theta_0
    p[0] = norm_p(p_0)

    # calculate the standard map, can't avoid for
    for curr in range(1, N + 1):
        last = curr - 1  # for readability
        theta[curr] = theta[last] + p[last]
        p[curr] = norm_p(p[last] + K * np.sin(theta[curr]))

    theta = theta % (2 * np.pi)  # do it once, for sin wraps automatically
    return theta, p


def _draw_on_click(event):
    """Click event handler.  Draws an orbit with the initial
    parameters from the clicked point.

    :param MouseEvent event: the click event
    """

    # only accept "normal", lef-clicks
    mode = event.canvas.toolbar.mode
    if not (event.button == 1 and event.inaxes == ax and mode == ""):
        return

    draw(get_standard_map(event.xdata, event.ydata, _K, _N))


def draw(points):
    """Draws points and refreshes the canvas.

    :param points: 2D Numpy array of x and y coordinates (2, N)
    """

    ax.plot(*points, linestyle="None", marker="o", markersize=_POINT_SIZE)
    fig.canvas.draw()
    fig.canvas.flush_events()  # don't draw twice


###########################################################################
#                                 Helpers                                 #
###########################################################################


def get_normalizer(interval, offset):
    """Gives a function that wraps a value in a given interval.

    :param Number interval: the interval length
    :param Number offset: the start point of the interval

    :returns: a normalizer function
    :rtype: function
    """

    def normalize(value):
        return ((value - offset) % interval) + offset

    return normalize


if __name__ == "__main__":
    # register events
    fig.canvas.mpl_connect("button_press_event", _draw_on_click)
    plt.show()  # block until figure closed


###########################################################################
#                                Diskussion                               #
###########################################################################

"""
Koordinaten: (theta, p)

Wir beginnen mit K=2.4.  Es zeichnen sich deutlich fuenf Hauptzentren
periodischer Bewegungen (Orbits) ab.  Die groesste Region befindet
sich um (pi, 0) wobei die Orbits zum Zentrum hin immer mehr einer
ellipse aehneln.  Es scheint sich im Zentrum ein Fixpunkt zu befinden.
Die Zentrale zone wird von einer zone der chaotischen (nicht
periodischen) Bewegung umgeben (Chaotisches Meer, es existieren aber
auch weiter aussen noch kleine Inseln ^^).  Die in dieser Zone
beginnenden Bahnen scheinen eine gewisse Zeit in der Naehe der fuenf
regulaeren 'Inseln' zu verweilen, um dann jedoch (teilweise!) in die
aeusseren gebiete abzudriften.

Die Orbits in den vier, die Zentrale Region umgebenden, Zentren der
regulaeren Bewegung springen zwischen eben diesen hin und her und
bilden dort Kreisaehnliche Formen.  Die Orbits in der Naehe der Mitte
dieser Zentren werden zunehmend Kreisfoermiger.  Die vier Nebenzentren
teilen sich einen Fixpunkt (der demnach natuerlich kein echter
Fixpunkt ist :P).

Ausserhalb dieser Zentren liegt eine Zone der nicht periodischen
(chaotischen) Bewegung, die mit zunehmenden K groesser wird.  Bei
K~=7.7 sind nach mehreren Wanderungen keine Zonen periodischer
Bewegung zu erkennen.

Fuer kleine K gibt es meist eine Zentrale regulaere Zone mit
wechselnden 'Satelitenzonen', die teilweise erstaunlich weit vom
Zentrum entfernt sind (Siehe K=1.55).  Mit steigendem K wird die
Zentrale Zone laenglicher.  Bis sie sich bei K~=4.55 in zwei 'Topfen'
aufspaltet.

Fuer K=0 ergeben sich die horizontalen Linien der gleichfoermigen
Bewegung.

Die Orbits in den ZP (Zohnen periodischer Bewegung) zweigen jeweils
individuelle Strukturen (Punktabstand, nicht exaktes
uebereinanderfallen) und besonders (aber nicht ausschlieslich) an den
Randzonen zum Chaotischen Meer scheinen in sich geschlossene Gebilde
fraktaler Natur (aehnlichkeit zu den groesseren ZP inkl.
Pseudofixpunkte) zu erscheinen.  Aufgrund der begraenzten
Rechenpraezision ist ein Detaillstudium dieser Zonen mit diesem
Programm nur begrenzt moeglich.  Da die Orbits zwischen einer Vielzahl
dieser kleinen ZP springen ist auch eine groessere Iterationsanzahl
noetig um deren Struktur aufzuzeigen.
"""
