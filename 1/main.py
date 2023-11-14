#!/usr/bin/env python3
"""Draw the standard map of the kicked rotor interactively."""

# NOTE: Barebones copied from ../0/main.py

# Meine Muttersprache ist Deutsch. Ich verwende in Kommentaren
# generell die Englische Sprache um verwirrendes 'Denglisch' zu
# vermeiden :)

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from functools import lru_cache
import matplotlib
from matplotlib.widgets import Slider

# use Cairo for performance
matplotlib.use("GTK3Agg", warn=False, force=True)

# I am leaving this function out of the below class, because it really
# has no reason being there.
@lru_cache(maxsize=None)
@jit(cache=True)
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
    theta = np.zeros(N + 1)
    p = np.zeros(N + 1)

    # set the initial parameters
    norm_p = get_normalizer(2 * np.pi, -np.pi)
    theta[0] = theta_0 % (2 * np.pi)
    p[0] = norm_p(p_0)

    # calculate the standard map
    for curr in range(1, N + 1):
        last = curr - 1  # for readability
        theta[curr] = (theta[last] + p[last]) % (2 * np.pi)
        p[curr] = norm_p(p[last] + K * np.sin(theta[curr]))

    return theta, p


# Just to avoid globals... again. I usually prefer to avoid too much
# useless Object orientation, but python kinda forces it on ya!

# This could be done nicer (remove hardcoded values)
class StandardMap:
    """
    Shows a plot window that interactively draws the standart map of
    the kicked rotor.
    """

    def __init__(self, iterations=1000, K=2.4):
        """Sets up the plot and its callbacks.

        :param Number iterations: the default number of iterations
        :param Number K: the default k parameter

        """

        # initialize state
        self.fig, self.ax = plt.subplots(1, 1, constrained_layout=True)
        self._orbits = []

        # initialize default parameters
        self._K = K
        self._N = iterations

        # initialize the axis
        self.clear()

        # set up sliders
        self._set_up_widgets()

        # register events
        self.button_cid = self.fig.canvas.mpl_connect(
            "button_press_event", self._draw_on_click
        )
        self.key_cid = self.fig.canvas.mpl_connect(
            "key_press_event", self._handle_key_press
        )

        # some explainatory test
        self.fig.text(
            0.01, 0.9, "press `c` to clear\nOrbits are redrawn automatically."
        )

        # show the figure
        self.fig.show()  # FIXME: that is bad style, put that into a method

    def _set_up_widgets(self):
        """Initializes the various widgets.
        """

        axcolor = "lightgoldenrodyellow"

        # K Slider
        self._ax_K = plt.axes([0.05, 0.1, 0.1, 0.03], facecolor=axcolor)
        self._slider_K = Slider(self._ax_K, "K", 0, 10.0, valinit=self._K, valstep=0.05)
        self._slider_K.on_changed(self._set_K)

        # N Slider
        self._ax_N = plt.axes([0.05, 0.15, 0.1, 0.03], facecolor=axcolor)
        self._slider_N = Slider(self._ax_N, "N", 10, 10000, valinit=self._N, valstep=1)

        self._slider_N.on_changed(self._set_N)

        self._status_txt = self.fig.text(0, 0, "")

    def _set_N(self, N):
        """Sets Iteration Count.
        Triggers Recalculation.

        :param Numer N: the new N
        """

        self._N = int(N)
        self.redraw_all_orbits()

    def _set_K(self, K):
        """Sets the K pramater.
        Triggers Recalculation.

        :param Numer K: the new K
        """

        self._K = K
        self.redraw_all_orbits()

    def _handle_key_press(self, event):
        """Handles key press events on the plot.

        :param KeyEvent event:
        """

        if event.key == "c":
            # delete all orbits
            self._orbits = []
            self.clear()
        elif event.key == "z":
            self._set_K(self._K + 0.1)
        elif event.key == "u":
            self._set_K(self._K - 0.1)

    def draw(self, points):
        """Draws points and refreshes the canvas.

        :param points: 2D Numpy array of x and y coordinates (2, N)
        """

        self.ax.plot(*points, linestyle="None", marker="o", markersize=0.5)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def clear(self):
        """Clears the axis and sets it up again.
        """

        # clear axis
        self.ax.clear()

        # set up the axis (again)
        self.ax.set_title("Standard Map")
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        # set up the axis limits and labels
        self.ax.set_xlim([0, 2 * np.pi])
        self.ax.set_xlabel(r"$\theta$")
        self.ax.set_ylim([-np.pi, np.pi])
        self.ax.set_ylabel(r"$p$")

        # redraw
        self.fig.canvas.draw()

    def redraw_all_orbits(self):
        """Recalculates and draws all saved orbits.
        """

        self.fig.canvas.draw_idle()

        orbits = [
            get_standard_map(theta_0, p_0, self._K, self._N)
            for theta_0, p_0 in self._orbits
        ]

        self.clear()

        for orbit in orbits:
            self.ax.plot(*orbit, linestyle="None", marker="o", markersize=0.5)

        self.fig.canvas.draw()

    def _draw_on_click(self, event):
        """Click event handler. Draws a new Spiral.

        :param MouseEvent event: the click event
        """

        mode = event.canvas.toolbar.mode
        if not (event.button == 1 and event.inaxes == self.ax and mode == ""):
            return

        p_0 = event.ydata
        theta_0 = event.xdata

        self._orbits.append((theta_0, p_0))
        self.draw(get_standard_map(theta_0, p_0, self._K, self._N))


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


###########################################################################
#                                Diskussion                               #
###########################################################################

DIS = """
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

###############################################################################
#                                     Main                                    #
###############################################################################

if __name__ == "__main__":
    print(DIS)

    plt.ion()
    _ = StandardMap()  # avoid garbage collection
    plt.show(block=True)  # block until figure closed
