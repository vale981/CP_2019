#!/usr/bin/env python3
"""
Compare numeric integration methods.

Plots the relative error of three numerical integration methods and
their expected dependence on the sub-interval length parameter `h` in respect to
that paramater.
"""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
#                                 Plot Set-Up                                 #
###############################################################################

def set_up_plot():
    """Sets up the global plot parameters."""
    plt.title("Numerical Integration Errors")
    plt.xlabel(r'$h$')
    plt.ylabel(r"$\Delta I(f)$")

    # logarithmic plot
    plt.yscale('log')
    plt.xscale('log')


###############################################################################
#                                    Logic                                    #
###############################################################################

# If called with compatible types, the following functions support
# vectored input (`f` must be vecotrized, `h` an numpy array).  In
# this case, a numpy array will be returned.

def trapez_int(func, interval, steps):
    """Calculate the integral of a function over an interval.

    :param func: the function to integrate
    :param tuple interval: the interval over which to integrate
    :param steps: the number of integration steps
    :returns: the integral
    :rtype: float

    """

    points, h = np.linspace(*interval, steps, retstep=True)

    partial_sums = func(points)
    partial_sums[0] /= 2
    partial_sums[-1] /= 2
    return h * partial_sums.sum()


def middlepoint_int(func, interval, steps):
    """Calculate the integral of a function over an interval.

    :param func: the function to integrate
    :param tuple interval: the interval over which to integrate
    :param steps: the number of integration steps
    :returns: the integral
    :rtype: float

    """

    points, h = np.linspace(*interval, steps, retstep=True)
    partial_sums = func(points[1:] - h/2)

    return h * partial_sums.sum()


def simpson_int(func, interval, steps):
    """Calculate the integral of a function over an interval.

    :param func: the function to integrate
    :param tuple interval: the interval over which to integrate
    :param steps: the number of integration steps
    :returns: the integral
    :rtype: float

    """

    # we need an even number of intervals
    steps = steps + 1 if steps % 2 == 0 else steps
    points, h = np.linspace(*interval, steps, retstep=True)

    partial_sums = func(points)
    partial_sums[1:-1:2] *= 4
    partial_sums[2:-2:2] *= 2

    return h/3 * partial_sums.sum()


def __main__():
    """Dispatch the main logic. Used as convenience.
    """

    # print the doc
    print(__doc__)

    set_up_plot()

    # integration interval
    interval = (-np.pi/2, np.pi/4)

    # initialize some parameters
    points = 1000
    hs = np.logspace(-4, 0, points)

    # we calculate the integer valued step counts
    N = np.int32(np.abs((interval[1] - interval[0]) / hs))

    # ... and update our interval lengths
    hs = (interval[1] - interval[0]) / N

    functions = [(lambda x: np.sinh(2*x), "sinh(2x)",
                  -4.541387398431731922871),
                 (lambda x: np.exp(-100*x**2), "exp(-100*x^2)",
                  0.17724538509055160272982),
                 (lambda x: 0.5 * (1.0+np.sign(x)), r"$\Theta(x)$",
                  interval[1]),
                 (lambda x: np.ones_like(x), r"$1$",
                  interval[1] - interval[0])]

    # choose the function to analyze
    function, name, true_int = functions[0]

    # test the integration methods
    for method, color, method_name, error_fun in [
            (trapez_int, "green", "trapez", lambda x: x**2),
            (middlepoint_int, "red", "middlepoint", lambda x: x**2),
            (simpson_int, "blue", "simpson", lambda x: x**4)]:

        # vectorize the step count argument for convenience
        method = np.vectorize(method, excluded=[0, 1])

        # get the numerical integrals
        integrals = method(function, interval, N)

        # calculate the relative errors
        errors = np.abs((integrals - true_int) / true_int)

        # calculate the expected h^alpha error curve
        expected_errors = np.abs(error_fun(hs)/error_fun(hs[0])
                                 * errors[np.nonzero(errors)[0][0]])

        # plot both into the diagram
        plt.plot(hs, expected_errors, color=color, linestyle='-',
                 label=f"{name}, {method_name}", alpha=0.5)
        plt.plot(hs, errors, color=color, linestyle="None", marker='o',
                 markersize=0.4, alpha=1,
                 label=f"Exp. Error ({name}, {method_name})")

    # show legend and plot
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    __main__()

###############################################################################
#                                  Diskussion                                 #
###############################################################################

r"""
a) sinh(2*x)
Analytisch: -4.541387398431731922871
Antisymetrisch mit geringem Anstieg.
Teilweise ausloeschung, aber gut zu approximieren.

Trapez:
 - groesster fehler, scaliert hier wie erwartet

Mittelpunkt:
 - um einen faktor geringerer fehler als bei tr.
 - skaliert ebenfalls wie erwartet.

Beise Methoden zeigen keine Fluktuationen im betrachteten Interval.

Simpson Regel:
 - skaliert wie erwartet
 - fehler zeigt Plateaubildung fuer kleine h aufgrund von diskretisierungsfehler
   (Aufwendigere Berechnung, h/3 multiplikator)

b) exp(-100*x^2)
Analytisch: 0.17724538509055160272982
In grossen Teilen des int. Bereiches ~= 0  [-pi/2, pi/4] \ (-0.2, 0.2)
Symetrisch

Aehnliches Bild fuer alle drei Methoden.
Erwartetes skallierungsverhalten wird nicht eingehalten.  Fehler bis h ~=
0.02 sehr gering auf einem plateau (~1e-17) und dann in parabelartigen
Bogen ansteigend. (maximum bei h~=0.2 bis 0.3, danach ~ plateau)
gerade da wo die schrittweite anfaengt die region mit f>>0 zu
ueberschreiten.

Trapez- und Mittelpunktregel sind annaehernd identisch.  Die Simpson
Regel liefert einen um einen faktor geringeren Fehler.

In der Plateauregion (h<0.02) ist ein leicht negativer Anstieg zu
erkennen, der sich auf Diskretisierungsfehler zurueckfuehren laesst,
wobei somit ein gewisses Minimum hier nicht unterschritten werden kann.
(~2e-16 mit simpson)

Es empfielt sich den Integrationsbereich einzuschraenken (effizienz).

c) Theta Funktion
Analytisch: pi/4
Unstetig bei x=0

Allgemeines Bild:
Fehler skalliert mit h^1. Zeigt bei h~<0.05 oszilationen.

Jeh nachdem ob die aufteilung der Intervalle x=0 erwischt oder nicht ergiebt
sich der Fehler aus dem Ueberhang, der mit kleineren Intervallen ~linear kleiner
wird.

Jeh nachdem, ob die Kante `erwischt` wird kann der Fehler auf das Niveau des
Rundungsfehlers hinabsinken. (Dies ware fuer die Trapezmethode bei symetrischen
Integrationsgrenzen oft der Fall.)

Tendentiell sind hier Mittelpunkt- und Simpsonregel aequivalent.

Die Trapezregel giebt einen um einen Faktor geringeren Fehler, osziliert aber im
Gegensatz zu den anderen Methoden nur geringfuegig nach oben.

Bei leichter variation des Integrationsintervalls laesst sich aehnliches
Verhalten beobachten.
"""
