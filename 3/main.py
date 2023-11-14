#!/usr/bin/env python3
"""
Compare numeric integration methods.

Plots the relative error of three numerical integration methods and
their expected dependence on the sub-interval length parameter `h` in
respect to that paramater.

Those integration methods are:

    - the trapezoidal rule -> I = h/2 * (f(x0) + 2*f(x+h) + 2*f(x+2*h)
      + ... + f(x1)

    - the midpoint method -> I = h * (f(x0+h/2) + f(x0+3/2*h) + ... +
      f(x0+h*(N-1/2))

    - the simpson rule -> I = h * (f(x0)/6 + f(x0+h/2)*4/6 + f(x0+h)/6
      + ... + f(x0+(N-1/2)*h) + f(x0+N*h=x1)

The three examined functions are:

    - sinh(2*x)

    - exp(-100*x^2)

    - the Theta Funktion

which are being integrated over the Interval (-pi/2, pi/2).
"""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
#                                 Plot Set-Up                                 #
###############################################################################


def set_up_plot():
    """Sets up the plot parameters.

    :returns: fig, ax
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, xscale="log", yscale="log")

    ax.set_title("Numerical Integration Errors")
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"Relative Error $I(f)$")

    return fig, ax


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

    :returns: the integral, stepsize
    :rtype: float, float
    """

    points, h = np.linspace(*interval, steps, retstep=True)

    partial_sums = func(points)
    partial_sums[0] /= 2
    partial_sums[-1] /= 2
    return h * partial_sums.sum(), h


def middlepoint_int(func, interval, steps):
    """Calculate the integral of a function over an interval.

    :param func: the function to integrate
    :param tuple interval: the interval over which to integrate
    :param steps: the number of integration steps

    :returns: the integral, stepsize
    :rtype: float, float
    """

    points, h = np.linspace(*interval, steps, retstep=True)

    # we begin counting at 1 (only midpoints)
    partial_sums = func(points[1:] - h / 2)

    return h * partial_sums.sum(), h


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

    return h / 3 * partial_sums.sum(), h


def main():
    """Dispatch the main logic. Used as convenience.
    """

    # print the doc
    print(__doc__)

    fig, ax = set_up_plot()

    # integration interval
    interval = (-np.pi / 2, np.pi / 4)

    # initialize some parameters
    max_steps = 1000
    hs = np.logspace(-4, 0, max_steps)

    # we calculate the integer valued step counts
    N = np.int32(np.abs((interval[1] - interval[0]) / hs))

    # the index where the error plot and the projected error plot touch
    touch_index = int(len(N) * 3 / 4)

    functions = [
        (lambda x: np.sinh(2 * x), "sinh(2x)", -4.541387398431731922871),
        (lambda x: np.exp(-100 * x ** 2), "exp(-100*x^2)", 0.17724538509055160272982),
        (lambda x: 0.5 * (1.0 + np.sign(x)), r"$\Theta(x)$", interval[1]),
        (np.ones_like, r"$1$", interval[1] - interval[0]),  # constant function
    ]

    # choose the function to analyze
    function, name, true_int = functions[2]

    # this represents the most concise and readable form of
    # implementing the evaluation and plotting of the integral errors
    # ~not~ involving a structured data type or a class!

    # test the integration methods int.
    # method function, plot color,
    # method name, exponent of the projected error
    for method, color, method_name, error_exp in [
        (trapez_int, "green", "trapez", 2),  # note, no lambda :)
        (middlepoint_int, "red", "middlepoint", 2),
        (simpson_int, "blue", "simpson", 4),
    ]:

        # vectorize the step count argument for convenience
        method = np.vectorize(method, excluded=[0, 1])

        # get the numerical integrals, unzip the list of tuples
        integrals, step_sizes = method(function, interval, N)

        # calculate the relative errors
        errors = np.abs((integrals - true_int) / true_int)

        # calculate the expected h^alpha error curve; we normalize the
        # error function so that it is close the the calculated values
        expected_errors = np.abs(
            (step_sizes) ** error_exp
            / (step_sizes[touch_index]) ** error_exp
            * errors[touch_index]
        )

        # plot both into the diagram
        ax.plot(
            step_sizes,
            expected_errors,
            color=color,
            linestyle="-",
            label=f"{name}, {method_name}",
            alpha=0.5,
        )
        ax.plot(
            step_sizes,
            errors,
            color=color,
            linestyle="None",
            marker="o",
            markersize=0.4,
            alpha=1,
            label=f"Exp. Error ({name}, {method_name})",
        )

    # show legend and plot
    ax.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()

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
