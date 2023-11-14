#!/usr/bin/env python3
"""
Compare numeric differentiation methods.

Plots the relative error of three numerical differentiation errors and
their expected dependence on the division parameter `h` in respect to
that paramater `h` == `dx`.
"""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
#                                 Plot Set-Up                                 #
###############################################################################


def set_up_plot():
    """Sets up the global plot parameters."""
    plt.title("Numerical differentiation Errors")
    plt.xlabel(r"$\log(h)$")
    plt.ylabel(r"$\log(\Delta f')$")

    # logarithmic plot
    plt.yscale("log")
    plt.xscale("log")


###############################################################################
#                                    Logic                                    #
###############################################################################

# If called with compatible types, the following functions support
# vectored input (`f` must be vecotrized, `h` an numpy array).  In
# this case, a numpy array will be returned.


def forward_diff(f, x, h):
    """Calculates the derivatife of `f` at the point `x` by the
    forward differentiation method.

    :param f: a function of one variable (should return a number)
    :param x: the point at which to differentiate the function
    :param h: the step size

    :returns: the derivative.

    :rtype: float
    """

    return (f(x + h) - f(x)) / h


def central_diff(f, x, h):
    """Calculates the derivatife of `f` at the point `x` by the central
    differentiation method.

    :param f: a function of one variable (should return a number)
    :param x: the point at which to differentiate the function
    :param h: the step size

    :returns: the derivative.
    :rtype: float
    """

    return (f(x + h / 2) - f(x - h / 2)) / h


def extapol_diff(f, x, h):
    """Calculates the derivatife of `f` at the point `x` by the central
    differentiation method.

    :param f: a function of one variable (should return a number)
    :param x: the point at which to differentiate the function
    :param h: the step size

    :returns: the derivative.
    :rtype: float
    """

    return (8 * (f(x + h / 4) - f(x - h / 4)) - f(x + h / 2) + f(x - h / 2)) / (3 * h)


def _f(x):
    """The function to differentiate.

    :param x: a number of or numpy array
    :returns: a number or numpy array
    """

    return np.arcsinh(-(x ** 2))


def __main__():
    """Dispatch the main logic. Used as convenience.
    """

    # print the bleeding doc
    print(__doc__)

    set_up_plot()

    # initialize some parameters
    hs = np.logspace(-10, 0, 1000)  # h values
    x = 1 / 5  # the evaluation point
    true_diff = -2 * x / np.sqrt(1 + x ** 4)  # the analytic value of the derivative

    # test the differentiation methods, it would be excessive to put
    # that code into a separate function!
    for method, color, name, error_fun in [
        (forward_diff, "green", "Forward Diff", lambda x: x),
        (central_diff, "red", "Central Diff", lambda x: x ** 2),
        (extapol_diff, "Blue", "Extrapolated Diff", lambda x: x ** 4),
    ]:

        # get the numerical derivatives
        differentials = method(_f, x, hs)

        # calculate the relative errors
        errors = np.abs((differentials - true_diff) / true_diff)

        # calculate the expected h^alpha error curve
        expected_errors = np.abs(error_fun(hs) * errors[-1])

        # plot both into the diagram
        plt.plot(hs, expected_errors, color=color, linestyle="-", label=name, alpha=0.5)
        plt.plot(
            hs,
            errors,
            color=color,
            linestyle="None",
            marker="o",
            markersize=0.4,
            alpha=1,
            label=f"Expected Error ({name})",
        )

    # show legend and plot
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    __main__()

###############################################################################
#                                  Diskussion                                 #
###############################################################################

"""
h^-1 fehler fuer forward diff
=============================
Alle formeln abschaetzungen mit eventuell weggelassenen
multiplikativen konnstanten.

Die analytische Fehlerabschaetzung lautet:
  f''(x) * h/2

Der numerische fehler bei der funktionsauswertung lautet ca.:
  epsilon * |f(x)|
bei kleinen werten, mit epsilon = 2−53 fuer float.

Der fehler von (f(x+h)-f(x))/h laesst sich durch:
  sqrt(2)/h * espilon * |f(x)|
abschaetzen.

In summe also:
  err(h) = f''(x) * h/2 + sqrt(2)/h * espilon * |f(x)| ~ 1/h fuer h klein

Somit erklaert sich die 1/h abhaengigkeit.

Ein optimum ergibt sich aus err'(h) == 0 zu:
  h ~= sqrt(epsilon * 2 * sqrt(2) * |f(x)| / |f''(x)|)
     = 3.56e-9 (fuer den speziellen fall)

Dieser Wert liegt nahe an dem empirisch gefunden (siehe unten), laesstsich aber
hoechtens als untere grenze ansehen.

Offenbar darf f'' nicht null in x sein. Die formel nimmt speziellen bezug auf f
und liefert somit ein spezielles optimum.

Fuer eine Abschaetzung koennte gelten:
  h ~ sqrt(espilon)


Optimale Werte:
===============

| Method      |       h | Rel.  Error |
|-------------+---------+-------------|
| forward     |  2.8e-8 |     4.49e-8 |
| central     | 4.15e-5 |    2.99e-11 |
| extrapolted |  4.3e-3 |    2.53e-12 |

Alle werte sind nicht die Minima der des relativen Fehlers, sonder
wurden an den jeweiligen Enden (grob!) der h^alpha abhaengigkeit
abgelesen, da dort der fehler stabil bleibt.  Somit lassen sich diese
Werte auch als grobe Richtwerte fuer den allgemeinen Fall verwenden.

Fuer polynomielle ausdruecke bis zum 4. Grad sollte die extrapolierte differenz
sogar exakt sein und man beobachtet nur noch den Rundungsfehler.
"""
