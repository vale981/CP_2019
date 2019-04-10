#!/usr/bin/env python3
"""Compare numeric differentiation methos."""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
#                                 Plot Set-Up                                 #
###############################################################################

def set_up_plot():
    plt.title("Numerical differentiation Errors")
    plt.xscale('log')
    plt.xlabel(r'$\log(h)$')
    plt.yscale('log')
    plt.ylabel(r"$\log(\Delta f')$")

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

    return (f(x+h) - f(x))/h


def central_diff(f, x, h):
    """Calculates the derivatife of `f` at the point `x` by the central
    differentiation method.

    :param f: a function of one variable (should return a number)
    :param x: the point at which to differentiate the function
    :param h: the step size

    :returns: the derivative.
    :rtype: float
    """

    return (f(x+h/2) - f(x-h/2))/h


def extapol_diff(f, x, h):
    """Calculates the derivatife of `f` at the point `x` by the central
    differentiation method.

    :param f: a function of one variable (should return a number)
    :param x: the point at which to differentiate the function
    :param h: the step size

    :returns: the derivative.
    :rtype: float
    """

    return (8*(f(x+h/4) - f(x-h/4)) - f(x+h/2) + f(x-h/2))/(3*h)


def _f(x):
    """The function to differentiate.

    :param x: a number of or numpy array
    :returns: a number or numpy array
    """

    return np.arcsinh(x**2)


def __main__():
    """Dispatch the main logic. Used as convenience.
    """

    set_up_plot()

    expect_style = '-'
    _hs = np.logspace(-20, 0, 1000)
    _x = 1/5
    true_diff = 2*_x/np.sqrt(1+_x**4)

    # test the differentiation methods
    for method, color, name, error_fun in [
            (forward_diff, "green", "Forward Diff", lambda x: x),
            (central_diff, "red", "Central Diff", lambda x: x**2),
            (extapol_diff, "Blue", "Extrapolated Diff", lambda x: x**4)]:

        m = method(_f, _x, _hs)
        errors = np.abs((m - true_diff)) / m
        expected_errors = np.abs(error_fun(_hs)) * errors[-1]
        plt.plot(_hs, expected_errors, color=color, linestyle=expect_style,
                 label=name, alpha=0.5)
        plt.plot(_hs, errors, color=color, linestyle="None", marker='o',
                 markersize=0.4, alpha=1, label=f"Exp. Error ({name})")

    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    __main__()
