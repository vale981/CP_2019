#!/usr/bin/env python3
"""
# TODO: Write
"""

import numpy as np
import matplotlib.pyplot as plt
import functools
from scipy.integrate import odeint

###############################################################################
#                                 Plot Set-Up                                 #
###############################################################################

def set_up_plots():
    """Sets up the plots and their parameters.

    :returns: fig, ax
    """

    # Create two subplots (trajectory, stroboscope) sharing the y axis.
    fig, (ax_tr, ax_st) = plt.subplots(1, 2, sharey=True)
    ax_tr.set_title('Phase Space Trajectory')
    ax_st.set_title('Stroboscopic Phase Space ')

    ax_tr.set_xlabel(r'$x$')
    ax_tr.set_ylabel(r"$t$")

    ax_st.set_xlabel(r'$x$')
    ax_st.set_ylabel(r"$t$")

    return fig, (ax_tr, ax_st)

###############################################################################
#                                    Logic                                    #
###############################################################################

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
    return np.array([p, -4*x**3 + 2*x - A - B*np.sin(omega*t)])

def add_trajectory(ax_tr, ax_st, periods,
                   steps_per_period, args, event):

    # only react if not zooming
    mode = event.canvas.toolbar.mode
    if not (event.button == 1 and event.inaxes == ax_tr or event.inaxes ==
            ax_st and mode == ''):
        return

    # solve the DE
    _, _, omega = args
    t_range = np.linspace(0, periods*2*np.pi/omega, steps_per_period*periods)
    trajectory = odeint(rhs, (event.xdata, event.ydata), t_range, args=args)

    # unpack the solution
    x, p = trajectory.T

    # plot the trajecotry
    ax_tr.plot(x, p, linestyle='none', marker='o', markersize=0.5)

    # and the stroboscopic view
    ax_st.plot(x[::steps_per_period], p[::steps_per_period], linestyle='none',
               marker='o', markersize=0.8)

    event.canvas.draw()

def main():
    """Dispatch the main logic. Used as convenience.
    """

    # print the doc
    print(__doc__)

    # System Constants
    A = 0 # 0.2
    B = 0 # 0.1
    omega = 1

    # Solver Parameters
    periods = 200  # how many drive periods to solve for
    steps_per_period = 20

    fig, (ax_tr, ax_st) = set_up_plots()
    on_click = functools.partial(add_trajectory, ax_tr, ax_st, periods,
                                 steps_per_period, (A, B, omega))

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

if __name__ == '__main__':
    main()

###############################################################################
#                                  Diskussion                                 #
###############################################################################

# TODO: write
