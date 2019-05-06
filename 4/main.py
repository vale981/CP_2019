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

    # Create two subplots (trajectory, stroboscope) sharing the x,y axis.
    fig, (ax_tr, ax_st) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax_tr.set_title('Phase Space Trajectory')
    ax_st.set_title('Stroboscopic Phase Space ')

    for ax in ax_tr, ax_st:
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$p$')

        # set fixed aspect ratio
        ax.set_aspect('equal')

        # fix the limits
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)

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

def add_trajectory(ax_tr, ax_st, periods, steps_per_period,
                   initial_conditions, args):
    # solve the DE
    _, _, omega = args
    t_range = np.arange(0, steps_per_period*periods)/steps_per_period \
        * 2 * np.pi / omega
    trajectory = odeint(rhs, initial_conditions, t_range,
                        args=args)

    # unpack the solution
    x, p = trajectory.T

    # plot the trajecotry
    ax_tr.plot(x, p, linestyle='none', marker='o', markersize=0.5)

    # and the stroboscopic view
    ax_st.plot(x[::steps_per_period], p[::steps_per_period], linestyle='none',
               marker='o', markersize=0.8)

def handle_mouse_click(ax_tr, ax_st, periods,
                   steps_per_period, args, event):

    # only react if not zooming
    mode = event.canvas.toolbar.mode
    if event.button != 1 or mode != '' or (event.inaxes != ax_tr and
                                       event.inaxes != ax_st):
        return


    add_trajectory(ax_tr, ax_st, periods,
                   steps_per_period, (event.xdata, event.ydata), args)
    event.canvas.draw()

def handle_key_press(ax_tr, ax_st, event):
    if event.key == 'c':
        ax_tr.lines = []
        ax_st.lines = []

    event.canvas.draw()

def main():
    """Dispatch the main logic. Used as convenience.
    """

    # print the doc
    print(__doc__)

    # System Constants
    A = 0.2
    B = 0.1
    omega = 1

    # Solver Parameters
    periods = 2000  # how many drive periods to solve for
    steps_per_period = 1

    # set up figures and listeners
    fig, (ax_tr, ax_st) = set_up_plots()
    on_click = functools.partial(handle_mouse_click, ax_tr, ax_st, periods,
                                 steps_per_period, (A, B, omega))

    fig.canvas.mpl_connect('button_press_event', on_click)

    # we do it in here, because it is minor
    on_key_press = functools.partial(handle_key_press, ax_tr, ax_st)
    fig.canvas.mpl_connect('key_press_event',
                           on_key_press)

    add_trajectory(ax_tr, ax_st, periods, steps_per_period, (0.231704,
                                                             -0.349683), (A, B, omega))
    plt.show()

if __name__ == '__main__':
    main()

###############################################################################
#                                  Diskussion                                 #
###############################################################################

# TODO: write
