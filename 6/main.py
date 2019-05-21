#!/usr/bin/env python3
"""
TODO
"""

import numpy as np
import quantenmechanik as qm
from functools import partial
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import time
from threading import Thread
from matplotlib.animation import FuncAnimation
import gc

def gauss_wavelet(sigma, p0, h_eff, x0, x):
    return 1/(2*np.pi*sigma**2)**1/4* \
        np.exp(-(x-x0)**2/(4*(sigma)**2))*np.exp(1j/h_eff*p0*x)


def two_well_potential(x, A):
    """A scewed two-well potential.

    :param x: the spatial coordinate
    :returns: the potential at point x, i.e. V(x)
    :rtype: number
    """

    return x**4 - x**2 - A*x

def projection_coeff(base, initial, points):
    dx = np.abs(points[1] - points[0])
    return dx * base.conjugate().T @ initial(points)

def initial_error(base, coeff, initial, points):
    diffs = base @ coeff - initial(points)
    return np.linalg.norm(diffs)

def expected_energy_value(energies, coeff):
    return energies @ np.abs(coeff)

def time_evolution(base, energies, coeff, h_eff):
    def wave_function(t):
        return base @ (np.exp(-1j*energies/h_eff * t) * coeff)

    return wave_function

def set_up_plot(interval):
    """Sets up the plots and their parameters.
    """

    fig, ax = plt.subplots(1, 1)
    ax.set_title('Time evolution of a Wavelet')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'Propability Density')
    ax.set_xlim(*interval)

    return fig, ax

def animate_time_evolution(base, energies, points, h_eff,
                           wavelet_params, fig, ax, event):

    mode = event.canvas.toolbar.mode
    if event.button != 1 or mode != '' or (event.inaxes != ax):
        return

    wavelet = partial(gauss_wavelet, *wavelet_params, h_eff, event.xdata)
    coeff = projection_coeff(base, wavelet, points)
    energy = expected_energy_value(energies, coeff)
    time_dep_wavelet = time_evolution(base, energies, coeff, h_eff)

    ax.axhline(energy, color='gray')
    ax.lines = [ax.lines[0]]
    line = ax.plot(points, np.zeros_like(points))
    start = time.time()
    exit_flag = False

    def set_exit_flag(_):
        nonlocal exit_flag
        exit_flag = True

    close_cid = fig.canvas.mpl_connect('close_event', set_exit_flag)
    click_cid = fig.canvas.mpl_connect("button_press_event", set_exit_flag)


    while 1:
        if exit_flag or not line:
            print("here")
            break
        dt = time.time() - start

        wave = np.abs(time_dep_wavelet(dt))**2

        line[0].set_ydata(energy + wave)
        event.canvas.flush_events()
        event.canvas.draw()

    fig.canvas.mpl_disconnect(close_cid)
    fig.canvas.mpl_disconnect(click_cid)

def main():
    """Dispatch the main logic. Used as convenience.
    """

    # Potential arameters
    A = 0.16

    # numeric parameters
    h_eff = 0.07
    interval = (-2, 2)
    N = 500  # discretization point-count
    max_t = 20

    # wavelet parameters
    sigma = 0.1
    wavelet_params = (sigma, 0.8)

    print(__doc__)

    # apply the parameters to the potential
    potential = partial(two_well_potential, A=A)

    points, dx = qm.diskretisierung(*interval, N, retstep=True)
    e, phi = qm.diagonalisierung(h_eff, points, potential)

    fig, ax = set_up_plot(interval)
    ax.plot(points, potential(points))

    on_click = partial(animate_time_evolution, phi, e, points, h_eff,
                       wavelet_params, fig, ax)
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

if __name__ == '__main__':
    main()

###############################################################################
#                                  Diskussion                                 #
###############################################################################

"""
TODO
"""
