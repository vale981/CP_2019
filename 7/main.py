#!/usr/bin/env python3
"""
This programm plots some classical phase space trajecotries (contours)
and the husimi phase space representation for a wavelet over timer.

Here a scewed two-well potential: x**4 - x**2 - A*x is considered for
A=0.06 and h_eff=0.07.  The coherent state used has a symtric width of
sqrt(h_eff/2) in x and p.

Upon a mouse click at (x0, p0), a wavelet of the form of a coherent
state will be created and animated through time.
"""

import time
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import quantenmechanik as qm
import pdb

def coherent_state(h_eff, points, x0, p0):
    """A gaussian wavelet as the base of the coherent state.

    :param h_eff: effective plank constant
    :param points: the x coordinates of the discretisation
    :param x0: central value
    :param p0: central impulse

    :returns: The discretized coherent state at (x0, p0).

    """

    sigma = np.sqrt(h_eff/2)
    norm = 1/(2*np.pi*sigma**2)**(1/4)

    return norm*np.exp(-(points-x0)**2/(4*sigma**2)) \
        *np.exp(1j/h_eff*p0*points)

def two_well_potential(x, A):
    """A scewed two-well potential.

    :param x: the spatial coordinate
    :returns: the potential at point x, i.e. V(x)
    """

    return x**4 - x**2 - A*x

def projection_coeff(base, initial, points):
    """Calculate the projection coefficients for a wave function at
    t=0.

    :param base: the base to project on
    :param initial: wave function at t=0
    :param points: the x coordinates of the discretization
    :returns: the projection coefficients
    """

    dx = np.abs(points[1] - points[0])
    return dx * (base.conjugate().T @ initial)

def time_evolution(base, energies, coeff, h_eff):
    """Get the time evolution of a given wave function.

    :param base: the projection base
    :param energies: the eigenenergies of the problem
    :param coeff: the projection coefficients of the wave function
    :param h_eff: effective plank constant

    :returns: a function of one parameter (time) returning the
              discritsized wave function at arbitrary times

    :rtype: function
    """

    # close over the function parameters
    def wave_function(t):
        return base @ (np.exp(-1j*energies/h_eff * t) * coeff)

    return wave_function

def husimi_transform(points, h_eff, husimi_res, interval):
    """Calculate a discreized husimi field (2D/3D matrix).

    :param points: the x coordinates of the discretization
    :param h_eff: effective plank constant
    :param husimi_res: the (x, p) resolution of the husimi matrix
    :param interval: the (x, p) boundaries

    :returns: a function that calulates a 2D husimi matrix from a
              wavelet

    :rtype: function
    """

    # calculate the grid points
    dx = points[1] - points[0]
    loci = np.linspace(*interval[0], husimi_res[0] + 2)[1:-1]
    impulses = np.linspace(*interval[1], husimi_res[1] + 2)[1:-1]

    # generate a 3D matrix with a coherent state for each point as
    # discrete vector
    husimi_moll = coherent_state(h_eff, points[None, None, :],
                                 loci[:, None, None], impulses[None, :, None])
    husimi_moll = dx * husimi_moll.conjugate()

    # bind the calculated matrix to a function for efficient reuse, as
    # this is the only recurring calculation
    def fold(wavelet):
        return 1/h_eff \
            * np.abs(husimi_moll.dot(wavelet)).T**2

    return fold

def set_up_plot(interval):
    """Sets up the plots and their parameters.

    :returns: fig, ax
    """

    fig, ax = plt.subplots(1, 1)
    ax.set_title('Husimi Phase Space Representation')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$p$')

    # set fixed aspect ratio
    ax.set_aspect('equal')

    # fix the limits
    ax.set_xlim(*interval[0])
    ax.set_ylim(*interval[1])

    return fig, ax

def handle_mouse_click(fig, ax, base, energies, points, interval,
                       husimi_res, h_eff, tmax, t_stretch, event):
    """Starts a new dynamic plot of the time evolution of the husimi
    phase space.

    :param ax: a plot axis
    :param base: the projection base
    :param energies: the eigenenergies of the problem
    :param points: the x coordinates of the discretization
    :param interval: the (x, p) boundaries
    :param husimi_res: the (x, p) resolution of the husimi matrix
    :param h_eff: effective plank constant
    :param tmax: the maximum evolution time
    :param t_stretch: speed factor (time stretch)
    :param event: the click event
    """

    # only react if not zooming
    mode = event.canvas.toolbar.mode
    if event.button != 1 or mode != '' or event.inaxes is not ax:
        return

    # create the initial wavelet and project it to the eigenbase
    initial_wavelet = coherent_state(h_eff, points, event.xdata, event.ydata)
    coeff = projection_coeff(base, initial_wavelet, points)

    # generate the time evolution function
    wavelet = time_evolution(base, energies, coeff, h_eff)

    # calulate the husimi mollifier and transform the initial state
    husini = husimi_transform(points, h_eff, husimi_res, interval)
    initial_transform = husini(initial_wavelet)

    # plot the husimi matrix
    hus_plot = ax.imshow(initial_transform, extent=(*interval[0],
                                                    *interval[1]),
                         origin='lower', cmap='binary')
    # bar = fig.colorbar(hus_plot, orientation='horizontal')
    event.canvas.draw()

    # set up timing
    start = time.time()
    dt = 0

    # animate unit the exit flag is signaled
    while dt < tmax:
        # calculate the time difference to get a realtime plot
        dt = time.time() - start
        wave = wavelet(dt*t_stretch)

        # plot the wave at its energy
        transform = husini(wave)
        hus_plot.set_data(transform)
         # trigger a redraw
        event.canvas.flush_events()
        event.canvas.draw()

    # clean up after ourselves
    # bar.remove()
    hus_plot.remove()

def draw_contour(fig, ax, potential, interval):
    """Plot contour lines for the potential.

    :param ax: a plot axis
    :param potential: the potential to plot
    :param interval: the (x, p) boundaries
    """

    # create a grid for the energy calculation
    coordinates = np.linspace(*interval[0], 1000), \
        np.linspace(*interval[1], 1000)
    p_x, p_p = np.meshgrid(*coordinates)

    # calculate the energies from the hamiltonian
    energies = p_p**2/2 + potential(p_x)
    levels = np.percentile(energies, np.linspace(0,100,17))
    norm = matplotlib.colors.BoundaryNorm(levels,256)

    # draw the contour + colorbar
    ct = ax.contour(p_x, p_p, energies, levels=levels, cmap='YlOrRd', norm=norm)
    fig.colorbar(ct, format= '%.3f')

def main():
    """Dispatch the main logic. Used as convenience.
    """

    # Potential arameters
    A = 0.06  # skewnes

    # numeric parameters
    h_eff = 0.01  # effective reduced plank constant
    interval = ((-2, 2), (-2, 2))  # the discretization interval (for
                                   # V < oo) and the impulse interval
    N = 500  # discretization point-count
    husimi_res = (100, 100)  # grid steps per dimension (xres, pres)

    t_max = 12
    t_stretch = 1  # real-time

    # wavelet parameters
    print(__doc__)

    # apply the parameters to the potential
    potential = partial(two_well_potential, A=A)

    # solve the schroedinger equation
    points = qm.diskretisierung(*interval[0], N)
    e, phi = qm.diagonalisierung(h_eff, points, potential)

    fig, ax = set_up_plot(interval)

    draw_contour(fig, ax, potential, interval)

    on_click = partial(handle_mouse_click, fig, ax, phi, e, points, interval,
                       husimi_res, h_eff, t_max, t_stretch)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()

if __name__ == '__main__':
    main()

###############################################################################
#                                  Diskussion                                 #
###############################################################################

"""
a) Startet man das Wellenpacket in den Potential-Minima, so bleiben
sie dort im allgemeinen konzentriert.  Die bewegung des punktes
groesster Dichte erfolg ungefaehr auf der klassischen bahn, wenn auch
die geschwindigkeit der Veraenderung der darstellung fuer grosse p
nicht unnbedingt die schnellste ist.  In x richtung ist das packet an
den umkehrpunten langsam und in p richtung am schnellsten, wie es
klassisch zu erwarten ist.  Dieses verhalten zeigt sich bei den
meisten Energien.  Allerdings ueberdeckt das Packet viele verschieden
trajektorien.

Fuer groessere Energien unterhalb (-0.55, -0.47) der Separatrix ist
eine zunehmnde Auteilung auf bahnen um beide Minima zu beobachten.
Das packet bewegt sich immer im Uhrzeigersinn, wie es zu erwarten ist.
In der Naehe der Separatrix scheint sich das Packet wiederholt
aufzuteilen und zu rekombinieren.  Dies waere klassisch nicht
moeglich.

Auf Bahnen in der Naehe der Separatrix (-0.6, -0.7), die klassisch um
beide Minima fuehren, bewegt sich das Packet kurz klassisch, um sich
dann aufzuteilen, als wuerde es die Minima `umbranden`, was ebenfalls
klassich kein Analog findet.

Genau auf der Separatrix zerlaeuft das Packet gleichmaessig in beide
Richtungen.  (klassische wuerde es sich fuer eine entscheiden).  Bei
energien weiter oberhalb der Sep.  folgt das packet recht lang (ca.
eine periode) der klassischen Trajektorie in kompakter Form um sich
dann weiter auf die gesammte bahn aufzuteilen.  Dabei bleibt das
Packet auf einen Schlauch um die Klassische Bahn begrenzt.  Fuer
grosse zeiten konzentriert sich das Packet wieder kurz zu einer
Kompakten Form und der zueklus beginnt von neuem.  (=> Grosse Energie
=> klassik)

Bei laengerer Beobachtung faellt auf, dass sich `verbotene` bereiche
(an denne sich das teilchen nie befindet) relativ formfest bleiben
(gauss punkt) und auch teilweise den klassischen Bahnen folgen.

Auch hat die wahl des Anfangspunktes auf der einer Klassischen
Trajektorie einen effekt, der jedoch mit der Zeit verschwimmt (das
Packet vergisst...).

b) h_eff ist nun relativ klein, die Eigenenrgien sind dicht.  In den
minima bewegt sich das Packet auch bei kleinen Energien energien in
relativ kompakter Form und zerfasert/rekombiniert periodisch.
Quanteneffekte treten weiterhin in der Naehe der Separatrix auf.

Bei hoeheren energien bleibt das Packet nun sehr lange in fester form
und relativ konzentriert.  Es folgt den klassischen bahnen sehr genau.
Es findet also ein Uebergang zur klassischen Mechanik stat.

Bemerkung:
==========

Es ist darauf zu achten die Wellenpackete nicht mit zu grosser Energie
zu starten, da dann Numerische ungenauigkeiten gross werden koennen.
(Normierung, endliche eigenwerte...)
"""
