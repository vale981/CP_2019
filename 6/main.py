#!/usr/bin/env python3
"""
This programm solves the time independent schroedinger equation
(tiseq) by discretisation of space and plots an animation of the time
evolution of a gaussian wavelet (see implementation for details).

Here a scewed two-well potential: x**4 - x**2 - A*x is considered for
A=0.06 and h_eff=0.07.  The parameters for the gaussian wavelet are:
width (sigma) = 0.1, p0 (impulse) = 0.

Upon a click on the canvas, a gaussian wavelet with its center at the
x coordinate of the click will be createt and projected onto the
computed eigenfunctions to be plottet as an animation at the height of
its energy (dashed line).  A second click will freeze the animation
and a third one will start a new one.
"""

import time
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import quantenmechanik as qm


def gauss_wavelet(sigma, p0, h_eff, x0, x):
    """A gaussian wavelet.

    :param sigma: standart deviation
    :param p0: central impulse
    :param h_eff: effective plank constant
    :param x0: central value
    :param x: evaluation point / array
    """

    return (
        1
        / (2 * np.pi * sigma ** 2) ** (1 / 4)
        * np.exp(-((x - x0) ** 2) / (4 * sigma ** 2))
        * np.exp(1j / h_eff * p0 * x)
    )


def two_well_potential(x, A):
    """A scewed two-well potential.

    :param x: the spatial coordinate
    :returns: the potential at point x, i.e. V(x)
    """

    return x ** 4 - x ** 2 - A * x


def projection_coeff(base, initial, points):
    """Calculate the projection coefficients for a wave function at
    t=0.

    :param base: the base to project on
    :param initial: wave function at t=0
    :param points: the x coordinates of the discretization
    :returns: the projection coefficients
    """

    dx = np.abs(points[1] - points[0])
    return dx * (base.conjugate().T @ initial(points))


def initial_error(base, coeff, initial, points):
    """Calculate the deviation of the reconstructed wave function from
    its original at t=0.

    :param base: the projection base
    :param initial: wave function at t=0
    :param coeff: the projection coefficients of the wave function
    :param points: the x coordinates of the discretization

    :returns: the scalar-product norm of the deviation `reconstructed - oriinial`
    """

    diffs = base @ coeff - initial(points)  # discetized difference function
    return np.sqrt((points[1] - points[0]) * np.sum(np.abs(diffs) ** 2))  # integral


def expected_energy_value(energies, coeff):
    """
    Calculate the expected energy value.
    :param energies: the eigenenergies of the problem
    :param coeff: the projection coefficients of the wave function
    :returns: the expected energy value
    """
    return energies @ np.abs(coeff) ** 2


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
        return base @ (np.exp(-1j * energies / h_eff * t) * coeff)

    return wave_function


def animate_time_evolution(
    base, energies, points, h_eff, wavelet_params, scaling, fig, ax
):
    """Plot an animation of the evolution of a wavelet.

    :param base: the projection base
    :param energies: the eigenenergies of the problem
    :param points: the x coordinates of the discretization
    :param h_eff: effective plank constant
    :param wavelet_params: a tuple of (sigma, p0)
    :param scaling: the scaling factor for the plot
    :param fig: the figure to draw on
    :param ax: the axis to draw on
    """

    # escape parameters for the infinite loop
    exit_flag = False
    running = False

    # save the number of lines present to clear the plot
    line_num = len(ax.lines)

    # to display the wave energy and quality
    status_text = fig.text(0, 0, "")

    # because pyplot does not support threading, yet somehow the click
    # events come from another thread (gui toolkit), we can allow
    # inifinite animations and arbitrary restarts of the animation if
    # we take some care

    # the first click starts the animation, the second stops it and
    # the third on starts a new one (and so on)
    def on_click(event):
        nonlocal exit_flag  # we need to modify the outer scope

        # check for a valid click
        mode = event.canvas.toolbar.mode
        if not (event.button == 1 and mode == "" and (event.inaxes == ax)):
            return

        # the loop has not been stopped yet, so do nothing
        if exit_flag and running:
            return

        # we want to stop
        if not exit_flag and running:
            exit_flag = True
            return

        # clear the previous line
        ax.lines = ax.lines[0:line_num]

        # start a new animation
        exit_flag = False
        if event is not None:
            run(event)

    # the worker function that does the actual animation
    def run(event):
        nonlocal exit_flag, running  # we need to modify the outer scope

        # create the initial wavelet
        wavelet = partial(gauss_wavelet, *wavelet_params, h_eff, event.xdata)

        # calculate the projection and energy
        coeff = projection_coeff(base, wavelet, points)
        energy = expected_energy_value(energies, coeff)
        quality = initial_error(base, coeff, wavelet, points)
        print(f"Projection Error: {quality}")

        # get the time evolution
        time_dep_wavelet = time_evolution(base, energies, coeff, h_eff)

        # plot the energy and an initial line
        ax.axhline(energy, color="gray", linestyle="--")
        line = ax.plot(points, wavelet(points), color="green")

        # set the status text
        delta = np.round(points[1] - points[0], 6)
        status_text.set_text(
            fr"$N={len(points)}$, $\delta = {delta}$ "
            + fr"Error=${quality}$,  Energy=${energy}$"
        )

        running = True  # signal running state
        start = time.time()  # set the initial time

        # animate unit the exit flag is signaled
        while not exit_flag:
            # calculate the time difference to get a realtime plot
            dt = time.time() - start
            wave = np.abs(time_dep_wavelet(dt)) ** 2
            wave = scaling * wave

            # plot the wave at its energy
            line[0].set_ydata(energy + wave)

            # trigger a redraw
            event.canvas.flush_events()
            event.canvas.draw()

        # signal the exit
        running = False

    # signal the exit flag upon closing the plot window
    def set_exit_flag(_):
        nonlocal exit_flag
        exit_flag = True

    fig.canvas.mpl_connect("close_event", set_exit_flag)

    # listen to clicks on the canvas
    fig.canvas.mpl_connect("button_press_event", on_click)


def main():
    """Dispatch the main logic. Used as convenience.
    """

    # Potential arameters
    A = 0.06  # skewnes

    # numeric parameters
    h_eff = 0.01  # effective reduced plank constant
    interval = (-2, 2)  # the discretization intervalc (for V < oo)
    N = 500  # discretization point-count

    # wavelet parameters
    sigma = 0.1  # width
    p0 = 0  # impulse
    wavelet_params = (sigma, p0)  # combine them for easier handling
    scaling = 0.01  # plot scaling

    print(__doc__)

    # apply the parameters to the potential
    potential = partial(two_well_potential, A=A)

    # solve the schroedinger equation
    points = qm.diskretisierung(*interval, N)
    e, phi = qm.diagonalisierung(h_eff, points, potential)

    # st up the plot and draw the potenial
    fig, ax = plt.subplots(1, 1)
    qm.plot_eigenfunktionen(
        ax, e, phi, points, potential, betragsquadrat=True, fak=scaling
    )
    ax.set_title("Asym. Two-Well Potential, Time evolution of a Wavelet")
    ax.set_ylabel("Potential, Eigenfunctions and Wavelet (norm squared)")

    # start the animation listener
    animate_time_evolution(phi, e, points, h_eff, wavelet_params, scaling, fig, ax)
    plt.show()


if __name__ == "__main__":
    main()

###############################################################################
#                                  Diskussion                                 #
###############################################################################

"""
a) Minimum
==========

Im rechten Maximum liegt die Energie des Gauss packetes nur knapp
ueber der Energie der Fundamentalloesung (unterhalb des linken pot.
minimums) und osziliert ohne sonderlich in den verbotenen Bereich
einzutreten.  Dabei behaelt das Packet weitestgehend seine Form
(aehnlich wie beim HO).  Im linken Minimum wiederum liegt die Energie
abermals leicht ueber der des 1.  Zustandes (beginne Zaehlen bei 0)
zeigt jedoch Ausschlaege in die Rechte Mulde.  Dies ist auf das
Tunneln der nun relativ zur Teilchenenergie kleineren
Potentialbariere.  Auch hier mutet die Bewegung periodisch an,
wenngleich dies vielleicht nur eine Taeuschung an.  Zumindest leuft
das Packet nicht breit.

Maximum
=======

Das Packet laeuft zuerst breit und zerfaellt dann in verschiendene
teile, die nur wenig symetrie aufweisen.  Es scheint als wuerde dei
Welle zwischen zustaenden starker Fragementierung (breite) und
konzentration in der Mitter schwanken.  Zeitweise ist die Ausgangsform
des Wellenpacketes wieder zu erahnen.

Allgmeienes:

    - fuer kleinerer sigma nehmen die Energien der Wellenpackete zu.
      (orts impuls unschaerfe)

    - mehr p => mehr energie

    - der fehler in der Entwicklung in eigenfunktionen ist sehr klein
      (mehr als akzeptabel) in der GO von 10^-30

    - mit groesseren energien wird das wellenpacket unzuverlaessiger,
      da nun auch terme hoeherer Energie mit groesserem Fehler
      einfluss nehmen (zerlaufen), bei packeten in den Minima ist
      dieses Problem jedoch gering

    - packete die am rand (mit grosser energie) gestartet werden sind
      verfaelscht, da dass gauss packet nicht genug raum zum abfallen
      hat und somit die Normierung nicht mehr stimmt

b) Allgemein: das Packet breitet sich zuerst nach rechts aus und
'prallt' dann an der Potentialwand ab.

Minimum
=======

Die energie des Pack.  im rechten Minimum liegt nun ueber der des
linken pot.  Minimums.  Die oszilationen des Packetes sind kraeftiger
und sogar leichtes tunneln ist zu erkennen.  Das maximum bricht immer
wieder auf und 'springt'.

Startet man das Packet im rechten Maximum, so beaobachtet man gleich
zu Anfang ein staerkeres tunneln, da nun der Nafangs impuls die
energie hebt und die Welle gegen die Bariere wirft.  (Energie nun
groesser als die dritte Eigeneenergie.)

Maximum
=======

Startet man in der Mitte, so liegt die energie nun ueber der sechten
eigenenergie.  Es zeigen sich nun im allgemeinen mehr maxima in der
Welle die sich nach einiger zeit gut verteilen, wobei aber immer noch
eine oszilation der hoechsten warscheinlichkeitsdichte zwischen den
umkehrpunkten zu erkennen ist (teilweise Sprunghaft).

c) Zeitraffer: Das wellenpacket weist zuerst eine lokalisierung in
einem der Minima auf.  Fuer grosse Zweiten symetrisiert sich die
Wellenfunktion zuerst im anderen Minnimum, bis sich der Zuecklus in
umgekehrter Richtung wiederholt (usw.).  Man erkennt also ein Tunneln
des 'ganzen' Wellenpacketes zwischen den beiden Minima.

Zusatz
======

Ungewichtete Superposition der Zeitentwicklung der ersten beiden
eigenfunktionen gibt:

|phi_1|^2 + 2*Re(phi_1 * phi_2 * exp(i/h*(E1-E0)*t)) (==> cosinus) +
|phi_2|^2

Fuer t=0 ergibt sich der ausgangszustand.  Bei (E1-E0)/h * t = 3/2*pi,
also t= 3/2*pi*h/(E1-E0) hier ca.  4255 wird der cosinus -1 und es
ergibt sich das andere extrem des zustands, also in falle des
Doppelmuldenpotentials eine rel.  symetrische wellenfunktion in
gleichen Teilen in beiden minima (gegenueber konzetration in einem
Minimum).  Tatsaechlich ist dieser effekt zu beobachten da die ersten
beiden coeffizienten der Entwicklung gegenueber den restlichen
domieren.  Wobei hoehere energien fuer die Unmittelbar sichtbare
dynamik sortgen waehrend die ersten loesungen eine sehr langsame
verschiebung hervorrufen.  Werden in dieser Simulation die beiden
ersten koeffizienten auf eins und alle anderen auf Null gesetzt so
laesst sich dies gut erkennen.
"""
