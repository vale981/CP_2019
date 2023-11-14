#!/usr/bin/env python3
"""
This programm solves the time independent schroedinger equation
(tiseq) by discretisation of space.

Here the tiseq scewed two-well potential: x**4 - x**2 - A*x

is considered for A=0.06 and h_eff=0.07.

The obtained solutions (eigenenergies, eigenfunctions) are being
renormalized and plotted in an energy-ladder diagram.  (The
eigenfunctions are plotted with a horizontal line at their
eigenenergies as their x-axis.)
"""

import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from scipy.linalg import eigh


def two_well_potential(x, A):
    """A scewed two-well potential.

    :param x: the spatial coordinate
    :returns: the potential at point x, i.e. V(x)
    :rtype: number
    """

    return x ** 4 - x ** 2 - A * x


def get_coeff_matrix(potential, h_eff, interval, N):
    """Get the coefficient matrix for the discreticized tiseq.

    :param potential: A function of one parameter acting on np.arrays.
    :param Tuple interval: the interval for which the potential is <
        infty
    :param N: matrix size, for how many points the tiseq may be solved
              (excluding the endpoints), this matches the nomenclature
              of `N` in the lecture.

    :returns: the coefficient matrix, points, step size

    :rtype: np.array, np.array, number
    """

    # evaluation points
    x, dx = np.linspace(*interval, N + 2, retstep=True)
    x = x[1:-1]  # discard endpoints, leaves `N=N` points

    # get the potential and z-parameter
    v = potential(x)
    z = h_eff ** 2 / (2 * dx ** 2)

    # create vectors for the secondary diagonals
    v = v + 2 * z
    z_vec = z * np.ones(N - 1)

    # build the matrix
    coeffs = np.diag(v) - np.diag(z_vec, k=-1) - np.diag(z_vec, k=1)

    return coeffs, x, dx


def calculate_eigen(potential, h_eff, interval, N, eigenrange):
    """Calculate the eigenenergies and discrete eigenfunctions of the
    tiseq.

    :param potential: A function of one parameter acting on np.arrays.
    :param Tuple interval: the interval for which the potential is <
        infty
    :param N: matrix size, for how many points the tiseq may be solved
              (excluding the endpoints), this matches the nomenclature
              of `N` in the lecture.
    :param Tuple eigenrange: for which eigenenergies E_n to solve

    :returns: eigenenergies, eigenfunctions (normalized), x
              coordinates, step size for as many points as `N`, but
              excluding the endpoints which are zero

    :rtype: np.array, np.array, np.array, number
    """

    # get the coefficients
    matrix, x, dx = get_coeff_matrix(potential, h_eff, interval, N)

    # solve the tiseq
    e, phi = eigh(matrix, eigvals=eigenrange)

    # normalize the eigenfunctions, riemann intgral ~= sum(phi_i)*delta
    phi = phi / np.sqrt(dx)

    return e, phi, x, dx


def plot_eigenfunctions(e, phi, interval, x, potential):
    """Sets up the plots and their parameters.

    :param e: Eigen Energies
    :param phi: Eigenvectors (Eigenfunctions)
    :param interval: The solution Interval
    :param x: the solution coordinates
    :param potential: A function of one parameter acting on np.arrays.

    :returns: fig, (trajecotry axis, stroboscope axis)
    """

    # initialize plot
    fig, ax = plt.subplots(1, 1)

    ax.set_title("Eigenfunctions")
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"Energy / Propability-Density")

    # set limits, ylim is a rough measure
    ax.set_xlim(*interval)

    # use the maximum energy difference as a measure for the overlap
    # because the eigenfunctions are renormalized into their bound
    max_energy_diff = np.max(e[1:] - e[:-1])
    ax.set_ylim((e[0] - max_energy_diff, e[-1] + max_energy_diff))

    # renormalize the eigenfunctions to the median of the energy level
    # differences, so that they dont vanish if energies are
    # pseudo-degenerate
    med_energy_diff = np.median(e[1:] - e[:-1])
    phi = phi / np.max(phi) * med_energy_diff * 0.8

    # plot the eigenenergies and eigenfunctions
    level = 0
    for eigenval, eigen_phi in zip(e, phi):
        ax.axhline(eigenval, color="gray")
        ax.plot(
            x,
            eigen_phi + eigenval,
            label=r"$\varphi$ for $E_{}={}$".format(level, np.round(eigenval, 4)),
        )
        level += 1

    # plot the potential as a reference
    ax.plot(x, potential(x), color="gray", label="Potential, Energy Levels")

    ax.legend(loc="upper right")

    return fig, ax


def main():
    """Dispatch the main logic. Used as convenience.
    """

    # Model Parameters
    h_eff = 0.07
    A = 0 * 0.06
    interval = (-2, 2)
    N = 500  # matrix size
    eigen_range = (0, 5)  # eigenvalue selection

    print(__doc__)

    # apply the parameters to the potential
    potential = partial(two_well_potential, A=A)

    # solve the tiseq
    e, phi, x, dx = calculate_eigen(potential, h_eff, interval, N, eigen_range)

    # plot the result
    fig, _ = plot_eigenfunctions(e, phi.T, interval, x, potential)

    # as this has nothing to do with the vizualization, i'll put it
    # here
    param_string = fr"$A={A}$ $N={N}$ $\delta={np.round(dx, 4)}$ $h_{{eff}}={h_eff}$"
    fig.text(0, 0, param_string)

    plt.show()


if __name__ == "__main__":
    main()

###############################################################################
#                                  Diskussion                                 #
###############################################################################

"""
a) Begruendung

    - Schritte, delta_x: N=500 ergibt Loesungen, die sich zu denen von
      N=2000 sehr wenig unterscheiden.  Unter ca.  200 Schritten
      treten aber zunehmend numerische Ungenauigkeiten zu tage.  Es
      muessen mindestens so viele Schritte gewaehlt werden, wie man
      Eigenwerte erhalten moechte.  Fuer N=500 ist die Rechenzeit
      annehmbar und reicht fuer diese einfache Visualisierung aus.
      Zwischen 500 und 2000 Schritten aendern sich die
      Energieeigenwerte nur noch in der 4.  Nachkommastelle und die
      Eigenfunktionen zeigen in dieser Darstellung keine Unterschiede.
      Kritisch allerdings sind die Nulldurchgaenge der phi_i, die bei
      kleinen N verschwinden und somit die Knotenregel verletzen.
      Werden allerdings genaue werte fuer die Energien benoetigt so
      muss N wesentlich groesser als 500 gewaehlt werden.

    - Grenzen: das interval -2, 2 wurde so gewahlt, dass die Loesung
      genuegen Raum zum Natuerlichen abfallen hat.  Eine variation
      ueber diese Grenzen hinaus zeigt wenig Unterschiede.

b) Die betrachteten ersten sieben Eigenfunktionen folgen im algemeinen
der Knotenregel, wobei der Nulldurchgang der 1.  und 2.  (2.)
Eigenfunktion (wir beginnen die Zaehlung bei 0, 0.  Eigenfunktion)
sehr schlecht sichtbar ist.  Bei hoeheren energien (hier nicht
geplottet) bildet sich sogar ein paritaets muster heraus.

Also: jede geplottete loesung E_i (i=1...n) hat genau I nullstellen
(die Raender nicht mitgezaehlt).  Die beiden untersten Loesungen (0,
1) aehneln sich (bis auf den Nulldurchgang, den der Knotensatz
erzwingt) sehr, da sie jeweils ueber aehnlich geformten
potentialminima gross werden.  Superposition dieser Loesungen laesst
einen Tunneleffekt zu.  (Ebenso Loesung 2, 3) Die Loesungen sind
gebunden und fallen in den Klassisch verbotenen bereichen schnell ab,
sodass die einschraenkung auf ein endliches Interval eine gute
naeherung darstellt.  In allgemeiner Form aber aehneln sie beig den
Eigenfunktionen des Harm.  Oszi (bis auf asymetrische Verzerrungen.)

Fuer groessere h_eff werden die Energieeigenwerte groesser (und liegen
weniger dicht) und die Wellenfunktionen immer Symetrischer, da die
Potentialasymetrie nur noch eine kleine Stoerung darstellt.  Die
Loesungen dringen immer weniger in die klassich verbotenen bereiche
ein => uebergang zur klassischen physik.

Fuer h_eff < 0.07 ergeben sich kleinere, dichtere eigenwerte.  Es
ergeben sich mehr energien im Rechten Minimum.  Der knotensatz wird
weiterhin eingehalten, wobei die Nulldurchgange fuer die ersten
loesungen im Linken minimum sehr flach sind.

c) A=0 Das Potential ist nun symetrisch und somit sind auch die
Eigenfunktionen abwechselnd gerade bzw.  ungerade.  Da die beiden
Potentialminima aequivalent sind, liegen jeweils zwei eigenwerte (fuer
kleine eigenwerte) sehr nah bei einander (und ihre eigenfunktionen
gehen durch gewichtete addition ineinander ueber under erfuellen somit
die Symetrieanforderungen, leichte variation von h_eff vertauscht
eigenfunktionen).  Numerisch wuerde man Entartung vermuten, die aber
analytisch nicht moeglich ist -> grenzen/gefahren der Numerik.  (Fuer
kleine h_eff wird der Effekt deutlicher!, fuer hohe N streben diese
Abstaende aber gegen endliche werte.)
"""
