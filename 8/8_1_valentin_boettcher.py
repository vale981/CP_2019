#!/usr/bin/env python3
"""
This program calulates the time-averaged pressure for a system
involving `N` particles `R` times.

The obtained pressure values are being plotted as histogram, along
with a reference normal distribution with the parameters (calculated
from the pressure values):

    - mu :: mean

    - sigma :: standard deviatiobs

The values of the following parameters are also displayed on the plot:

    - N :: Particle Count

    - R :: Realization Count

    - Delta t :: averaging time interval
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def refl_count(x0, v, dt):
    """Calculate the reflection count for one wall.

    :param x0: start positions
    :param v: velocities
    :param dt: time interval
    :returns: the relfection count in the same shape as x0/v0
    """

    # wee add an offset, because we dont start at x=0 but at x=1
    #      floor      way-length  +  offset / max free way length
    return np.uint32((np.abs(x0 + v * dt) + 1) / 2)


def pressure(particles, dt):
    """Calculate the pressure on the unit surface averaged over time.

    :param particles: a particle ensemble (x0, v0)
    :param dt: time interval

    :returns: the pressures
    """

    x0, v = particles
    N = x0.shape[1]

    # sum over one axis only -> pressure per realization
    pressures = np.abs(v) * refl_count(x0, v, dt)
    return 2 / (dt * N) * np.sum(pressures, axis=1)


def make_ensemle(R, N):
    """Generates a particle ensemble of `N` particles in `R`
    relizations.

    The returned velocities follow a normal distribution and the start
    positions are evenly distributed.

    :param R: realization count
    :param N: particle count

    :returns: tuple of x0, vo in the shape `(R, N)`
    """

    # normal distributed velocities
    v = np.random.normal(size=(R, N))

    # evenly distributed locations
    x0 = np.random.uniform(low=0, high=1, size=(R, N))

    return x0, v


def set_up_plot(xlim):
    """Sets up the plot and its parameters.

    :returns: fig, ax
    """

    fig, ax = plt.subplots(1, 1)
    ax.set_title("$p_A$ Histogram")

    ax.set_xlabel(r"$p_A$")
    ax.set_ylabel(r"$P(p_A)$")

    ax.set_xlim(*xlim)
    return fig, ax


def main(N=6000, R=10000, dt=5):
    """Set model parameters and Dispatch the main logic.  Used as
    convenience.

    :param R: realization count
    :param N: particle count
    :param dt: averaging time interval
    """

    # print user guide, whether he wants it or not :)
    print(__doc__)

    # craft our particles and calculate the pressures
    ensemple = make_ensemle(R, N)
    pressures = pressure(ensemple, dt)

    # calculate the moments
    mu = pressures.mean()
    sigma = pressures.std(ddof=1)  # empirical

    # set the number of bins estimated by the Freedman–Diaconis rule
    # statically, to make results comparable
    bins = 80

    # plotting
    fig, ax = set_up_plot((pressures.min(), pressures.max()))
    ax.hist(
        pressures,
        bins=bins,
        density=True,
        label="$p_A$ propability density",
        color="orange",
    )

    # gaussian referece curve
    x = np.linspace(0, pressures.max(), 1000)
    ax.plot(
        x,
        stats.norm.pdf(x, mu, sigma),
        label="reference normal distribution",
        color="blue",
    )

    ax.legend()

    print(pressures.max() / (stats.iqr(pressures) * 2 / ((len(pressures)) ** (1 / 3))))
    # print out parameters
    fig.text(
        0,
        0,
        fr"$N={N}$ $R={R}$ $\Delta t = {dt}$ "
        + fr"$\mu={np.round(mu, 5)}$ $\sigma={np.round(sigma, 5)}$ "
        + fr"bins=${bins}$",
    )

    plt.show()


if __name__ == "__main__":
    main()

###############################################################################
#                                  Diskussion                                 #
###############################################################################

"""
Anzahl der Bins: es soll eine glatte kurve mit einigem detail zu
erahnen sein.  80 Bins scheinen mir ein guter kompromiss aus
Uebersichtlichkeit und massvollem Detail darzustellen.

Die Freedman–Diaconis Regel liefert einen aehnlichen Wert.  (ca 70)
Die Wurzel-Regel wiederum liefert einen Wert von 100.  Andere regeln
liefern tiefere Werte, allerdings wird dann die Form der Verteilung
unklar.

a)

Form
====

Fuer N=60 stimmt die Form schon erheblich besser mit der
normalverteilung ueberein.  Wo das maximum bei N=6 noch weit links
~0.6 neben dem der Normalverteilung liegt decken sich die maxima bei
N=60 fast wobei auch hier ein leichter bias nach links zu erkennen
ist.  Im allgemeinen liegt (an den Raendern!) die Verteilung bei
kleinen Druecken unter und bei hoeheren ueber der gauss vert.

Erwartungswert
==============

Fuer beide Teilchenzahlen sind die Mittelwerte sehr nah an mu=1 (wie
es zu erwarten ist).

Standardabweichung
==================

Wo fuer N=6 die Standardabweichung noch ~0.6 betraegt sinkt selbige
erheblich fuer N=60 auf ~0.2.

Bei mehrfahausfuehrung veraendern sich diese werte sehr wenig (etwas
anderes waehre auch bedenklich).

Allgemeines
===========

Fuer kleine N hat der Stosszahlfaktor einen recht Grossen einfluss auf
die eig.  schon normalverteilten geschwindigkeiten (man entferne ihn
und sehe).  Dieser mindert sich wenn man die Zeit dt groesser waehlt.
Fuer grosse N vermute ich das wirken Zentralen grenzwertsatzes (in
ermangelung von ordentlichen statistischen Grundlagen meinerseits :P)
der bei einer Addition von unabh.  gleichartigen zuf.  Variablen
(|v_i|*n_i) greift.  (es mitteln sich die |v_i|*n_i aus, die rechte
Flanke wird angehoben da oefter grosse druecke entstehen) Bei wenig
teilchen ist bei der rel.  kurzen zeit dt=5 der einfluss des geschw.
vorzeichens noch erheblich und somit ergibt sich eine Tendenz zu
geringeren druecken.  Der konstante mittelwert ergibt vermutlich sich
aus den gesetzen des idealen gaases deren vorraussetzung wir hier
modellieren.  (v - verteilung in lieu fuer kollisionen...)

b)

Form
====

Anneaherung an eine Symetrische, sehr eng um den Mittelwert
konzentrierten Verteilung, sehr aehnlich oder aequivalent zur
Normalverteilung.  Fuer N=6 zu N=60 ist diese Entwicklung berteits zu
erkennen und selbige setzt sich auch fuer N=600 bzw.  N=6000 fort
(siehe auch obige punkte zur erklaerung).

Fuer grosse N ist das system weniger empfindlich fuer schwankungen im
unstetigen Stosszahlfaktor (die sich nun herausmitteln) und somit
symetrischer verteilt.

Erwartungswert
==============

Es gibt keinen Grund das abweichen von mu=1 zu erwarten.  Auch fuer
N=6000 (oder N=1243) ist mu~=1.  An der Situation des idealen gaases
aendert sich nichts, auch wenn die Realitaet evtl.  nicht mehr
einwandfrei modelliert wird.

Standardabweichung
==================

Diese wird sich, wie zu vermuten ist, auf einen sehr kleinen (aber
nicht verschwindenden) wert einstellen (zumindest um ein paar
groessendordnung kleiner als bei N=6).  Dies schliesse ich aus dem
allgemeinen Trend, da fuer N=6000 immerhin noch sigma~=0.02 zu
beobachten ist.
"""
