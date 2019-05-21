import numpy as np
import matplotlib.pyplot as plt
import functools

def start_sinus(event, ax, phi_t):
    """Plotte Sinus-Kurve."""
    x = np.linspace(0.0, 2.0*np.pi, 100)
    sinus = ax.plot(x, np.sin(x))
    for phi in phi_t:
        sinus_t = np.sin(x-phi)
        sinus[0].set_ydata(sinus_t)
        event.canvas.flush_events()
        event.canvas.draw()

def main():
    """Hauptprogramm."""
    phi_t = np.linspace(0.0, 6.0, 100)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Maustaste zum Starten klicken")
    klick_funktion = functools.partial(start_sinus, ax=ax,
                                       phi_t=phi_t)
    fig.canvas.mpl_connect("button_press_event", klick_funktion)
    fig.canvas.mpl_connect("button_press_event", klick_funktion)
    plt.show()
