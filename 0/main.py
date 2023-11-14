#!/usr/bin/env python3
"""Draws spirals on mouse click."""

import numpy as np
import matplotlib.pyplot as plt


def spiral(windings=5, s=1 / 2, omega=2 * np.pi):
    """Creates a 2D numpy array of points on a spiral.

    :param windings: how many windings the spiral shall have (defaults to 5)
    :param s: the scaling factor (distance between the arms of the
              spiral) (defaults to 1/2)
    :param omega: the angular speed of the spiral (defaults to 2pi)

    :returns: 2D numpy array of x and y

    :rtype: numpy array
    """

    times = np.arange(0, windings, 0.01)  # initialize the spiral parameter
    angles = omega * times

    return (s ** times) * np.array([np.cos(angles), np.sin(angles)])


# Just to avoid globals...
class SpiralPlotter:
    """Plots spirals on click."""

    def __init__(self):
        """Sets up the plot and its callbacks."""

        # initialize state
        self.fig, self.ax = plt.subplots(1, 1, constrained_layout=True)

        # clear the axis and plot the initial spiral
        self.clear()
        self.ax.plot(*spiral())

        # register events
        self.cid = self.fig.canvas.mpl_connect(
            "button_press_event", self._draw_on_click
        )
        self.cid = self.fig.canvas.mpl_connect(
            "key_press_event", self._handle_key_press
        )

        # show the figure
        self.fig.show()

    def _handle_key_press(self, event):
        """Handles key press events on the plot.

        :param KeyEvent event:
        """

        if event.key == "c":
            self.clear()

    def draw_at(self, points, coords=(0, 0)):
        """Draws points shifted by coords.

        :param points: 2D Numpy array of x and y coordinates (2, N)
        :param Tuple coords: (x, y) Coordinates (defaults to (0,0))

        """

        self.ax.plot(*(points + [[coords[0]], [coords[1]]]))
        self.fig.canvas.draw()

    def clear(self):
        """Clears the axis and sets it up again.
        """

        self.ax.clear()

        # set up the axis (again)
        self.ax.set_title("Spiral Plotter")
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        # red
        self.fig.canvas.draw()

    def _draw_on_click(self, event):
        """Click event handler. Draws a new Spiral.

        :param MouseEvent event: the click event
        """

        ax = event.inaxes
        if not ax:
            return

        self.draw_at(spiral(), (event.xdata, event.ydata))


if __name__ == "__main__":
    plt.ion()
    SpiralPlotter()
    plt.show(block=True)  # block until figure closed
