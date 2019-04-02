#!/usr/bin/env python3
"""Draw the standard map of the kicked rotor interactively."""
# NOTE: Barebones copied from ../0/main.py

import numpy as np
import matplotlib.pyplot as plt

# I am leaving this function out of the below class, because it really
# has no reason being there.
def get_standard_map(theta_0, p_0, K, N):
    """Calculates the standard map for the given parameters.

    :param theta_0: initial theta
    :param p_0: initial p
    :param K: the K parameter
    :param N: the number of iterations
    :returns: a tuple of the (thetas, ps)
    :rtype: tuple[NumpyArray, NumpyArray]
    """

    # Initialize the arrays
    theta = np.zeros(N)
    p = np.zeros(N)

    # set the initial parameters
    norm_p = get_normalizer(2 * np.pi, -np.pi)
    theta[0] = theta_0 % 2 * np.pi
    p[0] = norm_p(p_0)

    # calculate the standard map
    for curr in range(1, N):
        last = curr - 1 # for readability
        theta[curr] = (theta[last] + p[last]) % 2*np.pi
        p[curr] = norm_p(p[last] + K * np.sin(theta[curr]))

    return theta, p

# Just to avoid globals... again. I usually prefer to avoid to much
# useless Object orientation, but python kinda forces it on ya!
class StandardMap():
    """
    Shows a plot window that interactively draws the standart map of
    the kicked rotor
    """

    def __init__(self, iterations=1000, K=2.4):
        """Sets up the plot and its callbacks.

        :param Number iterations: the default number of iterations
        :param Number K: the default k parameter

        """
        # initialize state
        self.fig, self.ax = plt.subplots(1,1, constrained_layout=True)

        # initialize default parameters
        self._theta_0 = np.pi
        self._p_0 = 0
        self._K = K
        self._N = iterations

        # clear the axis and plot the initial spiral
        self.clear()
        self.ax.plot(*spiral())

        # register events
        self.cid = self.fig.canvas.mpl_connect('button_press_event',
                                               self._draw_on_click)
        self.cid = self.fig.canvas.mpl_connect('key_press_event',
                                               self._handle_key_press)

        # show the figure
        self.fig.show() # FIXME: that is bad style, put that into a method

    def _handle_key_press(self, event):
        """Handles key press events on the plot.

        :param KeyEvent event:
        """

        if event.key == 'c':
            self.clear()

    def draw_at(self, points, coords=(0,0)):
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
        self.ax.set_aspect('equal')
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


if __name__ == '__main__':
    StandardMap()
    plt.show(block=True) # block until figure closed



###########################################################################
#                                 Helpers                                 #
###########################################################################

def get_normalizer(interval, offset):
    """Gives a function that wraps a value in a given interval.

    :param Number interval: the interval length
    :param Number offset: the start point of the interval

    :returns: a normalizer function
    :rtype: function
    """
    def normalize(value):
        return ((value - offset) % interval) + offset

    return normalize
