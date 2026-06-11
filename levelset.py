"""
Level-set of the computational domain used by the phi-FD solvers.

Bean domain (slightly non-convex bent ellipse):

    phi(x, y) = 0.8 x^2 + (y + 1.4 x^2)^2 - Rb^2 ,   Rb = 0.5 ,

where the term (y + 1.4 x^2) bends the ellipse along the parabola y = -1.4 x^2,
producing a gentle concavity at the bottom.

Omega = { phi < 0 }.
"""

import numpy as np


def make_phi(Rb=0.5):
    """Return a vectorized level-set function phi(x, y) of the bean domain."""
    def phi(x, y):
        # the -1e-10 shift moves the interface slightly off the grid nodes so
        # that the test genuinely exercises weakly cut cells (a node landing
        # very close to the interface) rather than nodes lying exactly on it.
        return 0.8 * x**2 + (y + 1.4 * x**2)**2 - Rb**2 - 1e-10
    return phi
