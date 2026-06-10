"""
Level-set of the computational domain used by the phi-FD solvers.

Bean domain (slightly non-convex bent ellipse):

    phi_raw(x, y) = 0.8 x^2 + (y + 1.4 x^2)^2 - Rb^2 ,   Rb = 0.5 ,

where the term (y + 1.4 x^2) bends the ellipse along the parabola y = -1.4 x^2,
producing a gentle concavity at the bottom. The raw level set is far from a
signed distance, so it is normalized by |grad phi_raw|: this is essential for
the phi-FD boundary/Neumann stencils to converge.

Omega = { phi < 0 }.
"""

import numpy as np


def make_phi(Rb=0.5):
    """Return a vectorized level-set function phi(x, y) of the bean domain."""
    def raw(x, y):
        return 0.8 * x**2 + (y + 1.4 * x**2)**2 - Rb**2

    def phi(x, y):
        e = 1e-5
        g  = raw(x, y)
        gx = (raw(x + e, y) - raw(x - e, y)) / (2 * e)
        gy = (raw(x, y + e) - raw(x, y - e)) / (2 * e)
        return g / np.sqrt(gx * gx + gy * gy + 1e-30)

    return phi
