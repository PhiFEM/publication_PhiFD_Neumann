"""How many boundary-isolated points are there?

A boundary node is isolated in a direction when the last branch of the discrete
gradient (3) fires there: fewer than three nodes of Omega_h line up in that
direction. They can only occur within O(h) of the finitely many points where
dOmega is tangent to a grid direction, so their number should stay bounded as
h -> 0. Count them, and count the boundary nodes for comparison.
"""
import numpy as np
from levelset import make_phi
from phiFD_poisson_neumann_bean import solve

for N in (20, 40, 80, 160, 320):
    F = solve(N, make_phi(), return_fields=True)
    ind, indOut = F["ind"], F["indOut"]
    bd = (indOut == 0) & (ind == 0)
    J, I = np.where(bd)
    O = lambda i, j: indOut[j, i] == 1
    iso = 0
    for i, j in zip(I, J):
        for (di, dj) in ((1, 0), (0, 1)):
            p = lambda k: O(i + k*di, j + k*dj)
            if not (p(1) and not p(-1) and not p(-2)) and \
               not (p(-1) and not p(1) and not p(2)) and \
               not (not p(1) and not p(-1)):
                iso += 1; break
    print(f"N={N:4d}  h={2.0/N:.5f}  boundary nodes {len(I):4d}  isolated {iso:3d}")
