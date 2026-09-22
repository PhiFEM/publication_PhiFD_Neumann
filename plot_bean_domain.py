"""
Regenerate fig_elastic_domains.png: the bean computational domain Omega_h on the
Cartesian grid. Black dots are the nodes of Omega_h, the red curve is the
interface {phi = 0}; the boundary-isolated points (last branch of the discrete
gradient (3): fewer than three nodes of Omega_h line up in some direction) are
drawn in blue. Pure Neumann case: no Gamma_D/Gamma_N split line.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from levelset import make_phi

if __name__ == "__main__":
    N   = 14                       # grid resolution (matches the original figure)
    box = (-1.0, 1.0)
    phi = make_phi()

    a, b = box
    x = np.linspace(a, b, N + 1)
    h = x[1] - x[0]
    X, Y = np.meshgrid(x, x)
    ind = phi(X, Y) < 0            # interior nodes

    # Omega_h = interior nodes dilated by the 4-neighbourhood {(0,0),(+-1,0),(0,+-1)}
    # (the scalar definition: horizontal/vertical neighbours only, no diagonals).
    omega = np.zeros_like(ind)
    for dj, di in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        sj = slice(max(dj, 0), N + 1 + min(dj, 0))
        si = slice(max(di, 0), N + 1 + min(di, 0))
        tj = slice(max(-dj, 0), N + 1 + min(-dj, 0))
        ti = slice(max(-di, 0), N + 1 + min(-di, 0))
        omega[sj, si] |= ind[tj, ti]

    fig, ax = plt.subplots(figsize=(5, 5))
    # Cartesian grid
    for xi in x:
        ax.axvline(xi, color="black", lw=0.7)
        ax.axhline(xi, color="black", lw=0.7)
    # interface {phi = 0}, finely sampled
    xf = np.linspace(a, b, 600)
    Xf, Yf = np.meshgrid(xf, xf)
    ax.contour(Xf, Yf, phi(Xf, Yf), levels=[0.0], colors="red", linewidths=3)
    # boundary-isolated points: boundary nodes for which, in some direction,
    # neither the centered nor a second-order one-sided stencil fits in Omega_h
    ok = lambda j, i: 0 <= j <= N and 0 <= i <= N and bool(omega[j, i])
    iso = np.zeros_like(omega)
    for j, i in zip(*np.where(omega & ~ind)):
        for dj, di in ((0, 1), (1, 0)):
            p1, m1 = ok(j+dj, i+di), ok(j-dj, i-di)
            p2, m2 = ok(j+2*dj, i+2*di), ok(j-2*dj, i-2*di)
            if not (p1 and m1) and not (p1 and p2) and not (m1 and m2):
                iso[j, i] = True
    # nodes of Omega_h
    ax.plot(X[omega & ~iso], Y[omega & ~iso], "o", color="black", ms=7)
    ax.plot(X[iso], Y[iso], "o", color="blue", ms=7)

    ax.set_xlim(a, b); ax.set_ylim(a, b)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "fig_elastic_domains.png"), dpi=120, bbox_inches="tight", pad_inches=0.02)
    print("wrote fig_elastic_domains.png  (N =", N, ", nodes in Omega_h =", int(omega.sum()),
          ", boundary-isolated =", int(iso.sum()), ")")
