"""
Regenerate fig_elastic_domains.png: the bean computational domain Omega_h on the
Cartesian grid. Black dots are the nodes of Omega_h, the red curve is the
interface {phi = 0}. Pure Neumann case: no Gamma_D/Gamma_N split line.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from levelset import make_phi

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
# nodes of Omega_h
ax.plot(X[omega], Y[omega], "o", color="black", ms=7)

ax.set_xlim(a, b); ax.set_ylim(a, b)
ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(pad=0)
fig.savefig("fig_elastic_domains.png", dpi=120, bbox_inches="tight", pad_inches=0.02)
print("wrote fig_elastic_domains.png  (N =", N, ", nodes in Omega_h =", int(omega.sum()), ")")
