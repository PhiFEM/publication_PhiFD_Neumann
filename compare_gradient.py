"""
Accuracy of the gradient: phi-FD against the cut-cell scheme of Arias, Bochkov
& Gibou (2018), whose stated selling point is precisely a second-order accurate
gradient.

The article compares the two schemes on the L2 error of u only. Here we compare
the gradient of the computed solution, which for phi-FD is the quantity the
boundary condition acts on: the scheme relaxes grad u . grad phi, not u.

Each scheme is post-processed with its own natural gradient:

    phi-FD    grad_h, formula (3) of the article -- centered inside, one-sided
              second order near the boundary, and the derivative of the coupled
              node x_{alpha_0} at a boundary-isolated point;
    cut-cell  the derivative at the centre of each wetted grid edge, given by
              the very stencils ddx_vface / ddy_hface that assemble the fluxes.
              This is the gradient the method claims second-order accurate
              ("by discretizing the flux between cells by approximating the
              derivatives at the centers of cut edges"), and it lives on edges,
              not at cell centres -- comparing against a cell-centred finite
              difference post-processing would be a straw man.

Each scheme is therefore judged on its own gradient, at the points where that
gradient is defined, against the exact derivative at the same points. Three
relative errors are reported:

    L2(Omega)     discrete L2 over all the nodes / wetted faces,
    Linf(Omega)   worst node / face anywhere,
    Linf(bd)      worst one in the boundary layer -- dOmega_h for phi-FD, the
                  faces of the cut cells for the finite-volume scheme.

The last two expose the boundary treatment; Arias et al. measure in Linf.
"""

import warnings

import numpy as np

from arias_poisson_neumann_bean import K
from arias_poisson_neumann_bean import solve as arias_solve
from levelset import make_phi
from phiFD_poisson_neumann_bean import grad_h
from phiFD_poisson_neumann_bean import solve as phifd_solve

warnings.filterwarnings("ignore")

ALPHA = 1.0
# "refined" gives the cut-cell scheme a consistent, sub-resolved interface
# reconstruction, in place of the piecewise-linear one. Without it its
# gradient is only first order in Linf (0.97 instead of 1.54), which would
# be an artifact of our reconstruction rather than a property of the method.
GEOMETRY = "refined"
uex = lambda x, y: np.sin(K * x) * np.cos(K * y)
dxu = lambda x, y: K * np.cos(K * x) * np.cos(K * y)
dyu = lambda x, y: -K * np.sin(K * x) * np.sin(K * y)
f = lambda x, y: (2 * K * K + ALPHA) * uex(x, y)


def g(x, y):
    s = y + 1.4 * x ** 2
    phix, phiy = 1.6 * x + 5.6 * x * s, 2.0 * s
    return (dxu(x, y) * phix + dyu(x, y) * phiy) / np.hypot(phix, phiy)


def errors(ex, ey, gx, gy, dom, bd):
    """Relative L2 over dom, Linf over dom, Linf over the boundary layer bd."""
    e = np.hypot(gx - ex, gy - ey)
    n = np.hypot(ex, ey)
    return (np.sqrt(np.sum(e[dom] ** 2) / np.sum(n[dom] ** 2)),
            e[dom].max() / n[dom].max(),
            e[bd].max() / n[dom].max() if bd.any() else np.nan)


def phifd_gradient_errors(N, phi):
    r = phifd_solve(N, phi, alpha=ALPHA, return_fields=True)
    gx, gy = grad_h(r["u"], r["ind"], r["indOut"], r["a0"], r["h"])
    X, Y = r["X"], r["Y"]
    dom = (1 - r["indOut"]) > 0                   # Omega_h
    bd = dom & (r["ind"] == 0)                    # dOmega_h
    return errors(dxu(X, Y), dyu(X, Y), gx, gy, dom, bd)


def arias_gradient_errors(N, phi):
    """Error of the scheme's own edge-centred gradient, at the wetted faces."""
    *_, faces = arias_solve(N, phi, f, g, ALPHA, return_faces=True,
                            geometry=GEOMETRY)
    x = np.array([p[0] for p in faces])
    y = np.array([p[1] for p in faces])
    v = np.array([p[2] for p in faces])
    isx = np.array([p[3] for p in faces])
    bd = np.array([p[4] for p in faces])
    ex = np.where(isx, dxu(x, y), dyu(x, y))      # exact derivative at the face
    e = np.abs(v - ex)
    n = np.hypot(dxu(x, y), dyu(x, y))
    return (np.sqrt(np.sum(e ** 2) / np.sum(n ** 2)),
            e.max() / n.max(),
            e[bd].max() / n.max() if bd.any() else np.nan)


if __name__ == "__main__":
    phi = make_phi()
    Ns = [20, 40, 80, 160]
    hs = np.array([2.0 / N for N in Ns])
    P, A = [], []
    print(f"{'h':>8} | {'phi-FD L2':>10} {'Linf(Om)':>10} {'Linf(bd)':>10}"
          f" | {'cut L2':>10} {'Linf(Om)':>10} {'Linf(bd)':>10}")
    print("-" * 82)
    for N, h in zip(Ns, hs):
        p, a = phifd_gradient_errors(N, phi), arias_gradient_errors(N, phi)
        P.append(p); A.append(a)
        print(f"{h:8.4f} | " + " ".join(f"{v:10.3e}" for v in p)
              + " | " + " ".join(f"{v:10.3e}" for v in a))
    P, A = np.array(P), np.array(A)
    sl = lambda v: np.polyfit(np.log(hs), np.log(v), 1)[0]
    print("-" * 82)
    print(f"{'order':>8} | " + " ".join(f"{sl(P[:, k]):10.2f}" for k in range(3))
          + " | " + " ".join(f"{sl(A[:, k]):10.2f}" for k in range(3)))
    np.savetxt("compare_gradient.dat", np.column_stack([hs, P, A]),
               header="h  phiFD_L2 phiFD_Linf phiFD_Linf_bd  "
                      "cut_L2 cut_Linf cut_Linf_bd", fmt="%.6e")
    print("\nsaved compare_gradient.dat")
