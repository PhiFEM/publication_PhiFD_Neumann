"""
Robustness to the smallest cut cell: phi-FD versus the cut-cell finite-volume
scheme of Arias, Bochkov & Gibou (2018), on the same geometries.

Table (c) of the article compares the two condition numbers at a *fixed*
geometry while h varies, which is precisely the situation in which the small
cut cell does not appear. Here we do the converse: h is fixed and the bean
radius is swept, so that the interface moves relative to the grid and the
smallest cut cell spans three decades. This is the classical weak spot of
cut-cell finite volumes -- a wetted cell of area eps*h^2 produces a row of
weight eps -- whereas phi-FD couples nodes, not volumes.

For every geometry we record, for both schemes, the relative L2 error and the
2-norm condition number (cond2 of arias_poisson_neumann_bean, a power iteration
on A^T A and on its inverse). Two abscissae are reported, each natural for one
scheme:

    s     = min |phi_alpha| / h  over the grid nodes      (phi-FD)
    theta = min (wetted area) / h^2 over the active cells (cut-cell)

The geometries are the 40 of Figure fig:cond, so the phi-FD columns can be
checked against sensitivity_kappa.dat.
"""

import warnings

import numpy as np

from arias_poisson_neumann_bean import K, cond2, rel_L2
from arias_poisson_neumann_bean import solve as arias_solve
from levelset import make_phi
from phiFD_poisson_neumann_bean import solve as phifd_solve
from sensitivity_cutcell import SMIN, SMAX, pgf_coords, smallest_cut, subsample

warnings.filterwarnings("ignore")

N = 40                                            # h = 0.05, as in fig:cond
BOX = (-1.0, 1.0)
ALPHA = 1.0
GEOMETRY = "refined"    # consistent sub-resolved interface reconstruction

uex = lambda x, y: np.sin(K * x) * np.cos(K * y)
f = lambda x, y: (2 * K * K + ALPHA) * uex(x, y)


def g(x, y):
    """du/dn on {phi = 0}; the gradient of the bean does not depend on Rb."""
    s = y + 1.4 * x ** 2
    phix, phiy = 1.6 * x + 5.6 * x * s, 2.0 * s
    nn = np.hypot(phix, phiy)
    ux = K * np.cos(K * x) * np.cos(K * y)
    uy = -K * np.sin(K * x) * np.sin(K * y)
    return (ux * phix + uy * phiy) / nn


def smallest_wetted(area, active, h):
    """Smallest wetted cell area, in units of h^2 -- the cut-cell small cell."""
    a = area[active]
    return a.min() / h ** 2 if a.size else np.nan


def pick_geometries():
    """The 40 geometries of Figure fig:cond (see sensitivity_kappa.py)."""
    radii = np.linspace(0.40, 0.60, 240)
    s = np.array([smallest_cut(make_phi(Rb=R), N=N, box=BOX) for R in radii])
    keep = (s >= SMIN) & (s <= SMAX)
    radii, s = radii[keep], s[keep]
    order = np.argsort(s)
    radii, s = radii[order], s[order]
    idx = subsample(s)
    return radii[idx], s[idx]


def run():
    """Return (s, theta, L2_phifd, kappa_phifd, L2_arias, kappa_arias)."""
    radii, s = pick_geometries()
    h = 2.0 / N
    print(f"{len(s)} geometries, h = {h:.4f}, "
          f"cut range [{s.min():.2e}, {s.max():.2e}]\n")
    print(f"{'s=min|phi|/h':>12} {'theta=minV/h2':>13} | "
          f"{'phi-FD L2':>10} {'phi-FD kap':>11} | {'cut L2':>10} {'cut kappa':>11}")
    rows = []
    for Rb, si in zip(radii, s):
        phi = make_phi(Rb=Rb)
        e_p = phifd_solve(N, phi, alpha=ALPHA, box=BOX)[0]
        k_p = cond2(phifd_solve(N, phi, alpha=ALPHA, box=BOX, return_A=True))
        xc, u, area, active, A = arias_solve(N, phi, f, g, ALPHA, box=BOX,
                                            geometry=GEOMETRY)
        e_a, k_a = rel_L2(xc, u, area, active, uex), cond2(A)
        th = smallest_wetted(area, active, h)
        rows.append((si, th, e_p, k_p, e_a, k_a))
        print(f"{si:12.3e} {th:13.3e} | {e_p:10.3e} {k_p:11.3e} "
              f"| {e_a:10.3e} {k_a:11.3e}")
    return np.array(sorted(rows))


def report(a):
    s, th = a[:, 0], a[:, 1]
    print(f"\n{'':>16} {'min':>11} {'max':>11} {'max/min':>9} "
          f"{'slope in s':>11} {'slope in theta':>15}")
    for lab, v in (("phi-FD  L2", a[:, 2]), ("phi-FD  kappa", a[:, 3]),
                   ("cut-cell L2", a[:, 4]), ("cut-cell kappa", a[:, 5])):
        print(f"{lab:>16} {v.min():11.3e} {v.max():11.3e} {v.max()/v.min():9.2f} "
              f"{np.polyfit(np.log(s), np.log(v), 1)[0]:+11.2f} "
              f"{np.polyfit(np.log(th), np.log(v), 1)[0]:+15.2f}")
    print(f"\nsmallest wetted cell theta in [{th.min():.2e}, {th.max():.2e}]")


if __name__ == "__main__":
    a = run()
    report(a)
    s, th = a[:, 0], a[:, 1]
    np.savetxt("compare_cutsize.dat", a,
               header="s=min|phi|/h  theta=minV/h2  L2(phiFD)  kappa(phiFD)  "
                      "L2(cutcell)  kappa(cutcell)", fmt="%.6e")

    print("\n% --- kappa vs smallest cut, phi-FD and cut-cell ---")
    print("% phi-FD\n\\addplot[only marks,mark=x,mark size=1.5pt,color=blue] "
          f"coordinates {{{pgf_coords(s, a[:, 3])}}};")
    print("% cut-cell\n\\addplot[only marks,mark=square,mark size=1pt,color=red] "
          f"coordinates {{{pgf_coords(s, a[:, 5])}}};")
    print("\n% --- L2 error vs smallest cut ---")
    print("% phi-FD\n\\addplot[only marks,mark=x,mark size=1.5pt,color=blue] "
          f"coordinates {{{pgf_coords(s, a[:, 2])}}};")
    print("% cut-cell\n\\addplot[only marks,mark=square,mark size=1pt,color=red] "
          f"coordinates {{{pgf_coords(s, a[:, 4])}}};")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
        for ax, (cp, ca, lab) in zip((ax1, ax2),
                                     ((2, 4, r"$L^2$ error"), (3, 5, r"$\kappa$"))):
            ax.loglog(s, a[:, cp], "x", ms=5, color="tab:blue", label=r"$\varphi$-FD")
            ax.loglog(s, a[:, ca], "s", ms=4, mfc="none", color="tab:red",
                      label="cut-cell [Arias et al.]")
            ax.set_xlabel(r"$\min|\varphi|/h$"); ax.set_ylabel(lab)
            ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(fontsize=8)
        fig.suptitle(rf"Robustness to the smallest cut cell, $h={2.0/N:.3f}$",
                     fontsize=10)
        fig.tight_layout(); fig.savefig("compare_cutsize.png", dpi=130)
        print("\nsaved compare_cutsize.png and compare_cutsize.dat")
    except ImportError:
        print("\n(matplotlib not available -- skipped PNG)")
