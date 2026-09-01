"""
Sensitivity of phi-FD Neumann to the size of the smallest cut cell -- the data
behind Figure fig:cond(a)-(b) of the article.

Same protocol as sensitivity_cutcell.py: the mesh size is fixed (h = 2/N) and
the bean radius Rb is swept, which shifts the interface with respect to the
Cartesian grid so that the smallest coupled cell

    s = min_{(x_i,y_j) in Omega} |phi_ij| / h

spans three decades. For every geometry we record the relative L2 error and the
2-norm condition number of the assembled matrix, for two schemes:

    (i)   the scheme as first written: relaxation rows all scaled by h^-4 and
          alpha0 taken as the nearest interior node;
    (ii)  the same, with each relaxation row normalized by its own sup-norm;
    (iii) the same, with alpha0 additionally required to satisfy
          |phi_alpha0| >= 0.1 h.

Variant (i) shows kappa growing as the cut shrinks. Most of that growth is an
artifact of the uniform h^-4 normalization, removed by (ii). What remains comes
from interior nodes lying essentially on Gamma being chosen as alpha0 for two
different boundary nodes, which makes their two relaxation equations
proportional; (iii) removes it and leaves kappa flat in s. Variants (i) and (ii)
differ by a row scaling only, so they give the same discrete solution; (iii)
changes which interior node is coupled, hence the second error curve.

The script prints the curves as pgfplots "coordinates {...}" lists ready to
paste into main.tex, and saves sensitivity_kappa.dat and sensitivity_kappa.png.
"""

import warnings

import numpy as np

from levelset import make_phi
from phiFD_poisson_neumann_bean import solve
from sensitivity_cutcell import (SMIN, SMAX, pgf_coords, smallest_cut,
                                 subsample)

warnings.filterwarnings("ignore")                 # scipy sparse dtype FutureWarnings

N = 40                                            # h = 0.05, as in fig:cond
BOX = (-1.0, 1.0)
CFG = [("h4",   "uniform $h^{-4}$",  dict(row_scaling="h4", alpha0_rule="nearest")),
       ("norm", "row-normalized",    dict(row_scaling="norm", alpha0_rule="nearest")),
       ("safe", "+ $\\alpha_0$ rule", dict(row_scaling="norm", alpha0_rule="safe"))]


def pick_geometries():
    """The *exact* 40 geometries of the published Figure fig:cond.

    They are the subsample (sensitivity_cutcell.subsample) of the radius sweep
    np.linspace(0.40, 0.60, 240) restricted to SMIN <= min|phi|/h <= SMAX, i.e.
    the ones already plotted in panel (a). Reusing them makes the stabilized and
    unstabilized curves a paired comparison on identical geometries, and leaves
    panel (a) of the article unchanged.
    """
    radii = np.linspace(0.40, 0.60, 240)
    s = np.array([smallest_cut(make_phi(Rb=R), N=N, box=BOX) for R in radii])
    keep = (s >= SMIN) & (s <= SMAX)
    radii, s = radii[keep], s[keep]
    order = np.argsort(s)                         # subsample() expects sorted s
    radii, s = radii[order], s[order]
    idx = subsample(s)
    return radii[idx], s[idx]


def run():
    """Return (s, L2_nearest, L2_safe, kappa_h4, kappa_norm, kappa_safe, drift)."""
    radii, s = pick_geometries()
    print(f"{len(s)} geometries, h = {2.0 / N:.4f}, "
          f"cut range [{s.min():.2e}, {s.max():.2e}]\n")
    print(f"{'min|phi|/h':>11} {'L2 near':>11} {'L2 safe':>11} "
          + " ".join(f"{k:>12}" for k, _, _ in CFG))
    rows = []
    for Rb, si in zip(radii, s):
        phi = make_phi(Rb=Rb)
        kap = [np.linalg.cond(solve(N, phi, box=BOX, return_A=True, **kw).toarray())
               for _, _, kw in CFG]
        e = [solve(N, phi, box=BOX, **kw)[0] for _, _, kw in CFG]
        # (i) and (ii) differ by a row scaling only, so their solutions coincide;
        # (iii) changes which interior node is coupled, hence a different scheme.
        drift = abs(e[0] - e[1]) / e[0]      # round-off only, in exact arithmetic 0
        rows.append((si, e[0], e[2], *kap, drift))
        print(f"{si:11.3e} {e[0]:11.3e} {e[2]:11.3e} " + " ".join(f"{k:12.3e}" for k in kap)
              + f"  drift={drift:8.1e}")
    return np.array(sorted(rows))


def report(a):
    """Print the summary statistics quoted in the article."""
    s = a[:, 0]
    slope = lambda v: np.polyfit(np.log(s), np.log(v), 1)[0]
    print(f"\n{'':>18} {'min':>11} {'max':>11} {'max/min':>9} {'slope in s':>11}")
    for c, (_, lab, _) in enumerate(CFG):
        v = a[:, 3 + c]
        print(f"{lab:>18} {v.min():11.3e} {v.max():11.3e} "
              f"{v.max() / v.min():9.2f} {slope(v):+11.2f}")
    print(f"{'scaling drift':>18} {a[:,6].min():11.3e} {a[:,6].max():11.3e}"
          "   (relative L2 difference between the h^-4 and the normalized rows)")
    for lab, e in (("L2 nearest", a[:, 1]), ("L2 alpha0 rule", a[:, 2])):
        print(f"{lab:>18} {e.min():11.3e} {e.max():11.3e} "
          f"{e.max() / e.min():9.2f} {slope(e):+11.2f}")


if __name__ == "__main__":
    a = run()
    report(a)
    s, e, e2 = a[:, 0], a[:, 1], a[:, 2]
    np.savetxt("sensitivity_kappa.dat", a,
               header="min|phi|/h   L2(nearest)   L2(alpha0 rule)   "
                      "kappa(h4)   kappa(norm)   kappa(safe)",
               fmt="%.6e")

    print("\n% --- fig:cond(a): L2 error vs smallest cut ---")
    print("% nearest alpha0\n\\addplot[only marks,mark=*,mark size=1pt,color=blue] "
          f"coordinates {{{pgf_coords(s, e)}}};")
    print("% alpha0 rule\n\\addplot[only marks,mark=o,mark size=1pt,color=black] "
          f"coordinates {{{pgf_coords(s, e2)}}};")
    print("\n% --- fig:cond(b): condition number vs smallest cut ---")
    for c, (key, lab, _) in enumerate(CFG):
        mark = ("*", "o", "x")[c]
        col = ("red", "black", "blue")[c]
        print(f"% {lab}\n\\addplot[only marks,mark={mark},mark size=1pt,color={col}] "
              f"coordinates {{{pgf_coords(s, a[:, 3 + c])}}};")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
        ax1.semilogx(s, e, "o", ms=4, color="tab:blue", label="nearest")
        ax1.semilogx(s, e2, "o", ms=4, mfc="none", color="k", label=r"$\alpha_0$ rule")
        ax1.legend(fontsize=8)
        ax1.set_ylim(0, 1.4 * max(e.max(), e2.max())); ax1.set_ylabel(r"$L^2$ error")
        for c, (key, lab, _) in enumerate(CFG):
            ax2.loglog(s, a[:, 3 + c], ("o", "s", "^")[c], ms=4,
                       mfc="none" if c else None,
                       color=("tab:red", "k", "tab:blue")[c], label=lab)
        ax2.set_ylabel(r"$\kappa$"); ax2.legend(fontsize=8)
        for ax in (ax1, ax2):
            ax.set_xlabel(r"$\min|\varphi|/h$")
            ax.grid(True, which="both", ls=":", alpha=0.5)
        fig.suptitle(rf"$\varphi$-FD Neumann, $h={2.0 / N:.3f}$", fontsize=10)
        fig.tight_layout(); fig.savefig("sensitivity_kappa.png", dpi=130)
        print("\nsaved sensitivity_kappa.png and sensitivity_kappa.dat")
    except ImportError:
        print("\n(matplotlib not available -- skipped PNG)")
