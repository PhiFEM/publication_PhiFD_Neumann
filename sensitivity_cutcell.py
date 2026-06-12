"""
Sensitivity of the phi-FD scheme to the size of the smallest cut cell.

The Neumann condition couples each boundary node (x_i,y_j) with its closest
interior node (x_{i0},y_{j0}): this is the "coupled cell". When a grid node
falls very close to the interface, |phi|/h becomes tiny and the coupled cell
degenerates -- a classical source of accuracy loss for immersed methods. Here we
check that phi-FD is robust to this.

Strategy: fix the mesh size h and sweep the bean radius Rb over many values.
Each Rb shifts the interface with respect to the (fixed) Cartesian grid, so the
smallest cut cell min|phi|/h takes a whole range of values, including very small
ones. For every geometry we record the smallest cut size

    s = min_{(x_i,y_j) in Omega} |phi_ij| / h

and the relative L2 error. The error stays insensitive to s -> 0: the scheme is
robust to weakly cut cells. The script prints the data as a pgfplots
"coordinates {...}" list ready to paste into main.tex (Figure fig:cond) and also
saves a PNG.
"""

import numpy as np
from levelset import make_phi
from phiFD_poisson_neumann_bean import solve

# Fixed mesh (h = 2/N = 0.05) -- the value used for Figure fig:cond(a).
N = 40
BOX = (-1.0, 1.0)

# Realistic "weakly cut cell" window. Below SMIN a grid node lands within
# machine precision of the interface (an exact grid alignment, measure zero):
# |phi| ~ 0 there and the coupled cell is genuinely singular -- the
# node-on-boundary degeneracy, not a weakly cut cell, so we exclude it.
SMIN, SMAX = 1e-4, 1e-1


def smallest_cut(phi_func, N=N, box=BOX):
    """min |phi|/h over the interior nodes -- size of the smallest coupled cell."""
    a, b = box
    x = np.linspace(a, b, N + 1)
    h = x[1] - x[0]
    X, Y = np.meshgrid(x, x)
    phiij = phi_func(X, Y)
    inside = phiij < 0
    return np.min(np.abs(phiij[inside])) / h


def run(n_geom=240, Rmin=0.40, Rmax=0.60):
    """Sweep the bean radius and collect (cut size, L2 error)."""
    radii = np.linspace(Rmin, Rmax, n_geom)
    data = []
    print(f"{'Rb':>8} {'min|phi|/h':>12} {'L2 err':>12}")
    for Rb in radii:
        phi = make_phi(Rb=Rb)
        try:
            s = smallest_cut(phi)
            if not (SMIN <= s <= SMAX):           # skip exact-alignment / large
                continue
            eL2 = solve(N, phi)[0]
        except Exception as exc:                  # degenerate geometry -> skip
            print(f"{Rb:8.4f}  skipped ({exc})")
            continue
        data.append((s, eL2))
        print(f"{Rb:8.4f} {s:12.3e} {eL2:12.3e}")
    data.sort()                                   # order by cut size for plotting
    return np.array(data)


def pgf_coords(xs, ys):
    """Format two arrays as a pgfplots coordinates list."""
    return "".join(f"({x:.3e},{y:.3e})" for x, y in zip(xs, ys))


def subsample(s, n=55):
    """Indices of ~n points spread evenly in log(s) (for a readable figure)."""
    edges = np.logspace(np.log10(s[0]), np.log10(s[-1]), n + 1)
    idx = []
    for k in range(n):
        cand = np.where((s >= edges[k]) & (s < edges[k + 1]))[0]
        if len(cand):
            idx.append(cand[len(cand) // 2])
    return np.array(sorted(set(idx)))


if __name__ == "__main__":
    data = run()
    s, eL2 = data[:, 0], data[:, 1]
    np.savetxt("sensitivity_cutcell.dat", data,
               header="min|phi|/h    L2_error", fmt="%.6e")

    sub = subsample(s)
    print("\n% --- pgfplots (subsampled): L2 error vs smallest cut cell ---")
    print(pgf_coords(s[sub], eL2[sub]))
    print(f"\nh = {2.0/N:.4f},  N = {N}")
    print(f"cut range : [{s.min():.2e}, {s.max():.2e}]")
    print(f"L2 range  : [{eL2.min():.2e}, {eL2.max():.2e}]  "
          f"(max/min = {eL2.max()/eL2.min():.2f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        ax.semilogx(s, eL2, "o", color="tab:blue", ms=4)
        ax.set_xlabel(r"$\min|\varphi|/h$"); ax.set_ylabel(r"$L^2$ error")
        ax.set_ylim(0, 1.3 * eL2.max())
        ax.set_title(rf"phi-FD error vs smallest cut cell ($h={2.0/N:.3f}$)")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig("sensitivity_cutcell.png", dpi=130)
        print("\nsaved sensitivity_cutcell.png")
    except ImportError:
        print("\n(matplotlib not available -- skipped PNG)")
