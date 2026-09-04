"""Does the alpha_0 rule ALONE (uniform h^-4 scaling, no row normalization)
already make kappa independent of the cut?  The main sweep tests h4+nearest,
norm+nearest and norm+safe; this adds h4+safe, on the same 40 geometries."""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from levelset import make_phi
from phiFD_poisson_neumann_bean import solve
from sensitivity_kappa import pick_geometries, N, BOX

def masks(phi, N=N, box=BOX):
    """Omega_h (in + boundary) and dOmega_h, as flat boolean masks."""
    a, b = box
    x = np.linspace(a, b, N + 1)
    X, Y = np.meshgrid(x, x)
    ind = (phi(X, Y) < 0).astype(float)
    out = 1.0 - ind
    dx = ind[:, 1:] - ind[:, :-1]
    out[:, :-1][dx == 1] = 0
    out[:, 1:][dx == -1] = 0
    dy = ind[1:, :] - ind[:-1, :]
    out[:-1, :][dy == 1] = 0
    out[1:, :][dy == -1] = 0
    return (1 - out).ravel() > 0, ((1 - out) * (1 - ind)).ravel() > 0
radii, s = pick_geometries()
rows = []
for Rb, si in zip(radii, s):
    ph = make_phi(Rb=Rb)
    k = {}
    for tag, kw in (("h4+nearest", dict(row_scaling="h4", alpha0_rule="nearest")),
                    ("h4+safe",    dict(row_scaling="h4", alpha0_rule="safe")),
                    ("norm+safe",  dict(row_scaling="norm", alpha0_rule="safe"))):
        M = solve(N, ph, return_A=True, box=BOX, **kw)
        om, _ = masks(ph, N=N, box=BOX)
        i = np.where(om)[0]
        k[tag] = np.linalg.cond(M.toarray()[np.ix_(i, i)])
    rows.append((si, k["h4+nearest"], k["h4+safe"], k["norm+safe"]))
a = np.array(rows)
np.savetxt("alpha0_alone.dat", a, header="s kappa_h4_nearest kappa_h4_safe kappa_norm_safe")
print(f"{'':>14} {'min':>10} {'max':>10} {'max/min':>8} {'slope':>7}")
for j, lab in ((1, "h4+nearest"), (2, "h4+safe"), (3, "norm+safe")):
    v = a[:, j]; sl = np.polyfit(np.log(a[:, 0]), np.log(v), 1)[0]
    print(f"{lab:>14} {v.min():10.3e} {v.max():10.3e} {v.max()/v.min():8.2f} {sl:+7.2f}")
pg = lambda ys: "".join(f"({x:.3e},{y:.3e})" for x, y in zip(a[:,0], ys))
print("\n% h4+safe\n\\addplot[only marks,mark=square,mark size=1pt,color=green!60!black] coordinates {" + pg(a[:,2]) + "};")
