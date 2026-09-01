"""
phi-FD against the ghost-point scheme of Coco & Russo (see cocorusso.py), on the
bean domain and the same manufactured solution as everywhere else.

The two methods sit in the same family -- Cartesian grid, finite differences,
geometry by a level-set, no cut-cell geometry -- and differ in how the boundary
condition reaches the grid: Coco & Russo introduce unknowns outside Omega and
write the condition at the orthogonal projection of each of them onto Gamma,
interpolating there from a 3x3 block; phi-FD writes the relaxed condition at a
boundary node and at its coupled interior node and eliminates the auxiliary
field between the two.

Three things are measured on the same grids:

  * the convergence of the solution (L2, H1, Linf);
  * the 2-norm condition number, both schemes being row-normalized the same way;
  * the robustness to the smallest cut cell, at fixed h over the 40 geometries
    of Figure fig:cond.

The p = 1 variant of the ghost-point scheme is also run, since the drop to first
order it produces for Neumann conditions is what forces the 3x3 stencil.
"""

import warnings

import numpy as np

from arias_poisson_neumann_bean import cond2
from cocorusso import errors as cr_errors
from cocorusso import solve as cr_solve
from levelset import make_phi
from phiFD_poisson_neumann_bean import solve as phifd_solve
from sensitivity_cutcell import SMIN, SMAX, pgf_coords, smallest_cut, subsample

warnings.filterwarnings("ignore")

K = np.pi / 0.8
ALPHA = 1.0
BOX = (-1.0, 1.0)

ue = lambda x, y: np.sin(K * x) * np.cos(K * y)
f = lambda x, y: (2 * K * K + ALPHA) * ue(x, y)


def gradphi(x, y):
    """Gradient of the bean level set; independent of the radius Rb."""
    s = y + 1.4 * x ** 2
    return 1.6 * x + 5.6 * x * s, 2.0 * s


def g(x, y):
    gx, gy = gradphi(x, y)
    n = np.hypot(gx, gy)
    return (K * np.cos(K * x) * np.cos(K * y) * gx
            - K * np.sin(K * x) * np.sin(K * y) * gy) / n


def cr_run(N, phi, p):
    u, X, Y, omega, inside, ghost, failed, reduced = cr_solve(
        N, phi, gradphi, f, g, ALPHA, box=BOX, p=p)
    return cr_errors(u, ue(X, Y), omega, 2.0 / N), reduced


def convergence():
    phi = make_phi()
    Ns = [20, 40, 80, 160]
    hs = np.array([2.0 / N for N in Ns])
    print(f"{'h':>8} | {'phi-FD L2':>10} {'kappa':>10} "
          f"| {'ghost p=2 L2':>13} {'kappa':>10} | {'ghost p=1 L2':>13}")
    print("-" * 76)
    P, G2, G1, KP, KG = [], [], [], [], []
    for N, h in zip(Ns, hs):
        p_e = phifd_solve(N, phi, alpha=ALPHA, box=BOX)
        p_k = cond2(phifd_solve(N, phi, alpha=ALPHA, box=BOX, return_A=True))
        g2, fail2 = cr_run(N, phi, 2)
        g1, _ = cr_run(N, phi, 1)
        g_k = cond2(cr_solve(N, phi, gradphi, f, g, ALPHA, box=BOX, p=2,
                             return_A=True))
        P.append(p_e); G2.append(g2); G1.append(g1); KP.append(p_k); KG.append(g_k)
        star = "*" if fail2 else " "
        print(f"{h:8.4f} | {p_e[0]:10.3e} {p_k:10.3e} "
              f"| {g2[0]:13.3e}{star} {g_k:10.3e} | {g1[0]:13.3e}")
    P, G2, G1 = np.array(P), np.array(G2), np.array(G1)
    sl = lambda v: np.polyfit(np.log(hs), np.log(v), 1)[0]
    print("-" * 76)
    print(f"{'order':>8} | " + " ".join(
        f"{sl(v):>10.2f}" for v in (P[:, 0], KP)) +
        " | " + " ".join(f"{sl(v):>13.2f}" for v in (G2[:, 0],)) +
        f" {sl(KG):10.2f} | {sl(G1[:, 0]):13.2f}")
    print(f"\nphi-FD    orders L2/H1/Linf: "
          f"{sl(P[:,0]):.2f} {sl(P[:,1]):.2f} {sl(P[:,2]):.2f}")
    print(f"ghost p=2 orders L2/H1/Linf: "
          f"{sl(G2[:,0]):.2f} {sl(G2[:,1]):.2f} {sl(G2[:,2]):.2f}")
    print(f"ghost p=1 orders L2/H1/Linf: "
          f"{sl(G1[:,0]):.2f} {sl(G1[:,1]):.2f} {sl(G1[:,2]):.2f}")
    return hs, P, G2, G1, np.array(KP), np.array(KG)


def cut_sweep(N=40):
    """kappa and error against the smallest cut cell, both schemes, fixed h."""
    radii = np.linspace(0.40, 0.60, 240)
    s = np.array([smallest_cut(make_phi(Rb=R), N=N, box=BOX) for R in radii])
    keep = (s >= SMIN) & (s <= SMAX)
    radii, s = radii[keep], s[keep]
    o = np.argsort(s)
    radii, s = radii[o], s[o]
    idx = subsample(s)
    radii, s = radii[idx], s[idx]
    print(f"\n{len(s)} geometries at h = {2.0/N:.4f}")
    print(f"{'min|phi|/h':>11} | {'phi-FD L2':>10} {'kappa':>10} "
          f"| {'ghost L2':>10} {'kappa':>10}")
    rows = []
    for Rb, si in zip(radii, s):
        phi = make_phi(Rb=Rb)
        ep = phifd_solve(N, phi, alpha=ALPHA, box=BOX)[0]
        kp = cond2(phifd_solve(N, phi, alpha=ALPHA, box=BOX, return_A=True))
        (eg, _, _), _ = cr_run(N, phi, 2)
        kg = cond2(cr_solve(N, phi, gradphi, f, g, ALPHA, box=BOX, p=2,
                            return_A=True))
        rows.append((si, ep, kp, eg, kg))
        print(f"{si:11.3e} | {ep:10.3e} {kp:10.3e} | {eg:10.3e} {kg:10.3e}")
    a = np.array(sorted(rows))
    print(f"\n{'':>14} {'min':>11} {'max':>11} {'max/min':>9} {'slope in s':>11}")
    for lab, v in (("phi-FD L2", a[:, 1]), ("phi-FD kappa", a[:, 2]),
                   ("ghost L2", a[:, 3]), ("ghost kappa", a[:, 4])):
        print(f"{lab:>14} {v.min():11.3e} {v.max():11.3e} {v.max()/v.min():9.2f} "
              f"{np.polyfit(np.log(a[:,0]), np.log(v), 1)[0]:+11.2f}")
    return a


if __name__ == "__main__":
    hs, P, G2, G1, KP, KG = convergence()
    a = cut_sweep()
    np.savetxt("compare_ghostpoint_conv.dat",
               np.column_stack([hs, P, G2, G1, KP, KG]),
               header="h  phiFD_L2 phiFD_H1 phiFD_Linf  g2_L2 g2_H1 g2_Linf  "
                      "g1_L2 g1_H1 g1_Linf  kappa_phiFD kappa_ghost",
               fmt="%.6e")
    np.savetxt("compare_ghostpoint_cut.dat", a,
               header="min|phi|/h  L2(phiFD) kappa(phiFD)  L2(ghost) kappa(ghost)",
               fmt="%.6e")
    print("\n% --- convergence, ghost-point p=2 ---")
    for name, col in (("L2", 0), ("H1", 1), ("Linf", 2)):
        print(f"% {name}: " + "".join(f"({h:g},{v:.6e})"
                                      for h, v in zip(hs, G2[:, col])))
    print("\n% --- kappa vs smallest cut ---")
    print("% ghost-point\n\\addplot[only marks,mark=triangle,mark size=1.5pt,"
          f"color=orange] coordinates {{{pgf_coords(a[:, 0], a[:, 4])}}};")
    print("\nsaved compare_ghostpoint_conv.dat and compare_ghostpoint_cut.dat")
