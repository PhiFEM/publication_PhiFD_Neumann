"""
Arias et al. (2018) cut-cell finite-volume scheme on the SAME test case as
phiFD_poisson_neumann_bean.py, for a head-to-head comparison.

    V. Arias, D. Bochkov, F. Gibou, "Poisson equations in irregular domains with
    Robin boundary conditions -- Solver with second-order accurate gradients",
    J. Comput. Phys. 365 (2018) 1-6.

Problem (identical to the phi-FD Poisson test):

    -Delta u + alpha u = f      in  Omega = { phi < 0 }   (bean domain),
     du/dn = g                  on  Gamma = { phi = 0 }    (pure Neumann),

with alpha = 1, u = sin(Kx) cos(Ky), K = pi/0.8.

The cut-cell geometry (wetted edge lengths, cell areas, interface segments) is
reused from arias2018.py. Compared to the Robin solver there, two changes make
the problem coincide with the phi-FD test:
  * the PDE carries a reaction term  +alpha*u  ->  diagonal contribution
    alpha*Area_ij. This also removes the constant null space, so the pure-Neumann
    system is non-singular (no need to pin a cell, exactly as in phi-FD);
  * the boundary condition is pure Neumann, i.e. the Robin coefficient is 0, so
    the only interface contribution is the imposed flux  int_Gamma g.

We report the relative L2 error and the 2-norm condition number kappa(h), to be
compared with the phi-FD scheme on the same bean and the same h.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu

from arias2018 import wetted, cell_cut
from levelset import make_phi

K = np.pi / 0.8                                   # same wavelength as phi-FD


# ----------------------------------------------------------------------------
# Cut-cell FV solver for  -Delta u + alpha u = f ,  du/dn = g  on {phi<0}
# ----------------------------------------------------------------------------
def solve(N, phi, f, g, alpha=1.0, box=(-1.0, 1.0)):
    """Cell-centered cut-cell FV solver on an N x N grid (h = 2/N).
    Returns (xc, u, area, active, A): cell centers, solution, cell areas,
    active-cell mask and the assembled system matrix."""
    a, b = box
    h = (b - a) / N
    xv = a + h * np.arange(N + 1)                 # vertices
    xc = a + h * (np.arange(N) + 0.5)             # cell centers

    Xv, Yv = np.meshgrid(xv, xv, indexing="ij")
    Phi = phi(Xv, Yv)
    eps = 1e-6 * h                                # article's anti micro-cell shift
    Phi[np.abs(Phi) < eps] = eps

    # cut-cell geometry: area, interface length and midpoint of each cell
    area = np.zeros((N, N)); seglen = np.zeros((N, N))
    segmid = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            corners = [(xv[i], xv[j]), (xv[i + 1], xv[j]),
                       (xv[i + 1], xv[j + 1]), (xv[i], xv[j + 1])]
            pc = [Phi[i, j], Phi[i + 1, j], Phi[i + 1, j + 1], Phi[i, j + 1]]
            area[i, j], seglen[i, j], segmid[i, j] = cell_cut(corners, pc)
    active = area > 0.0

    def dof(i, j): return i + N * j
    inv = 1.0 / h
    rows, cols, vals = [], [], []
    rhs = np.zeros(N * N)

    def combine(terms):
        d = {}
        for (i, j), c in terms:
            d[(i, j)] = d.get((i, j), 0.0) + c
        return d

    def scale(d, s): return {k: s * v for k, v in d.items()}

    def plus(d1, d2):
        d = dict(d1)
        for k, v in d2.items():
            d[k] = d.get(k, 0.0) + v
        return d

    def ddx_vface(i, j, ycut):
        """d u/dx at the cut point (height ycut) of the vertical edge between
        cells (i,j) and (i+1,j); linear interpolation across rows (eq. (5)/(6))."""
        base = combine([((i + 1, j), inv), ((i, j), -inv)])
        yj = xc[j]
        if ycut >= yj and j + 1 < N and active[i, j + 1] and active[i + 1, j + 1]:
            w = (ycut - yj) / h
            up = combine([((i + 1, j + 1), inv), ((i, j + 1), -inv)])
            return plus(scale(base, 1 - w), scale(up, w))
        if ycut < yj and j - 1 >= 0 and active[i, j - 1] and active[i + 1, j - 1]:
            w = (yj - ycut) / h
            dn = combine([((i + 1, j - 1), inv), ((i, j - 1), -inv)])
            return plus(scale(base, 1 - w), scale(dn, w))
        return base

    def ddy_hface(i, j, xcut):
        """d u/dy at the cut point (abscissa xcut) of the horizontal edge
        between cells (i,j) and (i,j+1)."""
        base = combine([((i, j + 1), inv), ((i, j), -inv)])
        xi = xc[i]
        if xcut >= xi and i + 1 < N and active[i + 1, j] and active[i + 1, j + 1]:
            w = (xcut - xi) / h
            rt = combine([((i + 1, j + 1), inv), ((i + 1, j), -inv)])
            return plus(scale(base, 1 - w), scale(rt, w))
        if xcut < xi and i - 1 >= 0 and active[i - 1, j] and active[i - 1, j + 1]:
            w = (xi - xcut) / h
            lf = combine([((i - 1, j + 1), inv), ((i - 1, j), -inv)])
            return plus(scale(base, 1 - w), scale(lf, w))
        return base

    for i in range(N):
        for j in range(N):
            eq = dof(i, j)
            if not active[i, j]:
                rows.append(eq); cols.append(eq); vals.append(1.0)   # u = 0
                continue

            terms = {}   # {(i,j): coef} for the left-hand side

            # fluxes through the 4 (possibly cut) grid edges: -sum signe*du/dn*L
            if i + 1 < N:
                L, ycut = wetted(Phi[i + 1, j], Phi[i + 1, j + 1], xv[j], xv[j + 1])
                if L > 0: terms = plus(terms, scale(ddx_vface(i, j, ycut), -L))
            L, ycut = wetted(Phi[i, j], Phi[i, j + 1], xv[j], xv[j + 1])
            if L > 0: terms = plus(terms, scale(ddx_vface(i - 1, j, ycut), +L))
            if j + 1 < N:
                L, xcut = wetted(Phi[i, j + 1], Phi[i + 1, j + 1], xv[i], xv[i + 1])
                if L > 0: terms = plus(terms, scale(ddy_hface(i, j, xcut), -L))
            L, xcut = wetted(Phi[i, j], Phi[i + 1, j], xv[i], xv[i + 1])
            if L > 0: terms = plus(terms, scale(ddy_hface(i, j - 1, xcut), +L))

            # reaction  +alpha int_{C cap Omega} u  ~  alpha*Area*u_ij
            terms[(i, j)] = terms.get((i, j), 0.0) + alpha * area[i, j]

            # right-hand side: Area*f  +  pure-Neumann interface flux int_Gamma g
            rhs[eq] = area[i, j] * f(xc[i], xc[j])
            if seglen[i, j] > 0:
                mx, my = segmid[i, j]
                rhs[eq] += g(mx, my) * seglen[i, j]

            for (ii, jj), c in terms.items():
                rows.append(eq); cols.append(dof(ii, jj)); vals.append(c)

    A = sp.coo_matrix((vals, (rows, cols)), shape=(N * N, N * N)).tocsr()
    u = spsolve(A, rhs).reshape((N, N), order="F")
    return xc, u, area, active, A


# ----------------------------------------------------------------------------
# Relative L2 error (area-weighted) and 2-norm condition number
# ----------------------------------------------------------------------------
def rel_L2(xc, u, area, active, uex):
    """Area-weighted relative L2 error over the active cut cells."""
    Xc, Yc = np.meshgrid(xc, xc, indexing="ij")
    ue = uex(Xc, Yc)
    e = u - ue
    return np.sqrt(np.sum(area[active] * e[active] ** 2) /
                   np.sum(area[active] * ue[active] ** 2))


def cond2(A, iters=400, tol=1e-9):
    """2-norm condition number sigma_max/sigma_min via power iteration on A^T A
    and on (A^T A)^{-1} (sparse LU of A). Works for nonsymmetric A."""
    A = A.tocsc()
    n = A.shape[0]
    rng = np.random.default_rng(0)

    # sigma_max^2 = lambda_max(A^T A)
    v = rng.standard_normal(n); v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = A.T @ (A @ v)
        nw = np.linalg.norm(w)
        if abs(nw - lam) <= tol * nw:
            lam = nw; break
        lam = nw; v = w / nw
    smax = np.sqrt(lam)

    # sigma_min^2 = 1/lambda_max((A^T A)^{-1}),  (A^T A)^{-1} v = A^{-1}(A^{-T} v)
    LU = splu(A)
    v = rng.standard_normal(n); v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = LU.solve(LU.solve(v, trans='T'))
        nw = np.linalg.norm(w)
        if abs(nw - lam) <= tol * nw:
            lam = nw; break
        lam = nw; v = w / nw
    smin = 1.0 / np.sqrt(lam)
    return smax / smin


# ----------------------------------------------------------------------------
# Comparison phi-FD vs Arias on the bean, same manufactured data and same h
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from phiFD_poisson_neumann_bean import solve as phifd_solve

    phi = make_phi()                                   # bean (incl. -1e-10 shift)
    alpha = 1.0
    uex = lambda x, y: np.sin(K * x) * np.cos(K * y)
    f = lambda x, y: (2 * K * K + alpha) * uex(x, y)

    def g(x, y):                                        # du/dn = grad u . grad phi/|.|
        s = y + 1.4 * x ** 2
        phix = 1.6 * x + 5.6 * x * s
        phiy = 2.0 * s
        nn = np.hypot(phix, phiy)
        nx, ny = phix / nn, phiy / nn
        ux = K * np.cos(K * x) * np.cos(K * y)
        uy = -K * np.sin(K * x) * np.sin(K * y)
        return ux * nx + uy * ny

    Ns = [20, 40, 80, 160]                             # h = 2/N = 0.1, 0.05, ...
    print(f"{'h':>8} | {'phiFD L2':>11} {'phiFD kappa':>12} "
          f"| {'Arias L2':>11} {'Arias kappa':>12}")
    print("-" * 64)
    rows = []
    for N in Ns:
        h = 2.0 / N
        # phi-FD (separate module: errors and matrix come from two calls)
        eL2_phi = phifd_solve(N, phi, alpha=alpha)[0]
        kap_phi = cond2(phifd_solve(N, phi, alpha=alpha, return_A=True))
        # Arias cut-cell (single solve gives both the solution and the matrix)
        xc, u, area, active, A_ar = solve(N, phi, f, g, alpha)
        eL2_ar = rel_L2(xc, u, area, active, uex)
        kap_ar = cond2(A_ar)
        rows.append((h, eL2_phi, kap_phi, eL2_ar, kap_ar))
        print(f"{h:8.4f} | {eL2_phi:11.3e} {kap_phi:12.3e} "
              f"| {eL2_ar:11.3e} {kap_ar:12.3e}")

    hs = np.array([r[0] for r in rows])
    sl = lambda col: np.polyfit(np.log(hs), np.log([r[col] for r in rows]), 1)[0]
    print("-" * 64)
    print(f"order(L2):   phiFD = {sl(1):.2f}   Arias = {sl(3):.2f}")
    print(f"order(kappa): phiFD = {sl(2):.2f}   Arias = {sl(4):.2f}  "
          f"(O(h^-2) -> slope -2)")
