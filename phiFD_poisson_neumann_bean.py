"""
Scalar Poisson problem with (non-homogeneous) pure Neumann boundary conditions
on the bean domain, solved with the phi-FD scheme.

    -Delta u + alpha u = f   in Omega = { phi < 0 },
     du/dn = g                on Gamma = { phi = 0 }.

Interior nodes use the standard 5-point Laplacian. At each boundary node the
Neumann condition is relaxed as  grad u . grad phi = G + p phi  (with
G = g|grad phi| obtained from the exact solution through the same discrete
stencil); eliminating p between the boundary node (i,j) and its closest interior
node (i0,j0) gives, as in the elasticity case,

    phi_{i0j0}[(grad_h U . grad_h Phi)_{ij}   - G_{ij}]
  - phi_{ij}  [(grad_h U . grad_h Phi)_{i0j0} - G_{i0j0}] = 0.

This is the scalar analogue of phiFD_elastic_mixte_levelset.py.

Two options of `solve` govern the conditioning without changing the accuracy:

    row_scaling="norm"   normalize each relaxation equation by its own sup-norm
                         rather than scaling all of them by h^-4. A row scaling,
                         so the discrete solution is unchanged.
    alpha0_rule="safe"   never couple a boundary node to an interior node lying
                         essentially on Gamma (|phi_alpha0| < alpha0_tol * h).
                         Such a node kills the second bracket of the relaxation,
                         and two boundary nodes sharing it then carry two
                         proportional equations.

With both (the defaults) the 2-norm condition number stays within a factor 1.1
over three decades of min|phi|/h, against 82 with row_scaling="h4" and
alpha0_rule="nearest"; see sensitivity_kappa.py. The ghost-penalty term of the
Dirichlet phi-FD paper is also available (sigma > 0, see _stabilization) but is
not needed once alpha0 is chosen this way.
"""

import numpy as np
import scipy.sparse as sp
from levelset import make_phi

K = np.pi / 0.8                                   # same wavelength as elasticity


def _stab_1d(N, order):
    """1D raw finite-difference operator used by the ghost-penalty stabilization.

    order=1: second difference  [-1, 2, -1]  on nodes (i-1, i, i+1)
             -> jump of the two one-sided first-order approximations of d/dx,
                O(h) on a smooth function (basic phi-FD stabilization j_h).
    order=2: third difference   [-1, 3, -3, 1] on nodes (i-1, i, i+1, i+2)
             -> jump between the centered and the one-sided *second order*
                approximations of d/dx, hence O(h^2) on a smooth function.
                This is the term (21) of the phi-FD Dirichlet paper (scheme
                "phi-FD2"), the one giving optimal H1 convergence.
    Rows for which the stencil does not fit in the grid are left empty; the
    activation mask below discards them anyway.
    """
    stencil = (-1.0, 2.0, -1.0) if order == 1 else (-1.0, 3.0, -3.0, 1.0)
    shift   = (-1, 0, 1) if order == 1 else (-1, 0, 1, 2)
    lo, hi  = -min(shift), N - max(shift)          # rows where the stencil fits
    rows = np.arange(lo, hi + 1)
    r, c, v = [], [], []
    for sh, st in zip(shift, stencil):
        r.append(rows); c.append(rows + sh); v.append(np.full(rows.size, st))
    return sp.coo_array((np.concatenate(v), (np.concatenate(r), np.concatenate(c))),
                        shape=(N + 1, N + 1)).tocsr()


def _stabilization(N, h, w, ind, sigma, order):
    """Ghost-penalty stabilization of phi-FD, adapted to the Neumann setting.

        j_h(u,v) = sigma * sum_{active stencils} (T u / h) * (T v / h)

    i.e. the matrix  (sigma/h^2) * T^T diag(mask) T,  assembled separately in x
    and y. A stencil is active when all its nodes lie in Omega_h (w=1) and at
    least one of them lies outside Omega (ind=0, i.e. in the discrete boundary
    dOmega_h) -- the index set J of the phi-FD paper.
    """
    T1  = _stab_1d(N, order)
    Tx  = sp.kron(sp.eye(N + 1), T1)               # flat index = i + (N+1)*j
    Ty  = sp.kron(T1, sp.eye(N + 1))
    shift = (-1, 0, 1) if order == 1 else (-1, 0, 1, 2)

    def mask(axis):
        """1 where the stencil at (i,j) is active, along x (axis=1) or y (axis=0)."""
        inside_all = np.ones_like(w, dtype=bool)   # all stencil nodes in Omega_h
        touches_bd = np.zeros_like(w, dtype=bool)  # at least one outside Omega
        for sh in shift:
            inside_all &= np.roll(w, -sh, axis=axis).astype(bool)
            touches_bd |= (np.roll(w, -sh, axis=axis) - np.roll(ind, -sh, axis=axis)) > 0
        m = (inside_all & touches_bd).astype(float)
        # kill the wrapped-around rows/columns introduced by np.roll
        lo, hi = -min(shift), N - max(shift)
        if axis == 1:
            m[:, :lo] = 0.0; m[:, hi + 1:] = 0.0
        else:
            m[:lo, :] = 0.0; m[hi + 1:, :] = 0.0
        return sp.diags(m.ravel())

    return (sigma / h**2) * (Tx.T @ mask(1) @ Tx + Ty.T @ mask(0) @ Ty)



def grad_h(v, ind, indOut, a0, h):
    """The discrete gradient of the scheme, formula (3) of the article.

    Centered where both neighbours belong to Omega_h; second-order one-sided
    where only one side is available; and, at a boundary-isolated point (no
    usable stencil in that direction), the derivative of its coupled interior
    node x_{alpha_0}. Returns (dx, dy) on the whole grid, meaningful on Omega_h.
    """
    N = v.shape[0] - 1
    w = 1 - indOut                                # Omega_h
    out = [np.zeros_like(v), np.zeros_like(v)]
    for axis in (1, 0):                           # 1 = x (index i), 0 = y (j)
        d = out[1 - axis]
        for j in range(1, N):
            for i in range(1, N):
                if not w[j, i]:
                    continue
                p1 = (j, i+1) if axis == 1 else (j+1, i)
                m1 = (j, i-1) if axis == 1 else (j-1, i)
                p2 = (j, i+2) if axis == 1 else (j+2, i)
                m2 = (j, i-2) if axis == 1 else (j-2, i)
                ok = lambda t: 0 <= t[0] <= N and 0 <= t[1] <= N and w[t]
                if ok(p1) and ok(m1):
                    d[j, i] = (v[p1] - v[m1]) / (2*h)
                elif ok(p1) and ok(p2):
                    d[j, i] = (-3*v[j, i] + 4*v[p1] - v[p2]) / (2*h)
                elif ok(m1) and ok(m2):
                    d[j, i] = (3*v[j, i] - 4*v[m1] + v[m2]) / (2*h)
                else:                             # boundary-isolated point
                    i0, j0 = a0[j, i]
                    if i0 < 0:
                        continue
                    q1 = (j0, i0+1) if axis == 1 else (j0+1, i0)
                    n1 = (j0, i0-1) if axis == 1 else (j0-1, i0)
                    if ok(q1) and ok(n1):
                        d[j, i] = (v[q1] - v[n1]) / (2*h)
    return out[0], out[1]


def solve(N, phi_func, alpha=1.0, box=(-1.0, 1.0), return_A=False,
          row_scaling="norm", alpha0_rule="safe", alpha0_tol=0.1,
          sigma=0.0, stab_order=2, stab_rows="all", return_fields=False):
    """Pure Neumann scalar Poisson on Omega={phi<0}.
    If return_A, return the assembled system matrix (A+B+C) instead of the errors.
    Omega_h uses the 4-neighborhood (the 5-point Laplacian only needs the
    horizontal/vertical neighbours); this keeps the system well-conditioned."""
    ue = lambda x, y: np.sin(K*x) * np.cos(K*y)
    f  = lambda x, y: (2*K*K + alpha) * ue(x, y)  # -Delta u + alpha u
    a, b = box
    x    = np.linspace(a, b, N+1)
    h    = x[1] - x[0]
    Ndof = (N+1)**2
    X, Y = np.meshgrid(x, x)
    phiij = phi_func(X, Y)
    fij   = f(X, Y)
    dfx = np.zeros_like(phiij); dfx[:, 1:-1] = phiij[:, 2:] - phiij[:, :-2]
    dfy = np.zeros_like(phiij); dfy[1:-1, :] = phiij[2:, :] - phiij[:-2, :]
    ind    = (phiij < 0).astype(float)
    indOut = 1.0 - ind

    # interior 5-point Laplacian + reaction
    D2 = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(N+1, N+1)) / h**2
    L  = sp.kron(sp.eye(N+1), D2) + sp.kron(D2, sp.eye(N+1))
    A  = sp.diags(ind.ravel()) @ (L + alpha*sp.eye(Ndof))

    # exterior penalization (cut edges)
    dx = ind[:, 1:] - ind[:, :-1]
    indOut[:, :-1][dx ==  1] = 0
    indOut[:, 1: ][dx == -1] = 0
    dy = ind[1:, :] - ind[:-1, :]
    indOut[:-1, :][dy ==  1] = 0
    indOut[1:,  :][dy == -1] = 0
    B = sp.diags(indOut.ravel())
    w_h = 1 - indOut                              # mask of Omega_h (in + boundary)

    # boundary nodes and nearest interior neighbors
    J, I = np.where(ind + indOut == 0)
    dirs = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]])
    I0 = I[None, :] + dirs[:, 1, None]
    J0 = J[None, :] + dirs[:, 0, None]
    dots = (X[J0, I0]-X[J, I])**2 + (Y[J0, I0]-Y[J, I])**2
    dots = np.where(ind[J0, I0], dots, np.inf)
    if alpha0_rule == "safe":
        # Skip interior candidates sitting essentially on Gamma. Such a node has
        # phi_{alpha0} ~ 0, which kills the second bracket of the relaxation; if
        # it is the closest interior node of two boundary nodes, their two
        # equations collapse onto the same vector and the system is nearly
        # singular. Preferring a slightly farther candidate keeps
        # ||x_alpha - x_alpha0|| = O(h), hence the consistency estimate.
        ok = np.abs(phiij[J0, I0]) >= alpha0_tol * h
        safe = np.where(ok, dots, np.inf)
        best = np.argmin(np.where(np.isfinite(safe).any(axis=0), safe, dots), axis=0)
    else:
        best = np.argmin(dots, axis=0)
    I0b  = I0[best, np.arange(len(I))]
    J0b  = J0[best, np.arange(len(J))]

    rhs     = (ind * fij).ravel().copy()
    ue_flat = ue(X, Y).ravel()
    row, col, coef = [], [], []
    rhs_bdr = np.zeros(Ndof)

    def Add(eq, i, j, a_):
        fc = i + (N+1)*j
        row.append(eq); col.append(fc); coef.append(a_)
        rhs_bdr[eq] += a_ * ue_flat[fc]

    def ddx_bdf(eq, i, j, i0, j0, c):
        if   indOut[j, i+1]==1 and indOut[j, i-1]!=1 and indOut[j, i-2]!=1:
            Add(eq, i, j, 3*c); Add(eq, i-1, j, -4*c); Add(eq, i-2, j, c)
        elif indOut[j, i-1]==1 and indOut[j, i+1]!=1 and indOut[j, i+2]!=1:
            Add(eq, i, j, -3*c); Add(eq, i+1, j, 4*c); Add(eq, i+2, j, -c)
        elif indOut[j, i+1]!=1 and indOut[j, i-1]!=1:
            Add(eq, i+1, j, c); Add(eq, i-1, j, -c)
        else:
            Add(eq, i0+1, j0, c); Add(eq, i0-1, j0, -c)

    def ddy_bdf(eq, i, j, i0, j0, c):
        if   indOut[j+1, i]==1 and indOut[j-1, i]!=1 and indOut[j-2, i]!=1:
            Add(eq, i, j, 3*c); Add(eq, i, j-1, -4*c); Add(eq, i, j-2, c)
        elif indOut[j-1, i]==1 and indOut[j+1, i]!=1 and indOut[j+2, i]!=1:
            Add(eq, i, j, -3*c); Add(eq, i, j+1, 4*c); Add(eq, i, j+2, -c)
        elif indOut[j+1, i]!=1 and indOut[j-1, i]!=1:
            Add(eq, i, j+1, c); Add(eq, i, j-1, -c)
        else:
            Add(eq, i0, j0+1, c); Add(eq, i0, j0-1, -c)

    for i, j, i0, j0 in zip(I, J, I0b, J0b):
        eq    = i + (N+1)*j
        phi_b = phiij[j,  i ]
        phi_i = phiij[j0, i0]
        # Neumann relaxation: phi_out*(grad u.grad phi)_{i0j0} - phi_in*(...)_{ij}
        phi_out, phi_in = phi_b, phi_i
        c = phi_out * dfx[j0, i0]; Add(eq, i0+1, j0, c); Add(eq, i0-1, j0, -c)
        c = phi_out * dfy[j0, i0]; Add(eq, i0, j0+1, c); Add(eq, i0, j0-1, -c)
        ddx_bdf(eq, i, j, i0, j0, -phi_in * dfx[j, i])
        ddy_bdf(eq, i, j, i0, j0, -phi_in * dfy[j, i])

    # The Neumann relaxation rows are scaled by h^-4 so that they balance the
    # interior Laplacian (~ h^-2). This makes the whole system well-conditioned,
    # kappa = O(h^-2), without any preconditioner. The scaling does not change
    # the solution (the boundary rows are imposed as LHS = RHS).
    C = sp.coo_array((coef, (row, col)), shape=(Ndof, Ndof)).tocsr()
    scal = np.full(Ndof, 1.0 / h**4)              # generic magnitude |phi| ~ h
    if row_scaling == "norm":
        # Normalize each relaxation equation by its own sup-norm and bring it to
        # the h^-2 magnitude of the Laplacian rows. A row whose phi_alpha and
        # phi_{alpha_0} are both << h would otherwise carry an artificially
        # small weight and drive the condition number up. Being a row scaling,
        # this leaves the discrete solution unchanged.
        nrm  = abs(C).max(axis=1).toarray().ravel()
        eqs  = I + (N+1)*J
        scal[eqs] = 1.0 / (h**2 * np.maximum(nrm[eqs], 1e-300))
    C   = sp.diags(scal) @ C
    rhs = rhs + scal * rhs_bdr

    # Ghost-penalty stabilization (phi-FD Dirichlet paper, term (21) for
    # stab_order=2). Added to the left-hand side only: it is weakly consistent
    # (O(h^2) on the exact solution for stab_order=2), so the right-hand side is
    # left untouched. stab_rows="interior" restricts it to the Laplacian rows,
    # leaving the Neumann relaxation rows exactly as they are.
    if sigma > 0.0:
        S = _stabilization(N, h, w_h, ind, sigma, stab_order)
        if stab_rows == "interior":
            S = sp.diags(ind.ravel()) @ S
        M = (A + B + C + S).tocsr()
    else:
        M = (A + B + C).tocsr()
    if return_A:
        return M
    u = sp.linalg.spsolve(M, rhs).reshape(N+1, N+1)
    if return_fields:
        # everything a post-processing needs: the solution, the grid, the two
        # masks and the coupled-node map alpha -> alpha_0 (see grad_h below).
        a0 = np.full((N+1, N+1, 2), -1, dtype=int)
        a0[J, I, 0], a0[J, I, 1] = I0b, J0b
        return dict(u=u, X=X, Y=Y, h=h, ind=ind, indOut=indOut, a0=a0,
                    ue=ue(X, Y))

    # relative L2, H1 (semi-norm, edge-based), Linf errors over domain nodes
    w     = w_h
    ueij  = ue_flat.reshape(N+1, N+1)
    e     = u - ueij
    eL2   = np.sqrt(np.sum(e*e*w)) / np.sqrt(np.sum(ueij*ueij*w))
    hm = (w[:, 1:] == 1) & (w[:, :-1] == 1)
    vm = (w[1:, :] == 1) & (w[:-1, :] == 1)
    num = np.sum((e[:, 1:]-e[:, :-1])[hm]**2) + np.sum((e[1:, :]-e[:-1, :])[vm]**2)
    den = np.sum((ueij[:, 1:]-ueij[:, :-1])[hm]**2) + np.sum((ueij[1:, :]-ueij[:-1, :])[vm]**2)
    eH1 = np.sqrt(num / den)
    mask = w > 0
    eLinf = np.max(np.abs(e)[mask]) / np.max(np.abs(ueij)[mask])
    return eL2, eH1, eLinf


if __name__ == "__main__":
    phi_b = make_phi()                            # bean
    hs, L2, H1, Li = [], [], [], []
    print(f"{'N':>5} {'eL2':>11} {'eH1':>11} {'eLinf':>11}")
    SIGMA = 0.0                                   # 1e-2 to switch the (optional)
    for N in [20, 40, 80, 160, 320]:              # ghost penalty on
        hs.append(2.0/N)
        a, b, c = solve(N, phi_b, sigma=SIGMA)
        L2.append(a); H1.append(b); Li.append(c)
        print(f"{N:5d} {a:11.3e} {b:11.3e} {c:11.3e}")
    hs = np.array(hs)
    sl = lambda e: np.polyfit(np.log(hs), np.log(e), 1)[0]
    print(f"slopes:  L2={sl(L2):.2f}  H1={sl(H1):.2f}  Linf={sl(Li):.2f}")
