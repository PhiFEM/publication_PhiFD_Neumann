"""
The ghost-point finite-difference scheme of Coco & Russo, for the Neumann
problem, reimplemented for comparison with phi-FD.

    A. Coco, G. Russo, "Finite-difference ghost-point multigrid methods on
    Cartesian grids for elliptic problems in arbitrary domains",
    J. Comput. Phys. 241 (2013) 464-501.

    -Delta u + alpha u = f   in Omega = { phi < 0 },
     du/dn = g                on Gamma = { phi = 0 }.

Method. Internal nodes (phi < 0) carry the standard 5-point Laplacian. Each
ghost node -- an outside node with at least one internal neighbour, i.e. exactly
the set dOmega_h of phi-FD -- carries instead the boundary condition, written at
the orthogonal projection B of the ghost node onto Gamma:

    n(B) . grad u(B) = g(B),   n = grad phi / |grad phi|,

where grad u(B) is obtained by differentiating a tensor-product Lagrange
interpolant of degree p built on a (p+1)x(p+1) block of nodes of Omega_h
surrounding B. With p = 1 (4-point bilinear interpolant) the scheme is second
order for Dirichlet but only *first* order for Neumann; second order for Neumann
requires p = 2, the 9-point biquadratic interpolant. The coefficients are those
of Astuto, Coco & Zerbinati (2025), eqs. (11)-(13):

    p=1   l(t)  = (1-t, t),                l'(t) = (-1, 1)/h
    p=2   l(t)  = ((1-t)(2-t)/2, t(2-t), t(t-1)/2)
          l'(t) = ((2t-3)/2, 2(1-t), (2t-1)/2)/h

Implementation notes, stated because they matter for a fair comparison:

  * the projection B is computed by Newton iterations on the analytic phi, which
    is favourable to the method -- phi-FD uses only the nodal values of phi;
  * among the admissible (p+1)x(p+1) blocks of Omega_h nodes, the stencil is the
    one minimizing the sup-norm distance from B to the block centre, in units of
    h: this keeps the evaluation inside the block whenever the geometry allows
    it. Choosing instead the block whose centre is closest in Euclidean distance
    leaves, on the non-convex bean, one ghost node per grid whose projection
    falls a full cell outside its stencil; the quadratic is then extrapolated
    over that cell and the solution is destroyed (92% error at h = 0.05);
  * the ghost rows are normalized like the boundary rows of phi-FD (each divided
    by its sup-norm, then brought to the h^-2 magnitude of the Laplacian rows),
    so that the two condition numbers are comparable. Being a row scaling this
    does not change the solution;
  * we do not reproduce the multigrid solver of the original paper, which is its
    main contribution on the efficiency side; we compare accuracy and
    conditioning, both solved by a sparse direct factorization.
"""

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

warnings.filterwarnings("ignore")


def _lagrange(t, p):
    """Values and derivatives (in units of 1/h) of the degree-p Lagrange basis."""
    if p == 1:
        return np.array([1.0 - t, t]), np.array([-1.0, 1.0])
    return (np.array([(1 - t) * (2 - t) / 2, t * (2 - t), t * (t - 1) / 2]),
            np.array([(2 * t - 3) / 2, 2 * (1 - t), (2 * t - 1) / 2]))



def _best_block(p, ic, jc, xb, yb, omega, N, a, h, reach=0):
    """Admissible (p+1)x(p+1) block of Omega_h minimizing the sup-norm distance
    from B to the block centre, in units of h. A score <= 1 means B lies inside
    the block; above that the interpolant is extrapolated."""
    best = None
    for di in range(-p - reach, 1 + reach):
        for dj in range(-p - reach, 1 + reach):
            i0, j0 = ic + di, jc + dj
            if i0 < 0 or j0 < 0 or i0 + p > N or j0 + p > N:
                continue
            if not omega[j0:j0 + p + 1, i0:i0 + p + 1].all():
                continue
            d = max(abs((xb - (a + i0 * h)) / h - p / 2),
                    abs((yb - (a + j0 * h)) / h - p / 2))
            if best is None or d < best[0]:
                best = (d, i0, j0)
    return best


def project(phi, gradphi, x, y, iters=40):
    """Orthogonal projection of (x, y) onto {phi = 0}, by Newton iterations."""
    for _ in range(iters):
        v = phi(x, y)
        gx, gy = gradphi(x, y)
        n2 = gx * gx + gy * gy
        x = x - v * gx / n2
        y = y - v * gy / n2
    return x, y


def solve(N, phi, gradphi, f, g, alpha=1.0, box=(-1.0, 1.0), p=2,
          row_scaling="norm", return_A=False, return_fields=False):
    """Coco-Russo ghost-point scheme on Omega = {phi < 0}. Returns the relative
    discrete L2, H1 and Linf errors, or the matrix / the fields."""
    a, b = box
    x = np.linspace(a, b, N + 1)
    h = x[1] - x[0]
    Ndof = (N + 1) ** 2
    X, Y = np.meshgrid(x, x)
    P = phi(X, Y)

    inside = P < 0.0
    ghost = np.zeros_like(inside)
    for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        sh = np.roll(np.roll(inside, dj, axis=0), di, axis=1)
        if dj:                                    # kill the wrap-around
            sh[0 if dj > 0 else -1, :] = False
        if di:
            sh[:, 0 if di > 0 else -1] = False
        ghost |= sh
    ghost &= ~inside
    omega = inside | ghost                        # Omega_h

    rows, cols, vals = [], [], []
    rhs = np.zeros(Ndof)
    dof = lambda i, j: i + (N + 1) * j

    # ---- interior: standard 5-point Laplacian + reaction --------------------
    J, I = np.where(inside)
    for i, j in zip(I, J):
        eq = dof(i, j)
        rows += [eq] * 5
        cols += [dof(i, j), dof(i + 1, j), dof(i - 1, j),
                 dof(i, j + 1), dof(i, j - 1)]
        vals += [4.0 / h ** 2 + alpha, -1.0 / h ** 2, -1.0 / h ** 2,
                 -1.0 / h ** 2, -1.0 / h ** 2]
        rhs[eq] = f(x[i], x[j])

    # ---- ghost nodes: the boundary condition at the projected point ---------
    Jg, Ig = np.where(ghost)
    xb, yb = project(phi, gradphi, X[Jg, Ig], Y[Jg, Ig])
    gxb, gyb = gradphi(xb, yb)
    nrm = np.hypot(gxb, gyb)
    nx, ny = gxb / nrm, gyb / nrm
    ghost_eqs, failed, reduced = [], 0, 0

    for k, (i, j) in enumerate(zip(Ig, Jg)):
        eq = dof(i, j)
        ghost_eqs.append(eq)
        # candidate (p+1)x(p+1) blocks of Omega_h nodes; keep the one whose
        # centre is closest to B, i.e. the most centred interpolation available
        ic = int(np.floor((xb[k] - a) / h))
        jc = int(np.floor((yb[k] - a) / h))
        # The block need not contain B: where the domain curves away, the
        # only admissible blocks are shifted inwards ("upwind"), and the
        # interpolant is then evaluated slightly outside its own span. We widen
        # the search until an admissible block is found.
        # Degree actually used at this node: p if a (p+1)^2 block of Omega_h
        # keeps B inside it, otherwise the largest degree that does. Where the
        # domain is concave, a ghost node can have no admissible 3x3 block at
        # all containing its projection; extrapolating the quadratic over the
        # missing cell destroys the solution (92% error at h = 0.05 on the
        # bean), whereas dropping to the bilinear stencil there costs only a
        # local order reduction at isolated nodes -- the same trade-off as the
        # boundary-isolated points of phi-FD.
        best, pk = None, p
        for pk in range(p, 0, -1):
            best = _best_block(pk, ic, jc, xb[k], yb[k], omega, N, a, h)
            if best is not None and best[0] <= 1.0:
                break
        if best is None or best[0] > 1.0:
            # last resort: the nearest bilinear block, searched in a wider
            # window. In a thin concave region no block adjacent to the cell of
            # B lies entirely in Omega_h; extrapolating a bilinear interpolant
            # over one cell is a local first-order treatment, which is what the
            # geometry allows there.
            for reach in (1, 2, 3, 4):
                cand = _best_block(1, ic, jc, xb[k], yb[k], omega, N, a, h,
                                   reach=reach)
                if cand is not None:
                    best, pk = cand, 1
                    break
        if False:
            for di in range(0):
                for dj in range(0):
                    pass
        if best is None:                          # no admissible block at all
            failed += 1
            rows.append(eq); cols.append(eq); vals.append(1.0)
            rhs[eq] = np.nan
            continue
        if pk < p:
            reduced += 1
        _, i0, j0 = best
        tx = (xb[k] - (a + i0 * h)) / h
        ty = (yb[k] - (a + j0 * h)) / h
        lx, dlx = _lagrange(tx, pk)
        ly, dly = _lagrange(ty, pk)
        for aa in range(pk + 1):
            for bb in range(pk + 1):
                c = (nx[k] * dlx[aa] * ly[bb] + ny[k] * lx[aa] * dly[bb]) / h
                rows.append(eq); cols.append(dof(i0 + aa, j0 + bb)); vals.append(c)
        rhs[eq] = g(xb[k], yb[k])

    # ---- outside: identity ---------------------------------------------------
    Jo, Io = np.where(~omega)
    for i, j in zip(Io, Jo):
        eq = dof(i, j)
        rows.append(eq); cols.append(eq); vals.append(1.0)

    M = sp.coo_matrix((vals, (rows, cols)), shape=(Ndof, Ndof)).tocsr()

    if row_scaling == "norm" and ghost_eqs:
        scal = np.ones(Ndof)
        nrm_rows = abs(M).max(axis=1).toarray().ravel()
        eqs = np.array(ghost_eqs)
        scal[eqs] = 1.0 / (h ** 2 * np.maximum(nrm_rows[eqs], 1e-300))
        M = sp.diags(scal) @ M
        rhs = scal * rhs

    if return_A:
        return M.tocsr()
    u = spsolve(M.tocsr(), rhs).reshape(N + 1, N + 1)
    if return_fields:
        return dict(u=u, X=X, Y=Y, h=h, inside=inside, ghost=ghost,
                    reduced=reduced,
                    omega=omega, failed=failed)
    return u, X, Y, omega, inside, ghost, failed, reduced


def errors(u, ue, omega, h):
    """Relative discrete L2, H1 (edge-based semi-norm) and Linf over Omega_h."""
    w = omega
    e = u - ue
    eL2 = np.sqrt(np.sum(e[w] ** 2)) / np.sqrt(np.sum(ue[w] ** 2))
    hm = w[:, 1:] & w[:, :-1]
    vm = w[1:, :] & w[:-1, :]
    num = np.sum((e[:, 1:] - e[:, :-1])[hm] ** 2) + \
          np.sum((e[1:, :] - e[:-1, :])[vm] ** 2)
    den = np.sum((ue[:, 1:] - ue[:, :-1])[hm] ** 2) + \
          np.sum((ue[1:, :] - ue[:-1, :])[vm] ** 2)
    return eL2, np.sqrt(num / den), np.abs(e[w]).max() / np.abs(ue[w]).max()
