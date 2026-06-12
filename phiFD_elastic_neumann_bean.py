"""
Linear elasticity system with pure Neumann (traction) boundary conditions on the
bean domain, solved with the phi-FD scheme.

    -div sigma(u) + alpha u = f      in Omega = { phi < 0 },
     sigma(u) n = g                  on Gamma = { phi = 0 },

with sigma(u) = lam (div u) I + 2 mu eps(u), eps(u) = (grad u + grad u^T)/2,
and the displacement u = (u1, u2). In Navier form,

    -(lam+2mu) u1_xx - mu u1_yy - (lam+mu) u2_xy + alpha u1 = f1,
    -(lam+mu) u1_xy - mu u2_xx - (lam+2mu) u2_yy + alpha u2 = f2.

Manufactured (div-free) solution u1 = cos(Kx) sin(Ky), u2 = -sin(Kx) cos(Ky),
giving f = (2 mu K^2 + alpha) u. The reaction term alpha > 0 makes the pure
Neumann problem coercive (no rigid-body kernel), so it is well posed.

Interior nodes use the standard five-point stencils for u_xx, u_yy and the
centered tensor-product stencil for the mixed derivative u_xy. At EVERY boundary
node the traction condition sigma(u).grad phi = G (with G = g|grad phi| taken
from the exact solution through the same discrete stencil) is relaxed as
sigma(u).grad phi - G = p phi; eliminating p between the boundary node (i,j) and
its closest interior node (i0,j0) gives

    phi_{i0j0}[(sigma_h(U) grad_h Phi)_k - G_k]_{ij}
  - phi_{ij}  [(sigma_h(U) grad_h Phi)_k - G_k]_{i0j0} = 0,   k = 1, 2.

This is the pure-Neumann counterpart of phiFD_elastic_mixte_levelset.py (every
node Neumann, no Dirichlet branch) and the vector analogue of
phiFD_poisson_neumann_bean.py.
"""

import numpy as np
import scipy.sparse as sp
from levelset import make_phi

K = np.pi / 0.8                         # fixed wavelength (independent of geometry)


def solve(N, phi_func, lam=1.0, mu=1.0, alpha=1.0, box=(-1.0, 1.0), return_A=False):
    """Pure Neumann linear elasticity on Omega = {phi < 0}.
    If return_A, return the assembled system matrix instead of the errors."""
    ue1 = lambda x, y:  np.cos(K*x) * np.sin(K*y)
    ue2 = lambda x, y: -np.sin(K*x) * np.cos(K*y)
    c_f = 2*mu*K*K + alpha
    f1  = lambda x, y: c_f * ue1(x, y)
    f2  = lambda x, y: c_f * ue2(x, y)

    a, b = box
    x    = np.linspace(a, b, N+1)
    Ndof = (N+1)**2
    h    = x[1] - x[0]
    X, Y = np.meshgrid(x, x)
    phiij = phi_func(X, Y)
    f1ij  = f1(X, Y)
    f2ij  = f2(X, Y)
    dfx = np.zeros_like(phiij); dfx[:, 1:-1] = phiij[:, 2:] - phiij[:, :-2]
    dfy = np.zeros_like(phiij); dfy[1:-1, :] = phiij[2:, :] - phiij[:-2, :]
    ind    = (phiij < 0).astype(int)
    indOut = 1 - ind

    # 1D finite difference operators
    D2_1d = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N+1, N+1)) / h**2
    D1_1d = sp.diags([-1, 0,  1], [-1, 0, 1], shape=(N+1, N+1)) / (2*h)

    # 2D elastic stiffness (interior nodes only)
    I_N  = sp.eye(N+1)
    Lx   = sp.kron(I_N, D2_1d)
    Ly   = sp.kron(D2_1d, I_N)
    Dxy  = sp.kron(D1_1d, D1_1d)
    Ind  = sp.diags(ind.ravel())
    I_N2 = sp.eye(Ndof)

    A11 = Ind @ ((lam+2*mu)*Lx + mu*Ly          + alpha*I_N2)
    A22 = Ind @ (mu*Lx          + (lam+2*mu)*Ly + alpha*I_N2)
    A12 = Ind @ (-(lam+mu)*Dxy)
    A   = sp.bmat([[A11, A12], [A12, A22]], format='csr')

    # exterior penalization (H/V cut edges)
    dx = ind[:, 1:] - ind[:, :-1]
    indOut[:, :-1][dx ==  1] = 0
    indOut[:, 1: ][dx == -1] = 0
    dy = ind[1:, :] - ind[:-1, :]
    indOut[:-1, :][dy ==  1] = 0
    indOut[1:,  :][dy == -1] = 0

    # diagonal exterior nodes adjacent to interior nodes (needed for Dxy stencil)
    J_int, I_int = np.where(ind == 1)
    for dj, di in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        j2 = J_int + dj;  i2 = I_int + di
        valid = (j2 >= 0) & (j2 <= N) & (i2 >= 0) & (i2 <= N)
        j2v, i2v = j2[valid], i2[valid]
        mk = indOut[j2v, i2v] == 1
        indOut[j2v[mk], i2v[mk]] = 0

    B_block = sp.diags(indOut.ravel())
    B = sp.bmat([[B_block, None], [None, B_block]], format='csr')

    # boundary nodes and nearest interior neighbors
    J, I = np.where(ind + indOut == 0)
    dirs = np.array([[-1, -1], [-1, 0], [-1, 1],
                     [ 0, -1],          [ 0, 1],
                     [ 1, -1], [ 1, 0], [ 1, 1]])
    I0 = I[None, :] + dirs[:, 1, None]
    J0 = J[None, :] + dirs[:, 0, None]
    dots = (X[J0, I0] - X[J, I])**2 + (Y[J0, I0] - Y[J, I])**2
    dots = np.where(ind[J0, I0], dots, np.inf)
    best = np.argmin(dots, axis=0)
    I0b  = I0[best, np.arange(len(I))]
    J0b  = J0[best, np.arange(len(J))]

    # pure Neumann boundary conditions: sigma(u).grad phi = G at every node
    rhs1 = (ind * f1ij).ravel().copy()
    rhs2 = (ind * f2ij).ravel().copy()
    ue1_flat = ue1(X, Y).ravel()
    ue2_flat = ue2(X, Y).ravel()
    ue_all   = np.concatenate([ue1_flat, ue2_flat])

    row_n, col_n, coef_n = [], [], []
    rhs_bdr = np.zeros(2*Ndof)

    def AddN(eq_row, i, j, a_, offset=0):
        fc = offset + i + (N+1)*j
        row_n.append(eq_row); col_n.append(fc); coef_n.append(a_)
        rhs_bdr[eq_row] += a_ * ue_all[fc]

    def add_dx_bdf(eq_row, i, j, i0, j0, c_val, offset):
        """d/dx at (i,j) with BDF2 or central, fallback to central at (i0,j0)."""
        if indOut[j, i+1] == 1 and indOut[j, i-1] != 1 and indOut[j, i-2] != 1:
            AddN(eq_row, i,   j, 3*c_val, offset); AddN(eq_row, i-1, j, -4*c_val, offset); AddN(eq_row, i-2, j, c_val, offset)
        elif indOut[j, i-1] == 1 and indOut[j, i+1] != 1 and indOut[j, i+2] != 1:
            AddN(eq_row, i,   j, -3*c_val, offset); AddN(eq_row, i+1, j, 4*c_val, offset); AddN(eq_row, i+2, j, -c_val, offset)
        elif indOut[j, i+1] != 1 and indOut[j, i-1] != 1:
            AddN(eq_row, i+1, j, c_val, offset); AddN(eq_row, i-1, j, -c_val, offset)
        else:
            AddN(eq_row, i0+1, j0, c_val, offset); AddN(eq_row, i0-1, j0, -c_val, offset)

    def add_dy_bdf(eq_row, i, j, i0, j0, c_val, offset):
        """d/dy at (i,j) with BDF2 or central, fallback to central at (i0,j0)."""
        if indOut[j+1, i] == 1 and indOut[j-1, i] != 1 and indOut[j-2, i] != 1:
            AddN(eq_row, i, j,   3*c_val, offset); AddN(eq_row, i, j-1, -4*c_val, offset); AddN(eq_row, i, j-2, c_val, offset)
        elif indOut[j-1, i] == 1 and indOut[j+1, i] != 1 and indOut[j+2, i] != 1:
            AddN(eq_row, i, j,  -3*c_val, offset); AddN(eq_row, i, j+1, 4*c_val, offset); AddN(eq_row, i, j+2, -c_val, offset)
        elif indOut[j+1, i] != 1 and indOut[j-1, i] != 1:
            AddN(eq_row, i, j+1, c_val, offset); AddN(eq_row, i, j-1, -c_val, offset)
        else:
            AddN(eq_row, i0, j0+1, c_val, offset); AddN(eq_row, i0, j0-1, -c_val, offset)

    for i, j, i0, j0 in zip(I, J, I0b, J0b):
        eq = i + (N+1)*j
        phi_out = phiij[j,  i ]     # > 0  (phi at the boundary node)
        phi_in  = phiij[j0, i0]     # < 0  (phi at the interior node)

        # Terms at (i0,j0): central differences, coefficient phi_out
        # Component 1 (row = eq) : sigma_11 dphi_x + sigma_12 dphi_y
        c0 = phi_out * (lam+2*mu) * dfx[j0, i0]
        AddN(eq, i0+1, j0, c0, 0);    AddN(eq, i0-1, j0, -c0, 0)
        c0 = phi_out * lam * dfx[j0, i0]
        AddN(eq, i0, j0+1, c0, Ndof); AddN(eq, i0, j0-1, -c0, Ndof)
        c0 = phi_out * mu * dfy[j0, i0]
        AddN(eq, i0, j0+1, c0, 0);    AddN(eq, i0, j0-1, -c0, 0)
        c0 = phi_out * mu * dfy[j0, i0]
        AddN(eq, i0+1, j0, c0, Ndof); AddN(eq, i0-1, j0, -c0, Ndof)

        # Component 2 (row = eq+Ndof) : sigma_21 dphi_x + sigma_22 dphi_y
        c0 = phi_out * mu * dfx[j0, i0]
        AddN(eq+Ndof, i0, j0+1, c0, 0);    AddN(eq+Ndof, i0, j0-1, -c0, 0)
        c0 = phi_out * mu * dfx[j0, i0]
        AddN(eq+Ndof, i0+1, j0, c0, Ndof); AddN(eq+Ndof, i0-1, j0, -c0, Ndof)
        c0 = phi_out * lam * dfy[j0, i0]
        AddN(eq+Ndof, i0+1, j0, c0, 0);    AddN(eq+Ndof, i0-1, j0, -c0, 0)
        c0 = phi_out * (lam+2*mu) * dfy[j0, i0]
        AddN(eq+Ndof, i0, j0+1, c0, Ndof); AddN(eq+Ndof, i0, j0-1, -c0, Ndof)

        # Terms at (i,j): BDF2/central with coefficient -phi_in
        add_dx_bdf(eq, i, j, i0, j0, -phi_in*(lam+2*mu)*dfx[j, i], 0)
        add_dy_bdf(eq, i, j, i0, j0, -phi_in*lam        *dfx[j, i], Ndof)
        add_dy_bdf(eq, i, j, i0, j0, -phi_in*mu         *dfy[j, i], 0)
        add_dx_bdf(eq, i, j, i0, j0, -phi_in*mu         *dfy[j, i], Ndof)

        add_dy_bdf(eq+Ndof, i, j, i0, j0, -phi_in*mu         *dfx[j, i], 0)
        add_dx_bdf(eq+Ndof, i, j, i0, j0, -phi_in*mu         *dfx[j, i], Ndof)
        add_dx_bdf(eq+Ndof, i, j, i0, j0, -phi_in*lam        *dfy[j, i], 0)
        add_dy_bdf(eq+Ndof, i, j, i0, j0, -phi_in*(lam+2*mu)*dfy[j, i], Ndof)

    # The Neumann relaxation rows are scaled by h^-4 to balance the interior
    # operator (~ h^-2); this does not change the solution.
    C = sp.coo_array((coef_n, (row_n, col_n)), shape=(2*Ndof, 2*Ndof)).tocsr() / h**4
    rhs = np.concatenate([rhs1, rhs2]) + rhs_bdr / h**4

    M = (A + B + C).tocsr()
    if return_A:
        return M
    u_vec = sp.linalg.spsolve(M, rhs)
    u1h   = u_vec[:Ndof].reshape(N+1, N+1)
    u2h   = u_vec[Ndof:].reshape(N+1, N+1)

    # relative L2, H1 (semi-norm) and Linf errors over the domain nodes
    w      = 1 - indOut
    ue1ij  = ue1_flat.reshape(N+1, N+1)
    ue2ij  = ue2_flat.reshape(N+1, N+1)
    e1, e2 = u1h - ue1ij, u2h - ue2ij

    eL2 = (np.sqrt(np.sum((e1**2 + e2**2)*w))
           / np.sqrt(np.sum((ue1ij**2 + ue2ij**2)*w)))

    hm = (w[:, 1:] == 1) & (w[:, :-1] == 1)
    vm = (w[1:, :] == 1) & (w[:-1, :] == 1)
    dxe = (e1[:, 1:]-e1[:, :-1])**2 + (e2[:, 1:]-e2[:, :-1])**2
    dye = (e1[1:, :]-e1[:-1, :])**2 + (e2[1:, :]-e2[:-1, :])**2
    dxu = (ue1ij[:, 1:]-ue1ij[:, :-1])**2 + (ue2ij[:, 1:]-ue2ij[:, :-1])**2
    dyu = (ue1ij[1:, :]-ue1ij[:-1, :])**2 + (ue2ij[1:, :]-ue2ij[:-1, :])**2
    eH1 = np.sqrt((np.sum(dxe[hm]) + np.sum(dye[vm]))
                  / (np.sum(dxu[hm]) + np.sum(dyu[vm])))

    mask  = w > 0
    mag_e = np.sqrt(e1**2 + e2**2)
    mag_u = np.sqrt(ue1ij**2 + ue2ij**2)
    eLinf = mag_e[mask].max() / mag_u[mask].max()

    return eL2, eH1, eLinf


if __name__ == "__main__":
    phi_bean = make_phi()
    hs, L2, H1, Li = [], [], [], []
    print("Bean -- pure Neumann elasticity  (lam=mu=alpha=1)")
    print(f"{'N':>5} {'h':>9} {'eL2':>12} {'rate':>6} {'eH1':>12} {'eLinf':>12}")
    prev = None
    for N in [20, 40, 80, 160, 320]:
        h = 2.0/N
        eL2, eH1, eLinf = solve(N, phi_bean)
        rate = np.log2(prev/eL2) if prev else float('nan')
        print(f"{N:5d} {h:9.5f} {eL2:12.4e} {rate:6.2f} {eH1:12.4e} {eLinf:12.4e}")
        hs.append(h); L2.append(eL2); H1.append(eH1); Li.append(eLinf); prev = eL2
    hs = np.array(hs)
    sl = lambda e: np.polyfit(np.log(hs), np.log(e), 1)[0]
    print(f"slopes:  L2={sl(L2):.2f}  H1={sl(H1):.2f}  Linf={sl(Li):.2f}")
    print("\n% pgfplots coordinates (L2, H1, Linf):")
    for name, e in [("L2", L2), ("H1", H1), ("Linf", Li)]:
        print(f"% {name}: " + "".join(f"({hh:.4g},{ee:.6e})" for hh, ee in zip(hs, e)))
