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
"""

import numpy as np
import scipy.sparse as sp
from levelset import make_phi

K = np.pi / 0.8                                   # same wavelength as elasticity


def solve(N, phi_func, alpha=1.0, box=(-1.0, 1.0), return_A=False):
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
    ind    = (phiij < 0).astype(int)
    indOut = 1 - ind

    # interior 5-point Laplacian + reaction
    D2 = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N+1, N+1)) / h**2
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

    # boundary nodes and nearest interior neighbors
    J, I = np.where(ind + indOut == 0)
    dirs = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]])
    I0 = I[None, :] + dirs[:, 1, None]
    J0 = J[None, :] + dirs[:, 0, None]
    dots = (X[J0, I0]-X[J, I])**2 + (Y[J0, I0]-Y[J, I])**2
    dots = np.where(ind[J0, I0], dots, np.inf)
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
    C = sp.coo_array((coef, (row, col)), shape=(Ndof, Ndof)).tocsr() / h**4
    rhs = rhs + rhs_bdr / h**4

    M = (A + B + C).tocsr()
    if return_A:
        return M
    u = sp.linalg.spsolve(M, rhs).reshape(N+1, N+1)

    # relative L2, H1 (semi-norm, edge-based), Linf errors over domain nodes
    w     = 1 - indOut
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
    for N in [20, 40, 80, 160, 320]:
        hs.append(2.0/N)
        a, b, c = solve(N, phi_b)
        L2.append(a); H1.append(b); Li.append(c)
        print(f"{N:5d} {a:11.3e} {b:11.3e} {c:11.3e}")
    hs = np.array(hs)
    sl = lambda e: np.polyfit(np.log(hs), np.log(e), 1)[0]
    print(f"slopes:  L2={sl(L2):.2f}  H1={sl(H1):.2f}  Linf={sl(Li):.2f}")
