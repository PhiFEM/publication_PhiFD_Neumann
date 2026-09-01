import numpy as np
import scipy.sparse as sp
from levelset import make_phi

# 1) Parameters, exact solution, levelset
#    PDE: -(λ+2μ)∂_xx u1 - μ∂_yy u1 - (λ+μ)∂_xy u2 + α u1 = f1
#         -(λ+μ)∂_xy u1 - μ∂_xx u2 - (λ+2μ)∂_yy u2 + α u2 = f2
#    Manufactured solution (div-free): u1=cos(Kx)sin(Ky), u2=-sin(Kx)cos(Ky)
#    -> f = (2μK²+α) u
#    Mixed BC on ∂Ω = {φ=0}, outward (unnormalized) normal ∇φ:
#      Dirichlet where dirichlet_crit(x,y) is True : u = ue
#      Neumann  elsewhere                          : σ(u)·∇φ = σ(ue)·∇φ
#    The geometry ∂Ω is the zero level-set of Phi below; the scheme only uses
#    φ and its finite-difference gradient (dfx, dfy), so it is level-set
#    agnostic and needs no analytic normal.

K   = np.pi / 0.8                       # fixed wavelength (independent of geometry)


def solve(N, phi_func, lam=1.0, mu=1.0, alpha=1.0, dirichlet_crit=None,
          return_A=False, row_scaling="norm", alpha0_rule="nearest", alpha0_tol=0.1):
    """
    Solve the linear elasticity Neumann-Dirichlet mixed problem on Ω = {φ < 0}.

    Parameters
    ----------
    N              : grid size (N+1 points per side)
    phi_func       : level-set function (x, y) -> φ, vectorized
    lam, mu, alpha : elastic and reaction parameters
    dirichlet_crit : function (x, y) -> bool, True = Dirichlet node.
                     Default: x <= 0 (left half Dirichlet, right half Neumann).
    """
    if dirichlet_crit is None:
        dirichlet_crit = lambda x, y: x <= 0

    ue1 = lambda x, y:  np.cos(K*x) * np.sin(K*y)
    ue2 = lambda x, y: -np.sin(K*x) * np.cos(K*y)
    c_f = 2*mu*K*K + alpha
    f1  = lambda x, y: c_f * ue1(x, y)
    f2  = lambda x, y: c_f * ue2(x, y)

    # 2) Mesh
    x    = np.linspace(-1, 1, N+1)
    Ndof = (N+1)**2
    h    = x[1] - x[0]
    X, Y = np.meshgrid(x, x)
    phiij      = phi_func(X, Y)
    f1ij       = f1(X, Y)
    f2ij       = f2(X, Y)
    dfx = np.zeros_like(phiij); dfx[:,1:-1] = phiij[:,2:] - phiij[:,:-2]
    dfy = np.zeros_like(phiij); dfy[1:-1,:] = phiij[2:,:] - phiij[:-2,:]
    ind    = (phiij < 0).astype(float)
    indOut = 1.0 - ind

    # 3) 1D finite difference operators
    D2_1d = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(N+1, N+1)) / h**2
    D1_1d = sp.diags([-1.0, 0.0, 1.0], [-1, 0, 1], shape=(N+1, N+1)) / (2*h)

    # 4) 2D elastic stiffness (interior nodes only)
    I_N  = sp.eye(N+1)
    Lx   = sp.kron(I_N, D2_1d)
    Ly   = sp.kron(D2_1d, I_N)
    Dxy  = sp.kron(D1_1d, D1_1d)
    Ind  = sp.diags(ind.ravel())
    I_N2 = sp.eye(Ndof)

    A11 = Ind @ ((lam+2*mu)*Lx + mu*Ly        + alpha*I_N2)
    A22 = Ind @ (mu*Lx          + (lam+2*mu)*Ly + alpha*I_N2)
    A12 = Ind @ (-(lam+mu)*Dxy)
    A   = sp.bmat([[A11, A12], [A12, A22]], format='csr')

    # 5) Exterior penalization (H/V cut edges)
    dx = ind[:, 1:] - ind[:, :-1]
    indOut[:, :-1][dx ==  1] = 0
    indOut[:, 1: ][dx == -1] = 0
    dy = ind[1:, :] - ind[:-1, :]
    indOut[:-1, :][dy ==  1] = 0
    indOut[1:,  :][dy == -1] = 0

    # 5b) Diagonal exterior nodes adjacent to interior nodes (needed for Dxy stencil)
    J_int, I_int = np.where(ind == 1)
    for dj, di in [(-1,-1), (-1,1), (1,-1), (1,1)]:
        j2 = J_int + dj;  i2 = I_int + di
        valid = (j2 >= 0) & (j2 <= N) & (i2 >= 0) & (i2 <= N)
        j2v, i2v = j2[valid], i2[valid]
        mk = indOut[j2v, i2v] == 1
        indOut[j2v[mk], i2v[mk]] = 0

    B_block = sp.diags(indOut.ravel())
    B = sp.bmat([[B_block, None], [None, B_block]], format='csr')

    # 6) Boundary nodes and nearest interior neighbors
    J, I = np.where(ind + indOut == 0)
    dirs = np.array([[-1,-1],[-1, 0],[-1, 1],
                     [ 0,-1],        [ 0, 1],
                     [ 1,-1],[ 1, 0],[ 1, 1]])
    I0 = I[None,:] + dirs[:,1,None]
    J0 = J[None,:] + dirs[:,0,None]
    dots = (X[J0,I0] - X[J,I])**2 + (Y[J0,I0] - Y[J,I])**2
    dots = np.where(ind[J0, I0], dots, np.inf)
    # alpha0_rule="safe" (see phiFD_poisson_neumann_bean) is specific to the
    # Neumann relaxation and is OFF by default here: the Dirichlet relaxation
    # rests on u ~ phi*p, whose accuracy degrades directly when x_alpha0 is
    # pushed away from x_alpha (a factor 5 to 13 on the errors of the pure
    # Dirichlet and mixed cases below), whereas the Neumann one only needs
    # ||x_alpha - x_alpha0|| = O(h).
    if alpha0_rule == "safe":
        ok   = np.abs(phiij[J0, I0]) >= alpha0_tol * h
        safe = np.where(ok, dots, np.inf)
        best = np.argmin(np.where(np.isfinite(safe).any(axis=0), safe, dots), axis=0)
    else:
        best = np.argmin(dots, axis=0)
    I0b  = I0[best, np.arange(len(I))]
    J0b  = J0[best, np.arange(len(J))]

    # 7) Mixed boundary conditions
    #
    #    Dirichlet (dirichlet_crit True): phi-FD interpolation
    #      -phi_in/denom * u(i,j) + phi_b/denom * u(i0,j0) = ue(xb,yb)
    #
    #    Neumann (else): phi-FD condition σ(u)·∇φ = G
    #      phi_out*(σ·∇φ)_k(i0,j0) − phi_in*(σ·∇φ)_k(i,j) = RHS_k
    #      C_neu (2Ndof×2Ndof, scaled /h³); RHS = same stencil on ue

    rhs1 = (ind * f1ij).ravel().copy()
    rhs2 = (ind * f2ij).ravel().copy()

    ue1_flat = ue1(X, Y).ravel()
    ue2_flat = ue2(X, Y).ravel()
    ue_all   = np.concatenate([ue1_flat, ue2_flat])

    # Dirichlet: block (Ndof×Ndof), same for u1 and u2
    row_d, col_d, coef_d = [], [], []

    # Neumann: full (2Ndof×2Ndof), cross-component coupling
    row_n, col_n, coef_n = [], [], []
    rhs_bdr = np.zeros(2*Ndof)

    def AddD(eq, i, j, a):
        row_d.append(eq); col_d.append(i + (N+1)*j); coef_d.append(a)

    def AddN(eq_row, i, j, a, offset=0):
        fc = offset + i + (N+1)*j
        row_n.append(eq_row); col_n.append(fc); coef_n.append(a)
        rhs_bdr[eq_row] += a * ue_all[fc]

    def add_dx_bdf(eq_row, i, j, i0, j0, c_val, offset):
        """∂u/∂x at (i,j) with BDF2 or central, fallback to central at (i0,j0)."""
        if indOut[j,i+1]==1 and indOut[j,i-1]!=1 and indOut[j,i-2]!=1:
            AddN(eq_row,i,  j, 3*c_val,offset); AddN(eq_row,i-1,j,-4*c_val,offset); AddN(eq_row,i-2,j,c_val,offset)
        elif indOut[j,i-1]==1 and indOut[j,i+1]!=1 and indOut[j,i+2]!=1:
            AddN(eq_row,i,  j,-3*c_val,offset); AddN(eq_row,i+1,j, 4*c_val,offset); AddN(eq_row,i+2,j,-c_val,offset)
        elif indOut[j,i+1]!=1 and indOut[j,i-1]!=1:
            AddN(eq_row,i+1,j, c_val,offset); AddN(eq_row,i-1,j,-c_val,offset)
        else:
            AddN(eq_row,i0+1,j0, c_val,offset); AddN(eq_row,i0-1,j0,-c_val,offset)

    def add_dy_bdf(eq_row, i, j, i0, j0, c_val, offset):
        """∂u/∂y at (i,j) with BDF2 or central, fallback to central at (i0,j0)."""
        if indOut[j+1,i]==1 and indOut[j-1,i]!=1 and indOut[j-2,i]!=1:
            AddN(eq_row,i,j,  3*c_val,offset); AddN(eq_row,i,j-1,-4*c_val,offset); AddN(eq_row,i,j-2,c_val,offset)
        elif indOut[j-1,i]==1 and indOut[j+1,i]!=1 and indOut[j+2,i]!=1:
            AddN(eq_row,i,j, -3*c_val,offset); AddN(eq_row,i,j+1, 4*c_val,offset); AddN(eq_row,i,j+2,-c_val,offset)
        elif indOut[j+1,i]!=1 and indOut[j-1,i]!=1:
            AddN(eq_row,i,j+1, c_val,offset); AddN(eq_row,i,j-1,-c_val,offset)
        else:
            AddN(eq_row,i0,j0+1, c_val,offset); AddN(eq_row,i0,j0-1,-c_val,offset)

    for i, j, i0, j0 in zip(I, J, I0b, J0b):
        eq    = i + (N+1)*j
        phi_b = phiij[j,  i ]   # > 0
        phi_i = phiij[j0, i0]   # < 0
        denom = phi_b - phi_i
        t     = phi_b / denom
        xb    = X[j, i]  + t*(X[j0, i0] - X[j, i])
        yb    = Y[j, i]  + t*(Y[j0, i0] - Y[j, i])

        if dirichlet_crit(X[j, i], Y[j, i]):
            # ---- Dirichlet ----
            AddD(eq, i,  j,  -phi_i / denom)
            AddD(eq, i0, j0,  phi_b / denom)
            rhs1[eq] = ue1(xb, yb)
            rhs2[eq] = ue2(xb, yb)

        else:
            # ---- Neumann: σ(u)·∇φ = G ----
            phi_out, phi_in = phi_b, phi_i

            # Terms at (i0,j0): central differences
            # Component 1 (row = eq)
            c0 = phi_out * (lam+2*mu) * dfx[j0,i0]
            AddN(eq, i0+1,j0, c0,0);    AddN(eq, i0-1,j0,-c0,0)
            c0 = phi_out * lam * dfx[j0,i0]
            AddN(eq, i0,j0+1, c0,Ndof); AddN(eq, i0,j0-1,-c0,Ndof)
            c0 = phi_out * mu * dfy[j0,i0]
            AddN(eq, i0,j0+1, c0,0);    AddN(eq, i0,j0-1,-c0,0)
            c0 = phi_out * mu * dfy[j0,i0]
            AddN(eq, i0+1,j0, c0,Ndof); AddN(eq, i0-1,j0,-c0,Ndof)

            # Component 2 (row = eq+Ndof)
            c0 = phi_out * mu * dfx[j0,i0]
            AddN(eq+Ndof, i0,j0+1, c0,0);    AddN(eq+Ndof, i0,j0-1,-c0,0)
            c0 = phi_out * mu * dfx[j0,i0]
            AddN(eq+Ndof, i0+1,j0, c0,Ndof); AddN(eq+Ndof, i0-1,j0,-c0,Ndof)
            c0 = phi_out * lam * dfy[j0,i0]
            AddN(eq+Ndof, i0+1,j0, c0,0);    AddN(eq+Ndof, i0-1,j0,-c0,0)
            c0 = phi_out * (lam+2*mu) * dfy[j0,i0]
            AddN(eq+Ndof, i0,j0+1, c0,Ndof); AddN(eq+Ndof, i0,j0-1,-c0,Ndof)

            # Terms at (i,j): BDF2/central with coefficient -phi_in
            add_dx_bdf(eq, i,j,i0,j0, -phi_in*(lam+2*mu)*dfx[j,i], 0)
            add_dy_bdf(eq, i,j,i0,j0, -phi_in*lam        *dfx[j,i], Ndof)
            add_dy_bdf(eq, i,j,i0,j0, -phi_in*mu         *dfy[j,i], 0)
            add_dx_bdf(eq, i,j,i0,j0, -phi_in*mu         *dfy[j,i], Ndof)

            add_dy_bdf(eq+Ndof, i,j,i0,j0, -phi_in*mu         *dfx[j,i], 0)
            add_dx_bdf(eq+Ndof, i,j,i0,j0, -phi_in*mu         *dfx[j,i], Ndof)
            add_dx_bdf(eq+Ndof, i,j,i0,j0, -phi_in*lam        *dfy[j,i], 0)
            add_dy_bdf(eq+Ndof, i,j,i0,j0, -phi_in*(lam+2*mu)*dfy[j,i], Ndof)

    # Assemble C = C_dir (O(1)) + C_neu. Each Neumann relaxation row is
    # normalized by its own sup-norm and brought to the h^-2 magnitude of the
    # interior operator (row_scaling="norm", the default); row_scaling="h4"
    # restores the uniform h^-4 factor. Being a row scaling, neither changes
    # the solution.
    C_dir_block = sp.coo_array((coef_d, (row_d, col_d)), shape=(Ndof, Ndof)).tocsr()
    C_dir = sp.bmat([[C_dir_block, None], [None, C_dir_block]], format='csr')
    C_neu = sp.coo_array((coef_n, (row_n, col_n)), shape=(2*Ndof, 2*Ndof)).tocsr()
    scal  = np.full(2*Ndof, 1.0 / h**4)
    if row_scaling == "norm":                     # see phiFD_poisson_neumann_bean
        nrm = abs(C_neu).max(axis=1).toarray().ravel()
        nz  = nrm > 0.0                           # rows carrying a Neumann equation
        scal[nz] = 1.0 / (h**2 * nrm[nz])
    C_neu = sp.diags(scal) @ C_neu
    C = C_dir + C_neu

    rhs = np.concatenate([rhs1, rhs2]) + scal * rhs_bdr

    # 8) Linear solve
    M = (A + B + C).tocsr()
    if return_A:
        return M
    u_vec = sp.linalg.spsolve(M, rhs)
    u1h   = u_vec[:Ndof].reshape(N+1, N+1)
    u2h   = u_vec[Ndof:].reshape(N+1, N+1)

    # 9) Relative L2, H1 (semi-norm) and Linf errors over the domain nodes
    w      = 1 - indOut
    ue1ij  = ue1_flat.reshape(N+1, N+1)
    ue2ij  = ue2_flat.reshape(N+1, N+1)
    e1, e2 = u1h - ue1ij, u2h - ue2ij

    eL2 = (np.sqrt(np.sum((e1**2 + e2**2)*w))
           / np.sqrt(np.sum((ue1ij**2 + ue2ij**2)*w)))

    # H1 semi-norm : edge-based finite differences, using only edges whose two
    # endpoints are both domain nodes (avoids straddling the boundary).
    hm = (w[:, 1:] == 1) & (w[:, :-1] == 1)        # horizontal edges
    vm = (w[1:, :] == 1) & (w[:-1, :] == 1)        # vertical edges
    dxe = (e1[:, 1:]-e1[:, :-1])**2 + (e2[:, 1:]-e2[:, :-1])**2
    dye = (e1[1:, :]-e1[:-1, :])**2 + (e2[1:, :]-e2[:-1, :])**2
    dxu = (ue1ij[:, 1:]-ue1ij[:, :-1])**2 + (ue2ij[:, 1:]-ue2ij[:, :-1])**2
    dyu = (ue1ij[1:, :]-ue1ij[:-1, :])**2 + (ue2ij[1:, :]-ue2ij[:-1, :])**2
    num = np.sum(dxe[hm]) + np.sum(dye[vm])
    den = np.sum(dxu[hm]) + np.sum(dyu[vm])
    eH1 = np.sqrt(num / den)

    mask  = w > 0
    mag_e = np.sqrt(e1**2 + e2**2)
    mag_u = np.sqrt(ue1ij**2 + ue2ij**2)
    eLinf = mag_e[mask].max() / mag_u[mask].max()

    return eL2, eH1, eLinf


def convergence_table(label, phi_func, **kwargs):
    print(f"\n{label}")
    print(f"{'N':>6}  {'eL2':>12}  {'rate':>6}  {'eH1':>12}  {'eLinf':>12}")
    prev_e = None
    for N in [20, 40, 80, 160]:
        eL2, eH1, eLinf = solve(N, phi_func, **kwargs)
        rate = np.log2(prev_e / eL2) if prev_e else float('nan')
        print(f"{N:>6}  {eL2:>12.4e}  {rate:>6.2f}  {eH1:>12.4e}  {eLinf:>12.4e}")
        prev_e = eL2


if __name__ == "__main__":
    phi_bean = make_phi()

    convergence_table("Bean — mixed: x≤0 Dirichlet / x>0 Neumann  (λ=μ=α=1)",
                      phi_bean, dirichlet_crit=lambda x, y: x <= 0)
    convergence_table("Bean — pure Dirichlet (reference)",
                      phi_bean, dirichlet_crit=lambda x, y: True)
    convergence_table("Bean — pure Neumann (reference)",
                      phi_bean, dirichlet_crit=lambda x, y: False)
