"""
Cost of not meshing: phi-FD against a boundary-fitted P1 finite element method
on the same problem.

Convergence curves against h say nothing about what an immersed method is for.
The argument is a cost argument, so the comparison has to be error against total
CPU time, mesh generation included -- that is where a Cartesian grid, which
costs nothing to build, is supposed to pay for the accuracy it gives up near the
boundary.

The fitted reference is a plain P1 method on a conforming triangulation of
Omega = {phi < 0}:

  * boundary vertices are placed at roughly equal arc length along {phi = 0},
    each obtained by Newton projection of a coarse sample;
  * interior vertices sit on a triangular lattice of spacing hmesh, keeping
    those far enough from the boundary;
  * the triangulation is the Delaunay triangulation of the union, restricted to
    the triangles whose centroid lies in Omega;
  * -Delta u + alpha u = f with du/dn = g is assembled in the usual way, the
    Neumann datum entering as a boundary integral over the exterior edges,
    integrated by two-point Gauss.

Both solvers are timed end to end, from nothing to the solution: for the fitted
method that includes generating the mesh, for phi-FD it includes evaluating the
level set on the grid. Both errors are relative L2, the FEM one computed with
its mass matrix, phi-FD's the discrete l2 over the nodes of Omega_h; they are
not the same norm, but both are relative and the difference between them is far
below the effects at stake here.
"""

import time
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay

from levelset import make_phi
from phiFD_poisson_neumann_bean import solve as phifd_solve

warnings.filterwarnings("ignore")

K = np.pi / 0.8
ALPHA = 1.0
ue = lambda x, y: np.sin(K * x) * np.cos(K * y)
f = lambda x, y: (2 * K * K + ALPHA) * ue(x, y)


def gradphi(x, y):
    s = y + 1.4 * x ** 2
    return 1.6 * x + 5.6 * x * s, 2.0 * s


def g_of(x, y):
    gx, gy = gradphi(x, y)
    n = np.hypot(gx, gy)
    return (K * np.cos(K * x) * np.cos(K * y) * gx
            - K * np.sin(K * x) * np.sin(K * y) * gy) / n


def boundary_points(phi, hmesh, oversample=40):
    """Points on {phi = 0}, at roughly equal arc length hmesh."""
    # a dense polyline of the level set, from the marching-squares contour of a
    # fine auxiliary grid, then projected exactly
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = max(400, int(oversample / hmesh))
    g = np.linspace(-1.0, 1.0, n)
    Xg, Yg = np.meshgrid(g, g)
    cs = plt.contour(Xg, Yg, phi(Xg, Yg), levels=[0.0])
    segs = cs.allsegs[0]
    plt.close("all")
    pts = max(segs, key=len)
    # resample at equal arc length
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    m = max(8, int(round(d[-1] / hmesh)))
    s = np.linspace(0.0, d[-1], m, endpoint=False)
    x = np.interp(s, d, pts[:, 0])
    y = np.interp(s, d, pts[:, 1])
    for _ in range(20):                           # project back onto {phi = 0}
        v = phi(x, y)
        gx, gy = gradphi(x, y)
        n2 = gx * gx + gy * gy
        x, y = x - v * gx / n2, y - v * gy / n2
    return np.column_stack([x, y])


def make_mesh(phi, hmesh):
    """Conforming triangulation of {phi < 0}: vertices, triangles."""
    bnd = boundary_points(phi, hmesh)
    # interior vertices on a triangular lattice, kept away from the boundary
    ny = int(np.ceil(2.0 / (hmesh * np.sqrt(3) / 2)))
    rows = []
    for j in range(ny + 1):
        y = -1.0 + j * hmesh * np.sqrt(3) / 2
        off = 0.5 * hmesh * (j % 2)
        xs = np.arange(-1.0 + off, 1.0, hmesh)
        rows.append(np.column_stack([xs, np.full_like(xs, y)]))
    lat = np.vstack(rows)
    v = phi(lat[:, 0], lat[:, 1])
    gx, gy = gradphi(lat[:, 0], lat[:, 1])
    dist = -v / np.hypot(gx, gy)                  # signed distance, >0 inside
    inner = lat[dist > 0.65 * hmesh]
    pts = np.vstack([bnd, inner])
    tri = Delaunay(pts)
    c = pts[tri.simplices].mean(axis=1)
    keep = phi(c[:, 0], c[:, 1]) < 0.0
    return pts, tri.simplices[keep]


def p1_solve(pts, tris, alpha=ALPHA):
    """P1 assembly and solve for -Delta u + alpha u = f, du/dn = g."""
    n = len(pts)
    x, y = pts[:, 0], pts[:, 1]
    i0, i1, i2 = tris.T
    x0, y0 = x[i0], y[i0]
    b = np.column_stack([y[i1] - y[i2], y[i2] - y[i0], y[i0] - y[i1]])
    c = np.column_stack([x[i2] - x[i1], x[i0] - x[i2], x[i1] - x[i0]])
    det = (x[i1] - x0) * (y[i2] - y0) - (x[i2] - x0) * (y[i1] - y0)
    area = 0.5 * np.abs(det)
    rows, cols, vals = [], [], []
    Mloc = (np.ones((3, 3)) + np.eye(3)) / 24.0   # P1 mass matrix / (2*area)
    for p in range(3):
        for q in range(3):
            k = (b[:, p] * b[:, q] + c[:, p] * c[:, q]) / (4 * area) \
                + alpha * Mloc[p, q] * 2 * area
            rows.append(tris[:, p]); cols.append(tris[:, q]); vals.append(k)
    A = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()
    # load vector: f at the vertices, mass-lumped on the P1 mass matrix
    fv = f(x, y)
    rhs = np.zeros(n)
    for p in range(3):
        contrib = sum(Mloc[p, q] * 2 * area * fv[tris[:, q]] for q in range(3))
        np.add.at(rhs, tris[:, p], contrib)
    # Neumann term: the exterior edges, by two-point Gauss
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.sort(e, axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    be = uniq[cnt == 1]
    pa, pb = pts[be[:, 0]], pts[be[:, 1]]
    L = np.hypot(*(pb - pa).T)
    for w, s in ((0.5, 0.5 - 0.5 / np.sqrt(3)), (0.5, 0.5 + 0.5 / np.sqrt(3))):
        q = pa + s * (pb - pa)
        gq = g_of(q[:, 0], q[:, 1])
        np.add.at(rhs, be[:, 0], w * L * gq * (1 - s))
        np.add.at(rhs, be[:, 1], w * L * gq * s)
    u = spsolve(A, rhs)
    # relative L2 error with the P1 mass matrix
    Mm = sp.coo_matrix((np.concatenate([Mloc[p, q] * 2 * area
                                        for p in range(3) for q in range(3)]),
                        (np.concatenate([tris[:, p] for p in range(3)
                                         for q in range(3)]),
                         np.concatenate([tris[:, q] for p in range(3)
                                         for q in range(3)]))),
                       shape=(n, n)).tocsr()
    err = u - ue(x, y)
    exact = ue(x, y)
    return u, np.sqrt(err @ (Mm @ err)) / np.sqrt(exact @ (Mm @ exact)), len(tris)


if __name__ == "__main__":
    phi = make_phi()
    print(f"{'method':>10} {'dofs':>8} {'h':>9} {'L2 error':>11} "
          f"{'mesh s':>8} {'solve s':>9} {'total s':>9}")
    print("-" * 74)
    rows = []
    for N in (20, 40, 80, 160, 320):
        t0 = time.perf_counter()
        e = phifd_solve(N, phi, alpha=ALPHA)[0]
        t1 = time.perf_counter()
        ndof = int(((phi(*np.meshgrid(np.linspace(-1, 1, N + 1),
                                      np.linspace(-1, 1, N + 1))) < 0).sum()))
        rows.append((0, ndof, 2.0 / N, e, 0.0, t1 - t0))
        print(f"{'phi-FD':>10} {ndof:8d} {2.0/N:9.5f} {e:11.3e} "
              f"{0.0:8.3f} {t1-t0:9.3f} {t1-t0:9.3f}")
    for hm in (0.08, 0.04, 0.02, 0.01, 0.005):
        t0 = time.perf_counter()
        pts, tris = make_mesh(phi, hm)
        t1 = time.perf_counter()
        u, e, nt = p1_solve(pts, tris)
        t2 = time.perf_counter()
        rows.append((1, len(pts), hm, e, t1 - t0, t2 - t1))
        print(f"{'P1 fitted':>10} {len(pts):8d} {hm:9.5f} {e:11.3e} "
              f"{t1-t0:8.3f} {t2-t1:9.3f} {t2-t0:9.3f}   ({nt} triangles)")
    np.savetxt("compare_fitted.dat", np.array(rows),
               header="is_fem  dofs  h  L2_error  mesh_s  solve_s",
               fmt="%.6e")
    print("\nsaved compare_fitted.dat")
