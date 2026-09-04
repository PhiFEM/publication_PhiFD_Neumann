"""
Cut-cell geometry primitives for the Arias et al. (2018) finite-volume scheme,
used by arias_poisson_neumann_bean.py.

    V. Arias, D. Bochkov, F. Gibou, "Poisson equations in irregular domains with
    Robin boundary conditions -- Solver with second-order accurate gradients",
    J. Comput. Phys. 365 (2018) 1-6.

Omega = { phi < 0 }. The interface {phi = 0} is approximated piecewise-linearly
inside each Cartesian cell by linear interpolation of phi along the cell edges
(marching-squares / linear cut). Two helpers are exposed:

    wetted(phi_a, phi_b, c_a, c_b)
        One grid edge between coordinates c_a < c_b (length h), carrying phi_a at
        c_a and phi_b at c_b. Returns (L, c) where L is the length of the wetted
        part (phi < 0) of the edge and c is the centroid (midpoint) of that wetted
        part -- the point at which the flux through the face is evaluated.

    cell_cut(corners, phi_corners)
        One Cartesian cell given by its 4 corners (CCW: bottom-left, bottom-right,
        top-right, top-left) and the values of phi there. Returns
        (area, seglen, segmid):
          area   : area of the wetted polygon {phi < 0} inside the cell,
          seglen : length of the interface segment {phi = 0} inside the cell,
          segmid : (mx, my) midpoint of that interface segment (nan if seglen=0).
"""


def wetted(phi_a, phi_b, c_a, c_b):
    """Wetted length and its centroid on a single edge [c_a, c_b]."""
    mid = 0.5 * (c_a + c_b)
    a_in = phi_a < 0.0
    b_in = phi_b < 0.0
    if a_in and b_in:                       # fully inside Omega
        return c_b - c_a, mid
    if not a_in and not b_in:               # fully outside
        return 0.0, mid
    cstar = c_a + (c_b - c_a) * phi_a / (phi_a - phi_b)   # linear crossing
    if a_in:                                # wet part is [c_a, cstar]
        return cstar - c_a, 0.5 * (c_a + cstar)
    return c_b - cstar, 0.5 * (cstar + c_b)               # wet part is [cstar, c_b]


def _polygon_area(pts):
    """Shoelace area of a simple polygon given as a list of (x, y)."""
    s = 0.0
    n = len(pts)
    for k in range(n):
        x0, y0 = pts[k]
        x1, y1 = pts[(k + 1) % n]
        s += x0 * y1 - x1 * y0
    return 0.5 * abs(s)


def cell_cut(corners, phi_corners):
    """Wetted area, interface-segment length and midpoint of one cut cell."""
    nan = float("nan")
    inside = [p < 0.0 for p in phi_corners]
    if all(inside):
        return _polygon_area(corners), 0.0, (nan, nan)
    if not any(inside):
        return 0.0, 0.0, (nan, nan)

    # Sutherland-Hodgman clip of the cell polygon by the half-space {phi < 0};
    # the inserted edge-crossings are the endpoints of the interface segment.
    n = len(corners)
    poly, cross = [], []
    for k in range(n):
        a, b = corners[k], corners[(k + 1) % n]
        pa, pb = phi_corners[k], phi_corners[(k + 1) % n]
        if pa < 0.0:
            poly.append(a)
        if (pa < 0.0) != (pb < 0.0):                     # sign change on this edge
            t = pa / (pa - pb)
            c = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            poly.append(c)
            cross.append(c)

    area = _polygon_area(poly)
    if len(cross) >= 2:                                  # simple (2-crossing) cut
        p, q = cross[0], cross[1]
        seglen = ((p[0] - q[0])**2 + (p[1] - q[1])**2) ** 0.5
        segmid = (0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]))
        return area, seglen, segmid
    return area, 0.0, (nan, nan)
