"""Poisson, pure Neumann, bean domain: relative errors vs h, printed as
pgfplots coordinates for the note. Uses the solver defaults (row-normalized
boundary rows, alpha_0 rule)."""
import numpy as np
from levelset import make_phi
from phiFD_poisson_neumann_bean import solve

phi = make_phi()
Ns = [20, 40, 80, 160, 320]
hs, L2, H1, Li = [], [], [], []
for N in Ns:
    a, b, c = solve(N, phi)
    hs.append(2.0/N); L2.append(a); H1.append(b); Li.append(c)
    print(f"N={N:4d}  h={2.0/N:.5f}  L2={a:.6e}  H1={b:.6e}  Linf={c:.6e}")
hs = np.array(hs)
sl = lambda e: np.polyfit(np.log(hs), np.log(e), 1)[0]
print(f"slopes: L2={sl(L2):.2f} H1={sl(H1):.2f} Linf={sl(Li):.2f}")
pg = lambda e: "".join(f"({h:.6g},{v:.6e})" for h, v in zip(hs, e))
print("\n% pgfplots")
print(f"\\addplot[color=blue,mark=triangle*] coordinates {{{pg(L2)}}};")
print(f"\\addplot[color=red,mark=*] coordinates {{{pg(H1)}}};")
print(f"\\addplot[color=green,mark=*] coordinates {{{pg(Li)}}};")
np.savetxt("convergence.dat", np.c_[hs, L2, H1, Li], header="h L2 H1 Linf")
