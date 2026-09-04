# $\varphi$-FD, a second order finite difference scheme for geometry defined by level-set function: the Neumann case

Code of the short note "$\varphi$-FD, a second order finite difference scheme
for geometry defined by level-set function: the Neumann case", Michel Duprez,
Vanessa Lleras, Alexei Lozinski, Vincent Vigon, Lisl Weynans.

Everything reported in the note is here, in plain `NumPy`/`SciPy`
(`matplotlib` only for the domain figure). Run any script from this directory
with `PYTHONPATH=.`, e.g. `PYTHONPATH=. python3 convergence.py`.

## This repository is for reproducibility purposes only

It is "frozen in time" and not maintained. To use our latest $\varphi$-FEM code
please refer to the [phiFEM repository](https://github.com/PhiFEM/Poisson-Dirichlet-fenicsx).
Everything reported in the note, in plain `NumPy`/`SciPy` (`matplotlib` for the
domain figure only). Run any script from this directory with `PYTHONPATH=.`.

| file | what it is |
|---|---|
| `levelset.py` | the bean level set `phi(x,y) = 0.8x² + (y+1.4x²)² − Rb²` |
| `phiFD_poisson_neumann_bean.py` | the scheme (eq. 2 of the note); `row_scaling="norm"` and `alpha0_rule="safe"` are the two precautions of Section 2, on by default. **`bc_data="neumann"` is the default here**: the right-hand side of the boundary rows is assembled from the data `G = g|grad phi|` alone, as in eq. (2) — the scheme as a user would run it. (`"exact"` uses `(row)·u_exact` and removes the boundary consistency error; not used in the note.) |
| `sensitivity_cutcell.py` | geometry sweep primitives: `pick_geometries`, `smallest_cut`, `pgf_coords` |
| `arias2018.py`, `arias_poisson_neumann_bean.py` | the cut-cell finite-volume scheme of Arias, Bochkov & Gibou (2018), for the one comparison quoted |

| what | script | output |
|---|---|---|
| Fig. 1(a), bean domain | `plot_bean_domain.py` | `fig_elastic_domains.png` |
| Fig. 1(b), Poisson convergence | `convergence.py` | `convergence.dat`, pgfplots lines printed |
| Fig. 2(a)–(b), error and κ vs smallest cut (h⁻⁴ / normalized / both) | `sensitivity_kappa.py` | `sensitivity_kappa.dat`, `.png`, `.log` |
| Fig. 2(b), α₀ rule alone (h⁻⁴ scaling) | `alpha0_alone.py` | `alpha0_alone.dat`, `.log` |
| §4, cut-cell numbers over the same sweep | `compare_cutsize.py` | `compare_cutsize.dat`, `.png`, `.log` |
| §2, count of boundary-isolated points | `isolated_points.py` | printed (none, at N = 20 … 320) |

The scripts print their results as `pgfplots` `coordinates {...}` lists, ready
to paste into `note_journal.tex`, and save the same numbers as `.dat`.

Numbers quoted in the note (h = 0.05, 40 geometries, smallest cut spanning
1.2e-4 … 8.8e-2): κ varies by a factor 82 (slope −0.54) with the uniform h⁻⁴
scaling, 17 (−0.43) with row normalization, 2.95 (+0.02) with the α₀ rule
alone, and 1.11 (+0.00) with both. The L² error varies by a factor 2.5
(nearest) or 2.2 (α₀ rule) with no trend (slope +0.03); over the same sweep the
cut-cell scheme varies by 3.2 in error and 13.8 in κ. Convergence on the bean:
L² 3.93e-2 … 1.60e-4 for N = 20 … 320, fitted slopes 1.99 / 2.02 / 1.98.
