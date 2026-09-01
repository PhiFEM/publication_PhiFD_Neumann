# $\varphi$-FD, a second order finite difference scheme for geometry defined by level-set function: the Neumann case

This repository contains the code used in the study "$\varphi$-FD, a second order finite difference scheme for geometry defined by level-set function: the Neumann case" Michel Duprez, Vanessa Lleras, Alexei Lozinski, Vincent Vigon, Lisl Weynans ([preprint](https://hal.science/hal-04731164)).

Everything reported in the article is here, including our reimplementations of
the two methods we compare against. Pure `NumPy`/`SciPy`; `matplotlib` is used
only to draw figures.

## This repository is for reproducibility purposes only

It is "frozen in time" and not maintained.
To use our latest $\varphi$-FEM code please refer to the [phiFEM repository](https://github.com/PhiFEM/Poisson-Dirichlet-fenicsx).

## Usage

### Prerequisites

- numpy
- scipy
- matplotlib (figures only)

### Example of usage

From the main directory:

```bash
python3 phiFD_poisson_neumann_bean.py     # scalar Poisson, pure Neumann
python3 phiFD_elastic_neumann_bean.py     # linear elasticity, pure traction
```

Each script prints its results as `pgfplots` `coordinates {...}` lists, ready to
paste into the article, and the comparison scripts also save them as `.dat`.

## The scheme

| file | what it is |
|---|---|
| `levelset.py` | the bean level set $\varphi(x,y) = 0.8x^2 + (y+1.4x^2)^2 - R_b^2$ |
| `phiFD_poisson_neumann_bean.py` | $\varphi$-FD for the scalar Poisson problem with pure Neumann conditions — the reference implementation, plus the discrete gradient `grad_h` |
| `phiFD_elastic_neumann_bean.py` | the same for linear elasticity with traction conditions |
| `phiFD_elastic_mixte_levelset.py` | mixed Dirichlet/Neumann elasticity (demo, not used in the article) |

Two options of `solve` govern the conditioning without changing the accuracy,
both on by default:

- **`row_scaling="norm"`** — normalize each boundary equation by its own
  sup-norm rather than scaling all of them by $h^{-4}$. Being a row scaling, the
  discrete solution is unchanged (verified to $2\cdot10^{-12}$ in relative $L^2$).
- **`alpha0_rule="safe"`** — never couple a boundary node to an interior node
  lying on the interface ($|\varphi_{\alpha_0}| < 0.1\,h$). Such a node cancels
  the second bracket of the relaxation, and two boundary nodes sharing it then
  carry two proportional equations.

Together they take the condition number from a factor 82 down to **1.11** over
three decades of $\min|\varphi|/h$ (see `sensitivity_kappa.py`).

> The `alpha0_rule` is **specific to Neumann**. In the Dirichlet case the ansatz
> is $u \approx \varphi p$ and the accuracy degrades directly when
> $x_{\alpha_0}$ is moved away, so the nearest node remains the right choice —
> hence the default `alpha0_rule="nearest"` in `phiFD_elastic_mixte_levelset.py`,
> where turning it on costs a factor 5 to 13 on the error.

`sigma > 0` switches on the second-order ghost-penalty term of the Dirichlet
$\varphi$-FD paper. It is not needed once `alpha0_rule="safe"` is set, and is
kept only because the article mentions it.

## The methods we compare against

| file | what it is |
|---|---|
| `arias2018.py` | cut-cell geometry primitives (marching squares) |
| `arias_poisson_neumann_bean.py` | the cut-cell finite-volume scheme of Arias, Bochkov & Gibou (2018), plus `cond2`, a $\kappa$ estimator by LU + power iteration |
| `cocorusso.py` | the ghost-point scheme of Coco & Russo (2013), degrees $p = 1$ and $p = 2$ |

Two implementation choices matter enough to be stated, because getting them
wrong makes either method look worse than it is:

- `arias_poisson_neumann_bean.solve` takes `geometry="linear"` (plain marching
  squares) or **`"refined"`** (each cut cell subdivided, so volumes, wetted edges
  and interface segments stay consistent at the finer scale). The article uses
  `"refined"`. With the linear reconstruction the scheme's own gradient is only
  first order in $L^\infty$ (0.97 instead of 1.54) — an artifact of the
  reconstruction we give it, not a property of the method: Arias et al. locate
  the intersections with a quadratic approximation of the level set.
- `cocorusso.solve` falls back to a reduced stencil at ghost nodes having no
  admissible $(p+1)^2$ block containing their projection. Without that fallback
  the solution is destroyed on the non-convex bean (92 % error at $h = 0.05$).

## Reproducing the article

| what | script | output |
|---|---|---|
| Fig. 1(a), bean domain | `plot_bean_domain.py` | `fig_elastic_domains.png` |
| Fig. 1(b), Poisson convergence | `phiFD_poisson_neumann_bean.py` | printed |
| Fig. 1(c), elasticity convergence | `phiFD_elastic_neumann_bean.py` | printed |
| §4.2, gradient accuracy | `compare_gradient.py` | `compare_gradient.dat` |
| §4.3, robustness vs cut cells | `compare_cutsize.py` | `compare_cutsize.dat` |
| §4.4, ghost-point comparison | `compare_ghostpoint.py` | `compare_ghostpoint_*.dat` |
| §4.5, cost vs a fitted P1 method | `compare_fitted.py` | `compare_fitted.dat` |
| Fig. 3(a)–(b), conditioning vs cut size | `sensitivity_kappa.py` | `sensitivity_kappa.dat` |
| Fig. 3(c), $\varphi$-FD vs cut-cell | `arias_poisson_neumann_bean.py` | printed |
| error vs cut size, legacy variant | `sensitivity_cutcell.py` | `sensitivity_cutcell.dat` |

The `.dat` files shipped here are the ones the article plots. Re-running a
script overwrites its own file; figures (`.png`) are by-products and are not
tracked.

`sensitivity_cutcell.py` deliberately calls the solver with
`row_scaling="h4", alpha0_rule="nearest"`, so that `sensitivity_cutcell.dat`
stays exactly the data plotted as the "nearest" series of Figure 3(a).

## Checks worth keeping

- `sensitivity_kappa.py` measures the drift between the $h^{-4}$ and the
  row-normalized variants — they differ by a row scaling, so it stays below
  $2\cdot10^{-12}$.
- `cocorusso.py` on a disk reproduces the published order distinction for
  Neumann conditions: $p = 1$ gives 1.04, $p = 2$ gives 2.25. That is what says
  the reimplementation is faithful.
- `compare_fitted.py` solves $-\Delta u + u = 1$, $\partial_n u = 0$ exactly
  ($u \equiv 1$) as a sanity check of the P1 assembly.

## Issues and support

Please use the issue tracker to report any issues.

## Authors (alphabetical)

[Michel Duprez](https://michelduprez.fr/), Inria Nancy Grand-Est  
[Vanessa Lleras](https://vanessalleras.wixsite.com/lleras), Université de Montpellier  
[Alexei Lozinski](https://orcid.org/0000-0003-0745-0365), Université de Franche-Comté  
[Vincent Vigon](https://irma.math.unistra.fr/~vigon/), Université de Strasbourg  
[Lisl Weynans](https://www.math.u-bordeaux.fr/~lweynans/), Université de Bordeaux
