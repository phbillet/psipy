# psipy – Semiclassical Analysis & Pseudo‑Differential Operators in Python

**psipy** is a research‑oriented Python library for **semiclassical analysis**, **microlocal analysis**, **pseudo‑differential operators** (ΨDOs), and **asymptotic methods** in wave propagation, quantum mechanics, and geometric PDEs. It provides a unified computational framework that spans rigorous asymptotic expansions, ray‑based WKB methods, caustic handling (Airy, Pearcey), Riemannian geometry, symplectic mechanics, and high‑order spectral PDE solvers.

## Overview

The library is organised into several self‑contained modules, each implementing a specific mathematical theory. Together they form an integrated toolkit that can:

- Evaluate oscillatory integrals via stationary phase, Laplace, and saddle‑point methods, including degenerate critical points (Airy, Pearcey).
- Construct and apply pseudo‑differential operators with constant or spatially varying symbols (Kohn–Nirenberg / Weyl quantisation).
- Solve eikonal and transport equations along rays (WKB), detect caustics, and apply uniform Airy / Pearcey corrections.
- Propagate semiclassical wavefunctions (Van Vleck–Pauli–Morette propagator) on flat or curved Riemannian manifolds.
- Parse symbolic PDEs (SymPy), separate linear/nonlinear terms, and solve them with Fourier spectral methods and exponential time integrators (ETD‑RK4).
- Perform Lagrangian–Hamiltonian transformations (Legendre, Legendre–Fenchel) and generate formal PDEs from Hamiltonian symbols.
- Analyse symplectic geometry, integrability, spectral statistics (Brody distribution), topological monodromy, and semiclassical quantisation (Bohr–Sommerfeld).

## Modules

| Module | Description |
|--------|-------------|
| [`asymptotic`](./asymptotic.md) | Stationary phase, Laplace method, saddle‑point, second‑order corrections, Airy/Pearcey degenerate asymptotics. |
| [`caustics`](./caustics.md) | Catastrophe classification (fold, cusp, swallowtail, butterfly), ray‑based caustic detection, Maslov phase shift, uniform approximations. |
| [`fio_bridge`](./fio_bridge.md) | Fourier integral operators (FIOs), Egorov’s theorem, quantum tunnelling (complex saddles), gravitational lensing, cross‑validation with spectral solvers. |
| [`microlocal`](./microlocal.md) | Principal symbol, characteristic variety, bicharacteristic flow, wavefront set, WKB, stability matrix, Bohr–Sommerfeld quantisation. |
| [`physics`](./physics.md) | Legendre & Legendre–Fenchel transforms, Hamiltonian decomposition (local vs non‑local), formal PDE generation (stationary, Schrödinger, wave). |
| [`propagator`](./propagator.md) | Semiclassical (Van Vleck) propagator for Schrödinger, heat, and wave equations; Jacobi determinant; uniform caustic corrections; Riemannian geodesic motion. |
| [`psiop`](./psiop.md) | Pseudo‑differential operator calculus: Kohn–Nirenberg/Weyl quantisation, symbol composition (Moyal), adjoint, parametrices, exponential, pseudospectrum. |
| [`riemannian`](./riemannian.md) | Riemannian metrics, Christoffel symbols, geodesic equation, curvature tensors, Laplace–Beltrami, Hodge theory (2D), parallel transport, Gauss–Bonnet, Nash–Kuiper. |
| [`solver`](./solver.md) | Spectral PDE solver: Fourier method, equation parsing, linear/nonlinear separation, dealiasing, ETD‑RK4 / exponential Euler / leapfrog, ΨDO evaluation, boundary conditions (periodic, Dirichlet/Neumann via ΨDO), stationary inverses. |
| [`symplectic`](./symplectic.md) | Symplectic geometry, Hamiltonian mechanics, Poisson bracket, Liouville, Darboux, Gromov non‑squeezing, action‑angle variables, KAM, spectral statistics (Brody), topological monodromy, Floer homology, Bohr–Sommerfeld. |
| [`wkb`](./wkb.md) | Multidimensional WKB: eikonal equation, transport equations (any order), stability matrix, caustic detection, Maslov index, uniform Airy/Pearcey corrections, ray tracing and interpolation. |

## Key Mathematical Features

### Asymptotics & Special Functions
- **Stationary phase** for real‑phase oscillatory integrals, including second‑order corrections.
- **Laplace method** for purely imaginary phase (exponential damping).
- **Saddle‑point method** for complex phases (tunnelling).
- **Degenerate critical points** – Airy ($\lambda^{-1/3}$ decay) and Pearcey ($\lambda^{-3/4}$ decay) asymptotics.
- **Catastrophe classification** (Arnold) – fold, cusp, swallowtail, butterfly with uniform Airy/Pearcey approximations.

### Pseudo‑Differential Operators
- **Kohn–Nirenberg** and **Weyl** quantisations.
- **Symbol classes** $S^m$, asymptotic expansions, principal symbol.
- **Composition** (Moyal star product) and **commutator** to leading order $\{p,q\}$.
- **Formal adjoint** and **elliptic parametrices** (asymptotic inverses).
- **Exponential of an operator** (formal series).
- **Semiclassical trace** formula $\operatorname{Tr}(P) \sim \frac{1}{(2\pi)^n}\iint p(x,\xi)\,dx\,d\xi$.
- **Pseudospectrum** computation for non‑normal operators.

### WKB, Rays & Caustics
- **Eikonal equation** $p(x,\nabla S)=0$ solved via Hamilton’s equations.
- **Transport equations** for amplitudes $a_k$ (any order).
- **Stability matrix** $J = \partial x / \partial q$, its variational equation, caustic detection ($\det J = 0$).
- **Maslov index** – phase shift $\exp(i\pi\mu/2)$ accumulated at caustics.
- **Uniform approximations** – Airy function (fold), Pearcey integral (cusp), parabolic cylinder (heat equation).
- **Van Vleck–Pauli–Morette propagator** for Schrödinger, heat, and wave equations.

### Riemannian & Symplectic Geometry
- **Riemannian metrics** – covariant components $g_{ij}$, Christoffel symbols, geodesic equations, parallel transport.
- **Curvature** – Riemann, Ricci, scalar, Gaussian (2D), Brioschi formula.
- **Laplace–Beltrami operator** $\Delta_g$ and its symbol.
- **Hodge theory in 2D** – Hodge star, codifferential, Hodge decomposition, Weitzenböck identity.
- **Symplectic structure** – symplectic form $\omega$, Hamiltonian vector fields, Poisson bracket.
- **Liouville integrability**, **KAM theory**, **action‑angle variables**.
- **Spectral statistics** – Berry–Tabor (Poisson) vs. BGS (Wigner) conjectures, Brody distribution.
- **Topological monodromy** – obstruction to global action‑angle coordinates.
- **Bohr–Sommerfeld quantisation** $I(E_n) = \hbar (n + \mu/4)$.

### PDE Solving Infrastructure
- **Symbolic parsing** – uses SymPy to extract linear operator, nonlinear terms, source terms.
- **Fourier spectral method** – periodic domains, dealiasing (2/3‑rule).
- **Time integrators** – exponential Euler, ETD‑RK4 (4th order), leapfrog for wave equations.
- **ΨDO evaluation** – constant symbols become Fourier multipliers; spatially varying symbols use Kohn–Nirenberg quadrature (non‑periodic boundaries).
- **Boundary conditions** – periodic (native); Dirichlet/Neumann via ΨDO reformulation.
- **Stationary problems** – asymptotic right inverse (parametrix) for elliptic ΨDOs.
- **Stability diagnostics** – CFL condition, symbol checks (dissipation, growth), energy monitoring.

## Applications & Examples

The library includes notebooks that demonstrate:

- **Harmonic oscillator** – exact Mehler kernel vs. semiclassical propagator, caustic at $t=\pi/4$ and Airy correction.
- **Gravitational lensing** – Einstein ring, two images, interference fringes, caustic enhancement.
- **Quantum tunnelling** – complex saddle points, transmission through a Gaussian barrier.
- **Geodesic motion on curved surfaces** – sphere, paraboloid, with Jacobi fields and caustics.
- **Spectral statistics** – Brody parameter for integrable vs. chaotic billiards.
- **Topological monodromy** – spherical pendulum, action lattice transformation.
- **Pseudospectra** – non‑normal operators, transient growth.

## Installation

```bash
git clone https://github.com/yourusername/psipy.git
cd psipy
pip install -e .
```

Dependencies: `numpy`, `scipy`, `sympy`, `matplotlib`.

## Documentation

Each module comes with a detailed theory document (like this one) that explains the underlying mathematics. In‑depth usage examples are provided as Jupyter notebooks in the `examples/` directory.

## Contributing

Contributions are welcome – especially for higher‑dimensional extensions, additional catastrophe integrals, or GPU acceleration of ΨDO evaluations.

## References

The mathematical foundations are drawn from:

- L. Hörmander, *The Analysis of Linear Partial Differential Operators* (4 vols), Springer.
- J.J. Duistermaat, *Fourier Integral Operators*, Birkhäuser.
- M. Zworski, *Semiclassical Analysis*, AMS.
- V.P. Maslov & M.V. Fedoriuk, *Semi‑Classical Approximation in Quantum Mechanics*, Reidel.
- V.I. Arnold, *Catastrophe Theory*, Springer.
- M. Berry & C. Howls, “High orders of the Weyl expansion for quantum billiards”, *Phys. Rev. E* 50(5) (1994).
- A.-K. Kassam & L.N. Trefethen, “Fourth‑order time‑stepping for stiff PDEs”, *SIAM J. Sci. Comput.* 26(4) (2005).

## License

MIT License (or as specified in the repository).

---

*psipy – bringing microlocal analysis to Python.*
