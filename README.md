# $\psi\pi$ (psipy) – Pseudodifferential Operator Toolkit
# ψπ (psipy) – Pseudodifferential Operator Toolkit

[![CI – Scripts + Coverage + Docs](https://github.com/phbillet/psipy/actions/workflows/ci.yml/badge.svg)](https://github.com/phbillet/psipy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/phbillet/psipy/branch/main/graph/badge.svg)](https://codecov.io/gh/phbillet/psipy)

**Documentation:** [https://phbillet.github.io/psipy/](https://phbillet.github.io/psipy/)

`psipy` is a unified Python framework for symbolic–numerical analysis of partial differential equations (PDEs), pseudo‑differential operators (ΨDOs), and semiclassical asymptotics. It bridges the gap between formal symbolic manipulation (SymPy) and high‑performance numerical computation (NumPy/SciPy), enabling deep exploration of phase‑space geometry, caustics, and spectral properties in both 1D and 2D.

The library provides a comprehensive toolkit for researchers in mathematical physics, including:
- Symbolic and numerical handling of pseudo‑differential operators.
- Spectral PDE solvers with automatic term classification and ETD‑RK4 time stepping.
- Asymptotic evaluation of oscillatory integrals (stationary phase, Laplace, saddle point).
- Rigorous caustic detection and Arnold classification via the stability matrix.
- Multidimensional WKB approximations with uniform caustic corrections.
- Hamiltonian and Lagrangian mechanics (Legendre transforms, symbolic PDE generation).
- A curated catalog of over 500 Hamiltonian systems for testing and research.
- Geometric analysis of 1D/2D Hamiltonian flows, periodic orbits, and semiclassical spectra.

---

## Core Features

### 🔹 Pseudo‑Differential Operators (ΨDOs)
- Define operators from a symbol $p(x,\xi)$ or derive them automatically from differential expressions.
- Symbolic calculus: asymptotic expansion, composition, commutators, formal adjoints, exponential, and left/right inverses.
- Microlocal analysis: ellipticity checks, characteristic sets, Hamiltonian flows, and pseudospectrum.

### 🔹 PDE Solving with Spectral Methods
- Solve **1D & 2D** linear/nonlinear time‑dependent or stationary PDEs.
- **Automatic parsing** of SymPy equations into linear, nonlinear, source, and pseudo‑differential terms.
- **Spectral (FFT) discretization** with periodic or Dirichlet boundary conditions.
- Advanced time‑stepping: default exponential integrator and **ETD‑RK4** (Exponential Time Differencing Runge–Kutta 4).
- Stationary problems solved via asymptotic inversion of elliptic ΨDOs.

### 🔹 Asymptotic Evaluation of Oscillatory Integrals
- Automatic detection of integration method (stationary phase, Laplace, saddle point) from the phase function.
- **High‑order asymptotics** for Morse, Airy (fold), and Pearcey (cusp) singularities.
- Full second‑order corrections for Morse points, including amplitude and phase anharmonicities.
- **Saddle‑point** continuation for complex phases (with cautionary warnings).

### 🔹 Caustic Detection and Arnold Classification
- **Algebraic classification** of 1D and 2D catastrophes up to \(A_5\) and \(D_4^\pm\).
- **Adaptive critical‑point solver** that combines symbolic solving and grid‑based Newton refinement.
- **Ray‑based caustic detection** using the stability matrix \(J(t)\) (corrected method); computes Maslov index and Arnold type.

### 🔹 Multidimensional WKB Approximation
- Ray tracing for eikonal and transport equations up to third order.
- Co‑integration of the **stability matrix** \(J\) for rigorous caustic detection.
- Uniform caustic corrections using Airy (fold) and Pearcey (cusp) functions.
- Automatic interpolation onto regular grids for 1D/2D solutions.

### 🔹 Geometric and Dynamical Analysis
- **Unified 1D/2D geometry engine** for Hamiltonian systems: compute geodesics, periodic orbits, Gutzwiller trace, semiclassical spectrum.
- **Symplectic integration** (symplectic Euler, Verlet) for arbitrary degrees of freedom.
- Poincaré sections, KAM tori detection, and Lyapunov exponents for 2‑DOF systems.
- **Riemannian geometry** (metric, Christoffel symbols, curvature, geodesic distance) in 1D/2D.

### 🔹 Hamiltonian & Lagrangian Mechanics
- **Symbolic Legendre transforms** between Lagrangian and Hamiltonian formulations (classical and convex‑analytic via Legendre–Fenchel).
- Automatic generation of formal PDEs (Schrödinger, wave, stationary) from a Hamiltonian symbol via ΨDOs.
- Decomposition of Hamiltonians into local (polynomial) and nonlocal (e.g., sqrt, Abs) parts.

### 🔹 Extensive Hamiltonian Catalog
- Over **500 pre‑defined symbolic Hamiltonians** covering classical mechanics, quantum systems, field theory, astrophysics, condensed matter, biophysics, and many interdisciplinary domains.
- Search, filter, and export utilities for systematic exploration and benchmarking.

### 🔹 Visualisation and Diagnostics
- Rich plotting: phase‑space portraits, caustic overlays, convergence plots, amplitude decompositions.
- **AsymptoticVisualizer** for stationary‑phase, Laplace, and saddle‑point integrals.
- **Geometry visualizers** producing multi‑panel atlases (up to 18 panels) for any symbol.

---

## Core Modules

| Module          | Description |
| :-------------- | :---------- |
| `psiop`         | Symbolic and numerical framework for pseudo‑differential operators. Defines operators, performs symbolic calculus, and applies them via FFT/Kohn‑Nirenberg. |
| `solver`        | Spectral PDE solver with automatic term classification, support for ΨDOs, and ETD‑RK4 time stepping. Solves both time‑dependent and stationary equations. |
| `asymptotic`    | Large‑parameter asymptotics for oscillatory integrals. Detects method, finds critical points, and evaluates leading terms + corrections for Morse/Airy/Pearcey. |
| `caustics`      | Arnold classification, adaptive critical‑point search, and ray‑based caustic detection with Maslov index. |
| `wkb`           | Multidimensional WKB approximation with ray tracing, amplitude transport, and uniform caustic corrections. |
| `microlocal`    | Unified microlocal toolkit: characteristic varieties, bicharacteristic flow, wavefront sets, and WKB integration. |
| `fio_bridge`    | Fourier Integral Operator bridge: applies ΨDOs to WKB states via asymptotic evaluation, with precomputation for speed. Includes validation tools (CrossValidator). |
| `symplectic`    | Hamiltonian mechanics for any number of degrees of freedom: symplectic integration, Poisson brackets, fixed points, action‑angle (1D), Poincaré sections (2D). |
| `riemannian`    | Riemannian geometry in 1D/2D: metric, geodesics, curvature, Laplace–Beltrami, exponential map, and visualisation. |
| `geometry`      | Comprehensive visualisation and analysis of 1D/2D Hamiltonian systems: geodesics, periodic orbits, caustics, Gutzwiller trace, semiclassical spectrum, KAM tori. |
| `physics`       | Symbolic toolkit for Lagrangian–Hamiltonian conversion (Legendre transforms, Legendre–Fenchel with numeric fallback) and automatic PDE generation from Hamiltonian symbols. |
| `hamiltonian_catalog` | Curated collection of over 500 symbolic Hamiltonians across many domains, with search, filtering, export, and metadata. |

---

## Installation

```bash
git clone https://github.com/phbillet/psipy.git
cd psipy
pip install -e .
```

Dependencies: `numpy`, `scipy`, `sympy`, `matplotlib` (automatically installed).

---

## Documentation

Detailed API documentation and tutorials are available in the `docs/` directory and online at [https://phbillet.github.io/psipy/](https://phbillet.github.io/psipy/). Each module is extensively documented with mathematical background and usage examples.

---

## License

Licensed under the **Apache License 2.0**. See [LICENSE](LICENSE).