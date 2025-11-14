# Symbol Geometry Analyzer — Semiclassical & Spectral Framework

[Back to README](../README.md)

## Overview

**`SymbolGeometryAnalyzer`** is a comprehensive Python toolkit for the geometric and semiclassical analysis of **1D pseudodifferential operator symbols (Hamiltonians)**.
It visualizes the connection between the classical geometric structures of a symbol $H(x, \xi)$ and its quantum spectral properties.

This package bridges **symbolic Hamiltonian definition (via SymPy)** and **numerical semiclassical simulation (via SciPy and NumPy)**. It allows users to compute classical trajectories (geodesics), identify periodic orbits, detect caustics, and ultimately compute the semiclassical energy spectrum using the Gutzwiller trace formula.

-----

## Key Features

  * 🔹 **Symbolic Hamiltonian Engine**: Accepts any 1D Hamiltonian $H(x, \xi)$ as a SymPy expression.
  * 🔹 **Automatic Differentiation**: Symbolically computes all necessary first and second-order derivatives ($\partial H/\partial x$, $\partial H/\partial \xi$, $\partial^2 H/\partial x \partial \xi$, etc.).
  * 🔹 **Hamiltonian Flow Integration**: Numerically solves Hamilton's equations to compute geodesics (classical trajectories).
  * 🔹 **Caustics Detection**: Solves the variational equations alongside the flow to compute the Jacobian $J = \partial x / \partial \xi_0$, identifying caustics where $J \approx 0$.
  * 🔹 **Periodic Orbit Finder**: Numerically searches the energy surface $H(x, \xi) = E$ to find periodic orbits, their actions, periods, and stability.
  * 🔹 **Semiclassical Spectrum**: Implements the **Gutzwiller trace formula** to compute the semiclassical trace $Tr[\exp(-iHt/\hbar)]$ from periodic orbits.
  * 🔹 **Spectral Analysis**: Computes the energy spectrum by performing a Fourier transform on the semiclassical trace.
  * 🔹 **Complete Visualization Atlas**: Generates a comprehensive 15-panel figure displaying all geometric and spectral properties simultaneously.

-----

## Capabilities

The analyzer provides a deep dive into the phase space geometry and its spectral implications:

  * **Geometric Analysis**:
      * 3D Hamiltonian surface visualization
      * Phase space foliation (energy level sets)
      * Hamiltonian vector field
      * Group velocity ($v_g = \partial H / \partial \xi$)
      * Spatial projection (x vs. t) with caustic (focal point) markers
      * Jacobian evolution (focusing/defocusing)
      * Sectional curvature ($\partial^2 H / \partial x \partial \xi$)
  * **Spectral & Semiclassical Analysis**:
      * Periodic orbit plotting in phase space
      * Period-Energy and Action-Energy diagrams
      * **EBK (Einstein-Brillouin-Keller) quantization** levels
      * Gutzwiller trace formula (time domain)
      * Semiclassical energy spectrum (energy domain)
      * Orbit stability analysis (Lyapunov exponents)
      * **Level spacing distribution** (Poisson vs. Wigner) for integrability analysis

-----

## Example Usage

```python
import sympy as sp
import matplotlib.pyplot as plt
from geometry_1d import visualize_symbol # Assuming the file is in the path

# 1. Define symbolic variables
x, xi = sp.symbols('x xi', real=True)

# 2. Define the Hamiltonian (e.g., Harmonic Oscillator)
H = xi**2 + x**2

# 3. Set visualization parameters
x_range = (-3, 3)
xi_range = (-3, 3)
hbar = 0.1

# Define initial conditions for geodesics: (x0, xi0, t_max, color)
geodesics_params = [
    (1.0, 0.0, 10.0, 'red'),
    (0.0, 1.5, 10.0, 'blue')
]

# Define energy range for spectral analysis (periodic orbits, etc.)
E_range = (0.5, 4.0)

# 4. Run the complete analysis
fig, geodesics, orbits, spectrum = visualize_symbol(
    symbol=H,
    x_range=x_range,
    xi_range=xi_range,
    geodesics_params=geodesics_params,
    E_range=E_range,
    hbar=hbar,
    resolution=100
)

plt.show()
```

-----

## Structure and Internals

| Class / Module | Description |
| --- | --- |
| `SymbolGeometry` | **Core Engine**: Handles symbolic differentiation (SymPy), numerical conversion (lambdify), and all core computations (geodesics, periodic orbits, trace formula). |
| `SymbolVisualizer` | **Visualization Engine**: Contains all 15 Matplotlib plotting methods. Consumes data from `SymbolGeometry` to generate the complete visual atlas. |
| `SpectralAnalysis` | **Utility Class**: Provides supplementary spectral tools like Weyl's Law, Berry-Tabor formula, and integrability analysis based on level spacing statistics. |
| `Geodesic` (dataclass) | Data structure for storing trajectory information, including position, momentum, energy, and the Jacobian. |
| `PeriodicOrbit` (dataclass) | Data structure for storing orbit properties, including period, action, energy, and stability. |
| `Spectrum` (dataclass) | Data structure for storing the semiclassical trace and the resulting energy spectrum. |
| `visualize_symbol` (func) | The primary user-facing function that initializes the engine and visualizer and returns the final figure and data. |

---

## 📜 License

Licensed under the **Apache License 2.0** — see [LICENSE](./LICENSE) for details.
Copyright © 2025 Philippe Billet

---
[Back to README](../README.md)