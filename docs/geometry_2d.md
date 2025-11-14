# SymbolVisualizer2D — Geometric and Semiclassical Analysis

[Back to README](../README.md)

## Overview

**`SymbolVisualizer2D`** is an advanced analysis and visualization toolkit for **2D Hamiltonian systems**, particularly those arising from the symbols of **pseudo-differential operators**.

This package bridges the gap between **symbolic Hamiltonian specification (via SymPy)** and **numerical phase space analysis (via SciPy/NumPy)**. It computes classical trajectories (geodesics) while rigorously tracking the system's full 4x4 Jacobian, enabling precise **caustic detection** and classification.

It integrates high-level concepts from dynamical systems, semiclassical mechanics, and quantum chaos—including **KAM theory**, **Bohr-Sommerfeld quantization**, **Maslov indices**, and **Weyl's Law**—into a comprehensive, multi-panel visualization suite.

-----

## Key Features

  * 🔹 **Symbolic Hamiltonian Parsing**: Directly accepts 2D Hamiltonian symbols `H(x, y, ξ, η)` defined in **SymPy**.
  * 🔹 **Rigorous Caustic Detection**: Computes trajectories with the **full 4x4 Jacobian** evolution to precisely locate and analyze caustics (where `det(∂(x,y)/∂(ξ₀,η₀)) = 0`).
  * 🔹 **Semiclassical Analysis**: Integrates **Maslov indices**, **Bohr-Sommerfeld quantization**, and **spectral density** (Poisson vs. Wigner) into the core analysis.
  * 🔹 **Periodic Orbit Search**: Algorithmically finds periodic orbits, calculating their **action (S)**, **period (T)**, stability, and Maslov index.
  * 🔹 **Dynamical Systems Tools**: Generates **Poincaré sections**, provides utilities for **KAM tori** detection, and computes winding/rotation numbers.
  * 🔹 **Comprehensive Visualization**: Generates a dynamic multi-panel "atlas" (up to 18 plots) showcasing all geometric and physical aspects of the system.
  * 🔹 **Phase Space Volume**: Estimates the phase space volume `Vol({H≤E})` using **Monte Carlo** methods to verify Weyl's Law.

-----

## Capabilities

`SymbolVisualizer2D` can compute and visualize a wide array of geometric and semiclassical properties, including:

  * **Geometric Projections**:
      * Configuration space `(x, y)` trajectories
      * Momentum space `(ξ, η)` trajectories
      * Phase space projections `(x, ξ)` and `(y, η)`
  * **Hamiltonian Geometry**:
      * 3D Energy surfaces `H(x, y, ξ₀, η₀)`
      * Configuration space vector fields `(vx, vy)`
      * Group velocity magnitude `|∇_p H|`
  * **Caustic & Singularity Analysis**:
      * Jacobian determinant `det(J)` evolution over time
      * Caustic curves and networks in configuration space
      * Maslov index phase shift visualization
  * **Dynamical System Analysis**:
      * Poincaré sections (e.g., at `x=0` or `y=0`)
      * Periodic orbits in 3D space-time `(x, y, t)`
      * Energy conservation verification
  * **Semiclassical & Quantum Analysis**:
      * Action-Energy `S(E)` diagrams
      * Torus (EBK) quantization levels
      * Spectral density `ρ(E)` with caustic corrections
      * Level spacing distribution (Poisson vs. Wigner)
      * Phase space volume `N(E)` (Weyl's Law)

-----

## Example Usage

```python
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# The main function is imported from the file
from geometry_2d import visualize_symbol_2d

x, y = sp.symbols('x y', real=True)
xi, eta = sp.symbols('xi eta', real=True)

# Define a 2D anharmonic oscillator Hamiltonian
H = xi**2 + eta**2 + x**2 + y**2 + 0.1*x**4

# Generate the complete analysis and visualization
fig, geodesics, orbits, caustics = visualize_symbol_2d(
    symbol=H,
    x_range=(-3, 3), y_range=(-3, 3),
    xi_range=(-3, 3), eta_range=(-3, 3),
    geodesics_params=[
        (1, 0, 0, 1, 2*np.pi, 'red'),      # Geodesic 1
        (0, 1, 1, 0, 2*np.pi, 'blue'),     # Geodesic 2
        (0.5, 0.5, 0.5, -0.5, 4*np.pi, 'green') # Geodesic 3
    ],
    E_range=(0.5, 5),  # Energy range for periodic orbit search
    hbar=0.1           # Set the reduced Planck constant
)

# Display the resulting 18-panel figure
plt.show()
```

-----

## Structure and Internals

| Class / Module | Description |
| :--- | :--- |
| **`SymbolGeometry2D`** | Core analysis engine. Handles symbolic differentiation (Hessian), lambdification, and solving the 20-dimensional augmented ODE (4 phase space + 16 Jacobian). |
| **`SymbolVisualizer2D`** | Main visualization class. Contains all 18+ plotting methods to generate the final atlas from geometry data. |
| **`visualize_symbol_2d`** | Top-level interface function that wires the geometry engine to the visualizer for a one-shot analysis. |
| **Data Structures** | `Geodesic2D`, `PeriodicOrbit2D`, `CausticStructure`: Dataclasses for storing analysis results (trajectories, orbits, caustic points). |
| **`Utilities2D`** | Helper class for post-analysis, including winding number calculation and KAM tori clustering. |

---

## 📜 License

Licensed under the **Apache License 2.0** — see [LICENSE](./LICENSE) for details.
Copyright © 2025 Philippe Billet

---
[Back to README](../README.md)