"""
Ψπ (psipy): Symbolic-Numerical Toolkit for PDEs and Hamiltonian Mechanics

Overview
--------
Welcome to `psipy`, a comprehensive Python ecosystem designed to bridge the gap
between formal symbolic mathematics (via SymPy) and high-performance numerical
simulation (via NumPy/SciPy).

This library provides a unified framework for defining, analyzing, solving, and
visualizing complex problems in:
- Partial Differential Equations (PDEs)
- Pseudo-Differential Operators (ΨDOs)
- Hamiltonian and Lagrangian Mechanics
- Semiclassical and Microlocal Analysis

The core philosophy is to allow users to move seamlessly from a formal symbolic
definition—such as a Lagrangian, a Hamiltonian from the included catalog, or a
PDE written in SymPy—to a robust numerical analysis, such as solving the PDE's
evolution, visualizing its phase-space geometry, or computing its semiclassical
spectrum.

Core Components
---------------
The `psipy` ecosystem is composed of several powerful, interoperable modules:

- **`PDESolver`**:
  The main numerical engine. It parses symbolic PDEs and solves
  1D/2D, linear/nonlinear, time-dependent or stationary equations.
  It uses spectral (FFT) methods with high-order exponential integrators
  (like ETD-RK4) for robust time evolution.

- **`ψOp` (psiop)**:
  A complete symbolic and numerical framework for Pseudo-Differential
  Operators (ΨDOs). It supports symbolic calculus (composition,
  commutators, adjoints) and microlocal analysis (ellipticity,
  characteristic sets), bridging formal definitions with numerical
  evaluation on grids.

- **`SymPhysics`**:
  A symbolic toolkit for analytical mechanics. It performs purely
  symbolic Legendre transforms ($L \leftrightarrow H$) and can automatically
  generate formal symbolic PDEs (e.g., Schrödinger, Wave) from any
  given Hamiltonian symbol.

- **`HamiltonianCatalog`**:
  A vast, curated, and searchable symbolic database of **over 600**
  Hamiltonian systems. It spans classical mechanics, quantum chaos,
  biophysics, and more, providing a rich testbed for research
  and education.

- **`SymbolGeometry1D`**:
  A comprehensive analysis and visualization suite for 1D Hamiltonian
  systems. It connects classical geometry to quantum spectra by
  computing classical trajectories, periodic orbits, and the
  semiclassical energy spectrum via the **Gutzwiller trace formula**
  and **EBK quantization**.

- **`SymbolGeometry2D`**:
  An advanced 2D analysis toolkit for visualizing dynamical systems.
  It performs rigorous **caustic detection** by tracking the full 4x4 Jacobian,
  generates **Poincaré sections**, and analyzes **KAM tori**, providing
  a deep dive into 2D phase space geometry.

Typical Workflow
----------------
A common use case involves combining all modules:

1.  **Select a System**: Fetch a complex Hamiltonian (e.g., "henon_heiles")
    from the `HamiltonianCatalog`.
2.  **Formulate the PDE**: Use `SymPhysics` to automatically generate the
    corresponding symbolic Schrödinger equation.
3.  **Analyze Geometry**: Pass the Hamiltonian symbol to `SymbolGeometry2D`
    to visualize its classical trajectories, Poincaré sections, and
    chaotic regions.
4.  **Solve Dynamics**: Pass the symbolic PDE to the `PDESolver` to
    simulate the quantum wave function's evolution in time.

Example: Solving a Pseudo-Differential PDE
-------------------------------------------
This example defines a 1D Schrödinger-type equation with a non-local,
relativistic kinetic term, $i \partial_t u = \sqrt{1 - \partial_x^2} u$.

```python
import sympy as sp
import numpy as np
from psipy.solver import PDESolver
from psipy.operators import psiOp
from sympy import symbols, Function, Eq, diff

# 1. Define symbolic variables
t, x, xi = symbols('t x xi', real=True)
u = Function('u')

# 2. Define the PDE symbolically
# The symbol for the operator sqrt(1 - d_xx) is p(ξ) = sqrt(1 + ξ²)
# (using the Fourier convention p(ξ) -> op(ξ) -> -d_xx)
p_symbol = sp.sqrt(1 + xi**2)

# The equation is: i * u_t = psiOp(p_symbol) * u
equation = Eq(sp.I * diff(u(t,x), t), psiOp(p_symbol, u(t,x)))

# 3. Create the solver
solver = PDESolver(equation)

# 4. Setup the simulation domain and initial condition
initial_packet = lambda x: np.exp(-(x - np.pi)**2 / 0.5) * np.exp(1j * 5.0 * x)

solver.setup(
    Lx=2*np.pi, Nx=256,fig, data
    Lt=4.0, Nt=1000,
    initial_condition=initial_packet,
    boundary_condition='periodic'
)

# 5. Solve the PDE
solver.solve()

# The solution is available in solver.solution
# An animation can be created with:
# ani = solver.animate(component='real', interval=20)
"""
from importlib.metadata import version

# Imports publics
from .physics import *
from .solver import *
from .geometry_1d import *
from .geometry_2d import *
from .psiop import *
from .hamiltonian_catalog import *

# Version du package
__version__ = version("psipy")

# Liste des noms exposés par `from psipy import *`
__all__ = [
    "PseudoDifferentialOperators",
    "PDESolver",
    "LagrangianHamiltonianConverter",
    "HamiltonianSymbolicConverter",
    "SymbolGeometry2D",
    "SymbolVisualizer2D",
    "Utilities2D",
    "SymbolGeometry",
    "SymbolVisualizer",
    "SpectralAnalysis",
]
