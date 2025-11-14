# Ψπ (psipy)

> Python library for solving PDEs using pseudodifferential operators and spectral methods, with applications to Hamiltonian mechanics.

`psipy` is a comprehensive, multi-layered Python ecosystem for advanced computational physics and symbolic mathematics. It bridges the gap between the formal symbolic representation of dynamical systems (using `SymPy`) and their numerical simulation and geometric analysis.

The library allows you to define, analyze, and solve complex partial differential equations (PDEs), especially those involving **pseudo-differential operators (ΨDOs)**. It provides tools to move seamlessly between Lagrangian, Hamiltonian, and PDE formulations, solve the resulting equations using spectral methods, and deeply analyze the underlying phase-space geometry and semiclassical properties.

-----

## Core Features

### 🔹 Pseudo-Differential Operators (ΨDOs)

  * **Symbolic & Numerical Framework**: Define operators from a symbol $p(x, \xi)$ or derive them automatically from differential expressions.
  * **Symbolic Calculus**: Perform symbolic composition, compute commutators `[A, B]`, formal adjoints, and asymptotic expansions.
  * **Microlocal Analysis**: Test for ellipticity, compute characteristic sets, and analyze Hamiltonian flows.

### 🔹 PDE Solving

  * **Solve 1D & 2D linear and nonlinear PDEs** (time-dependent or stationary).
  * **Spectral (FFT) methods** with support for periodic and Dirichlet boundary conditions.
  * **Symbolic PDE Parsing**: Automatically parses `SymPy` equations to classify linear, nonlinear, and pseudo-differential terms.
  * **Advanced Time-Stepping**: Includes exponential integrators like **ETD-RK4** (Exponential Time Differencing Runge–Kutta 4th order).

### 🔹 Symbolic Engine & Mechanics

  * **Symbolic Legendre Transforms**: Bidirectional, purely symbolic conversion between Lagrangian ($L$) and Hamiltonian ($H$) formulations.
  * **Hamiltonian-to-PDE Conversion**: Automatically generates formal symbolic PDEs (Schrödinger, Wave, or Stationary) from a given Hamiltonian symbol $H(x, \xi)$.
  * **Extensive Hamiltonian Catalog**: Includes a vast, curated symbolic library of **over 500 Hamiltonians** (e.g., Hénon–Heiles, chaotic, integrable) for research and testing.

### 🔹 Semiclassical & Geometric Analysis

  * **Phase Space Visualization**: Generate comprehensive 1D and 2D "visualization atlases" (up to 18 panels) for any Hamiltonian.
  * **Geometric & Caustic Detection**: Compute classical trajectories (geodesics) and rigorously detect caustics by tracking the full system Jacobian.
  * **Spectral Analysis**: Implements the **Gutzwiller trace formula** and **EBK quantization** to compute the semiclassical energy spectrum from periodic orbits.
  * **Dynamical Systems Tools**: Generate **Poincaré sections**, analyze **KAM tori**, and compute Maslov indices.

-----

## Core Components

The `psipy` ecosystem is composed of several key modules that work together:

| Module | Description |
| :--- | :--- |
| [**`psiop`**](docs/psiop.md) | Symbolic and numerical framework for pseudo-differential operators. Handles symbol definition, quantization, and analysis. |
| [**`solver`**](docs/solver.md) | The main numerical engine. Solves time-dependent and stationary PDEs using spectral methods and exponential integrators. |
| [**`physics`**](docs/physics.md) | A symbolic toolkit for analytical mechanics. Performs Legendre transforms ($L \leftrightarrow H$) and generates formal PDEs from Hamiltonian symbols. |
| [**`hamiltonian_catalog`**](docs/hamiltonian_catalog.md) | A vast, symbolic library of **over 500** Hamiltonian systems for testing, research, and education. |
| [**`geometry_1d`**](docs/geometry_1d.md) | A comprehensive visualization and analysis suite for 1D systems. Computes geodesics, periodic orbits, and the semiclassical spectrum. |
| [**`geometry_2d`**](docs/geometry_2d.md) | An advanced toolkit for 2D systems. Handles rigorous caustic detection, Poincaré sections, and KAM theory visualization. |

-----

## Example Usage

### 1\. Solve a Time-Dependent PDE

Quickly define a PDE symbolically and solve it numerically.

```python
from imports import * 
from solver import PDESolver

# Define symbolic variables
t, x = symbols('t x')
u = Function('u')

# 1. Define the PDE symbolically (e.g., u_t = u_xx + u^2)
eq = Eq(diff(u(t, x), t), diff(u(t, x), x, 2) + u(t, x)**2)

# Define an initial condition
def initial(x_grid):
    return np.sin(x_grid)

# 2. Set up and run the solver
solver = PDESolver(eq)
solver.setup(Lx=2*np.pi, Nx=128, Lt=1.0, Nt=1000, initial_condition=initial)
solver.solve()

# 3. Visualization
ani = solver.animate(component='abs')
HTML(ani.to_jshtml())

```

### 2\. Explore the Hamiltonian Catalog

Fetch and analyze a symbolic Hamiltonian from the built-in library.

```python
from hamiltonian_catalog import *

# 1. Get a specific system (e.g., Hénon–Heiles)
H, variables, metadata = get_hamiltonian("henon_heiles")
print(H)

# H -> (xi**2 + eta**2)/2 + (x**2 + y**2)/2 + alpha*(x**2*y - y**3/3)

# 2. Pretty-print its information
print_hamiltonian_info("henon_heiles")

# 3. Search the catalog by keyword
search_hamiltonians("pendulum")
# ['double_pendulum_reduced', 'driven_pendulum', ...]
```

### 3\. Analyze 1D Hamiltonian Geometry

Define a 1D Hamiltonian and generate its complete geometric and semiclassical analysis.

```python
import sympy as sp
import matplotlib.pyplot as plt
from geometry_1d import visualize_symbol

# 1. Define symbolic variables
x, xi = sp.symbols('x xi', real=True)

# 2. Define the Hamiltonian (e.g., Harmonic Oscillator)
H_symbol = xi**2 + x**2

# 3. Set visualization parameters
geodesics_params = [
    (1.0, 0.0, 10.0, 'red'),  # (x0, xi0, t_max, color)
    (0.0, 1.5, 10.0, 'blue')
]

# 4. Run the complete analysis and display the 15-panel figure
fig, geos, orbits, spec = visualize_symbol(
    symbol=H_symbol,
    x_range=(-3, 3),
    xi_range=(-3, 3),
    geodesics_params=geodesics_params,
    E_range=(0.5, 4.0),
    hbar=0.1
)

plt.show()
```

-----

## License

Licensed under the **Apache License 2.0**.