# PDE Solver — Spectral and Pseudo-Differential Operator Framework

[Back to README](../README.md)

## Overview

**`PDESolver`** is a general-purpose spectral solver for **linear and nonlinear partial differential equations (PDEs)** in one and two spatial dimensions.
It supports both **periodic and non-periodic (Dirichlet)** boundary conditions and integrates **pseudo-differential operators** through symbolic quantization methods (Kohn–Nirenberg formulation).

This package bridges **symbolic PDE specification (via SymPy)** and **numerical spectral simulation (via FFTs)**, enabling seamless analysis of differential and pseudo-differential dynamics.

---

## Key Features

* 🔹 **Symbolic PDE Parsing** with automatic classification of linear, nonlinear, and pseudo-differential terms.
* 🔹 **Spectral Discretization** using fast Fourier transforms (1D & 2D).
* 🔹 **Time Integration Schemes**:

  * Exponential time stepping (`default`)
  * **ETD-RK4** (Exponential Time Differencing Runge–Kutta 4th order)
  * Second-order leapfrog schemes for wave-like PDEs
* 🔹 **Pseudo-Differential Operator Support**:

  * Symbolic operators (`psiOp`) with full symbolic evaluation
  * **Kohn–Nirenberg quantization** (periodic & non-periodic)
  * Asymptotic inversion for stationary elliptic problems
* 🔹 **Boundary Conditions**:

  * Periodic (FFT-based)
  * Dirichlet (non-periodic quantization)
* 🔹 **Symbolic and Numerical Analysis Tools**:

  * Dispersion relation and stability analysis
  * Symbol visualization in Fourier space
  * Microlocal and Hamiltonian flow analysis
  * CFL condition verification

---

## Capabilities

* 1D and 2D **time-dependent** or **stationary** PDEs
* Linear or nonlinear PDEs (up to second-order nonlinearities)
* Automatic symbolic detection of:

  * Derivative orders and operators
  * Source terms
  * Pseudo-differential symbols
* Modular FFT setup with **dealiasing filters**
* Support for **exponential propagators** and **explicit pseudo-spectral** evolution
* Energy conservation diagnostics for second-order systems

---

## Example Usage

```python
from PDESolver import *
from sympy import symbols, Function, Eq, diff
import numpy as np

t, x = symbols('t x')
u = Function('u')

# Example: u_t = u_xx + u^2
eq = Eq(diff(u(t, x), t), diff(u(t, x), x, 2) + u(t, x)**2)

def initial(x):
    return np.sin(x)

solver = PDESolver(eq)
solver.setup(Lx=2*np.pi, Nx=128, Lt=1.0, Nt=1000, initial_condition=initial)
solver.solve()
```

---

## Pseudo-Differential Operator Example

```python
from sympy import symbols, Function
from psiop import psiOp

x, xi = symbols('x xi')
u = Function('u')
p_symbol = xi**2 + x*xi
eq = Eq(psiOp(p_symbol)(u(x)), 0)

solver = PDESolver(eq)
solver.setup(Lx=10, Nx=256, boundary_condition='dirichlet')
solver.solve_stationary_psiOp(order=3)
```

---

## Structure and Internals

| Module                       | Description                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------- |
| `solver.py`                  | Main PDE solver (symbolic parsing, setup, time integration)                     |
| `psiop.py`                   | Pseudo-differential operator definitions, symbolic evaluation, and quantization |
| `imports.py`                 | Global imports and numerical constants                                          |
| `PseudoDifferentialOperator` | Handles ψ-operator evaluation, inversion, and visualization                     |

---
---

## 📜 License

Licensed under the **Apache License 2.0** — see [LICENSE](./LICENSE) for details.
Copyright © 2025 Philippe Billet

---

[Back to README](../README.md)
