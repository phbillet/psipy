# ψOp — Symbolic and Numerical Pseudo-Differential Operators

[Back to README](../README.md)

> **ψOp** provides a unified symbolic–numerical framework for defining, analyzing, and visualizing pseudo-differential operators (ΨDOs) in 1D and 2D.
> It bridges *microlocal analysis* and *computational PDE solvers* using **SymPy**, **NumPy**, and **SciPy**.

---

## 🚀 Overview

The package defines a single class, **`PseudoDifferentialOperator`**, which represents general pseudo-differential operators
$$P(x, D)u(x) = \frac{1}{(2\pi)^d} \int e^{i x \xi} p(x,\xi) \hat{u}(\xi), d\xi$$
and supports both **symbolic manipulation** and **numerical evaluation**.

Operators can be:

* **Explicitly defined** by a symbol ( p(x,\xi) )
* **Automatically derived** from a differential expression ( P(u) ) via symbolic computation

This makes ψOp suitable for analytical exploration, PDE prototyping, and visualization of phase-space structures.

---

## ✨ Features

### 🔹 Core Functionality

* **Unified 1D / 2D operator representation**
* **Two definition modes**:

  * `mode='symbol'`: direct symbol input
  * `mode='auto'`: derive symbol from a differential operator
* **Dynamic FFT backend** for evaluating symbols on grids

### 🔹 Symbolic Analysis

* Compute **principal symbols** and **asymptotic expansions**
* Determine **homogeneity degree** and **symbol order**
* Perform **asymptotic composition**:

  * *Kohn–Nirenberg* (`mode='kn'`)
  * *Weyl* (`mode='weyl'`)
* Compute:

  * Symbolic **commutators** `[A,B]`
  * **Formal adjoints** and **left/right inverses**
  * **Exponential operators** `exp(tP)` via asymptotic series

### 🔹 Spectral & Microlocal Tools

* Evaluate and cache ( $p(x,\xi)$ ) or ( $p(x,y,\xi,\eta)$ )
* Test **ellipticity** and **self-adjointness**
* Compute **symplectic (Hamiltonian) flows**
* Derive **characteristic sets** and **gradient norms**

### 🔹 Visualization

* Plot amplitude `|p(x,ξ)|` and phase `arg(p(x,ξ))`
* Display **cotangent fibers** and **characteristic manifolds**
* Visualize **Hamiltonian trajectories** and **flow fields**
* Supports interactive exploration via `matplotlib`

### 🔹 Semiclassical & PDE Integration

* Implements **semiclassical trace formulas** (symbolic or numerical)
* Interfaces with an external **`PDESolver`** class to simulate:

  * First-order evolutions: ∂ₜu = ψOp(p) u
  * Second-order wave-type equations
* Produces **animated phase-space evolutions** of real, imaginary, or modulus components

---

## 🧠 Applications

* Microlocal and semiclassical analysis
* Symbolic study of PDE operators
* Propagation of singularities
* Quantum and wave mechanics
* Asymptotic expansions and quantization (Kohn–Nirenberg, Weyl)
* Numerical validation of symbolic calculus identities

---

## 🧩 Dependencies

| Library                  | Purpose                                       |
| ------------------------ | --------------------------------------------- |
| **SymPy**                | Symbolic differentiation and series expansion |
| **NumPy / SciPy**        | FFTs and numerical evaluation                 |
| **Matplotlib**           | Visualization and animation                   |
| **PDESolver (optional)** | Time integration for PDEs                     |

Python 3.9+ is recommended.

---

## 🧰 Example Usage

```python
from sympy import symbols, Function
from psiop import PseudoDifferentialOperator

# --- Example 1: 1D Laplacian operator (symbol mode)
x, xi = symbols('x xi', real=True)
laplacian = PseudoDifferentialOperator(expr=xi**2, vars_x=[x], mode='symbol')

# --- Example 2: 1D transport operator (auto mode)
u = Function('u')
expr = u(x).diff(x)
transport = PseudoDifferentialOperator(expr=expr, vars_x=[x], var_u=u(x), mode='auto')

# Evaluate the symbol numerically
import numpy as np
X, XI = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-10, 10, 100))
p_vals = laplacian.evaluate(X, None, XI, None)

# Visualize amplitude
laplacian.visualize_symbol_amplitude(x_grid=np.linspace(-2, 2, 100),
                                     xi_grid=np.linspace(-10, 10, 100))
```

---

## 📜 License

Licensed under the **Apache License 2.0** — see [LICENSE](./LICENSE) for details.
Copyright © 2025 Philippe Billet

---
[Back to README](../README.md)
