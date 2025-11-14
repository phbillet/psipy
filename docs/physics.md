# SymPhysics: Analytical & Symbolic Mechanics

[Back to README](../README.md)

## Overview

**`SymPhysics`** is a symbolic toolkit, built on `SymPy`, for operations in analytical mechanics and formal partial differential equation (PDE) theory.

This package provides a bridge between classical Lagrangian/Hamiltonian mechanics and the symbolic representation of PDEs. It allows users to perform bidirectional **Legendre transforms** ($L \leftrightarrow H$) and to **construct formal pseudo-differential PDEs** (e.g., Schrödinger, wave) from a given Hamiltonian symbol $H(x, \xi)$.

This module is designed for symbolic analysis, manipulation, and the generation of formal equations *prior* to numerical discretization.

-----

## Key Features

  * 🔹 **Symbolic Legendre Transform**: Bidirectional, purely symbolic conversion between Lagrangian $L(x, u, p)$ and Hamiltonian $H(x, u, \xi)$ formulations.
  * 🔹 **Hamiltonian-to-PDE Conversion**: Generates formal symbolic PDEs from a Hamiltonian symbol $H(x, \xi)$ using a `psiOp(H, u)` representation.
  * 🔹 **Multiple PDE Symmetries**: Supports "schrödinger" ($i \partial_t u = \psi Op(H, u)$), "wave" ($\partial_{tt} u + \psi Op(H, u) = 0$), and "stationary" ($\psi Op(H, u) = E u$) formulations.
  * 🔹 **Hamiltonian Analysis**: Automatically decomposes Hamiltonian symbols into polynomial (local, differential) and non-local (pseudo-differential) components.
  * 🔹 **1D & 2D Support**: All converters are compatible with one and two spatial dimensions.

-----

## Capabilities

  * Derive the Hamiltonian $H$ from a given Lagrangian $L$ (e.g., for a field theory or classical particle).
  * Derive the Lagrangian $L$ from a given Hamiltonian $H$.
  * **Generate a pseudo-differential symbol** $H(x, \xi)$ directly from a Lagrangian $L(p)$ that is quadratic in momentum.
  * Formulate symbolic, time-dependent PDEs (Schrödinger, Wave) from a given physical Hamiltonian.
  * Formulate symbolic stationary (eigenvalue) problems.
  * Analyze the symbolic structure of a Hamiltonian (e.g., separating $V(x) + \xi^2$ from $\sqrt{\xi^2 + m^2}$).

-----

## Example Usage

### 1\. Lagrangian to Hamiltonian (Legendre Transform)

Convert a 1D relativistic Lagrangian $L = -\sqrt{1-p^2}$ to its corresponding Hamiltonian $H$.

```python
import sympy as sp
from physics import LagrangianHamiltonianConverter

# Define 1D symbols
x = sp.Symbol('x', real=True)
u = sp.Function('u')(x)
p = sp.Symbol('p', real=True)

# Relativistic Lagrangian: L(p) = -m * sqrt(1 - p^2/c^2)
# (using m=1, c=1 for simplicity)
L_expr = -sp.sqrt(1 - p**2)
coords = (x,)
p_vars = (p,)

# Perform the conversion
H_expr, (xi,) = LagrangianHamiltonianConverter.L_to_H(L_expr, coords, u, p_vars)

print(f"Lagrangian: L = {L_expr}")
print(f"Hamiltonian: H = {H_expr}")
# Output:
# Lagrangian: L = -sqrt(1 - p**2)
# Hamiltonian: H = sqrt(xi**2 + 1)
```

### 2\. Hamiltonian to Symbolic PDE (Schrödinger-type)

Generate the formal Schrödinger equation for a 1D Hamiltonian $H(x, \xi) = \frac{1}{2}\xi^2 + V(x)$.

```python
import sympy as sp
from physics import HamiltonianSymbolicConverter

# Define 1D space, time, and phase space symbols
x, t, xi = sp.symbols('x t xi', real=True)
V = sp.Function('V')(x)
u = sp.Function('u')(x, t)

# Standard Hamiltonian: Kinetic + Potential
H_symbol = 0.5*xi**2 + V

# Generate the symbolic PDE
pde_data = HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(
    H_symbol, (x,), t, u, mode="schrodinger"
)

print(pde_data["pde"])
print(f"Formal string: {pde_data['formal_string']}")

# Output:
# Eq(I*Derivative(u(x, t), t), psiOp(0.5*xi**2 + V(x), u(x, t)))
# Formal string: i ∂_t u = ψOp(H, u)   (H = H(x; xi))
```

-----

## Structure and Internals

| Class | Description |
| :--- | :--- |
| `LagrangianHamiltonianConverter` | Handles symbolic Legendre transforms ($L \leftrightarrow H$) in 1D and 2D. |
| `HamiltonianSymbolicConverter` | Connects Hamiltonian symbols $H(x, \xi)$ to formal PDE representations (`psiOp`). |

---

## 📜 License

Licensed under the **Apache License 2.0** — see [LICENSE](./LICENSE) for details.
Copyright © 2025 Philippe Billet

---

[Back to README](../README.md)