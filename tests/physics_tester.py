#!/usr/bin/env python
# coding: utf-8

from physics import *
import sympy as sp

print("\n====================================================")
print("  TEST SUITE FOR Lagrangian–Hamiltonian CONVERTER")
print("====================================================\n")

# ------------------------------------------------------------
# 1) BASIC 1D QUADRATIC LAGRANGIAN (SHO)
# ------------------------------------------------------------

print("\n--- Test 1: 1D quadratic L = 1/2 p^2 + sin(u) (analytic path expected) ---")

x, u, p = sp.symbols('x u p', real=True)
L = 0.5*p**2 + sp.sin(u)

H, (xi,) = LagrangianHamiltonianConverter.L_to_H(L, (x,), u, (p,))
print("Computed Hamiltonian H(x,u,xi):", H)
print("Expected: 1/2 xi^2 - sin(u)")

# ------------------------------------------------------------
# 2) BASIC 2D QUADRATIC LAGRANGIAN
# ------------------------------------------------------------

print("\n--- Test 2: 2D quadratic L = 1/2 (p_x^2 + p_y^2) + 1/2 u^2 ---")

x, y, u, p_x, p_y = sp.symbols('x y u p_x p_y', real=True)
L = 0.5*(p_x**2 + p_y**2) + 0.5*u**2

H, (xi, eta) = LagrangianHamiltonianConverter.L_to_H(L, (x, y), u, (p_x, p_y))
print("Computed Hamiltonian H(x,y,u,xi,eta):", H)
print("Expected: 1/2 (xi^2 + eta^2) - 1/2 u^2")

# ------------------------------------------------------------
# 3) RETURN SYMBOL ONLY
# ------------------------------------------------------------

print("\n--- Test 3: return_symbol_only=True (u eliminated) ---")

H_sym, (xi, eta) = LagrangianHamiltonianConverter.L_to_H(
    L, (x, y), u, (p_x, p_y), return_symbol_only=True
)
print("H symbol-only:", H_sym)
print("Expected (no u): 1/2 (xi^2 + eta^2) - 1/2 * 0^2 == 1/2 (xi^2 + eta^2)")

# ------------------------------------------------------------
# 4) CONSISTENCY: L → H → L
# ------------------------------------------------------------

print("\n--- Test 4: Consistency check L → H → L ---")

x, u, p = sp.symbols('x u p', real=True)
L = 0.5*p**2 + 0.5*u**2

H, (xi,) = LagrangianHamiltonianConverter.L_to_H(L, (x,), u, (p,))
L_back, (p_back,) = LagrangianHamiltonianConverter.H_to_L(H, (x,), u, (xi,))
print("Original L:", sp.simplify(L))
print("Reconstructed L:", sp.simplify(L_back))
print("Note: L should match up to algebraic simplification.")

# ------------------------------------------------------------
# 5) NON-INVERTIBLE CASE (Hessian = 0) → EXPECT FAILURE
# ------------------------------------------------------------

print("\n--- Test 5: non-invertible Lagrangian L = p^4 (Hessian zero) ---")

x, u, p = sp.symbols('x u p', real=True)
L_noninv = p**4

try:
    H_bad, (xi,) = LagrangianHamiltonianConverter.L_to_H(L_noninv, (x,), u, (p,))
    print("ERROR: This should not be invertible but inversion succeeded:", H_bad)
except Exception as e:
    print("Expected failure:", e)

# ------------------------------------------------------------
# 6) FORCED INVERSION (force=True) ON NON-INVERTIBLE L
# ------------------------------------------------------------

print("\n--- Test 6: forced inversion for L = p^4 ---")

try:
    H_forced, (xi,) = LagrangianHamiltonianConverter.L_to_H(
        L_noninv, (x,), u, (p,), force=True
    )
    print("Forced inversion result (may be messy):", H_forced)
except Exception as e:
    print("Forced inversion still failed:", e)

# ------------------------------------------------------------
# 7) HAMILTONIAN → SYMBOLIC PDE GENERATION (1D)
# ------------------------------------------------------------

print("\n====================================================")
print("  TEST SUITE FOR Hamiltonian → Symbolic PDE")
print("====================================================\n")

print("\n--- Test 7: 1D PDE modes with nonlocal term (|xi|) ---")

x, t, xi = sp.symbols("x t xi", real=True)
u = sp.Function("u")(t, x)
V = sp.Function("V")(x)
H1D = 0.5*xi**2 + V + sp.Abs(xi)

print("\nStationary PDE:")
res_stationary = HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(
    H1D, (x,), t, u, mode="stationary"
)
print(res_stationary["pde"])

print("\nSchrödinger PDE:")
res_sch = HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(
    H1D, (x,), t, u, mode="schrodinger"
)
print(res_sch["pde"])

print("\nWave PDE:")
res_wave = HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(
    H1D, (x,), t, u, mode="wave"
)
print(res_wave["pde"])

# ------------------------------------------------------------
# 8) 2D PDE TEST
# ------------------------------------------------------------

print("\n--- Test 8: 2D PDE modes (with sqrt nonlocal term) ---")

x, y, t = sp.symbols("x y t", real=True)
u = sp.Function("u")(t, x, y)
V2 = sp.Function("V")(x, y)
xi, eta = sp.symbols("xi eta", real=True)
H2D = 0.5*(xi**2 + eta**2) + V2 + sp.sqrt(xi**2 + 1)

print("\n2D Stationary:")
print(HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(
    H2D, (x, y), t, u, mode="stationary"
)["pde"])

print("\n2D Schrödinger:")
print(HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(
    H2D, (x, y), t, u, mode="schrodinger"
)["pde"])

print("\n2D Wave:")
print(HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(
    H2D, (x, y), t, u, mode="wave"
)["pde"])

print("\n====================================================")
print("                 Test suite finished.")
print("====================================================\n")
