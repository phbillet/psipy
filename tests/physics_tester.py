#!/usr/bin/env python
# coding: utf-8







from physics import *


# ## Convert Lagrangian into Hamiltonian
# ### 1D



x, u, p = sp.symbols('x u p', real=True)
L = 0.5*p**2 + sp.sin(u)

H, (xi,) = LagrangianHamiltonianConverter.L_to_H(L, (x,), u, (p,))
print("H(x,u,ξ) =", H)
# H(x,u,ξ) = 0.5*xi**2 - sin(u)


# ### 2D



x, y, u, p_x, p_y = sp.symbols('x y u p_x p_y', real=True)
L = 0.5*(p_x**2 + p_y**2) + 0.5*u**2

H, (xi, eta) = LagrangianHamiltonianConverter.L_to_H(L, (x, y), u, (p_x, p_y))
print("H(x,y,u,ξ,η) =", H)
# H(x,y,u,ξ,η) = 0.5*(xi**2 + eta**2) - 0.5*u**2




x, y, u, p_x, p_y = sp.symbols('x y u p_x p_y', real=True)
L = 0.5*(p_x**2 + p_y**2) + 0.5*u**2

H, (xi, eta) = LagrangianHamiltonianConverter.L_to_H(L, (x, y), u, (p_x, p_y))
print("H(x,y,u,ξ,η) =", H)
# H(x,y,u,ξ,η) = 0.5*(xi**2 + eta**2) - 0.5*u**2


# ### Influence of `return_symbol_only`



x, y, u, p_x, p_y = sp.symbols('x y u p_x p_y', real=True)
L = 0.5*(p_x**2 + p_y**2) + 0.5*u**2

H, (xi, eta) = LagrangianHamiltonianConverter.L_to_H(L, (x, y), u, (p_x, p_y), return_symbol_only=True)
print("H(x,y,u,ξ,η) =", H)
# H(x,y,u,ξ,η) = 0.5*(xi**2 + eta**2) - 0.5*u**2


# ### From Lagrangian to Hamiltoniana nd vice versa



x, u, p = sp.symbols('x u p', real=True)
L = 0.5*p**2 + 0.5*u**2

# Aller de L → H → L
H, (xi,) = LagrangianHamiltonianConverter.L_to_H(L, (x,), u, (p,))
L_back, (p_back,) = LagrangianHamiltonianConverter.H_to_L(H, (x,), u, (xi,))
print("Original L:", sp.simplify(L))
print("Reconstructed L:", sp.simplify(L_back))


# ## Convert Hamiltonian into EDPs 
# ### 1D



# ======== 1D ========
x, t, xi = sp.symbols("x t xi", real=True)
u = sp.Function("u")(t, x)
V = sp.Function("V")(x)
H1D = 0.5*xi**2 + V + sp.Abs(xi)

print("\n--- 1D Stationary ---")
res2 = HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(H1D, (x,), t, u, mode="stationary")
print(res2["pde"])

print("\n--- 1D Schrödinger ---")
res1 = HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(H1D, (x,), t, u, mode="schrodinger")
print(res1["pde"])

print("\n--- 1D Wave ---")
res3 = HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(H1D, (x,), t, u, mode="wave")
print(res3["pde"])



# ### 2D



# ======== 2D ========
x, y, t = sp.symbols("x y t", real=True)
u = sp.Function("u")(t, x, y)
V2 = sp.Function("V")(x, y)
xi, eta = sp.symbols("xi eta", real=True)
H2D = 0.5*(xi**2 + eta**2) + V2 + sp.sqrt(xi**2 + 1)

print("\n--- 2D Stationary ---")
res5 = HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(H2D, (x, y), t, u, mode="stationary")
print(res5["pde"])

print("\n--- 2D Schrödinger ---")
res4 = HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(H2D, (x, y), t, u, mode="schrodinger")
print(res4["pde"])

print("\n--- 2D Wave ---")
res6 = HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(H2D, (x, y), t, u, mode="wave")
print(res6["pde"])






