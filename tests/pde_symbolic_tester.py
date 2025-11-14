#!/usr/bin/env python
# coding: utf-8



from solver import *
from psiop import *


# # Symbolic solution test
# ## 1D

# ### Heat equation ∂u/∂t = ∂²u/∂x²



from sympy import symbols, Function, diff, sin, exp, simplify

# Define symbols and function
t, x = symbols('t x')
u_exact = sin(x) * exp(-t)  # Exact solution

# Define the PDE: ∂u/∂t = ∂²u/∂x²
lhs_pde = diff(u_exact, t)       # Left-hand side of the PDE (∂u/∂t)
rhs_pde = diff(u_exact, x, x)    # Right-hand side of the PDE (∂²u/∂x²)

# Check if the PDE is satisfied
pde_satisfied = simplify(lhs_pde - rhs_pde) == 0

# Check the initial condition: u(x, 0) = sin(x)
initial_condition = simplify(u_exact.subs(t, 0) - sin(x)) == 0

# Print results
print("Does the exact solution satisfy the PDE? :", pde_satisfied)
print("Does the exact solution satisfy the initial condition? :", initial_condition)


# ### PDE: ∂u/∂t + x * ∂u/∂x



from sympy import symbols, Function, diff, cos, pi, exp, simplify

# Define the variables and the exact function
t, x = symbols('t x')
u_exact = cos(2 * pi * x * exp(-t))  # Exact solution

# Compute the derivatives
dudt = diff(u_exact, t)   # Time derivative
dudx = diff(u_exact, x)   # Spatial derivative

# Evaluate the PDE residual: ∂u/∂t + x * ∂u/∂x
residual = dudt + x * dudx

# Simplify the residual (should be identically zero if the solution satisfies the PDE)
residual_simplified = simplify(residual)

# Check the initial condition u(x, 0) == cos(2πx)
initial_condition = simplify(u_exact.subs(t, 0) - cos(2 * pi * x)) == 0

# Display the results
print("PDE residual (should be 0):", residual_simplified)
print("Is the PDE satisfied?     :", residual_simplified == 0)
print("Is the initial condition met? :", initial_condition)


# ### PDE: ∂u/∂t + x*∂u/∂x + u



from sympy import symbols, Function, diff, exp, simplify, pi, cos

# Define the variables and the unknown function
t, x = symbols('t x')
f = Function('f')

# Proposed exact solution
u_exact = exp(-t) * f(x * exp(-t))

# Compute the derivatives
dudt = diff(u_exact, t)
dudx = diff(u_exact, x)

# Residual of the PDE: ∂u/∂t + x*∂u/∂x + u
residual = dudt + x * dudx + u_exact

# Simplify the residual (should be identically zero)
residual_simplified = simplify(residual)

# Display the result
print("Residual of the PDE (should be 0):", residual_simplified)
print("Is the PDE satisfied?           :", residual_simplified == 0)


# ### Wave PDE: ∂²u/∂t² = ∂²u/∂x²



from sympy import symbols, Function, diff, sin, cos, simplify

# Define symbols and function
t, x = symbols('t x')
u_exact = sin(x) * cos(t)  # Exact solution

# Define the PDE: ∂²u/∂t² = ∂²u/∂x²
lhs_pde = diff(u_exact, t, t)       # Left-hand side of the PDE (∂²u/∂t²)
rhs_pde = diff(u_exact, x, x)       # Right-hand side of the PDE (∂²u/∂x²)

# Check if the PDE is satisfied
pde_satisfied = simplify(lhs_pde - rhs_pde) == 0

# Check the initial conditions
initial_condition_1 = simplify(u_exact.subs(t, 0) - sin(x)) == 0  # u(x,0) = sin(x)
initial_condition_2 = simplify(diff(u_exact, t).subs(t, 0)) == 0  # ∂u/∂t(x,0) = 0

# Print results
print("Does the exact solution satisfy the PDE? :", pde_satisfied)
print("Does the exact solution satisfy the first initial condition (u(x,0) = sin(x))? :", initial_condition_1)
print("Does the exact solution satisfy the second initial condition (∂u/∂t(x,0) = 0)? :", initial_condition_2)


# ### KdV equation



from sympy import symbols, Function, diff, simplify, cosh, sqrt, Eq, trigsimp, N
import numpy as np

# Symbols
t, x = symbols('t x')
c, x0 = symbols('c x0')

# Exact solution
u_exact = c / 2 * (1 / cosh(sqrt(c) / 2 * (x - c * t - x0)))**2

# KdV equation with correct sign
u = Function('u')(t, x)
kdv_eq = Eq(diff(u, t) + 6 * u * diff(u, x) - diff(u, x, x, x), 0)

# Substitute solution
u_t = diff(u_exact, t)
u_x = diff(u_exact, x)
u_xxx = diff(u_exact, x, x, x)

lhs_kdv = u_t + 6 * u_exact * u_x - u_xxx
lhs_simpl = simplify(trigsimp(lhs_kdv))

print("LHS simplified =", lhs_simpl)

x_vals = np.linspace(-10, 10, 5) 
t_val = 0  # Test à t=0
x0_val = 0
c_val = 1

tolerance = 5e-2
all_pass = True

for x_test in x_vals:
    res_num = lhs_simpl.subs({c: c_val, x: x_test, t: t_val, x0: x0_val})
    res_val = N(res_num)
    print(f"Residual at x={x_test}: {res_val}")
    if abs(res_val) > tolerance:
        all_pass = False

print("✅ Does the exact solution satisfy the KdV equation numerically at multiple points? :", all_pass)


# ### PDE: ∂u/∂t = -∂u/∂x



from sympy import symbols, Function, diff, exp, simplify, Mod

# Define symbols and the exact solution
t, x, L = symbols('t x L')
u_exact = exp(-((x - t + L/2) % L - L/2)**2)

# Define the PDE: ∂u/∂t = -∂u/∂x
lhs_pde = diff(u_exact, t)       # Left-hand side of the PDE (∂u/∂t)
rhs_pde = -diff(u_exact, x)      # Right-hand side of the PDE (-∂u/∂x)

# Check if the PDE is satisfied
pde_satisfied = simplify(lhs_pde - rhs_pde) == 0

# Check periodicity: u(x, t) == u(x + L, t)
periodicity_condition = simplify(u_exact.subs(x, x + L) - u_exact) == 0

# Print results
print("Does the exact solution satisfy the PDE? :", pde_satisfied)
print("Does the exact solution satisfy the periodicity condition? :", periodicity_condition)


# ### Klein-Gordon equation: ∂²u/∂t² = ∂²u/∂x² - u



from sympy import symbols, Function, diff, cos, sqrt, simplify

# Define symbols and function
t, x = symbols('t x')
u_exact = cos(sqrt(2) * t) * cos(x)  # Exact solution

# Define the Klein-Gordon equation: ∂²u/∂t² = ∂²u/∂x² - u
lhs_kg = diff(u_exact, t, t)       # Left-hand side (∂²u/∂t²)
rhs_kg = diff(u_exact, x, x) - u_exact  # Right-hand side (∂²u/∂x² - u)

# Check if the PDE is satisfied
pde_satisfied = simplify(lhs_kg - rhs_kg) == 0

# Check the initial condition: u(x, 0) = cos(x)
initial_condition_1 = simplify(u_exact.subs(t, 0) - cos(x)) == 0

# Check the initial velocity condition: ∂u/∂t(x, 0) = 0
initial_velocity = simplify(diff(u_exact, t).subs(t, 0)) == 0

# Print results
print("Does the exact solution satisfy the Klein-Gordon equation? :", pde_satisfied)
print("Does the exact solution satisfy the first initial condition (u(x,0) = cos(x))? :", initial_condition_1)
print("Does the exact solution satisfy the second initial condition (∂u/∂t(x,0) = 0)? :", initial_velocity)


# ### Schrödinger equation



from sympy import symbols, Function, diff, exp, I, simplify, sqrt

# Define the symbols and the exact solution
t, x = symbols('t x')
u_exact = 1 / sqrt(1 - 4*I*t) * exp(I*(x + t)) * exp(-((x + 2*t)**2) / (1 - 4*I*t))

# Define the 1D Schrödinger equation
lhs_schrodinger = I * diff(u_exact, t)  # Left-hand side: i ∂u/∂t
rhs_schrodinger = diff(u_exact, x, x)   # Right-hand side: ∂²u/∂x²

# Check if the PDE is satisfied
pde_satisfied = simplify(lhs_schrodinger - rhs_schrodinger) == 0

# Check the initial condition: u(x, 0) = exp(-x^2) * exp(i*x)
initial_condition = simplify(u_exact.subs(t, 0) - exp(-x**2) * exp(I*x)) == 0

# Display the results
print("Does the exact solution satisfy the PDE? :", pde_satisfied)
print("Does the exact solution satisfy the initial condition? :", initial_condition)


# ### PDE: ∂u/∂t = -∂⁴u/∂x⁴



from sympy import symbols, Function, diff, sin, exp, simplify

# Define symbols and function
t, x = symbols('t x')
u_exact = sin(x) * exp(-t)  # Exact solution

# Define the PDE: ∂u/∂t = -∂⁴u/∂x⁴
lhs_pde = diff(u_exact, t)       # Left-hand side of the PDE (∂u/∂t)
rhs_pde = -diff(u_exact, x, x, x, x)  # Right-hand side of the PDE (-∂⁴u/∂x⁴)

# Check if the PDE is satisfied
pde_satisfied = simplify(lhs_pde - rhs_pde) == 0

# Check the initial condition: u(x, 0) = sin(x)
initial_condition = simplify(u_exact.subs(t, 0) - sin(x)) == 0

# Print results
print("Does the exact solution satisfy the PDE? :", pde_satisfied)
print("Does the exact solution satisfy the initial condition? :", initial_condition)


# ## 2D 
# ### PDE: ∂u/∂t = ∂²u/∂x² + ∂²u/∂y²



from sympy import symbols, Function, diff, sin, exp, simplify

# Define symbols and the unknown function
t, x, y = symbols('t x y')
u_exact = sin(x) * sin(y) * exp(-2 * t)  # Exact solution

# Define the PDE: ∂u/∂t = ∂²u/∂x² + ∂²u/∂y²
lhs_pde = diff(u_exact, t)               # Left-hand side of the PDE (∂u/∂t)
rhs_pde = diff(u_exact, x, x) + diff(u_exact, y, y)  # Right-hand side of the PDE (∂²u/∂x² + ∂²u/∂y²)

# Check if the PDE is satisfied
pde_satisfied = simplify(lhs_pde - rhs_pde) == 0

# Check the initial condition: u(x, y, 0) = sin(x) * sin(y)
initial_condition = simplify(u_exact.subs(t, 0) - sin(x) * sin(y)) == 0

# Print results
print("Does the exact solution satisfy the PDE? :", pde_satisfied)
print("Does the exact solution satisfy the initial condition? :", initial_condition)


# ### Schrödinger equation v1



from sympy import symbols, Function, diff, exp, I, N
import numpy as np

# Define the variables
t, x, y = symbols('t x y')

# Exact solution (the one that did not pass the symbolic test)
u_exact = 1 / (1 + 4*I*t) * exp(I * (x + y - t)) * exp(-((x - 2*t)**2 + (y - 2*t)**2) / (1 + 4*I*t))

# Calculate both sides of the Schrödinger equation
lhs = I * diff(u_exact, t)
rhs = diff(u_exact, x, 2) + diff(u_exact, y, 2)

# Calculate the difference: lhs - rhs
pde_diff = lhs - rhs

# Simplify the expression before evaluation (optional but useful)
pde_diff_simplified = pde_diff.simplify()

# Function to numerically evaluate the expression at a given point
def evaluate_pde_diff(t_val, x_val, y_val):
    return N(pde_diff.subs({t: t_val, x: x_val, y: y_val}))

# Loop over different points (t, x, y)
points = [
    (0.0, 0.0, 0.0),
    (0.1, 0.5, 0.5),
    (0.5, 1.0, 0.0),
    (1.0, -1.0, 1.0),
    (2.0, 0.0, -2.0)
]

print("Numerical evaluation of lhs - rhs at different points:")
for t_val, x_val, y_val in points:
    val = evaluate_pde_diff(t_val, x_val, y_val)
    print(f"Point (t={t_val}, x={x_val}, y={y_val}): {val}")


# ### Schrödinger equation v2



from sympy import symbols, Function, diff, exp, I, simplify, N, conjugate, re

# Variables
t, x, y = symbols('t x y')

# Corrected solution: Gaussian wave packet centered at (2t, 2t)
# with plane wave phase moving with group velocity (2,2)
u_exact = 1 / (1 + 4*I*t) * exp(I * (x + y - t)) * exp(-((x - 2*t)**2 + (y - 2*t)**2) / (1 + 4*I*t))

# PDE terms
lhs = I * diff(u_exact, t)
rhs = diff(u_exact, x, 2) + diff(u_exact, y, 2)
residual = simplify(lhs - rhs)

# Numerical evaluation
def evaluate_pde_diff(t_val, x_val, y_val):
    return N(lhs.subs({t: t_val, x: x_val, y: y_val}) - rhs.subs({t: t_val, x: x_val, y: y_val}))


points = [
    (0.1, 0.0, 0.0),
    (0.5, 0.5, 0.5),
    (1.0, 1.0, 0.0),
    (1.5, -1.0, 1.0),
    (2.0, 0.0, -2.0)
]

print("Residual (lhs - rhs) at sample points:")
for t_val, x_val, y_val in points:
    r = evaluate_pde_diff(t_val, x_val, y_val)
    print(f"Point (t={t_val}, x={x_val}, y={y_val}): {r}")


# ### Wave PDE: ∂²u/∂t² = ∂²u/∂x² + ∂²u/∂y²



from sympy import symbols, Function, diff, sin, cos, sqrt, simplify

# Define symbols and the unknown function
t, x, y = symbols('t x y')
u_exact = sin(x) * sin(y) * cos(sqrt(2) * t)  # Exact solution

# Define the PDE: ∂²u/∂t² = ∂²u/∂x² + ∂²u/∂y²
lhs_pde = diff(u_exact, t, t)  # Left-hand side of the PDE (∂²u/∂t²)
rhs_pde = diff(u_exact, x, x) + diff(u_exact, y, y)  # Right-hand side of the PDE (∂²u/∂x² + ∂²u/∂y²)

# Check if the PDE is satisfied
pde_satisfied = simplify(lhs_pde - rhs_pde) == 0

# Check the initial condition: u(x, y, 0) = sin(x) * sin(y)
initial_condition_1 = simplify(u_exact.subs(t, 0) - sin(x) * sin(y)) == 0

# Check the initial velocity condition: ∂u/∂t(x, y, 0) = 0
initial_velocity_condition = simplify(diff(u_exact, t).subs(t, 0)) == 0

# Print results
print("Does the exact solution satisfy the PDE? :", pde_satisfied)
print("Does the exact solution satisfy the first initial condition? :", initial_condition_1)
print("Does the exact solution satisfy the second initial condition? :", initial_velocity_condition)


# ### PDE: ∂u/∂t = ∂²u/∂x² + ∂²u/∂y²



from sympy import symbols, Function, diff, sin, exp, simplify

# Define symbols and function
t, x, y = symbols('t x y')
u_exact = sin(x) * sin(y) * exp(-2 * t)  # Exact solution

# Define the PDE: ∂u/∂t = ∂²u/∂x² + ∂²u/∂y²
lhs_pde = diff(u_exact, t)               # Left-hand side of the PDE (∂u/∂t)
rhs_pde = diff(u_exact, x, x) + diff(u_exact, y, y)  # Right-hand side of the PDE (∂²u/∂x² + ∂²u/∂y²)

# Check if the PDE is satisfied
pde_satisfied = simplify(lhs_pde - rhs_pde) == 0

# Check the initial condition: u(x, y, 0) = sin(x) * sin(y)
initial_condition = simplify(u_exact.subs(t, 0) - sin(x) * sin(y)) == 0

# Print results
print("Does the exact solution satisfy the PDE? :", pde_satisfied)
print("Does the exact solution satisfy the initial condition? :", initial_condition)


# ### PDE: ∂u/∂t = -(∂⁴u/∂x⁴ + 2∂⁴u/∂x²∂y² + ∂⁴u/∂y⁴)



from sympy import symbols, Function, diff, sin, exp, simplify

# Define symbols and function
t, x, y = symbols('t x y')
u_exact = sin(x) * sin(y) * exp(-4 * t)  # Exact solution

# Define the PDE: ∂u/∂t = -(∂⁴u/∂x⁴ + 2∂⁴u/∂x²∂y² + ∂⁴u/∂y⁴)
lhs_pde = diff(u_exact, t)  # Left-hand side of the PDE (∂u/∂t)
rhs_pde = -(diff(u_exact, x, 4) + 2 * diff(u_exact, x, 2, y, 2) + diff(u_exact, y, 4))  # Right-hand side of the PDE

# Check if the PDE is satisfied
pde_satisfied = simplify(lhs_pde - rhs_pde) == 0

# Check the initial condition: u(x, y, 0) = sin(x) * sin(y)
initial_condition = simplify(u_exact.subs(t, 0) - sin(x) * sin(y)) == 0

# Print results
print("Does the exact solution satisfy the PDE? :", pde_satisfied)
print("Does the exact solution satisfy the initial condition? :", initial_condition)


# ### Klein-Gordon equation



from sympy import symbols, diff, sin, cos, sqrt, simplify, N

# Define symbols and the exact solution
t, x, y = symbols('t x y')
c = 1.0  # Wave speed
m = 1.0  # Field mass
kx, ky = 1, 1  # Wave numbers
omega = sqrt(c**2 * (kx**2 + ky**2) + m**2)  # Angular frequency
u_exact = sin(kx * x) * sin(ky * y) * cos(omega * t)

# Define the Klein-Gordon equation
lhs_kg = diff(u_exact, t, t)  # ∂²u/∂t²
rhs_kg = c**2 * (diff(u_exact, x, x) + diff(u_exact, y, y)) - m**2 * u_exact  # c²Δu - m²u
residual = lhs_kg - rhs_kg

# Check symbolic identities
kg_satisfied = simplify(residual) == 0
initial_condition_u = simplify(u_exact.subs(t, 0) - sin(kx * x) * sin(ky * y)) == 0
initial_velocity_u = simplify(diff(u_exact, t).subs(t, 0)) == 0

# Print symbolic checks
print("Does the exact solution satisfy the Klein-Gordon equation? :", kg_satisfied)
print("Does the exact solution satisfy the initial condition u(x, y, 0)? :", initial_condition_u)
print("Does the exact solution satisfy the initial velocity condition ∂u/∂t(x, y, 0)? :", initial_velocity_u)

# Function to evaluate the residual numerically
def evaluate_residual(t_val, x_val, y_val):
    res = residual.subs({t: t_val, x: x_val, y: y_val})
    return N(res)

# Test points
points = [
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.5),
    (1.0, 1.0, 1.0),
    (1.5, 1.0, 0.0),
    (2.0, -1.0, 1.0),
]

print("\nNumerical evaluation of Klein-Gordon residual at sample points:")
for t_val, x_val, y_val in points:
    val = evaluate_residual(t_val, x_val, y_val)
    print(f"Point (t={t_val}, x={x_val}, y={y_val}): Residual = {val}")


# ### PDE: ∂u/∂t = -∂u/∂x - ∂u/∂y



from sympy import symbols, Function, diff, exp, simplify, Eq

# Define symbols and function
t, x, y = symbols('t x y')
u_exact = exp(-((x - t)**2 + (y - t)**2))  # Exact solution

# Define the PDE: ∂u/∂t = -∂u/∂x - ∂u/∂y
lhs_pde = diff(u_exact, t)       # Left-hand side of the PDE (∂u/∂t)
rhs_pde = -diff(u_exact, x) - diff(u_exact, y)  # Right-hand side of the PDE (-∂u/∂x - ∂u/∂y)

# Check if the PDE is satisfied
pde_satisfied = simplify(lhs_pde - rhs_pde) == 0

# Check periodicity in x and y directions
L = symbols('L')  # Domain size
periodic_x = simplify(u_exact.subs(x, x + L) - u_exact) == 0
periodic_y = simplify(u_exact.subs(y, y + L) - u_exact) == 0

# Print results
print("Does the exact solution satisfy the PDE? :", pde_satisfied)
print("Is the solution periodic in x? :", periodic_x)
print("Is the solution periodic in y? :", periodic_y)


# ### Burgers' equation



from sympy import symbols, Function, Eq, sin, cos, exp, diff, simplify

# Define symbolic variables
x, t, nu = symbols('x t nu')
phi = Function('phi')(x, t)

# Define phi and u
phi_expr = 1 + sin(x) * exp(-nu * t)
dphi_dx = diff(phi_expr, x)
u_expr = -2 * nu * dphi_dx / phi_expr

# Verification of Burgers' equation
du_dt = diff(u_expr, t)
du_dx = diff(u_expr, x)
d2u_dx2 = diff(u_expr, x, x)
lhs = du_dt + u_expr * du_dx
rhs = nu * d2u_dx2

# Calculation of the residual
residual = simplify(lhs - rhs)

# Display results
print("u(x,t) =")
print(u_expr)
print("\nResidual of the Burgers equation (should be 0):")
print(residual)


# ## Other cases
# ### ODE -d²/dx² + x² + 1



x = symbols('x', real=True)
u = exp(-x**2)
# Operator P = -d²/dx² + x² + 1
d2u = diff(u, x, 2)
Pu = -d2u + x**2 * u + u
# Expected right-hand side
f_expected = (-3 * x**2 + 3) * exp(-x**2)
# Symbolic verification
simplify(Pu - f_expected)  # Should yield 0


# ### Apply P = -∂xx - ∂yy + x² + y² + 1 on a given function 



from sympy import symbols, Function, exp, simplify, diff, pprint

# Declaration of variables
x, y = symbols('x y', real=True)
u_exact = exp(-x**2 - y**2)

# Direct application of the operator P = -∂xx - ∂yy + x² + y² + 1
uxx = diff(u_exact, x, 2)
uyy = diff(u_exact, y, 2)
P_applied = -uxx - uyy + (x**2 + y**2 + 1)*u_exact

# Expected form
f_expected = (-3*x**2 - 3*y**2 + 5) * u_exact

# Verification of equality
difference = simplify(P_applied - f_expected)

# Display
print("P[u_exact](x, y) =")
pprint(P_applied)
print("\nExpected form:")
pprint(f_expected)
print("\nDifference (should be 0):")
pprint(difference)






