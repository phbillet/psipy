#!/usr/bin/env python
# coding: utf-8







from solver import *
from psiop import *




# Declaration of necessary variables
f = Function('f')
g = Function('g')


# # Symbol manipulation

# ## Basic input cases

# ### 1D



# Define SymPy variables
x = symbols('x', real=True)
xi = symbols('xi', real=True)

# Symbolic expression p(x, ξ) = x * ξ
expr_symbol = x * xi

# Creation of the operator in 'symbol' mode
try:
    op = PseudoDifferentialOperator(expr=expr_symbol, vars_x=[x], mode='symbol')

    # Check that p_func is well defined
    assert callable(op.p_func), "p_func should be a function"

    # Numerical evaluation test
    x_vals = np.array([1.0, 2.0])
    xi_vals = np.array([3.0, 4.0])
    result = op.p_func(x_vals[:, None], xi_vals[None, :])  # shape (2, 2)

    expected = x_vals[:, None] * xi_vals[None, :]
    assert np.allclose(result, expected), f"Incorrect result: {result}, expected: {expected}"
    print("✅ 'symbol' mode test successful")
except Exception as e:
    print(f"❌ 'symbol' mode test failed: {e}")




# Sympy Variables
x = symbols('x', real=True)
u = Function('u')

# Differential operator: ∂u/∂x
expr_auto = diff(u(x), x)

# Creation of the operator in 'auto' mode
try:
    op = PseudoDifferentialOperator(expr=expr_auto, vars_x=[x], var_u=u(x), mode='auto')

    # Check that p_func is well defined
    assert callable(op.p_func), "p_func should be a function"

    # Numerical evaluation test
    x_vals = np.array([1.0, 2.0])
    xi_vals = np.array([3.0, 4.0])
    result = op.p_func(x_vals[:, None], xi_vals[None, :])

    expected = 1j * xi_vals[None, :]  # Symbol: iξ
    assert np.allclose(result, expected, atol=1e-6), f"Incorrect result: {result}, expected: {expected}"
    print("✅ 'auto' mode test successful")
except Exception as e:
    print(f"❌ 'auto' mode test failed: {e}")




x = symbols('x', real=True)
expr = x  # Any expression
# Test 1: invalid mode
try:
    op = PseudoDifferentialOperator(expr=expr, vars_x=[x], mode='invalid_mode')
    print("❌ Test failed: no error raised for invalid mode")
except ValueError as e:
    if "mode must be 'auto' or 'symbol'" in str(e):
        print("✅ Correct exception raised for invalid mode")
    else:
        print(f"❌ Wrong exception raised for invalid mode: {e}")
# Test 2: 'auto' mode without var_u
try:
    op = PseudoDifferentialOperator(expr=expr, vars_x=[x], mode='auto')
    print("❌ Test failed: no error raised for missing var_u")
except ValueError as e:
    if "var_u must be provided in mode='auto'" in str(e):
        print("✅ Correct exception raised for missing var_u")
    else:
        print(f"❌ Wrong exception raised for missing var_u: {e}")


# ### 2D



# Sympy Variables
x, y = symbols('x y', real=True)
xi, eta = symbols('xi eta', real=True)

# Symbolic expression p(x, y, ξ, η) = x*y*ξ + η**2
expr_symbol = x * y * xi + eta**2

# Creation of the operator in 'symbol' mode
try:
    op = PseudoDifferentialOperator(expr=expr_symbol, vars_x=[x, y], mode='symbol')

    # Check that p_func is well defined
    assert callable(op.p_func), "p_func should be a function"

    # Numerical evaluation test
    x_vals = np.array([1.0, 2.0])
    y_vals = np.array([0.5, 1.5])
    xi_vals = np.array([2.0, 3.0])
    eta_vals = np.array([1.0, 4.0])

    result = op.p_func(
        x_vals[:, None, None, None],
        y_vals[None, :, None, None],
        xi_vals[None, None, :, None],
        eta_vals[None, None, None, :]
    )

    expected = (
        x_vals[:, None, None, None] *
        y_vals[None, :, None, None] *
        xi_vals[None, None, :, None] +
        eta_vals[None, None, None, :]**2
    )

    assert np.allclose(result, expected), f"Incorrect result: {result}, expected: {expected}"
    print("✅ 2D 'symbol' mode test successful")
except Exception as e:
    print(f"❌ 2D 'symbol' mode test failed: {e}")




# Sympy Variables
x, y = symbols('x y', real=True)
u = Function('u')

# Differential operator: Δu = ∂²u/∂x² + ∂²u/∂y²
expr_auto = diff(u(x, y), x, 2) + diff(u(x, y), y, 2)

# Creation of the operator in 'auto' mode
try:
    op = PseudoDifferentialOperator(expr=expr_auto, vars_x=[x, y], var_u=u(x, y), mode='auto')

    # Check that p_func is well defined
    assert callable(op.p_func), "p_func should be a function"

    # Numerical evaluation test
    x_vals = np.array([1.0, 2.0])
    y_vals = np.array([0.5, 1.5])
    xi_vals = np.array([2.0, 3.0])
    eta_vals = np.array([1.0, 4.0])

    result = op.p_func(
        x_vals[:, None, None, None],
        y_vals[None, :, None, None],
        xi_vals[None, None, :, None],
        eta_vals[None, None, None, :]
    )

    expected = -(xi_vals[None, None, :, None]**2 + eta_vals[None, None, None, :]**2)

    assert np.allclose(result, expected), f"Incorrect result: {result}, expected: {expected}"
    print("✅ 2D 'auto' mode test successful")
except Exception as e:
    print(f"❌ 2D 'auto' mode test failed: {e}")




# Sympy Variables
x, y = symbols('x y', real=True)
expr = x * y  # Any expression

try:
    op = PseudoDifferentialOperator(expr=expr, vars_x=[x, y], mode='invalid_mode')
    print("❌ Test failure: no error raised for invalid mode")
except ValueError as e:
    if "mode must be 'auto' or 'symbol'" in str(e):
        print("✅ Correct exception raised for invalid mode")
    else:
        print(f"❌ Wrong exception raised for invalid mode: {e}")




# Sympy variables in 3D
x, y, z = symbols('x y z', real=True)
u = Function('u')

# Arbitrary differential operator
expr_3d = u(x, y, z).diff(x, 2) + u(x, y, z).diff(y, 2) + u(x, y, z).diff(z, 2)

try:
    op = PseudoDifferentialOperator(expr=expr_3d, vars_x=[x, y, z], var_u=u(x, y, z), mode='auto')
    print("❌ Test failure: no error raised for dim = 3")
except NotImplementedError as e:
    if "Only 1D and 2D supported" in str(e):
        print("✅ Correct exception raised for dimension 3")
    else:
        print(f"❌ Wrong exception raised: {e}")


# ## Order (degree of homogeneity) of the symbol
# ### 1D



x, xi = symbols('x xi', real=True, positive=True)
op = PseudoDifferentialOperator(expr=xi**2, vars_x=[x], mode='symbol')
is_hom = op.is_homogeneous()
print(is_hom)
order = op.symbol_order()
print("1D Estimated order:", order)




x, xi = symbols('x xi', real=True, positive=True)
op = PseudoDifferentialOperator(expr=(x*xi)**2, vars_x=[x], mode='symbol')
is_hom = op.is_homogeneous()
print(is_hom)
order = op.symbol_order()
print("1D Estimated order:", order)




x, xi = symbols('x xi', real=True, positive=True)
op = PseudoDifferentialOperator(expr=sin(x)+xi**2, vars_x=[x], mode='symbol')
is_hom = op.is_homogeneous()
print(is_hom)
order = op.symbol_order()
print("1D Estimated order:", order)




x, xi = symbols('x xi', real=True, positive=True)

expr = exp(x*xi / (x**2 + xi**2))
op = PseudoDifferentialOperator(expr=expr, vars_x=[x], mode='symbol')
op_expanded = PseudoDifferentialOperator(expr=op.asymptotic_expansion(order=4), vars_x=[x], mode='symbol')

print("estimated order=", op.symbol_order())
print("estimated order (expanded)= ", op_expanded.symbol_order())




# code
expr = exp(1 / xi)
op = PseudoDifferentialOperator(expr=expr, vars_x=[x], mode='symbol')
print("estimated order =", op.symbol_order())




expr = xi**3
op = PseudoDifferentialOperator(expr=expr, vars_x=[x], mode='symbol')
print("estimated order =", op.symbol_order())  # ← devrait renvoyer 3


# ### 2D



x, y = symbols('x y', real=True)
xi, eta = symbols('xi eta', real=True, positive=True) 
op2d = PseudoDifferentialOperator(expr=xi**2 + eta**2, vars_x=[x, y], mode='symbol')
order = op2d.symbol_order()
is_hom = op2d.is_homogeneous()
print(is_hom)
print("2D Estimated order: ", order)




op = PseudoDifferentialOperator(expr=(xi**2 + eta**2)**(1/3), vars_x=[x, y])
is_hom = op.is_homogeneous()
print(is_hom)
print("2D Estimated order: ", op.symbol_order())




op = PseudoDifferentialOperator(expr=(xi**2 + eta**2 + 1)**(1/3), vars_x=[x, y])
is_hom = op.is_homogeneous()
print(is_hom)
print("2D Estimated order: ", op.symbol_order())




expr = exp(x*xi / (y**2 + eta**2))
op = PseudoDifferentialOperator(expr=expr, vars_x=[x, y], mode='symbol')

print("estimated order=", op.symbol_order(tol=0.1))




expr = sqrt(xi**2 + eta**2)

op = PseudoDifferentialOperator(expr=expr, vars_x=[x, y], mode='symbol')
print("estimated order =", op.symbol_order())




expr = exp(-(x**2 + y**2) / (xi**2 + eta**2))

op = PseudoDifferentialOperator(expr=expr, vars_x=[x, y], mode='symbol')
print("estimated order =", op.symbol_order(tol=1e-3))


# ## Homogeneous principal symbol of the operator
# ### 1D



x, xi = symbols('x xi')
xi = symbols('xi', real=True, positive=True)

expr = xi**2 + sqrt(xi**2 + x**2)
p = PseudoDifferentialOperator(expr=expr, vars_x=[x], mode='symbol')
print("Order 1:")
print(p.principal_symbol(order=1))  
print("Order 2:")
print(p.principal_symbol(order=2))


# ### 2D



x, y = symbols('x y')        # 2 spatial variables
xi, eta = symbols('xi eta', real=True, positive=True)

p_ordre_1 = (xi**2 + eta**2 + 1)**(1/3)

p = PseudoDifferentialOperator(expr=p_ordre_1, vars_x=[x, y], mode='symbol')

print("Order 1:")
print(p.principal_symbol(order=1))  

print("Order 2:")
print(p.principal_symbol(order=2))  


# ## Asymptotic expansion
# ### 1D



x, xi = symbols('x xi', real=True, positive=True)

expr = exp(x*xi / (x**2 + xi**2))
p = PseudoDifferentialOperator(expr=expr, vars_x=[x], mode='symbol')
print(p.asymptotic_expansion(order=4))




x, xi = symbols('x xi', real=True, positive=True)

p_ordre_2 = sqrt(xi**2 + 1) + x / (xi**2 + 1)
p = PseudoDifferentialOperator(expr=p_ordre_2, vars_x=[x], mode='symbol')
pprint(p.asymptotic_expansion(order=4))


# ### 2D



x, y, xi, eta = symbols('x y xi eta', real=True, positive=True)

p_ordre_1 = sqrt(xi**2 + eta**2) + x
p = PseudoDifferentialOperator(expr=p_ordre_1, vars_x=[x, y], mode='symbol')

print(p.asymptotic_expansion(order=4))




x, y, xi, eta = symbols('x y xi eta', real=True, positive=True)

p_ordre_2 = sqrt(xi**2 + eta**2) + x / (xi**2 + eta**2)
p = PseudoDifferentialOperator(expr=p_ordre_2, vars_x=[x, y], mode='symbol')
pprint(p.asymptotic_expansion(order=4))




x, y, xi, eta = symbols('x y xi eta', real=True, positive=True)

expr = exp(x*xi*y*eta / (x**2 + y**2 + xi**2 + eta**2))
p = PseudoDifferentialOperator(expr=expr, vars_x=[x], mode='symbol')
print(p.asymptotic_expansion(order=4))


# ## Asymptotic composition
# ### 1D



x, xi = symbols('x xi')
p1 = PseudoDifferentialOperator(expr=xi + x, vars_x=[x], mode='symbol')

p2 = PseudoDifferentialOperator(expr=xi + x**2, vars_x=[x], mode='symbol')

composition_1d = p1.compose_asymptotic(p2, order=2)
print("1D Asymptotic composition:")
print(composition_1d)


# ### 2D



x, y, xi, eta = symbols('x y xi eta')
p1 = PseudoDifferentialOperator(expr=xi**2 + eta**2 + x*y, vars_x=[x, y], mode='symbol')
p2 = PseudoDifferentialOperator(expr=xi + eta + x + y, vars_x=[x, y], mode='symbol')

composition_2d = p1.compose_asymptotic(p2, order=3)
print("2D Asymptotic composition:")
print(composition_2d)


# ### Asymptotic composition & order importance



# Sympy Variables
x, xi = symbols('x xi', real=True)
u = Function('u')
var_u=u(x)
L = - x * diff(u(x), x) - u(x)

op = PseudoDifferentialOperator(expr=L, vars_x=[x], var_u=u(x), mode='auto')

print(op.symbol)

compose = op.compose_asymptotic(op, order=0)
print("0th order :")
pprint(expand(compose))
compose = op.compose_asymptotic(op, order=1)
print("1st order :")
pprint(expand(compose))


# ## Commutator
# ### 1D



x, xi = symbols('x xi', real=True)
A = PseudoDifferentialOperator(expr=x*xi, vars_x=[x])
B = PseudoDifferentialOperator(expr=xi**2, vars_x=[x])
C = A.commutator_symbolic(B, order=1)
simplify(C)
# Expected result : 2*I*xi^2


# ### 2D



x, y, xi, eta = symbols('x y xi eta', real=True)
A = PseudoDifferentialOperator(expr=x*xi + y*eta, vars_x=[x, y])
B = PseudoDifferentialOperator(expr=xi**2 + eta**2, vars_x=[x, y])
C = A.commutator_symbolic(B, order=1)
simplify(C)
# Expected result : 2*I*(xi^2+eta^2)


# ### Commutator & order importance



# Symbolic variables
x, xi = symbols('x xi', real=True)
# Test symbols
a = sin(x) * xi
b = cos(2*x) * xi**2
# Definition of pseudo-differential operators
A = PseudoDifferentialOperator(expr=a, vars_x=[x])
B = PseudoDifferentialOperator(expr=b, vars_x=[x])
# Loop over the orders of the asymptotic expansion
for order in [1, 2, 3]:
    C = A.commutator_symbolic(B, order=order)
    print(f"\n--- Commutator symbolic expansion (order={order}) ---")
    print(simplify(C))


# ## Right asymptotic inverse
# ### 1D



p = PseudoDifferentialOperator(expr=xi + 1, vars_x=[x], mode='symbol')
right_inv_1d = p.right_inverse_asymptotic(order=2)
print("1D Right asymtotic inverse:")
print(right_inv_1d)
p2 = PseudoDifferentialOperator(expr=right_inv_1d, vars_x=[x], mode='symbol')
print("1D Composition rule:")
p.compose_asymptotic(p2, order=2)


# ### 2D



p = PseudoDifferentialOperator(expr=xi + eta + 1, vars_x=[x, y], mode='symbol')
right_inv_2d = p.right_inverse_asymptotic(order=2)
print("2D Right asymtotic inverse:")
print(right_inv_2d)
p2 = PseudoDifferentialOperator(expr=right_inv_2d, vars_x=[x, y], mode='symbol')
print("2D Composition rule:")
p.compose_asymptotic(p2, order=2)


# ## Left asymptotic inverse
# ### 1D



p = PseudoDifferentialOperator(expr=xi + 1, vars_x=[x], mode='symbol')
left_inv_1d = p.left_inverse_asymptotic(order=2)
print("1D Left asymtotic inverse:")
print(left_inv_1d)
p2 = PseudoDifferentialOperator(expr=left_inv_1d, vars_x=[x], mode='symbol')
print("1D Composition rule:")
p2.compose_asymptotic(p, order=2)


# ### 2D



p = PseudoDifferentialOperator(expr=xi + eta + 1, vars_x=[x, y], mode='symbol')
left_inv_2d = p.left_inverse_asymptotic(order=3)
print("2D Left asymtotic inverse:")
print(left_inv_2d)
p2 = PseudoDifferentialOperator(expr=left_inv_2d, vars_x=[x, y], mode='symbol')
print("2D Composition rule:")
p2.compose_asymptotic(p, order=3)


# ## Ellipticity
# ### 1D



x, xi = symbols('x xi', real=True)

x_vals = np.linspace(-1, 1, 100)
xi_vals = np.linspace(-10, 10, 100)

op1 = PseudoDifferentialOperator(expr=xi**2 + 1, vars_x=[x], mode='symbol')
print("1D elliptic test:", op1.is_elliptic_numerically(x_vals, xi_vals))  # True 

op2 = PseudoDifferentialOperator(expr=xi, vars_x=[x], mode='symbol')
print("1D non-elliptic test:", op2.is_elliptic_numerically(x_vals, xi_vals))  # False attendu


# ### 2D



x_vals = np.linspace(-1, 1, 50)
y_vals = np.linspace(-1, 1, 50)
xi_vals = np.linspace(-10, 10, 50)
eta_vals = np.linspace(-10, 10, 50)

op3 = PseudoDifferentialOperator(expr=xi**2 + eta**2 + 1, vars_x=[x, y], mode='symbol')
print("2D elliptic test:", op3.is_elliptic_numerically((x_vals, y_vals), (xi_vals, eta_vals)))  # True

op4 = PseudoDifferentialOperator(expr=xi + eta, vars_x=[x, y], mode='symbol')
print("2D non-elliptic test:", op4.is_elliptic_numerically((x_vals, y_vals), (xi_vals, eta_vals)))  # False


# ## Formal adjoint

# ### 1D



x, xi = symbols('x xi')
xi = symbols('xi', real=True, positive=True)
x = symbols('x', real=True, positive=True)

p_ordre_1 = xi**2
p = PseudoDifferentialOperator(expr=p_ordre_1, vars_x=[x], mode='symbol')
print(p.formal_adjoint())  
print(p.is_self_adjoint())




# Declaration of variables
x, xi = symbols('x xi', real=True)
xi = symbols('xi', real=True, positive=True)
x = symbols('x', real=True)

# Complex symbol depending on x and xi
p_expr = (1 + I * x) * xi + exp(-x) / xi

# Creation of the pseudo-differential operator
p = PseudoDifferentialOperator(expr=p_expr, vars_x=[x], mode='symbol')

# Calculation of the formal adjoint
adjoint = p.formal_adjoint()
print("Formal adjoint (1D):")
print(adjoint)

# Self-adjointness test
print("Is self-adjoint (1D)?", p.is_self_adjoint())


# ### 2D



xi, eta = symbols('xi eta', real=True, positive=True)
x, y = symbols('x y', real=True, positive=True)

p_ordre_1 = xi**2 + eta**2
p = PseudoDifferentialOperator(expr=p_ordre_1, vars_x=[x, y], mode='symbol')
print(p.formal_adjoint())  

print(p.is_self_adjoint())






xi, eta = symbols('xi eta', real=True, positive=True)
x, y = symbols('x y', real=True, positive=True)

p_ordre_1 = y * xi**2 + x * eta**2
p = PseudoDifferentialOperator(expr=p_ordre_1, vars_x=[x, y], mode='symbol')
print(p.formal_adjoint())  

print(p.is_self_adjoint())




# Declaration of variables
x, y, xi, eta = symbols('x y xi eta', real=True)
xi = symbols('xi', real=True, positive=True)
eta = symbols('eta', real=True, positive=True)
x, y = symbols('x y', real=True)

# Complex symbol in 2D
p_expr_2d = (x + I*y)*xi + (y - I*x)*eta + exp(-x - y)/(xi + eta)

# Creation of the 2D operator
p2 = PseudoDifferentialOperator(expr=p_expr_2d, vars_x=[x, y], mode='symbol')

# Calculation of the adjoint
adjoint_2d = p2.formal_adjoint()
print("Formal adjoint (2D):")
print(adjoint_2d)

# Self-adjointness test
print("Is self-adjoint (2D)?", p2.is_self_adjoint())


# ## $L ∘ P = Id$



order = 2
# Define symbolic variables
x, xi = symbols('x xi', real=True)
u = Function('u')(x)

# Define the pseudo-differential symbol p(x, xi)
p_symbol = xi**2 + x**2 + 1

# Create the pseudo-differential operator
p = PseudoDifferentialOperator(expr=p_symbol, vars_x=[x], mode='symbol')

# Compute the left asymptotic inverse of the operator
left_inv = p.left_inverse_asymptotic(order=order)
print("Left asymptotic inverse:")
pprint(left_inv, num_columns=150)

# Create a new operator from the left inverse
p_left_inv = PseudoDifferentialOperator(expr=left_inv, vars_x=[x], mode='symbol')

# Verify the composition rule: L ∘ P ≈ I (identity operator)
composition = p_left_inv.compose_asymptotic(p, order=order)
print("Composition rule (L ∘ P):")
pprint(composition, num_columns=150)

# Simplify the result to check if it approximates the identity operator
identity_approximation = simplify(composition)
print("Simplified composition (should approximate 1):")
pprint(identity_approximation, num_columns=150)

# Development around xi -> infinity
series_expansion = series(identity_approximation, xi, oo, n=3)
print("Series expansion:", series_expansion)

# Convert to numerical function
identity_func = lambdify((x, xi), identity_approximation, 'numpy')

# Grid of points
x_vals = np.linspace(-10, 10, 100)
xi_vals = np.linspace(1, 100, 100)  # Positive frequencies
X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')


# ## $P ∘ R = Id$



order = 2
# Define symbolic variables
x, xi = symbols('x xi', real=True)
u = Function('u')(x)

# Define the pseudo-differential symbol p(x, xi)
p_symbol = xi**2 + x**2 + 1

# Create the pseudo-differential operator
p = PseudoDifferentialOperator(expr=p_symbol, vars_x=[x], mode='symbol')

# Compute the left asymptotic inverse of the operator
right_inv = p.right_inverse_asymptotic(order=order)
print("Right asymptotic inverse:")
pprint(right_inv, num_columns=150)

# Create a new operator from the left inverse
p_right_inv = PseudoDifferentialOperator(expr=right_inv, vars_x=[x], mode='symbol')

# Verify the composition rule: L ∘ P ≈ I (identity operator)
composition = p.compose_asymptotic(p_right_inv, order=order)
print("Composition rule (P ∘ R ):")
pprint(composition, num_columns=150)

# Simplify the result to check if it approximates the identity operator
identity_approximation = simplify(composition)
print("Simplified composition (should approximate 1):")
pprint(identity_approximation, num_columns=150)

# Development around xi -> infinity
series_expansion = series(identity_approximation, xi, oo, n=3)
print("Series expansion:", series_expansion)

# Convert to numerical function
identity_func = lambdify((x, xi), identity_approximation, 'numpy')

# Grid of points
x_vals = np.linspace(-10, 10, 100)
xi_vals = np.linspace(1, 100, 100)  # Positive frequencies
X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')


# ## Trace
# ### 1D



x = symbols('x', real=True)
xi = symbols('xi', real=True)
p = exp(-(x**2 + xi**2))
P = PseudoDifferentialOperator(p, [x], mode='symbol')
trace = P.trace_formula()
print("Tr(P)=", trace)




trace_num = P.trace_formula(
     numerical=True,
     x_bounds=((-5, 5),),
     xi_bounds=((-5, 5),)
 )


# ### Trace & Hermite's polynomials



n=6
for order in range(n):
    H_n = exp(-(x**2 + xi**2)/2) * hermite(order, x) * hermite(order, xi)
    P_n = PseudoDifferentialOperator(H_n, [x], mode='symbol')
    tr = P_n.trace_formula() 
    print("Tr(H_", order,")=",tr)


# ### Trace & Laguerre's polynomials



def wigner_function_harmonic_oscillator(n, x_sym, xi_sym):
    """
    Wigner function (phase space representation) for the n-th Fock state
    of the quantum harmonic oscillator.

    W_n(x, ξ) = (-1)^n / π * exp(-(x² + ξ²)) * L_n(2(x² + ξ²))

    where L_n is the Laguerre polynomial.
    """
    r_squared = x_sym**2 + xi_sym**2

    wigner = ((-1)**n / pi) * exp(-r_squared) * laguerre(n, 2*r_squared)

    return simplify(wigner)

# Example usage
n = 1
x_sym = symbols('x', real=True)
xi_sym = symbols('xi', real=True)
# Wigner function for the ground state
W_0 = wigner_function_harmonic_oscillator(n, x_sym, xi_sym)
print("Wigner function W_0(x, ξ) =")
pprint(W_0)
# Create the pseudo-differential operator
P_n = PseudoDifferentialOperator(W_0, [x_sym], mode='symbol')
# Calculate the trace (should be 1)
tr = P_n.trace_formula()
print(f"\nTrace for n={n}: {tr}")
print(f"Simplified: {simplify(tr)}")

# Symbolic variables
x = symbols('x', real=True)
xi = symbols('xi', real=True)
# Wigner function for different Fock states
for n in range(3):
    print(f"\n{'='*60}")
    print(f"Fock state n = {n}")
    print('='*60)

    # Calculate the Wigner function
    r_squared = x**2 + xi**2
    W_n = ((-1)**n / pi) * exp(-r_squared) * laguerre(n, 2*r_squared)

    print(f"\nW_{n}(x, ξ) =")
    pprint(simplify(W_n))

    # Create the operator
    P_n = PseudoDifferentialOperator(W_n, [x], mode='symbol')

    # Calculate the trace
    print(f"\nCalculating trace...")
    try:
        tr = P_n.trace_formula()
        print(f"Trace = {simplify(tr)}")

        # The trace should be 1 (normalization)
        tr_numerical = float(tr.evalf())
        print(f"Numerical value: {tr_numerical:.6f}")

        if abs(tr_numerical - 1.0/(2*pi)) < 1e-6:
            print("✓ Trace correctly normalized to 1/(2π)")
        else:
            print(f"⚠ Warning: trace ≠ 1/(2π) (got {tr_numerical})")

    except Exception as e:
        print(f"⚠ Symbolic integration failed: {e}")
        print("Trying numerical integration...")

        tr_num = P_n.trace_formula(
            numerical=True,
            x_bounds=((-5, 5),),
            xi_bounds=((-5, 5),)
        )
        print(f"Numerical trace ≈ {tr_num:.6f}")


# ## Exponential of P
# ### 1D



x = symbols('x', real=True)
xi = symbols('xi', real=True)
H = xi**2 + x**2  # Hamiltonian symbol
H_op = PseudoDifferentialOperator(H, [x], mode='symbol')
t_sym = symbols('t', real=True)
U_symbol = H_op.exponential_symbol(t=-I*t_sym, order=3)
print(U_symbol)
t0 = 0.001
expr = U_symbol.subs(t_sym, t0)
print(expr)

p1d = PseudoDifferentialOperator(expr=expr, vars_x=[x], mode='symbol')

# #### Laplacian



Laplacian = -xi**2
L_op = PseudoDifferentialOperator(Laplacian, [x], mode='symbol')
heat_kernel = L_op.exponential_symbol(t=0.1, order=5)
print("Heat kernel:", heat_kernel)


# #### Fractional Schrödinger



x, xi = symbols('x xi', real=True)
alpha = 1.5  # fractional exponent
H_frac = xi**alpha
H_frac_op = PseudoDifferentialOperator(H_frac, [x], mode='symbol')
t_sym = symbols('t', real=True)
U_frac_symbol = H_frac_op.exponential_symbol(t=-I*t_sym, order=3)
print("Fractional Schrödinger:", U_frac_symbol)


# #### Gibbs partition function



x, xi = symbols('x xi', real=True)
H_classical = xi**2/2 + x**4/4  # Quartic Hamiltonian
H_classical_op = PseudoDifferentialOperator(H_classical, [x], mode='symbol')
beta = symbols('beta', real=True)  # Inverse temperature
Gibbs_symbol = H_classical_op.exponential_symbol(t=-beta, order=4)
print("Gibbs partition function:", Gibbs_symbol)


# ### 2D
# #### Heat kernel



x, y = symbols('x y', real=True)
xi, eta = symbols('xi eta', real=True)
# 2D Laplacian symbol
Laplacian_2D = - (xi**2 + eta**2)
# Define the 2D pseudodifferential operator
L_2D_op = PseudoDifferentialOperator(Laplacian_2D, [x, y], mode='symbol')
t_val = 0.05
# Compute the 2D heat kernel using exponential symbol
heat_kernel_2D = L_2D_op.exponential_symbol(t=t_val, order=4)
print("2D Diffusion:", heat_kernel_2D)


# #### Harmonic oscillator



x, y = symbols('x y', real=True)
xi, eta = symbols('xi eta', real=True)
H_2D = xi**2 + eta**2 + x**2 + y**2
H_2D_op = PseudoDifferentialOperator(H_2D, [x, y], mode='symbol')
t_sym = symbols('t', real=True)
U_2D_symbol = H_2D_op.exponential_symbol(t=-I*t_sym, order=3)
print("2D harmonic oscillator:", U_2D_symbol)









