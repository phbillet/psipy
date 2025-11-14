#!/usr/bin/env python
# coding: utf-8







from solver import *


# # 1D Tests in Dirichlet conditions

# ## Stationnary equation



# Define symbolic variables
x = symbols('x')
xi = symbols('xi', real=True)
u = Function('u')(x)

# Equation: psiOp(xi^2 + 1, u)(x) = sin(x)
equation = Eq(psiOp(xi**2 + 1, u), sin(x))

# Exact solution (for testing)
def u_exact(x_vals):
    return np.sin(x_vals) / 2  # Uses numpy, not sympy

# Create the solver
solver = PDESolver(equation)

# Grid parameters
Lx = 2 * np.pi
Nx = 256

# Configure the solver with Dirichlet boundary condition
solver.setup(
    Lx=Lx,
    Nx=Nx,
    boundary_condition='dirichlet',
    initial_condition=None,  # Not necessary for a stationary problem
    plot=False
)

# Solve the stationary problem
u_num = solver.solve_stationary_psiOp(order=1)

# Exact solution on grid
u_ref = u_exact(solver.X)

# Compute relative L2 error
error = np.linalg.norm(np.real(u_num) - u_ref) / np.linalg.norm(u_ref)

# Automatic test (uses the well-vectorized u_exact function)
solver.test(u_exact=u_exact, threshold=5e-3, component='real')

# ## Stationnary equation with psiOp depending on $x$ and on $\xi$



# Symbols
x = symbols('x')
xi = symbols('xi', real=True)
u = Function('u')(x)

# Pseudo-differential equation depending on x
equation = Eq(psiOp(x**2 * xi**2 + 1, u), sin(x))

# Exact solution (approximate here, for visual testing purposes)
def u_exact(x_vals):
    return np.sin(x_vals) / (x_vals**2 + 1)  # Corresponds to the symbol evaluated at xi=1 approximately

# Creation of the solver
solver = PDESolver(equation)

# Grid parameters
Lx = 2 * np.pi
Nx = 256

# Configuration of the solver with Dirichlet condition
solver.setup(
    Lx=Lx,
    Nx=Nx,
    boundary_condition='dirichlet',
    initial_condition=None, 
    plot=False
)

# Stationary solution with asymptotic inversion of order 1
u_num = solver.solve_stationary_psiOp(order=0)

# Comparison with approximate exact solution
solver.test(u_exact=u_exact, threshold=1, component='real')  # Larger tolerance here

# ## Diffusion equation



# Definition of symbols
t, x, xi = symbols('t x xi', real=True)
u = Function('u')

# Evolution equation (with psiOp): ∂u/∂t = -psiOp(ξ² + 1, u)
equation = Eq(diff(u(t,x), t), -psiOp(xi**2 + 1, u(t,x)))

# Creation of the solver
solver = PDESolver(equation)

# Parameters
Lx = 2 * np.pi
Nx = 128
Lt = 2.0
Nt = 400

# Initial condition
k0 = 1.0
initial_condition = lambda x: np.sin(k0 * x)

# Exact solution function
def u_exact(x, t):
    return np.sin(k0 * x) * np.exp(- (k0**2 + 1) * t)

# Setup with Dirichlet boundary condition
solver.setup(
    Lx=Lx,
    Nx=Nx,
    Lt=Lt,
    Nt=Nt,
    boundary_condition='dirichlet',
    initial_condition=initial_condition, 
    plot=False
)


# Solving
solver.solve()

# Automatic tests
n_test = 4
for i in range(n_test + 1):
    solver.test(u_exact=u_exact, t_eval=i * Lt / n_test, threshold=50, component='real')


# ## Wave equation



# Symbols
t, x, xi = symbols('t x xi', real=True)
u = Function('u')(t, x)

# Wave equation via psiOp
eq = Eq(diff(u, t, t), psiOp(-xi**2, u))

# Create the solver
solver = PDESolver(eq)

# Parameters
Lt = 5.0
Lx = 2 * np.pi
Nx = 512

# Setup with Dirichlet boundary conditions
solver.setup(
    Lx=Lx,
    Nx=Nx,
    Lt=Lt,
    Nt=1000,
    boundary_condition='dirichlet',
    initial_condition=lambda x: np.sin(x),
    initial_velocity=lambda x: np.zeros_like(x), 
    plot=False
)

# Solve
solver.solve()

# Energy visualization
#solver.plot_energy()
#solver.plot_energy(log=True)

# Exact solution (eigenmode)
def u_exact(x, t):
    return np.sin(x) * np.cos(t)
    
# Automatic tests
n_test = 10
for i in range(n_test + 1):
    solver.test(u_exact=u_exact, t_eval=i * Lt / n_test, threshold=7e-1, component='real')


# ## Schrödinger equation



# Definition of symbols
t, x = symbols('t x', real=True)
u = Function('u')(t, x)

# Schrödinger equation (form adapted to the solver)
equation = Eq(diff(u, t), psiOp(I * xi**2, u))

# Creation of the solver
solver = PDESolver(equation)

# Parameters
Lx = 2 * np.pi
Nx = 512
Lt = 5.0
Nt = 500

# Initial condition: localized Gaussian (or sine)
initial_condition = lambda x: np.sin(x)
initial_velocity = None  # not used here
# Setup with Dirichlet conditions
solver.setup(
    Lx=Lx,
    Nx=Nx,
    Lt=Lt,
    Nt=Nt,
    boundary_condition='dirichlet',
    initial_condition=initial_condition, 
    plot=False
)

# Solving
solver.solve()

# Exact solution 
def u_exact(x, t):
    return np.exp(1j * t) * np.sin(x) 
    
# Tests (if applicable)
n_test = 5
for i in range(n_test + 1):
    solver.test(u_exact=u_exact, t_eval=i * Lt / n_test, threshold=7e-1, component='real')


# ## Equation with psiOp depending on $x$ but not on $\xi$



# Symbols
t, x, xi = symbols('t x xi', real=True)
u = Function('u')
# Symbol dependent on x only
symbol_expr = 1 + x**2
eq = Eq(diff(u(t,x), t), -psiOp(symbol_expr, u(t,x)))
# Creation of the solver
solver = PDESolver(eq)
# Domain: Dirichlet on [-1, 1]
Lx = 2.0
Nx = 256
Lt = 2.0
Nt = 300
# Initial condition: sin(π x), vanishes at ±1
def initial_condition(x):
    return np.sin(np.pi * x / (Lx/2))
# Exact solution
def u_exact(x, t):
    return initial_condition(x) * np.exp(-t * (1 + x**2))
# Setup with Dirichlet
solver.setup(
    Lx=Lx,
    Nx=Nx,
    Lt=Lt,
    Nt=Nt,
    boundary_condition='dirichlet',
    initial_condition=initial_condition, 
    plot=False
)
# Solving
solver.solve()
# Tests
n_test = 5
for i in range(n_test + 1):
    t_eval = i * Lt / n_test
    solver.test(
        u_exact=u_exact,
        t_eval=t_eval,
        threshold=5e-2,
        component='real'
    )


# ## Hermite equation



# Definition of symbols
t, x, xi = symbols('t x xi', real=True)
u = Function('u')

# Evolution equation: ∂²u/∂t² = -ψOp(x² + ξ², u)
p_expr = x**2 + xi**2
equation = Eq(diff(u(t,x), t, t), -psiOp(p_expr, u(t,x)))

# Creation of the solver
solver = PDESolver(equation)

# Parameters
Lx = 12.0
Nx = 256
Lt = 3.0
Nt = 600
n = 2                     # Order of Hermite
lambda_n = 2 * n + 1

# Initial function: u₀(x) = Hₙ(x) * exp(-x² / 2)
initial_condition = lambda x: eval_hermite(n, x) * np.exp(-x**2 / 2)

# Zero initial velocity: ∂ₜ u(0,x) = 0
initial_velocity = lambda x: 0.0 * x

# Exact solution
def u_exact(x, t):
    return np.cos(np.sqrt(lambda_n) * t) * eval_hermite(n, x) * np.exp(-x**2 / 2)

# Solver setup
solver.setup(
    Lx=Lx,
    Nx=Nx,
    Lt=Lt,
    Nt=Nt,
    boundary_condition='dirichlet',
    initial_condition=initial_condition,
    initial_velocity=initial_velocity,
    plot=False
)

# Solving
solver.solve()

# Validation tests
n_test = 5
for i in range(n_test + 1):
    t_eval = i * Lt / n_test
    solver.test(u_exact=u_exact, t_eval=t_eval, threshold=50, component='real')


# ## Airy equation



# Symbols
t, x, xi = symbols('t x xi', real=True)
u = Function('u')
# Airy Symbol: p(x, ξ) = x + ξ²
p_expr = x + xi**2
equation = Eq(diff(u(t,x), t, t), -psiOp(p_expr, u(t,x)))
# Solver creation
solver = PDESolver(equation)
# Numerical parameters
Lx = 40.0     # Large domain for Ai(x)
Nx = 256
Lt = 2.0
Nt = 1000
# Initial condition: u(0,x) = Ai(x)
initial_condition = lambda x: airy(x)[0]
# Initial temporal derivative: ∂ₜ u(0,x) = 0
initial_velocity = lambda x: 0.0 * x
# Exact solution: Ai(x - t²/4)
def u_exact(x, t):
    return airy(x - t**2 / 4)[0]
# Setup
solver.setup(
    Lx=Lx,
    Nx=Nx,
    Lt=Lt,
    Nt=Nt,
    boundary_condition='dirichlet',
    initial_condition=initial_condition,
    initial_velocity=initial_velocity, 
    plot=False
)
# Solving
solver.solve()
# Automatic tests
n_test = 5
for i in range(n_test + 1):
    t_eval = i * Lt / n_test
    solver.test(u_exact=u_exact, t_eval=t_eval, threshold=50, component='real')


# ## Gaussian equation



# Definition of symbols
t, x, xi = symbols('t x xi', real=True)
u = Function('u')

# Evolution equation
p_expr = x**2 + xi**2
equation = Eq(diff(u(t,x), t, t), -psiOp(p_expr, u(t,x)))

# Creation of the solver
solver = PDESolver(equation)

# Numerical parameters
Lx = 10.0
Nx = 256
Lt = 2 * np.pi   # To observe a complete period
Nt = 1000

# Initial condition: u₀(x) = exp(-x²)
initial_condition = lambda x: np.exp(-x**2)

# Initial velocity is zero
initial_velocity = lambda x: 0.0 * x

# Exact solution: cos(t) * exp(-x²)
def u_exact(x, t):
    return np.cos(t) * np.exp(-x**2)

# Setup of the solver
solver.setup(
    Lx=Lx,
    Nx=Nx,
    Lt=Lt,
    Nt=Nt,
    boundary_condition='dirichlet',
    initial_condition=initial_condition,
    initial_velocity=initial_velocity, 
    plot=False
)

# Solving
solver.solve()

# Validation tests
n_test = 5
for i in range(n_test + 1):
    t_eval = i * Lt / n_test
    solver.test(u_exact=u_exact, t_eval=t_eval, threshold=50, component='real')


# ## Legendre equation



# Symbols
t, x, xi = symbols('t x xi', real=True)
u = Function('u')
# Order of the Legendre polynomial
n = 20
P_n = legendre(n)
lambda_n = n * (n + 1)
# Equation: ∂tt u = -ψOp(ξ², u)
p_expr = xi**2
equation = Eq(diff(u(t,x), t, t), -psiOp(p_expr, u(t,x)))
# Creating the solver
solver = PDESolver(equation)
# Domain [-1, 1] -> Lx = 2
Lx = 2.0
Nx = 256
Lt = 2 * np.pi / np.sqrt(lambda_n)  # period of the oscillation
Nt = 500
# Initial condition: Pₙ(x)
initial_condition = lambda x: P_n(2 * x / Lx)  # scales x ∈ [-1, 1]
initial_velocity = lambda x: 0 * x
# Exact solution
def u_exact(x, t):
    x_scaled = 2 * x / Lx  # scales x ∈ [-1,1]
    return np.cos(np.sqrt(lambda_n) * t) * P_n(x_scaled)
# Setup
solver.setup(
    Lx=Lx,
    Nx=Nx,
    Lt=Lt,
    Nt=Nt,
    boundary_condition='dirichlet',
    initial_condition=initial_condition,
    initial_velocity=initial_velocity, 
    plot=False
)
# Solving
solver.solve()
# Validation
n_test = 5
for i in range(n_test + 1):
    t_eval = i * Lt / n_test
    solver.test(u_exact=u_exact, t_eval=t_eval, threshold=50, component='real')






