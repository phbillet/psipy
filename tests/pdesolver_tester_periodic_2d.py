#!/usr/bin/env python
# coding: utf-8







from solver import *


# # 2D tests in periodic conditions

# ## Stationnary equation
# ### With psiOp()



# Symbolic variables
x, y, xi, eta = symbols('x y xi eta', real=True)
u = Function('u')(x, y)

# Symbolic equation
symbol = xi**2 + eta**2 + 1
equation = Eq(psiOp(symbol, u), -sin(x) * sin(y))

# Define exact solution
def u_exact(x, y):
    return -np.sin(x) * np.sin(y) / 3

# Solver creation
solver = PDESolver(equation)

# Domain
Lx, Ly = 2 * np.pi, 2 * np.pi
Nx, Ny = 128, 128

# Setup with exact solution as source
solver.setup(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, initial_condition=None, plot=False)

# Stationary solution
u_num = solver.solve_stationary_psiOp()

# Accuracy test
solver.test(u_exact=u_exact, threshold=20, component='real')


# ### With psiOp() depending on spatial variables



# 1) Define symbolic variables
x, y, xi, eta = symbols('x y xi eta', real=True)
u = Function('u')(x, y)

# 2) Define the 2D symbol p(x, y, xi, eta)
p_symbol = xi**2 + eta**2 + x**2 + y**2 + 1

# 3) Choose an exact solution u_exact(x, y)
u_exact_expr = exp(-x**2 - y**2)

def u_exact(x, y):
    return np.exp(-x**2 - y**2)

# 4) Compute the right-hand side f_expr = P[u_exact] symbolically
f_expr = (-3*x**2 - 3*y**2 + 5)*u_exact_expr

# 5) Build the stationary equation: psiOp(p_symbol, u) = f_expr
equation_2d = Eq(psiOp(p_symbol, u), f_expr)

# 6) Initialize and setup the solver
solver2d = PDESolver(equation_2d)
solver2d.setup(Lx=10.0, Ly=10.0, Nx=64, Ny=64, initial_condition=None, plot=False)

# 7) Solve stationary psiOp problem (first-order asymptotic inverse)
u_num2d = solver2d.solve_stationary_psiOp(order=1)

# 8) Test numerical vs exact
solver2d.test(u_exact=u_exact, threshold=3e-1, component='real')


# ## Transport equation



# Definition of the 2D transport equation
t, x, y, kx, ky, omega = symbols('t x y kx ky omega')
u_func = Function('u') 
u = u_func(t, x, y)

transport_eq = Eq(diff(u, t), -diff(u, x) - diff(u, y))  # Diagonal transport

# Creation of the solver with periodic conditions
solver = PDESolver(transport_eq)

# Configuration of the 2D domain and initial condition
L = 10.0  # Square domain [-5, 5] × [-5, 5]
N = 512   # Spatial resolution
Lt=3.0
solver.setup(
    Lx=L, Ly=L,
    Nx=N, Ny=N,
    Lt=Lt, Nt=2000,
    initial_condition=lambda x, y: np.exp(-(x**2 + y**2)),  # 2D Gaussian
    n_frames=100, 
    plot=False
)

# Exact solution with periodic wrapping
def u_exact(x, y, t):
    Lx = Ly = L  # Square domain
    x_shift = (x - t + Lx/2) % Lx - Lx/2  # Periodic wrapping in x
    y_shift = (y - t + Ly/2) % Ly - Ly/2  # Periodic wrapping in y
    return np.exp(-(x_shift**2 + y_shift**2))

# Solving and validation
solver.solve()

# Automatic testing
n_test = 4
test_set = [solver.test(u_exact=u_exact, t_eval=i * Lt / n_test, threshold=5e-2, component='real') for i in range(n_test + 1)]


# ## Heat equation
# ### With PDE



# Define the symbols and the unknown function
t, x, y, kx, ky, omega = symbols('t x y kx ky omega')
u_func = Function('u') 
u = u_func(t, x, y)

# Define the 2D heat equation
eq = Eq(diff(u, t), diff(u, x, x) + diff(u, y, y))

# Initialize the solver
solver = PDESolver(eq)
Lt=2.0
# Configure the solver with the problem parameters
solver.setup(
    Lx=2 * np.pi, Ly=2 * np.pi,  # Spatial domain: [0, 2π] × [0, 2π]
    Nx=512, Ny=512,             # Spatial resolution: 512×512 points
    Lt=Lt, Nt=100,             # Total time: 1.0, number of time steps: 100
    initial_condition=lambda x, y: np.sin(x) * np.sin(y),  # Initial condition
    n_frames=50, 
    plot=False
)

# Solve the equation
solver.solve()

# Exact solution at t = 1
def u_exact(x, y, t):
    """
    Analytical solution of the 2D Poisson equation.
    The solution is given by u(x, y, t) = sin(x) * sin(y) * exp(-2t).
    """
    return np.sin(x) * np.sin(y) * np.exp(-2.0 * t)

# Automatic testing
n_test = 4
test_set = [solver.test(u_exact=u_exact, t_eval=i * Lt / n_test, threshold=5e-2, component='real') for i in range(n_test + 1)]



# ### With psiOp



# Define symbols
t, x, y, xi, eta = symbols('t x y xi eta', real=True)
u_func = Function('u')
u = u_func(t, x, y)

# Define the equation using psiOp
eq = Eq(diff(u, t), psiOp(-(xi**2 + eta**2), u))

# Create the solver instance
solver = PDESolver(eq)

# Total simulation time
Lt = 2.0

# Setup the solver
solver.setup(
    Lx=2 * np.pi, Ly=2 * np.pi,
    Nx=512, Ny=512,
    Lt=Lt, Nt=100,
    initial_condition=lambda x, y: np.sin(x) * np.sin(y),
    n_frames=50, 
    plot=False
)

# Solve the PDE
solver.solve()

# Define the exact solution
def u_exact(x, y, t):
    return np.sin(x) * np.sin(y) * np.exp(-2.0 * t)

# Test the result
n_test = 4
test_set = [
    solver.test(
        u_exact=u_exact,
        t_eval=i * Lt / n_test,
        threshold=5e-2,
        component='real'
    )
    for i in range(n_test + 1)
]

# ## Schrödinger equation
# ### With PDE



# Define the symbols and the unknown function
t, x, y, kx, ky, omega = symbols('t x y kx ky omega')
u_func = Function('u') 
u = u_func(t, x, y)

# 2D Schrödinger equation: i ∂t u = ∂xx u + ∂yy u
eq = Eq(I * diff(u, t), (diff(u, x, x) + diff(u, y, y)))

# Create the solver
solver = PDESolver(eq, time_scheme='ETD-RK4', dealiasing_ratio=1)

# Domain
Lx = Ly = 20
Nx = Ny = 256
Lt = 2.0
Nt = 200

# Initial condition: Gaussian wave packet modulated e^{-x^2 - y^2} e^{i(x + y)}
solver.setup(
    Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, Lt=Lt, Nt=Nt,
    initial_condition=lambda x, y: np.exp(-x**2 - y**2) * np.exp(1j * (x + y)), 
    plot=False
)

# Solve the equation
solver.solve()

# Exact solution
def u_exact(x, y, t):
    """
    Analytical solution of the 2D Schrödinger equation.
    The solution is given by:
        u(x, y, t) = 1 / sqrt(1 + 4j * t)**2 * exp(i * (x + y - 2 * t)) * exp(-((x + 2t)^2 + (y + 2t)^2) / (1 + 4j * t)).
    """
    return 1 / np.sqrt(1 + 4j * t)**2 * np.exp(1j * (x + y - 2 * t)) * np.exp(-((x + 2 * t)**2 + (y + 2 * t)**2) / (1 + 4j * t))

# Automatic testing
n_test = 4
test_set = [solver.test(u_exact=u_exact, t_eval=i * Lt / n_test, threshold=0.3, component='abs') for i in range(n_test + 1)]


# ### With psiOp



# Define the symbols and the unknown function
t, x, y, xi, eta = symbols('t x y xi eta', real=True)
u_func = Function('u') 
u = u_func(t, x, y)

# 2D Schrödinger equation: i ∂t u = ∂xx u + ∂yy u
eq = Eq(I * diff(u, t), psiOp(-(xi**2 + eta**2), u))

# Create the solver
solver = PDESolver(eq)

# Domain
Lx = Ly = 20
Nx = Ny = 256
Lt = 2.0
Nt = 200

# Initial condition: Gaussian wave packet modulated e^{-x^2 - y^2} e^{i(x + y)}
solver.setup(
    Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, Lt=Lt, Nt=Nt,
    initial_condition=lambda x, y: np.exp(-x**2 - y**2) * np.exp(1j * (x + y)), 
    plot=False
)

# Solve the equation
solver.solve()

# Exact solution
def u_exact(x, y, t):
    """
    Analytical solution of the 2D Schrödinger equation.
    The solution is given by:
        u(x, y, t) = 1 / sqrt(1 + 4j * t)**2 * exp(i * (x + y - 2 * t)) * exp(-((x + 2t)^2 + (y + 2t)^2) / (1 + 4j * t)).
    """
    return 1 / np.sqrt(1 + 4j * t)**2 * np.exp(1j * (x + y - 2 * t)) * np.exp(-((x + 2 * t)**2 + (y + 2 * t)**2) / (1 + 4j * t))

# Automatic testing
n_test = 4
test_set = [solver.test(u_exact=u_exact, t_eval=i * Lt / n_test, threshold=0.3, component='abs') for i in range(n_test + 1)]


# ## Fractional diffusion equation



# Define the symbols and the equation
t, x, y, kx, ky, omega = symbols('t x y kx ky omega')
u_func = Function('u') 
u = u_func(t, x, y)

# Parameters
alpha = 1.5  # Fractional order
nu = 0.01    # Classical diffusion coefficient

# 2D fractional diffusion equation
eq = Eq(diff(u, t), -Op((kx**2 + ky**2)**(alpha/2), u))

# Create the solver
solver = PDESolver(eq, dealiasing_ratio=0.5)

# Domain and initial condition setup
Lx, Ly = 2 * np.pi, 2 * np.pi  # Domain size
Nx, Ny = 128, 128             # Grid resolution
Lt = 1.0                      # Total simulation time
Nt = 100                     # Number of time steps

# Initial condition: sinusoidal wave in 2D
initial_condition = lambda x, y: np.sin(x) * np.sin(y)

# Setup the solver
solver.setup(
    Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, Lt=Lt, Nt=Nt,
    initial_condition=initial_condition, 
    plot=False
)

# Solve the equation
solver.solve()

# Exact solution function at t = 1
def u_exact(x, y, t):
    return np.sin(x) * np.sin(y) * np.exp(-t * (1**2 + 1**2)**(alpha/2))

# Automatic testing
n_test = 4
test_set = [solver.test(u_exact=u_exact, t_eval=i * Lt / n_test, threshold=5e-2, component='real') for i in range(n_test + 1)]


# ## Non-linear equation



# Symbolic variables
t, x, y = symbols('t x y')
u_func = Function('u')
u = u_func(t, x, y)

# Definition of the equation ∂u/∂t = Δu + u²
eq = Eq(diff(u, t), diff(u, x, 2) + diff(u, y, 2) + u**2)

# Creation of the solver
solver = PDESolver(eq)

# Square domain centered around 0
L = 10.0
N = 256
Lt = 1   # Reduced time for improved stability (nonlinear term)
Nt = 1000   # More time steps for precision

# Initial condition: t = 0
def initial_condition(x, y):
    return np.exp(-(x**2 + y**2))  # Centered Gaussian

# ❌ No simple exact solution for this nonlinear equation.
# For validation: use numerical convergence
# But if you absolutely want an "exact solution", here is an approximate case:
# This is a heuristic approximation, valid for short durations:
def u_exact(x, y, t):
    """Empirical approximation for short duration"""
    denom = 4 * t + 1
    return 1 / denom * np.exp(-(x**2 + y**2) / denom)

# Solver setup
solver.setup(
    Lx=L, Ly=L,
    Nx=N, Ny=N,
    Lt=Lt, Nt=Nt,
    initial_condition=initial_condition, 
    plot=False
)

# Solving
solver.solve()

# Validation at different times
n_test = 4
test_results = [
    solver.test(u_exact=u_exact, t_eval=i * Lt / n_test, threshold=1, component='real')
    for i in range(n_test + 1)
]


# ## Wave equation
# ### With PDE



# Define the symbols and the unknown function
t, x, y, kx, ky, omega = symbols('t x y kx ky omega')
u_func = Function('u') 
u = u_func(t, x, y)

# 2D wave equation
eq = Eq(diff(u, t, t), diff(u, x, x) + diff(u, y, y))

# Initialize the solver
solver = PDESolver(eq, time_scheme='ETD-RK4')

# Parameters
Lx = Ly = 2 * np.pi
Nx = Ny = 512
Lt = 2.0
Nt = 400

# Configure the solver with the problem parameters
solver.setup(
    Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, Lt=Lt, Nt=Nt,
    initial_condition=lambda x, y: np.sin(x) * np.sin(y),
    initial_velocity=lambda x, y: np.zeros_like(x), 
    plot=False
)

# Solve the equation
solver.solve()

#solver.plot_energy()
#solver.plot_energy(log=True) 

# Exact solution: sin(x) sin(y) cos(ωt), with ω = sqrt(2)
def u_exact(x, y, t):
    """
    Analytical solution of the 2D wave equation.
    The solution is given by u(x, y, t) = sin(x) * sin(y) * cos(ωt),
    where ω = sqrt(kx^2 + ky^2) = sqrt(1^2 + 1^2).
    """
    omega = np.sqrt(1**2 + 1**2)
    return np.sin(x) * np.sin(y) * np.cos(omega * t)

# Automatic testing
n_test = 4
test_set = [solver.test(u_exact=u_exact, t_eval=i * Lt / n_test, threshold=5, component='abs') for i in range(n_test + 1)]


# ### With a source term



# Define the symbols and the wave equation with a source term
t, x, y, omega = symbols('t x y omega')
u_func = Function('u')
u = u_func(t, x, y)

# Source term
source_term = cos(x) * cos(y) * (5 * cos(sqrt(3) * t) - sqrt(3) * sin(sqrt(3) * t))

# Equation with source term
eq = Eq(diff(u, t, t), diff(u, x, x) + diff(u, y, y) - u + source_term)

# Create the solver with ETD-RK4 scheme
solver = PDESolver(eq, time_scheme='ETD-RK4', dealiasing_ratio=2/3)

# Simulation parameters
Lt = 2.0  # Total simulation time
Lx, Ly = 20, 20  # Spatial domain size
Nx, Ny = 128, 128  # Number of grid points
Nt = 200  # Number of time steps

# Initial conditions
initial_condition = lambda x, y: np.cos(x) * np.cos(y)
initial_velocity = lambda x, y: np.zeros_like(x)

# Setup the solver
solver.setup(
    Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, Lt=Lt, Nt=Nt,
    initial_condition=initial_condition,
    initial_velocity=initial_velocity, 
    plot=False
)

# Solve the equation
solver.solve()

# Plot energy evolution
#solver.plot_energy()
#solver.plot_energy(log=True)

# Exact solution at any time t
def u_exact(x, y, t):
    return np.cos(x) * np.cos(y) * np.cos(np.sqrt(3) * t)

# Automatic testing at multiple times
n_test = 4
test_set = [
    solver.test(
        u_exact=u_exact,
        t_eval=i * Lt / n_test,
        threshold=5,  # Adjust threshold if necessary
        component='real'
    )
    for i in range(n_test + 1)
]


# ### With psiOp



# Define the symbols and the unknown function
t, x, y, xi, eta = symbols('t x y xi eta', real=True)
u_func = Function('u') 
u = u_func(t, x, y)

# 2D wave equation
eq = Eq(diff(u, t, t), psiOp(-(xi**2 + eta**2), u))

# Initialize the solver
solver = PDESolver(eq)

# Parameters
Lx = Ly = 2 * np.pi
Nx = Ny = 512
Lt = 2.0
Nt = 400

# Configure the solver with the problem parameters
solver.setup(
    Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, Lt=Lt, Nt=Nt,
    initial_condition=lambda x, y: np.sin(x) * np.sin(y),
    initial_velocity=lambda x, y: np.zeros_like(x),
    plot=False
)

# Solve the equation
solver.solve()

#solver.plot_energy()
#solver.plot_energy(log=True) 

# Exact solution: sin(x) sin(y) cos(ωt), with ω = sqrt(2)
def u_exact(x, y, t):
    """
    Analytical solution of the 2D wave equation.
    The solution is given by u(x, y, t) = sin(x) * sin(y) * cos(ωt),
    where ω = sqrt(kx^2 + ky^2) = sqrt(1^2 + 1^2).
    """
    omega = np.sqrt(1**2 + 1**2)
    return np.sin(x) * np.sin(y) * np.cos(omega * t)

# Automatic testing
n_test = 4
test_set = [solver.test(u_exact=u_exact, t_eval=i * Lt / n_test, threshold=0.2, component='abs') for i in range(n_test + 1)]


# ## Klein-Gordon equation



# Physical parameters
c = 1.0  # Wave speed
m = 1.0  # Field mass

# Definition of the 2D Klein-Gordon equation
t, x, y, kx, ky, omega = symbols('t x y kx ky omega')
u_func = Function('u') 
u = u_func(t, x, y)

klein_gordon_eq = Eq(diff(u, t, t), c**2*(diff(u, x, x) + diff(u, y, y)) - m**2*u)

# Creation of the solver 
solver = PDESolver(klein_gordon_eq)

# Domain configuration
L = 2 * np.pi  # Square domain [0, 2π] × [0, 2π]
N = 512        # Grid points per dimension
T_final = 2.0  # Final time
Nt = 200       # Time steps

# Initial conditions
kx = 1
ky = 1
omega = np.sqrt(c**2*(kx**2 + ky**2) + m**2)  # Direct numerical calculation

solver.setup(
    Lx=L, Ly=L,
    Nx=N, Ny=N,
    Lt=T_final, Nt=Nt,
    initial_condition=lambda x, y: np.sin(x) * np.sin(y),
    initial_velocity=lambda x, y: np.zeros_like(x),  # Initial time derivative is zero
    plot=False
)

# Replace the definition of the exact solution with:
omega_val = float(np.sqrt(c**2*(kx**2 + ky**2) + m**2))  # Convert to float

def u_exact(x, y, t):
    return np.sin(x) * np.sin(y) * np.cos(omega_val * t)

# Solving and validation
solver.solve()

#solver.plot_energy()
#solver.plot_energy(log=True) 

# Automatic testing
n_test = 4
test_set = [solver.test(u_exact=u_exact, t_eval=i * Lt / n_test, threshold=1e-1, component='real') for i in range(n_test + 1)]


# ## Biharmonic equation



# Definition of the 2D biharmonic equation
t, x, y, kx, ky, omega = symbols('t x y kx ky omega')
u_func = Function('u') 
u = u_func(t, x, y)

biharmonic_eq = Eq(diff(u, t), -(diff(u, x, 4) + 2*diff(u, x, 2, y, 2) + diff(u, y, 4)))

# Creation of the solver with periodic boundary conditions
solver = PDESolver(biharmonic_eq)

# Configuration of the domain
L = 2 * np.pi  # Square domain [0, 2π] × [0, 2π]
N = 512         # Grid points per dimension
Lt = 2.0  # Final time
Nt = 400       # Time steps

# Initial sinusoidal 2D condition
initial_condition = lambda x, y: np.sin(x) * np.sin(y)

solver.setup(
    Lx=L, Ly=L,
    Nx=N, Ny=N,
    Lt=Lt, Nt=Nt,
    initial_condition=initial_condition, 
    plot=False
)

# Corresponding exact solution
def u_exact(x, y, t):
    return np.sin(x) * np.sin(y) * np.exp(-4*t)

# Solving 
solver.solve()

# Automatic testing
n_test = 4
test_set = [solver.test(u_exact=u_exact, t_eval=i * Lt / n_test, threshold=7e-2, component='real') for i in range(n_test + 1)]






