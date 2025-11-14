#!/usr/bin/env python
# coding: utf-8







from solver import *


# # 2D tests in Dirichlet conditions

# ## Stationnary equation



# Symbolic variables
x, y, xi, eta = symbols('x y xi eta', real=True)
u = Function('u')(x, y)

# Symbolic equation
symbol = xi**2 + eta**2 + 1
equation = Eq(psiOp(symbol, u), -sin(x) * sin(y))

# Define exact solution (vanishing at x= -pi, pi and y= -pi, pi)
def u_exact(x, y):
    return -np.sin(x) * np.sin(y) / 3
# Solver creation
solver = PDESolver(equation)

# Domain setup with Dirichlet boundary condition
Lx, Ly = 2 * np.pi, 2 * np.pi
N = 64
Nx, Ny = N, N

# Setup solver
solver.setup(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, boundary_condition='dirichlet', initial_condition=None, plot=False)

# Stationary solution
u_num = solver.solve_stationary_psiOp(order=6)

# Accuracy test
solver.test(u_exact=u_exact, threshold=1e-5, component='real')

# ## Stationnary equation with psiOp() depending on spatial variables



# Symbolic variables
x, y, xi, eta = symbols('x y xi eta', real=True)
u = Function('u')(x, y)
# Symbol dependent on x, y, xi, eta
symbol = x**2 + y**2 + xi**2 + eta**2 + 1  # Elliptic symbol
equation = Eq(psiOp(symbol, u), -exp(-(x**2 + y**2)))  # Source = Gaussian
# Exact solution (satisfies Dirichlet conditions on [-π, π]×[-π, π])
def u_exact(x, y):
    return -np.exp(-(x**2 + y**2)) / 3  # Inverse symbol applied to the source
# Solver configuration
solver = PDESolver(equation)
Lx, Ly = 2 * np.pi, 2 * np.pi  # Extended domain to satisfy Dirichlet
Nx, Ny = 64, 64
# Initialization with Dirichlet conditions
solver.setup(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, boundary_condition='dirichlet', plot=False)
solver.space_window = True
# Stationary solution
u_num = solver.solve_stationary_psiOp(order=0)
# Accuracy test
solver.test(u_exact=u_exact, threshold=1, component='abs')


# ## Diffusion equation



# Variables symboliques
t, x, y, xi, eta = symbols('t x y xi eta', real=True)
u = Function('u')

# Opérateur pseudo-différentiel : ∂u/∂t = -psiOp(ξ² + η² + 1, u)
equation = Eq(diff(u(t, x, y), t), -psiOp(xi**2 + eta**2 + 1, u(t, x, y)))

# Création du solveur
solver = PDESolver(equation)

# Paramètres
Lx, Ly = 2 * np.pi, 2 * np.pi  # Domaine [-π, π] × [-π, π]
Nx, Ny = 64, 64                # Résolution spatiale
Lt = 2.0                        # Durée temporelle
Nt = 500                        # Résolution temporelle

# Condition initiale : produit de sinus
k0, l0 = 1.0, 1.0
initial_condition = lambda x, y: np.sin(k0 * x) * np.sin(l0 * y)

# Solution exacte
def u_exact(x, y, t):
    return np.sin(k0 * x) * np.sin(l0 * y) * np.exp(- (k0**2 + l0**2 + 1) * t)

# Configuration avec conditions de Dirichlet
solver.setup(
    Lx=Lx, Ly=Ly,
    Nx=Nx, Ny=Ny,
    Lt=Lt, Nt=Nt,
    boundary_condition='dirichlet',
    initial_condition=initial_condition,
    n_frames=10, 
    plot=False
)

# Résolution
solver.solve()

# Tests de précision
n_test = 4
for i in range(n_test + 1):
    t_eval = i * Lt / n_test
    print(f"Test à t = {t_eval:.2f}")
    solver.test(u_exact=u_exact, t_eval=t_eval, threshold=1, component='real')






