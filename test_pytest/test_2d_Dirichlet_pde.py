import pytest
import numpy as np
from sympy import symbols, Function, Eq, sin, exp, diff
from solver import PDESolver, psiOp

def test_stationary_equation_constant_symbol():
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

    # Domain setup with Dirichlet boundary condition
    Lx, Ly = 2 * np.pi, 2 * np.pi
    N = 64
    Nx, Ny = N, N

    # Setup solver
    solver.setup(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, boundary_condition='dirichlet', initial_condition=None, plot=False)

    # Stationary solution
    u_num = solver.solve_stationary_psiOp(order=6)

    # Accuracy test
    threshold = 1e-5
    err = solver.test(u_exact=u_exact, threshold=threshold, component='real')
    
    assert err is not None and err < threshold

def test_stationary_equation_space_dependent_symbol():
    # Symbolic variables
    x, y, xi, eta = symbols('x y xi eta', real=True)
    u = Function('u')(x, y)
    
    # Symbol dependent on x, y, xi, eta
    symbol = x**2 + y**2 + xi**2 + eta**2 + 1  # Elliptic symbol
    equation = Eq(psiOp(symbol, u), -exp(-(x**2 + y**2)))  # Source = Gaussian
    
    # Exact solution
    def u_exact(x, y):
        return -np.exp(-(x**2 + y**2)) / 3  # Inverse symbol applied to the source
    
    # Solver configuration
    solver = PDESolver(equation)
    Lx, Ly = 2 * np.pi, 2 * np.pi
    Nx, Ny = 64, 64
    
    # Initialization with Dirichlet conditions
    solver.setup(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, boundary_condition='dirichlet', plot=False)
    solver.space_window = True
    
    # Stationary solution
    u_num = solver.solve_stationary_psiOp(order=0)
    
    # Accuracy test
    threshold = 1
    err = solver.test(u_exact=u_exact, threshold=threshold, component='abs')
    
    assert err is not None and err < threshold

def test_diffusion_equation():
    # Variables symboliques
    t, x, y, xi, eta = symbols('t x y xi eta', real=True)
    u = Function('u')

    # Opérateur pseudo-différentiel : ∂u/∂t = -psiOp(ξ² + η² + 1, u)
    equation = Eq(diff(u(t, x, y), t), -psiOp(xi**2 + eta**2 + 1, u(t, x, y)))

    # Création du solveur
    solver = PDESolver(equation)

    # Paramètres
    Lx, Ly = 2 * np.pi, 2 * np.pi
    Nx, Ny = 64, 64
    Lt = 2.0
    Nt = 500

    # Condition initiale
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
    threshold = 1
    
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=threshold, component='real')
        assert err is not None and err < threshold