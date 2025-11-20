# tests/test_all_pdes_2d.py
import numpy as np
import pytest
from sympy import symbols, Function, Eq, sin, exp, I, diff, cos, sqrt
from solver import PDESolver, psiOp, Op

# =====================================================================
# 2D Stationary psiOp (constant symbol)
# =====================================================================
def test_stationary_psiOp_constant_symbol_2d():
    x, y, xi, eta = symbols('x y xi eta', real=True)
    u = Function('u')(x, y)

    equation = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))

    def u_exact(x, y):
        return -np.sin(x) * np.sin(y) / 3

    solver = PDESolver(equation)
    solver.setup(Lx=2*np.pi, Ly=2*np.pi, Nx=128, Ny=128, initial_condition=None, plot=False)

    u_num = solver.solve_stationary_psiOp()
    ref = u_exact(solver.X, solver.Y)

    err = np.linalg.norm(np.real(u_num) - ref)/np.linalg.norm(ref)
    assert err < 2e1

# =====================================================================
# 2D Stationary psiOp (variable symbol)
# =====================================================================
def test_stationary_psiOp_variable_symbol_2d():
    x, y, xi, eta = symbols('x y xi eta', real=True)
    u = Function('u')(x, y)

    p_symbol = xi**2 + eta**2 + x**2 + y**2 + 1
    u_exact_expr = exp(-x**2 - y**2)

    def u_exact(x, y): return np.exp(-x**2 - y**2)

    f_expr = (-3*x**2 - 3*y**2 + 5)*u_exact_expr

    equation = Eq(psiOp(p_symbol, u), f_expr)

    solver = PDESolver(equation)
    solver.setup(Lx=10, Ly=10, Nx=64, Ny=64, initial_condition=None, plot=False)
    u_num = solver.solve_stationary_psiOp(order=1)

    ref = u_exact(solver.X, solver.Y)
    err = np.linalg.norm(np.real(u_num) - ref)/np.linalg.norm(ref)

    assert err < 3e-1

# =====================================================================
# 2D Transport
# =====================================================================
def test_transport_2d():
    t, x, y = symbols('t x y')
    u = Function('u')(t, x, y)
    eq = Eq(diff(u,t), -diff(u,x) - diff(u,y))

    L=10
    N=128
    Lt=1.0

    solver = PDESolver(eq)
    solver.setup(Lx=L, Ly=L, Nx=N, Ny=N, Lt=Lt, Nt=200,
                 initial_condition=lambda x,y: np.exp(-(x**2+y**2)),
                 plot=False)
    solver.solve()

    def u_exact(x,y,t):
        Lx=Ly=L
        xs=(x - t + L/2) % L - L/2
        ys=(y - t + L/2) % L - L/2
        return np.exp(-(xs**2+ys**2))

    err = solver.test(u_exact=u_exact, t_eval=Lt/2, threshold=5e-2, component='real')
    assert err is not None and err < 5e-2

# =====================================================================
# 2D Heat
# =====================================================================
def test_heat_2d():
    t,x,y = symbols('t x y')
    u = Function('u')(t,x,y)

    eq = Eq(diff(u,t), diff(u,x,2) + diff(u,y,2))

    solver = PDESolver(eq)
    solver.setup(Lx=2*np.pi, Ly=2*np.pi, Nx=128, Ny=128, Lt=1.0, Nt=50,
                 initial_condition=lambda x,y: np.sin(x)*np.sin(y), plot=False)
    solver.solve()

    def u_exact(x,y,t): return np.sin(x)*np.sin(y)*np.exp(-2*t)

    err = solver.test(u_exact=u_exact, t_eval=0.5, threshold=5e-2, component='real')
    assert err is not None and err < 5e-2

# =====================================================================
# 2D Heat with psiOp
# =====================================================================
def test_heat_psiop_2d():
    t,x,y,xi,eta = symbols('t x y xi eta', real=True)
    u = Function('u')(t,x,y)

    eq = Eq(diff(u,t), psiOp(-(xi**2 + eta**2), u))

    solver = PDESolver(eq)
    solver.setup(Lx=2*np.pi, Ly=2*np.pi, Nx=128, Ny=128, Lt=1.0, Nt=50,
                 initial_condition=lambda x,y: np.sin(x)*np.sin(y), plot=False)
    solver.solve()

    def u_exact(x,y,t): return np.sin(x)*np.sin(y)*np.exp(-2*t)

    err = solver.test(u_exact=u_exact, t_eval=0.5, threshold=5e-2, component='real')
    assert err is not None and err < 5e-2

# =====================================================================
# 2D Schrödinger PDE
# =====================================================================
def test_schrodinger_2d():
    t,x,y = symbols('t x y')
    u = Function('u')(t,x,y)
    eq = Eq(I*diff(u,t), diff(u,x,2)+diff(u,y,2))

    solver = PDESolver(eq, time_scheme='ETD-RK4')
    solver.setup(Lx=10, Ly=10, Nx=128, Ny=128, Lt=1.0, Nt=100,
                 initial_condition=lambda x,y: np.exp(-x**2-y**2)*np.exp(1j*(x+y)),
                 plot=False)
    solver.solve()

    def u_exact(x,y,t):
        return 1/np.sqrt(1+4j*t)**2 * np.exp(1j*(x+y-2*t)) * np.exp(-((x+2*t)**2+(y+2*t)**2)/(1+4j*t))

    err = solver.test(u_exact=u_exact, t_eval=0.5, threshold=0.3, component='abs')
    assert err is not None and err < 0.3

# =====================================================================
# 2D Fractional diffusion
# =====================================================================
def test_fractional_diffusion_2d():
    t,x,y,kx,ky = symbols('t x y kx ky')
    u = Function('u')(t,x,y)

    alpha=1.5
    eq = Eq(diff(u,t), -Op((kx**2+ky**2)**(alpha/2), u))

    solver = PDESolver(eq)
    solver.setup(Lx=2*np.pi, Ly=2*np.pi, Nx=128, Ny=128, Lt=1.0, Nt=80,
                 initial_condition=lambda x,y: np.sin(x)*np.sin(y), plot=False)
    solver.solve()

    def u_exact(x,y,t): return np.sin(x)*np.sin(y)*np.exp(-t*(2**(alpha/2)))

    err = solver.test(u_exact=u_exact, t_eval=0.5, threshold=5e-2, component='real')
    assert err is not None and err < 5e-2

# =====================================================================
# 2D Wave equation
# =====================================================================
def test_wave_2d():
    t,x,y = symbols('t x y')
    u = Function('u')(t,x,y)

    eq = Eq(diff(u,t,2), diff(u,x,2)+diff(u,y,2))

    solver = PDESolver(eq, time_scheme='ETD-RK4')
    solver.setup(Lx=2*np.pi, Ly=2*np.pi, Nx=128, Ny=128, Lt=1.0, Nt=100,
                 initial_condition=lambda x,y: np.sin(x)*np.sin(y),
                 initial_velocity=lambda x,y: np.zeros_like(x),
                 plot=False)
    solver.solve()

    def u_exact(x,y,t): return np.sin(x)*np.sin(y)*np.cos(np.sqrt(2)*t)

    err = solver.test(u_exact=u_exact, t_eval=0.5, threshold=5, component='abs')
    assert err is not None and err < 5

# =====================================================================
# 2D Klein-Gordon
# =====================================================================
def test_klein_gordon_2d():
    # Physical parameters
    c = 1.0
    m = 1.0

    # Symbols and PDE definition
    t, x, y = symbols('t x y')
    u = Function('u')(t, x, y)

    klein_gordon_eq = Eq(diff(u, t, 2),
                         c**2 * (diff(u, x, 2) + diff(u, y, 2)) - m**2 * u)

    # Domain configuration
    L = 2 * np.pi
    N = 512
    T_final = 2.0
    Nt = 200

    kx = 1
    ky = 1
    omega_val = float(np.sqrt(c**2 * (kx**2 + ky**2) + m**2))

    solver = PDESolver(klein_gordon_eq)
    solver.setup(
        Lx=L, Ly=L,
        Nx=N, Ny=N,
        Lt=T_final, Nt=Nt,
        initial_condition=lambda x, y: np.sin(x) * np.sin(y),
        initial_velocity=lambda x, y: np.zeros_like(x),
        plot=False
    )

    # Exact solution
    def u_exact(x, y, t):
        return np.sin(x) * np.sin(y) * np.cos(omega_val * t)

    solver.solve()

    err = solver.test(u_exact=u_exact, t_eval=0.5, threshold=1e-1, component='abs')
    assert err is not None and err < 5


# =====================================================================
# 2D Biharmonic
# =====================================================================
def test_biharmonic_2d():
    # PDE definition
    t, x, y = symbols('t x y')
    u = Function('u')(t, x, y)

    biharmonic_eq = Eq(
        diff(u, t),
        -(diff(u, x, 4) + 2 * diff(u, x, 2, y, 2) + diff(u, y, 4))
    )

    # Domain configuration
    L = 2 * np.pi
    N = 512
    Lt = 2.0
    Nt = 400

    solver = PDESolver(biharmonic_eq)
    solver.setup(
        Lx=L, Ly=L,
        Nx=N, Ny=N,
        Lt=Lt, Nt=Nt,
        initial_condition=lambda x, y: np.sin(x) * np.sin(y),
        plot=False
    )

    # Exact solution
    def u_exact(x, y, t):
        return np.sin(x) * np.sin(y) * np.exp(-4 * t)

    solver.solve()

    err = solver.test(u_exact=u_exact, t_eval=0.5, threshold=7e-2, component='abs')
    assert err is not None and err < 7e-2


