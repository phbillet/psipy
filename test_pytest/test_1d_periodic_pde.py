# tests/test_all_pdes.py
"""
Fichier pytest unique généré automatiquement à partir de
/mnt/data/pdesolver_tester_periodic_1d.py

Chaque bloc du script original a été transformé en une fonction
`test_...()` indépendante. Les tests utilisent l'API `PDESolver`
présente dans le dépôt (importée depuis `solver`).

Pour exécuter : `pytest -q tests/test_all_pdes.py`

Remarque : certains tests peuvent être lourds (grandes valeurs de Nx, Nt).
Pour exécution rapide en CI/locally, réduisez Nx, Nt, ou commentez des tests.
"""

from solver import *

# Utility: small helper to run solver and assert via solver.test
def _run_and_assert(solver, u_exact, t_eval=None, threshold=1e-3, component='real'):
    """Run the solver (if not already run) and use solver.test()
    Returns the value from solver.test() so tests can assert on it.
    """
    # Many solver.setup(...) calls in tests call solver.solve() themselves.
    # Here we just call solver.test() and assert the returned error is below threshold.
    result = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=threshold, component=component)
    assert result < threshold
    return result


# -----------------------------
# 1) Stationary: psiOp constant
# -----------------------------

def test_stationary_psiOp_constant_symbol():
    x, xi = symbols('x xi')
    u = Function('u')(x)
    equation = Eq(psiOp(xi**2 + 1, u), sin(x))

    def u_exact(x):
        return np.sin(x) / 2

    solver = PDESolver(equation)
    solver.setup(Lx=2*np.pi, Nx=256, initial_condition=None, plot=False)

    u_num = solver.solve_stationary_psiOp(order=1)
    u_ref = u_exact(solver.X)
    error = np.linalg.norm(np.real(u_num) - u_ref) / np.linalg.norm(u_ref)

    assert error < 5e-3


# ------------------------------------------------
# 2) Stationary: psiOp depending on x (non-const xi)
# ------------------------------------------------

def test_stationary_psiOp_x_dependent():
    x, xi = symbols('x xi', real=True)
    u = Function('u')(x)
    p_symbol = xi**2 + x**2 + 1
    f_expr = 3 * (1 - x**2) * exp(-x**2)
    equation = Eq(psiOp(p_symbol, u), f_expr)

    def u_exact(x):
        return np.exp(-x**2)

    solver = PDESolver(equation)
    solver.setup(Lx=30, Nx=512, initial_condition=None, plot=False)

    u_num = solver.solve_stationary_psiOp(order=1)
    # perform a sanity test (looser threshold because domain large)
    u_ref = u_exact(solver.X)
    err = np.linalg.norm(np.real(u_num) - u_ref) / np.linalg.norm(u_ref)
    assert err < 2.0


# --------------------------------------
# 3) Integro-differential with Op (time)
# --------------------------------------

def test_integro_differential_with_Op():
    t, x, kx, xi, omega = symbols('t x kx xi omega')
    u_func = Function('u')
    u = u_func(t, x)
    eps = 0.001
    integral_term = Op(1 / (I * (kx + eps)), u)

    eq = Eq(diff(u, t), diff(u, x) + integral_term - u)
    solver = PDESolver(eq, time_scheme='ETD-RK4')

    Lt = 0.5
    Lx = 20
    Nx = 256
    Nt = 50

    solver.setup(Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt, initial_condition=lambda x: np.sin(x), plot=False)
    solver.solve()

    def u_exact(x, t):
        return np.exp(-t) * np.sin(x)

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=0.6, component='real')
        assert err is not None and err < 0.6


# --------------------------------------
# 4) Integro-differential with psiOp
# --------------------------------------

def test_integro_differential_with_psiOp():
    t, x, xi, omega = symbols('t x xi omega')
    u_func = Function('u')
    u = u_func(t, x)
    eps = 0.001
    integral_term = psiOp(I*xi + 1 / (I * (xi + eps)) - 1, u)

    eq = Eq(diff(u, t), integral_term)
    solver = PDESolver(eq, dealiasing_ratio=2/3)

    Lt = 0.5
    Lx = 20
    Nx = 256
    Nt = 50

    solver.setup(Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt, initial_condition=lambda x: np.sin(x), plot=False)
    solver.solve()

    def u_exact(x, t):
        return np.exp(-t) * np.sin(x)

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=0.6, component='real')
        assert err is not None and err < 0.6


# ----------------------------------
# 5) Convolution equation with Op
# ----------------------------------

def test_convolution_with_Op():
    t, x, kx, xi, omega, lam = symbols('t x kx xi omega lambda')
    u_func = Function('u')
    u = u_func(t, x)
    lam_val = 1.0
    f_kernel = exp(-lam_val * Abs(x))

    # Define the kernel f(x) = exp(-lambda * |x|)
    f_kernel = exp(-lam_val * Abs(x))
    
    # Convolution equation: ∂t u = - OpConv(f_kernel, u)
    eq = Eq(diff(u, t), -Op(fourier_transform(f_kernel, x, kx/(2*pi)), u))

    solver = PDESolver(eq, time_scheme='ETD-RK4')
    Lt = 0.5
    solver.setup(Lx=2*np.pi, Nx=256, Lt=Lt, Nt=50, initial_condition=lambda x: np.cos(x), plot=False)
    solver.solve()

    def u_exact(x, t):
        lam_val = 1.0
        k_val = 1.0
        decay = 2 * lam_val / (lam_val**2 + k_val**2)
        return np.cos(x) * np.exp(-decay * t)

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=5e-3, component='real')
        assert err is not None and err < 5e-3

# ----------------------------------
# 5b) Convolution equation with psiOp
# ----------------------------------

def test_convolution_with_psiOp():
    # Define symbols
    t, x, kx, xi, omega, lam = symbols('t x kx xi omega lambda')
    u_func = Function('u')
    u = u_func(t, x)
    lam = 1.0
    # Define the kernel f(x) = exp(-lambda * |x|)
    f_kernel = exp(-lam * Abs(x))
    
    # Convolution equation: ∂t u = - OpConv(f_kernel, u)
    eq = Eq(diff(u, t), -psiOp(fourier_transform(f_kernel, x, xi/(2*pi)), u))
    
    # Create solver with lambda=1
    solver = PDESolver(eq.subs(lam, 1))
    
    # Simulation parameters
    Lt = 2.0
    solver.setup(
        Lx=2 * np.pi, Nx=256, Lt=Lt, Nt=100,
        initial_condition=lambda x: np.cos(x),
        plot=False
    )
    
    # Solve
    solver.solve()
    
    # Exact solution: u(x, t) = cos(x) * exp(- (2*lambda / (lambda^2 + k^2)) * t )
    def u_exact(x, t):
        lam_val = 1.0
        k_val = 1.0  # mode k=1 (cos(x))
        decay = 2 * lam_val / (lam_val**2 + k_val**2)
        return np.cos(x) * np.exp(-decay * t)
    
    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=5e-3, component='real')
        assert err is not None and err < 5e-3

# --------------------------
# 6) Transport (PDE simple)
# --------------------------

def test_transport_pde():
    t, x = symbols('t x')
    u = Function('u')(t, x)
    eq = Eq(diff(u, t), -diff(u, x))
    Lt = 1.0
    solver = PDESolver(eq, time_scheme='ETD-RK4')
    solver.setup(Lx=10, Nx=256, Lt=Lt, Nt=200, initial_condition=lambda x: np.exp(-x**2), plot=False)
    solver.solve()

    def u_exact(x, t):
        L = 10
        return np.exp(-((x - t + L/2) % L - L/2)**2)

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=5e-2, component='real')
        assert err is not None and err < 5e-2


# --------------------
# 7) Transport with Op
# --------------------

def test_transport_with_Op():
    t, x, kx, xi, omega = symbols('t x kx xi omega')
    u_func = Function('u')
    u = u_func(t, x)
    c = 1.0
    beta = 1.3
    eq = Eq(diff(u, t) + c * diff(u, x), -Op(abs(kx)**beta, u))
    Lt=0.5
    solver = PDESolver(eq, dealiasing_ratio=0.5, time_scheme='ETD-RK4')
    solver.setup(Lx=2*np.pi, Nx=256, Lt=Lt, Nt=200, initial_condition=lambda x: np.cos(x), plot=False)
    solver.solve()

    def u_exact(x, t):
        return np.cos(x - c * t) * np.exp(-t * abs(1)**beta)

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=7e-2, component='real')
        assert err is not None and err < 7e-2


# --------------------
# 8) Transport psiOp
# --------------------

def test_transport_with_psiOp():
    t, x, kx, xi, omega = symbols('t x kx xi omega')
    u_func = Function('u')
    u = u_func(t, x)
    c = 1.0
    beta = 1.3
    eq = Eq(diff(u, t), -psiOp(c*I*xi + abs(xi)**beta, u))
    Lt = 0.5
    solver = PDESolver(eq, dealiasing_ratio=2/3)
    solver.setup(Lx=2*np.pi, Nx=256, Lt=Lt, Nt=200, initial_condition=lambda x: np.cos(x), plot=False)
    solver.solve()

    def u_exact(x, t):
        return np.cos(x - c * t) * np.exp(-t * abs(1)**beta)

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=7e-2, component='real')
        assert err is not None and err < 7e-2


# ----------------
# 9) Heat (sin)
# ----------------

def test_heat_periodic_sin():
    t, x = symbols('t x')
    u_func = Function('u')
    u = u_func(t, x)
    eq = Eq(diff(u, t), diff(u, x, 2))
    Lt = 0.5

    solver = PDESolver(eq, time_scheme='ETD-RK4')
    solver.setup(Lx=2*np.pi, Nx=128, Lt=Lt, Nt=40, initial_condition=lambda x: np.sin(x), plot=False)
    solver.solve()

    def u_exact(x, t):
        return np.sin(x) * np.exp(-t)

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=5e-2, component='real')
        assert err is not None and err < 5e-2


# -----------------------------
# 10) Heat with Gaussian initial
# -----------------------------

def test_heat_gaussian_initial():
    t, x = symbols('t x')
    u = Function('u')(t, x)
    eq = Eq(diff(u, t), diff(u, x, 2))

    solver = PDESolver(eq, time_scheme='ETD-RK4')
    Lt = 0.5
    solver.setup(Lx=5*np.pi, Nx=256, Lt=Lt, Nt=40, initial_condition=lambda x: np.exp(-2 * x**2), plot=False)
    solver.solve()

    def u_exact(x, t):
        sigma = 0.5
        variance = sigma**2 + 2 * t
        return (np.exp(-x**2 / (2 * variance)) / np.sqrt(1 + 2 * t / sigma**2))

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=5e-2, component='real')
        assert err is not None and err < 5e-2


# -------------------------
# 11) Heat with Op variant
# -------------------------

def test_heat_with_Op():
    t, x, kx, xi, omega = symbols('t x kx xi omega')
    u_func = Function('u')
    u = u_func(t, x)
    eq = Eq(diff(u, t), Op((I*kx)**2, u)/2 + diff(u, x, 2)/2)
    Lt = 0.5

    solver = PDESolver(eq, time_scheme='ETD-RK4')
    solver.setup(Lx=2*np.pi, Nx=128, Lt=Lt, Nt=40, initial_condition=lambda x: np.sin(x), plot=False)
    solver.solve()

    def u_exact(x, t):
        return np.sin(x) * np.exp(-t)

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=5e-2, component='real')
        assert err is not None and err < 5e-2


# ---------------------------
# 12) Fractional heat / Op
# ---------------------------

def test_fractional_heat():
    t, x, kx, xi, omega = symbols('t x kx xi omega')
    u_func = Function('u')
    u = u_func(t, x)
    alpha = 1.8
    nu = 0.1
    eq = Eq(diff(u, t), -Op(abs(kx)**alpha, u) + nu * diff(u, x, 2))
    Lt = 0.5

    solver = PDESolver(eq, dealiasing_ratio=0.5, time_scheme='ETD-RK4')
    solver.setup(Lx=2*np.pi, Nx=128, Lt=Lt, Nt=200, initial_condition=lambda x: np.sin(x), plot=False)
    solver.solve()

    def u_exact(x, t):
        return np.sin(x) * np.exp(-t * (abs(1)**alpha + nu * 1**2))

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=2e-2, component='real')
        assert err is not None and err < 2e-2


# ----------------
# 13) Burgers'
# ----------------

def test_burgers_equation():
    t, x = symbols('t x')
    u = Function('u')(t, x)
    nu = 0.1
    eq = Eq(diff(u, t), -u * diff(u, x) + nu * diff(u, x, 2))

    def phi(x, t):
        return 2 + np.sin(x) * np.exp(-nu * t)
    def dphi_dx(x, t):
        return np.cos(x) * np.exp(-nu * t)
    def u_exact(x, t):
        return -2 * nu * dphi_dx(x, t) / phi(x, t)
    Lt = 0.5

    solver = PDESolver(eq, time_scheme='ETD-RK4')
    solver.setup(Lx=2*np.pi, Nx=128, Lt=Lt, Nt=100, initial_condition=lambda x: u_exact(x,0), plot=False)
    solver.solve()

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=5e-2, component='real')
        assert err is not None and err < 5e-2


# -----------------------------------
# 14) psiOp depending on x (multiplicative)
# -----------------------------------

def test_psiOp_spatial_potential():
    t, x, xi = symbols('t x xi', real=True)
    u = Function('u')
    symbol_expr = 1 + sin(x)
    eq = Eq(diff(u(t,x), t), -psiOp(symbol_expr, u(t,x)))
    Lt = 0.5
    solver = PDESolver(eq)
    solver.setup(Lx=2*np.pi, Nx=128, Lt=Lt, Nt=50, boundary_condition='periodic', initial_condition=lambda x: np.cos(x), plot=False)
    solver.solve()

    def u_exact(x, t):
        return np.cos(x) * np.exp(-t * (1 + np.sin(x)))

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=5e-2, component='real')
        assert err is not None and err < 5e-2


# ----------------
# 15) Wave equation
# ----------------

def test_wave_periodic_sin():
    t, x = symbols('t x')
    u_func = Function('u')
    u = u_func(t, x)
    eq = Eq(diff(u, t, t), diff(u, x, x))
    Lt = 0.5

    solver = PDESolver(eq, time_scheme='ETD-RK4')
    solver.setup(Lx=2*np.pi, Nx=256, Lt=Lt, Nt=100, initial_condition=lambda x: np.sin(x), initial_velocity=lambda x: np.zeros_like(x), plot=False)
    solver.solve()

    def u_exact(x, t):
        return np.sin(x) * np.cos(t)

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=0.2, component='real')
        assert err is not None and err < 0.2


# ----------------------
# 16) Wave with source
# ----------------------

def test_wave_with_source():
    t, x = symbols('t x')
    u_func = Function('u')
    u = u_func(t, x)
    source_term = cos(x) * (3 * cos(np.sqrt(2) * t) - np.sqrt(2) * sin(np.sqrt(2) * t))
    eq = Eq(diff(u, t, t), diff(u, x, x) - u + source_term)

    solver = PDESolver(eq, time_scheme='ETD-RK4', dealiasing_ratio=1)
    Lt = 0.5

    solver.setup(Lx=20, Nx=256, Lt=Lt, Nt=100, initial_condition=lambda x: np.cos(x), initial_velocity=lambda x: np.zeros_like(x), plot=False)
    solver.solve()

    def u_exact(x, t):
        return np.cos(x) * np.cos(np.sqrt(2) * t)

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=1, component='real')
        assert err is not None and err < 1


# ----------------
# 17) Schrödinger
# ----------------

def test_schrodinger_free_packet():
    t, x = symbols('t x')
    u_func = Function('u')
    u = u_func(t, x)
    eq = Eq(diff(u, t), -I * diff(u, x, x))
    Lt = 0.5

    solver = PDESolver(eq, time_scheme='ETD-RK4', dealiasing_ratio=1/2)
    solver.setup(Lx=20, Nx=512, Lt=Lt, Nt=200, initial_condition=lambda x: np.exp(-x**2) * np.exp(1j * x), plot=False)
    solver.solve()

    def u_exact(x, t):
        return 1 / np.sqrt(1 - 4j * t) * np.exp(1j * (x + t)) * np.exp(-((x + 2*t)**2) / (1 - 4j * t))

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=0.6, component='real')
        assert err is not None and err < 0.6


# ----------------
# 18) Klein-Gordon
# ----------------

def test_klein_gordon():
    t, x = symbols('t x')
    u_func = Function('u')
    u = u_func(t, x)
    eq = Eq(diff(u, t, t), diff(u, x, x) - u)
    Lt = 0.5

    solver = PDESolver(eq, time_scheme='ETD-RK4', dealiasing_ratio=1/4)
    solver.setup(Lx=20, Nx=256, Lt=Lt, Nt=100, initial_condition=lambda x: np.cos(x), initial_velocity=lambda x: np.zeros_like(x), plot=False)
    solver.solve()

    def u_exact(x, t):
        return np.cos(np.sqrt(2) * t) * np.cos(x)

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=5, component='real')
        assert err is not None and err < 5


# ---------------
# 19) KdV soliton
# ---------------

def test_kdv_soliton():
    t, x = symbols('t x')
    u_func = Function('u')
    u = u_func(t, x)
    eq = Eq(diff(u, t) + 6 * u * diff(u, x) - diff(u, x, x, x), 0)

    c = 0.5
    x0 = 0.0
    initial_condition = lambda x: c / 2 * (1 / np.cosh(np.sqrt(c)/2 * (x - x0)))**2
    Lt = 0.5

    solver = PDESolver(eq, time_scheme='ETD-RK4', dealiasing_ratio=2/3)
    solver.setup(Lx=40, Nx=1024, Lt=Lt, Nt=200, initial_condition=initial_condition, plot=False)
    solver.solve()

    def u_exact(x, t):
        return c / 2 * (1 / np.cosh(np.sqrt(c)/2 * (x - c*t - x0)))**2

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=0.5, component='real')
        assert err is not None and err < 0.5


# ------------------
# 20) Biharmonic
# ------------------

def test_biharmonic():
    t, x = symbols('t x')
    u_func = Function('u')
    u = u_func(t, x)
    biharmonic_eq = Eq(diff(u, t), -diff(u, x, x, x, x))
    Lt = 0.5

    solver = PDESolver(biharmonic_eq)
    solver.setup(Lx=2*np.pi, Nx=128, Lt=Lt, Nt=50, initial_condition=lambda x: np.sin(x), plot=False)
    solver.solve()

    def u_exact(x, t):
        return np.sin(x) * np.exp(-t)

    # Validation tests
    n_test = 5
    for i in range(n_test + 1):
        t_eval = i * Lt / n_test
        err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=5e-2, component='real')
        assert err is not None and err < 5e-2

# End of tests
