# Copyright 2025 Philippe Billet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from imports import *
from psiop import * 

class PDESolver:
    """
    A partial differential equation (PDE) solver based on **spectral methods** using Fourier transforms.

    This solver supports symbolic specification of PDEs via SymPy and numerical solution using high-order spectral techniques. 
    It is designed for both **linear and nonlinear time-dependent PDEs**, as well as **stationary pseudo-differential problems**.
    
    Key Features:
    -------------
    - Symbolic PDE parsing using SymPy expressions
    - 1D and 2D spatial domains with periodic boundary conditions
    - Fourier-based spectral discretization with dealiasing
    - Temporal integration schemes:
        - Default exponential time stepping
        - ETD-RK4 (Exponential Time Differencing Runge-Kutta of 4th order)
    - Nonlinear terms handled through pseudo-spectral evaluation
    - Built-in tools for:
        - Visualization of solutions and error surfaces
        - Symbol analysis of linear and pseudo-differential operators
        - Microlocal analysis (e.g., Hamiltonian flows)
        - CFL condition checking and numerical stability diagnostics

    Supported Operators:
    --------------------
    - Linear differential and pseudo-differential operators
    - Nonlinear terms up to second order in derivatives
    - Symbolic operator composition and adjoints
    - Asymptotic inversion of elliptic operators for stationary problems

    Example Usage:
    --------------
    >>> from PDESolver import *
    >>> u = Function('u')
    >>> t, x = symbols('t x')
    >>> eq = Eq(diff(u(t, x), t), diff(u(t, x), x, 2) + u(t, x)**2)
    >>> def initial(x): return np.sin(x)
    >>> solver = PDESolver(eq)
    >>> solver.setup(Lx=2*np.pi, Nx=128, Lt=1.0, Nt=1000, initial_condition=initial)
    >>> solver.solve()
    >>> ani = solver.animate()
    >>> HTML(ani.to_jshtml())  # Display animation in Jupyter notebook
    """
    def __init__(self, equation, time_scheme='default', dealiasing_ratio=2/3):
        """
        Initialize the PDE solver with a given equation.

        This method analyzes the input partial differential equation (PDE), 
        identifies the unknown function and its dependencies, determines whether 
        the problem is stationary or time-dependent, and prepares symbolic and 
        numerical structures for solving in spectral space.

        Supported features:
        
        - 1D and 2D problems
        - Time-dependent and stationary equations
        - Linear and nonlinear terms
        - Pseudo-differential operators via `psiOp`
        - Source terms and boundary conditions

        The equation is parsed to extract linear, nonlinear, source, and 
        pseudo-differential components. Symbolic manipulation is used to derive 
        the Fourier representation of linear operators when applicable.

        Parameters
        ----------
        equation : sympy.Eq 
            The PDE expressed as a SymPy equation.
        time_scheme : str
            Temporal integration scheme: 
                - 'default' for exponential 
                - time-stepping or 'ETD-RK4' for fourth-order exponential 
                - time differencing Runge–Kutta.
        dealiasing_ratio : float
            Fraction of high-frequency modes to zero out 
            during dealiasing (e.g., 2/3 for standard truncation).

        Attributes initialized:
        
        - self.u: the unknown function (e.g., u(t, x))
        - self.dim: spatial dimension (1 or 2)
        - self.spatial_vars: list of spatial variables (e.g., [x] or [x, y])
        - self.is_stationary: boolean indicating if the problem is stationary
        - self.linear_terms: dictionary mapping derivative orders to coefficients
        - self.nonlinear_terms: list of nonlinear expressions
        - self.source_terms: list of source functions
        - self.pseudo_terms: list of pseudo-differential operator expressions
        - self.has_psi: boolean indicating presence of pseudo-differential operators
        - self.fft / self.ifft: appropriate FFT routines based on spatial dimension
        - self.kx, self.ky: symbolic wavenumber variables for Fourier space

        Raises:
            ValueError: If the equation does not contain exactly one unknown function,
                        if unsupported dimensions are detected, or invalid dependencies.
        """
        self.time_scheme = time_scheme # 'default'  or 'ETD-RK4'
        self.dealiasing_ratio = dealiasing_ratio
        
        print("\n*********************************")
        print("* Partial differential equation *")
        print("*********************************\n")
        pprint(equation, num_columns=NUM_COLS)
        
        # Extract symbols and function from the equation
        functions = equation.atoms(Function)
        
        # Ignore the wrappers psiOp and Op
        excluded_wrappers = {'psiOp', 'Op'}
        
        # Extract the candidate fonctions (excluding wrappers)
        candidate_functions = [
            f for f in functions 
            if f.func.__name__ not in excluded_wrappers
        ]
        
        # Keep only user functions (u(x), u(x, t), etc.)
        candidate_functions = [
            f for f in functions
            if isinstance(f, AppliedUndef)
        ]
        
        # Stationary detection: no dependence on t
        self.is_stationary = all(
            not any(str(arg) == 't' for arg in f.args)
            for f in candidate_functions
        )
        
        if len(candidate_functions) != 1:
            print("candidate_functions :", candidate_functions)
            raise ValueError("The equation must contain exactly one unknown function")
        
        self.u = candidate_functions[0]

        self.u_eq = self.u

        args = self.u.args
        
        if self.is_stationary:
            if len(args) not in (1, 2):
                raise ValueError("Stationary problems must depend on 1 or 2 spatial variables")
            self.spatial_vars = args
        else:
            if len(args) < 2 or len(args) > 3:
                raise ValueError("The function must depend on t and at least one spatial variable (x [, y])")
            self.t = args[0]
            self.spatial_vars = args[1:]

        self.dim = len(self.spatial_vars)
        if self.dim == 1:
            self.x = self.spatial_vars[0]
            self.y = None
        elif self.dim == 2:
            self.x, self.y = self.spatial_vars
        else:
            raise ValueError("Only 1D and 2D problems are supported.")

        if self.dim == 1:
            self.fft = partial(fft, workers=FFT_WORKERS)
            self.ifft = partial(ifft, workers=FFT_WORKERS)
        else:
            self.fft = partial(fft2, workers=FFT_WORKERS)
            self.ifft = partial(ifft2, workers=FFT_WORKERS)
            
        # Parse the equation
        self.linear_terms = {}
        self.nonlinear_terms = []
        self.symbol_terms = []
        self.source_terms = []
        self.pseudo_terms = []
        self.temporal_order = 0  # Order of the temporal derivative
        self.linear_terms, self.nonlinear_terms, self.symbol_terms, self.source_terms, self.pseudo_terms = self.parse_equation(equation)
        # flag : pseudo‑differential operator present ?
        self.has_psi = bool(self.pseudo_terms)
        if self.has_psi:
            print('⚠️  Pseudo‑differential operator detected: all other linear terms have been rejected.')
            self.is_spatial = False
            for coeff, expr in self.pseudo_terms:
                if expr.has(self.x) or (self.dim == 2 and expr.has(self.y)):
                    self.is_spatial = True
                    break
    
        if self.dim == 1:
            self.kx = symbols('kx')
        elif self.dim == 2:
            self.kx, self.ky = symbols('kx ky')
    
        # Compute linear operator
        if not self.is_stationary:
            self.compute_linear_operator()
        else:
            self.psi_ops = []
            for coeff, sym_expr in self.pseudo_terms:
                psi = PseudoDifferentialOperator(sym_expr, self.spatial_vars, self.u, mode='symbol')
                self.psi_ops.append((coeff, psi))

    def parse_equation(self, equation):
        """
        Parse the PDE to separate linear and nonlinear terms, symbolic operators (Op), 
        source terms, and pseudo-differential operators (psiOp).
    
        This method rewrites the input equation in standard form (lhs - rhs = 0),
        expands it, and classifies each term into one of the following categories:
        
        - Linear terms involving derivatives or the unknown function u
        - Nonlinear terms (products with u, powers of u, etc.)
        - Symbolic pseudo-differential operators (Op)
        - Source terms (independent of u)
        - Pseudo-differential operators (psiOp)
    
        Parameters
            equation (sympy.Eq): The partial differential equation to be analyzed. 
                                 Can be provided as an Eq object or a sympy expression.
    
        Returns:
            tuple: A 5-tuple containing:
            
                - linear_terms (dict): Mapping from derivative/function to coefficient.
                - nonlinear_terms (list): List of terms classified as nonlinear.
                - symbol_terms (list): List of (coefficient, symbolic operator) pairs.
                - source_terms (list): List of terms independent of the unknown function.
                - pseudo_terms (list): List of (coefficient, pseudo-differential symbol) pairs.
    
        Notes:
            - If `psiOp` is present in the equation, expansion is skipped for safety.
            - When `psiOp` is used, only nonlinear terms, source terms, and possibly 
              a time derivative are allowed; other linear terms and symbolic operators 
              (Op) are forbidden.
            - Classification logic includes:
                - Detection of nonlinear structures like products or powers of u
                - Mixed terms involving both u and its derivatives
                - External symbolic operators (Op) and pseudo-differential operators (psiOp)
        """
        def is_nonlinear_term(term, u_func):
            # If the term contains functions (Abs, sin, exp, ...) applied to u
            if term.has(u_func):
                for sub in preorder_traversal(term):
                    if isinstance(sub, Function) and sub.has(u_func) and sub.func != u_func.func:
                        return True
            # If the term contains a nonlinear power of u
            if term.has(Pow):
                for pow_term in term.atoms(Pow):
                    if pow_term.base == u_func and pow_term.exp != 1:
                        return True
            # If the term is a product containing u and its derivative
            if term.func == Mul:
                factors = term.args
                has_u = any((f.has(u_func) and not isinstance(f, Derivative) for f in factors))
                has_derivative = any((isinstance(f, Derivative) and f.expr.func == u_func.func for f in factors))
                if has_u and has_derivative:
                    return True
            return False
    
        print("\n********************")
        print("* Equation parsing *")
        print("********************\n")
    
        if isinstance(equation, Eq):
            lhs = equation.lhs - equation.rhs
        else:
            lhs = equation
    
        print(f"\nEquation rewritten in standard form: {lhs}")
        if lhs.has(psiOp):
            print("⚠️ psiOp detected: skipping expansion for safety")
            lhs_expanded = lhs
        else:
            lhs_expanded = expand(lhs)
    
        print(f"\nExpanded equation: {lhs_expanded}")
    
        linear_terms = {}
        nonlinear_terms = []
        symbol_terms = []
        source_terms = []
        pseudo_terms = []
    
        for term in lhs_expanded.as_ordered_terms():
            print(f"Analyzing term: {term}")
    
            if isinstance(term, psiOp):
                expr = term.args[0]
                pseudo_terms.append((1, expr))
                print("  --> Classified as pseudo linear term (psiOp)")
                continue
    
            # Otherwise, look for psiOp inside (general case)
            if term.has(psiOp):
                psiops = term.atoms(psiOp)
                for psi in psiops:
                    try:
                        coeff = simplify(term / psi)
                        expr = psi.args[0]
                        pseudo_terms.append((coeff, expr))
                        print("  --> Classified as pseudo linear term (psiOp)")
                    except Exception as e:
                        print(f"  ⚠️ Failed to extract psiOp coefficient in term: {term}")
                        print(f"     Reason: {e}")
                        nonlinear_terms.append(term)
                        print("  --> Fallback: classified as nonlinear")
                continue
    
            if term.has(Op):
                ops = term.atoms(Op)
                for op in ops:
                    coeff = term / op
                    expr = op.args[0]
                    symbol_terms.append((coeff, expr))
                    print("  --> Classified as symbolic linear term (Op)")
                continue
    
            if is_nonlinear_term(term, self.u):
                nonlinear_terms.append(term)
                print("  --> Classified as nonlinear")
                continue
    
            derivs = term.atoms(Derivative)
            if derivs:
                deriv = derivs.pop()
                coeff = term / deriv
                linear_terms[deriv] = linear_terms.get(deriv, 0) + coeff
                print(f"  Derivative found: {deriv}")
                print("  --> Classified as linear")
            elif self.u in term.atoms(Function):
                coeff = term.as_coefficients_dict().get(self.u, 1)
                linear_terms[self.u] = linear_terms.get(self.u, 0) + coeff
                print("  --> Classified as linear")
            else:
                source_terms.append(term)
                print("  --> Classified as source term")
    
        print(f"Final linear terms: {linear_terms}")
        print(f"Final nonlinear terms: {nonlinear_terms}")
        print(f"Symbol terms: {symbol_terms}")
        print(f"Pseudo terms: {pseudo_terms}")
        print(f"Source terms: {source_terms}")
    
        if pseudo_terms:
            # Check if a time derivative is present among the linear terms
            has_time_derivative = any(
                isinstance(term, Derivative) and self.t in [v for v, _  in term.variable_count]
                for term in linear_terms
            )
            # Extract non-temporal linear terms
            invalid_linear_terms = {
                term: coeff for term, coeff in linear_terms.items()
                if not (
                    isinstance(term, Derivative)
                    and self.t in [v for v, _  in term.variable_count]
                )
                and term != self.u  # exclusion of the simple u term (without derivative)
            }
    
            if invalid_linear_terms or symbol_terms:
                raise ValueError(
                    "When psiOp is used, only nonlinear terms, source terms, "
                    "and possibly a time derivative are allowed. "
                    "Other linear terms and Ops are forbidden."
                )
    
        return linear_terms, nonlinear_terms, symbol_terms, source_terms, pseudo_terms


    def compute_linear_operator(self):
        """
        Compute the symbolic Fourier representation L(k) of the linear operator 
        derived from the linear part of the PDE.
    
        This method constructs a dispersion relation by applying each symbolic derivative
        to a plane wave exp(i(k·x - ωt)) and extracting the resulting expression.
        It handles arbitrary derivative combinations and includes symbolic and
        pseudo-differential terms.
    
        Steps:
        -------
        1. Construct a plane wave φ(x, t) = exp(i(k·x - ωt)).
        2. Apply each term from self.linear_terms to φ.
        3. Normalize by φ and simplify to obtain L(k).
        4. Include symbolic terms (e.g., psiOp) if present.
        5. Detect the temporal order from the dispersion relation.
        6. Build the numerical function L(k) via lambdify.
    
        Sets:
        -----
        - self.L_symbolic : sympy.Expr
            Symbolic form of L(k).
        - self.L : callable
            Numerical function of L(kx[, ky]).
        - self.omega : callable or None
            Frequency root ω(k), if available.
        - self.temporal_order : int
            Order of time derivatives detected.
        - self.psi_ops : list of (coeff, PseudoDifferentialOperator)
            Pseudo-differential terms present in the equation.
    
        Raises:
        -------
        ValueError if the dimension is unsupported or the dispersion relation fails.
        """
        print("\n*******************************")
        print("* Linear operator computation *")
        print("*******************************\n")
    
        # --- Step 1: symbolic variables ---
        omega = symbols("omega")
        if self.dim == 1:
            kvars = [symbols("kx")]
            space_vars = [self.x]
        elif self.dim == 2:
            kvars = symbols("kx ky")
            space_vars = [self.x, self.y]
        else:
            raise ValueError("Only 1D and 2D are supported.")
    
        kdict = dict(zip(space_vars, kvars))
        self.k_symbols = kvars
    
        # Plane wave expression
        phase = sum(k * x for k, x in zip(kvars, space_vars)) - omega * self.t
        plane_wave = exp(I * phase)
    
        # --- Step 2: build lhs expression from linear terms ---
        lhs = 0
        for deriv, coeff in self.linear_terms.items():
            if isinstance(deriv, Derivative):
                total_factor = 1
                for var, n in deriv.variable_count:
                    if var == self.t:
                        total_factor *= (-I * omega)**n
                    elif var in kdict:
                        total_factor *= (I * kdict[var])**n
                    else:
                        raise ValueError(f"Unknown variable {var} in derivative")
                lhs += coeff * total_factor * plane_wave
            elif deriv == self.u:
                lhs += coeff * plane_wave
            else:
                raise ValueError(f"Unsupported linear term: {deriv}")
    
        # --- Step 3: dispersion relation ---
        equation = simplify(lhs / plane_wave)
        print("\nCharacteristic equation before symbol treatment:")
        pprint(equation, num_columns=NUM_COLS)

        print("\n--- Symbolic symbol analysis ---")
        symb_omega = 0
        symb_k = 0
        
        for coeff, symbol in self.symbol_terms:
            if symbol.has(omega):
                # Ajouter directement les termes dépendant de omega
                symb_omega += coeff * symbol
            elif any(symbol.has(k) for k in self.k_symbols):
                 symb_k += coeff * symbol.subs(dict(zip(symbol.free_symbols, self.k_symbols)))

        print(f"symb_omega: {symb_omega}")
        print(f"symb_k: {symb_k}")
        
        equation = equation + symb_omega + symb_k         

        print("\nRaw characteristic equation:")
        pprint(equation, num_columns=NUM_COLS)

        # Temporal derivative order detection
        try:
            poly_eq = Eq(equation, 0)
            poly = poly_eq.lhs.as_poly(omega)
            self.temporal_order = poly.degree() if poly else 0
        except Exception as e:
            warnings.warn(f"Could not determine temporal order: {e}", RuntimeWarning)
            self.temporal_order = 0
        print(f"Temporal order from dispersion relation: {self.temporal_order}")
        print('self.pseudo_terms = ', self.pseudo_terms)
        if self.pseudo_terms:
            coeff_time = 1
            for term, coeff in self.linear_terms.items():
                if isinstance(term, Derivative) and any(var == self.t for var, _  in term.variable_count):
                    coeff_time = coeff
                    print(f"✅ Time derivative coefficient detected: {coeff_time}")
            self.psi_ops = []
            for coeff, sym_expr in self.pseudo_terms:
                # expr est le Sympy expr. différentiel, var_x la liste [x] ou [x,y]
                psi = PseudoDifferentialOperator(sym_expr / coeff_time, self.spatial_vars, self.u, mode='symbol')
                
                self.psi_ops.append((coeff, psi))
        else:
            dispersion = solve(Eq(equation, 0), omega)
            if not dispersion:
                raise ValueError("No solution found for omega")
            print("\n--- Solutions found ---")
            pprint(dispersion, num_columns=NUM_COLS)
        
            if self.temporal_order == 2:
                omega_expr = simplify(sqrt(dispersion[0]**2))
                self.omega_symbolic = omega_expr
                self.omega = lambdify(self.k_symbols, omega_expr, "numpy")
                self.L_symbolic = -omega_expr**2
            else:
                self.L_symbolic = -I * dispersion[0]
        
        
            self.L = lambdify(self.k_symbols, self.L_symbolic, "numpy")
  
            print("\n--- Final linear operator ---")
            pprint(self.L_symbolic, num_columns=NUM_COLS)   

    def linear_rhs(self, u, is_v=False):
        """
        Apply the linear operator (in Fourier space) to the field u or v.

        Parameters
        ----------
        u : np.ndarray
            Input solution array.
        is_v : bool
            Whether to apply the operator to v instead of u.

        Returns
        -------
        np.ndarray
            Result of applying the linear operator.
        """
        if self.dim == 1:
            self.symbol_u = np.array(self.L(self.KX), dtype=np.complex128)
            self.symbol_v = self.symbol_u  # même opérateur pour u et v
        elif self.dim == 2:
            self.symbol_u = np.array(self.L(self.KX, self.KY), dtype=np.complex128)
            self.symbol_v = self.symbol_u
        u_hat = self.fft(u)
        u_hat *= self.symbol_v if is_v else self.symbol_u
        u_hat *= self.dealiasing_mask
        return self.ifft(u_hat)

    def setup(self, Lx, Ly=None, Nx=None, Ny=None, Lt=1.0, Nt=100, boundary_condition='periodic',
              initial_condition=None, initial_velocity=None, n_frames=100, plot=True):
        """
        Configure the spatial/temporal grid and initialize the solution field.
    
        This method sets up the computational domain, initializes spatial and temporal grids,
        applies boundary conditions, and prepares symbolic and numerical operators.
        It also performs essential analyses such as:
        
            - CFL condition verification (for stability)
            - Symbol analysis (e.g., dispersion relation, regularity)
            - Wave propagation analysis for second-order equations
    
        If pseudo-differential operators (ψOp) are present, symbolic analysis is skipped
        in favor of interactive exploration via `interactive_symbol_analysis`.
    
        Parameters
        ----------
        Lx : float
            Size of the spatial domain along x-axis.
        Ly : float, optional
            Size of the spatial domain along y-axis (for 2D problems).
        Nx : int
            Number of spatial points along x-axis.
        Ny : int, optional
            Number of spatial points along y-axis (for 2D problems).
        Lt : float, default=1.0
            Total simulation time.
        Nt : int, default=100
            Number of time steps.
        initial_condition : callable
            Function returning the initial state u(x, 0) or u(x, y, 0).
        initial_velocity : callable, optional
            Function returning the initial time derivative ∂ₜu(x, 0) or ∂ₜu(x, y, 0),
            required for second-order equations.
        n_frames : int, default=100
            Number of time frames to store during simulation for visualization or output.
    
        Raises
        ------
        ValueError
            If mandatory parameters are missing (e.g., Nx not given in 1D, Ly/Ny not given in 2D).
    
        Notes
        -----
        - The spatial discretization assumes periodic boundary conditions by default.
        - Fourier transforms are computed using real-to-complex FFTs (`scipy.fft.fft`, `fft2`).
        - Frequency arrays (`KX`, `KY`) are defined following standard spectral conventions.
        - Dealiasing is applied using a sharp cutoff filter at a fraction of the maximum frequency.
        - For second-order equations, initial acceleration is derived from the governing operator.
        - Symbolic analysis includes plotting of the symbol's real/imaginary/absolute values
          and dispersion relation.
    
        See Also
        --------
        setup_1D : Sets up internal variables for one-dimensional problems.
        setup_2D : Sets up internal variables for two-dimensional problems.
        initialize_conditions : Applies initial data and enforces compatibility.
        check_cfl_condition : Verifies time step against stability constraints.
        plot_symbol : Visualizes the linear operator’s symbol in frequency space.
        analyze_wave_propagation : Analyzes group velocity.
        interactive_symbol_analysis : Interactive tools for ψOp-based equations.
        """
        
        # Temporal parameters
        self.Lt, self.Nt = Lt, Nt
        self.dt = Lt / Nt
        self.n_frames = n_frames
        self.frames = []
        self.initial_condition = initial_condition
        self.boundary_condition = boundary_condition
        self.plot = plot

        if self.boundary_condition == 'dirichlet' and not self.has_psi:
            raise ValueError(
                "Dirichlet boundary conditions require the equation to be defined via a pseudo-differential operator (psiOp). "
                "Please provide an equation involving psiOp for non-periodic boundary treatment."
            )
    
        # Dimension checks
        if self.dim == 1:
            if Nx is None:
                raise ValueError("Nx must be specified in 1D.")
            self.setup_1D(Lx, Nx)
        else:
            if None in (Ly, Ny):
                raise ValueError("In 2D, Ly and Ny must be provided.")
            self.setup_2D(Lx, Ly, Nx, Ny)
    
        # Initialization of solution and velocities
        if not self.is_stationary:
            self.initialize_conditions(initial_condition, initial_velocity)
            
        # Symbol analysis if present
        if self.has_psi:
            print("⚠️ For psiOp, use interactive_symbol_analysis.")
        else:
            if self.L_symbolic == 0:
                print("⚠️ Linear operator is null.")
            else:
                self.check_cfl_condition()
                self.check_symbol_conditions()
                if plot:
                	self.plot_symbol()
                	if self.temporal_order == 2:
                		self.analyze_wave_propagation()

    def setup_1D(self, Lx, Nx):
        """
        Configure internal variables for one-dimensional (1D) problems.
    
        This private method initializes spatial and frequency grids, applies dealiasing,
        and prepares either pseudo-differential symbols or linear operators for use in time evolution.
        
        It assumes periodic boundary conditions and uses real-to-complex FFT conventions.
        The spatial domain is centered at zero: [-Lx/2, Lx/2].
    
        Parameters
        ----------
        Lx : float
            Physical size of the spatial domain along the x-axis.
        Nx : int
            Number of grid points in the x-direction.
    
        Attributes Set
        --------------
        - self.Lx : float
            Size of the spatial domain.
        - self.Nx : int
            Number of spatial points.
        - self.x_grid : np.ndarray
            1D array of spatial coordinates.
        - self.X : np.ndarray
            Alias to `self.x_grid`, used in physical space computations.
        - self.kx : np.ndarray
            Array of wavenumbers corresponding to the Fourier transform.
        - self.KX : np.ndarray
            Alias to `self.kx`, used in frequency space computations.
        - self.dealiasing_mask : np.ndarray
            Boolean mask used to suppress aliased frequencies during nonlinear calculations.
        - self.exp_L : np.ndarray
            Exponential of the linear operator scaled by time step: exp(L(k) · dt).
        - self.omega_val : np.ndarray
            Frequency values ω(k) = Re[√(L(k))] used in second-order time stepping.
        - self.cos_omega_dt, self.sin_omega_dt : np.ndarray
            Cosine and sine of ω(k)·dt for dispersive propagation.
        - self.inv_omega : np.ndarray
            Inverse of ω(k), used to avoid division-by-zero in time stepping.
    
        Notes
        -----
        - Frequencies are computed using `scipy.fft.fftfreq` and then shifted to center zero frequency.
        - Dealiasing is applied using a sharp cutoff filter based on `self.dealiasing_ratio`.
        - If pseudo-differential operators (ψOp) are present, symbolic tables are precomputed via `prepare_symbol_tables`.
        - For second-order equations, the dispersion relation ω(k) is extracted from the linear operator L(k).
    
        See Also
        --------
        setup_2D : Equivalent setup for two-dimensional problems.
        prepare_symbol_tables : Precomputes symbolic arrays for ψOp evaluation.
        setup_omega_terms : Sets up terms involving ω(k) for second-order evolution.
        """
        self.Lx, self.Nx = Lx, Nx
        self.x_grid = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
        self.X = self.x_grid
        self.kx = 2 * np.pi * fftfreq(Nx, d=Lx / Nx)
        self.KX = self.kx
    
        # Dealiasing mask
        k_max = self.dealiasing_ratio * np.max(np.abs(self.kx))
        self.dealiasing_mask = (np.abs(self.KX) <= k_max)
    
        # Preparation of symbol or linear operator
        if self.has_psi:
            self.prepare_symbol_tables()
        else:
            L_vals = np.array(self.L(self.KX), dtype=np.complex128)
            self.exp_L = np.exp(L_vals * self.dt)
            if self.temporal_order == 2:
                omega_val = self.omega(self.KX)
                self.setup_omega_terms(omega_val)
    
    def setup_2D(self, Lx, Ly, Nx, Ny):
        """
        Configure internal variables for two-dimensional (2D) problems.
    
        This private method initializes spatial and frequency grids, applies dealiasing,
        and prepares either pseudo-differential symbols or linear operators for use in time evolution.
        
        It assumes periodic boundary conditions and uses real-to-complex FFT conventions.
        The spatial domain is centered at zero: [-Lx/2, Lx/2] × [-Ly/2, Ly/2].
    
        Parameters
        ----------
        Lx : float
            Physical size of the spatial domain along the x-axis.
        Ly : float
            Physical size of the spatial domain along the y-axis.
        Nx : int
            Number of grid points along the x-direction.
        Ny : int
            Number of grid points along the y-direction.
    
        Attributes Set
        --------------
        - self.Lx, self.Ly : float
            Size of the spatial domain in each direction.
        - self.Nx, self.Ny : int
            Number of spatial points in each direction.
        - self.x_grid, self.y_grid : np.ndarray
            1D arrays of spatial coordinates in x and y directions.
        - self.X, self.Y : np.ndarray
            2D meshgrids of spatial coordinates for physical space computations.
        - self.kx, self.ky : np.ndarray
            Arrays of wavenumbers corresponding to Fourier transforms in x and y directions.
        - self.KX, self.KY : np.ndarray
            Meshgrids of wavenumbers used in frequency space computations.
        - self.dealiasing_mask : np.ndarray
            Boolean mask used to suppress aliased frequencies during nonlinear calculations.
        - self.exp_L : np.ndarray
            Exponential of the linear operator scaled by time step: exp(L(kx, ky) · dt).
        - self.omega_val : np.ndarray
            Frequency values ω(kx, ky) = Re[√(L(kx, ky))] used in second-order time stepping.
        - self.cos_omega_dt, self.sin_omega_dt : np.ndarray
            Cosine and sine of ω(kx, ky)·dt for dispersive propagation.
        - self.inv_omega : np.ndarray
            Inverse of ω(kx, ky), used to avoid division-by-zero in time stepping.
    
        Notes
        -----
        - Frequencies are computed using `scipy.fft.fftfreq` and then shifted to center zero frequency.
        - Dealiasing is applied using a sharp cutoff filter based on `self.dealiasing_ratio`.
        - If pseudo-differential operators (ψOp) are present, symbolic tables are precomputed via `prepare_symbol_tables`.
        - For second-order equations, the dispersion relation ω(kx, ky) is extracted from the linear operator L(kx, ky).
    
        See Also
        --------
        setup_1D : Equivalent setup for one-dimensional problems.
        prepare_symbol_tables : Precomputes symbolic arrays for ψOp evaluation.
        setup_omega_terms : Sets up terms involving ω(kx, ky) for second-order evolution.
        """
        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.x_grid = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
        self.y_grid = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')
        self.kx = 2 * np.pi * fftfreq(Nx, d=Lx / Nx)
        self.ky = 2 * np.pi * fftfreq(Ny, d=Ly / Ny)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
    
        # Dealiasing mask
        kx_max = self.dealiasing_ratio * np.max(np.abs(self.kx))
        ky_max = self.dealiasing_ratio * np.max(np.abs(self.ky))
        self.dealiasing_mask = (np.abs(self.KX) <= kx_max) & (np.abs(self.KY) <= ky_max)
    
        # Preparation of symbol or linear operator
        if self.has_psi:
            self.prepare_symbol_tables()
        else:
            L_vals = self.L(self.KX, self.KY)
            self.exp_L = np.exp(L_vals * self.dt)
            if self.temporal_order == 2:
                omega_val = self.omega(self.KX, self.KY)
                self.setup_omega_terms(omega_val)
    
    def setup_omega_terms(self, omega_val):
        """
        Initialize terms derived from the angular frequency ω for time evolution.
    
        This private method precomputes and stores key trigonometric and inverse quantities
        based on the dispersion relation ω(k), used in second-order time integration schemes.
        
        These values are essential for solving wave-like equations with dispersive behavior:
            cos(ω·dt), sin(ω·dt), 1/ω
        
        The inverse frequency is computed safely to avoid division by zero.
    
        Parameters
        ----------
        omega_val : np.ndarray
            Array of angular frequency values ω(k) evaluated at discrete wavenumbers.
            Can be one-dimensional (1D) or two-dimensional (2D) depending on spatial dimension.
    
        Attributes Set
        --------------
        - self.omega_val : np.ndarray
            Copy of the input angular frequency array.
        - self.cos_omega_dt : np.ndarray
            Cosine of ω(k) multiplied by time step: cos(ω(k) · dt).
        - self.sin_omega_dt : np.ndarray
            Sine of ω(k) multiplied by time step: sin(ω(k) · dt).
        - self.inv_omega : np.ndarray
            Inverse of ω(k), with zeros where ω(k) == 0 to avoid division by zero.
    
        Notes
        -----
        - This method is typically called during setup when solving second-order PDEs
          involving dispersive waves (e.g., Klein-Gordon, Schrödinger, or water wave equations).
        - The safe computation of 1/ω ensures numerical stability even when low frequencies are present.
        - These precomputed arrays are used in spectral propagators for accurate time stepping.
    
        See Also
        --------
        setup_1D : Sets up internal variables for one-dimensional problems.
        setup_2D : Sets up internal variables for two-dimensional problems.
        solve : Time integration using the computed frequency terms.
        """
        self.omega_val = omega_val
        self.cos_omega_dt = np.cos(omega_val * self.dt)
        self.sin_omega_dt = np.sin(omega_val * self.dt)
        self.inv_omega = np.zeros_like(omega_val)
        nonzero = omega_val != 0
        self.inv_omega[nonzero] = 1.0 / omega_val[nonzero]

    def evaluate_source_at_t0(self):
        """
        Evaluate source terms at initial time t = 0 over the spatial grid.
    
        This private method computes the total contribution of all source terms at the initial time,
        evaluated across the entire spatial domain. It supports both one-dimensional (1D) and
        two-dimensional (2D) configurations.
    
        Returns
        -------
        np.ndarray
            A numpy array representing the evaluated source term at t=0:
            - In 1D: Shape (Nx,), evaluated at each x in `self.x_grid`.
            - In 2D: Shape (Nx, Ny), evaluated at each (x, y) pair in the grid.
    
        Notes
        -----
        - The symbolic expressions in `self.source_terms` are substituted with numerical values at t=0.
        - In 1D, each term is evaluated at (t=0, x=x_val).
        - In 2D, each term is evaluated at (t=0, x=x_val, y=y_val).
        - Evaluated using SymPy's `evalf()` to ensure numeric conversion.
        - This method assumes that the source terms have already been lambdified or are compatible with symbolic substitution.
    
        See Also
        --------
        setup : Initializes the spatial grid and source terms.
        solve : Uses this evaluation during the first time step.
        """
        if self.dim == 1:
            # Evaluation on the 1D spatial grid
            return np.array([
                sum(term.subs(self.t, 0).subs(self.x, x_val).evalf()
                    for term in self.source_terms)
                for x_val in self.x_grid
            ], dtype=np.float64)
        else:
            # Evaluation on the 2D spatial grid
            return np.array([
                [sum(term.subs({self.t: 0, self.x: x_val, self.y: y_val}).evalf()
                      for term in self.source_terms)
                 for y_val in self.y_grid]
                for x_val in self.x_grid
            ], dtype=np.float64)
    
    def initialize_conditions(self, initial_condition, initial_velocity):
        """
        Initialize the solution and velocity fields at t = 0.
    
        This private method sets up the initial state of the solution `u_prev` and, if applicable,
        the time derivative (velocity) `v_prev` for second-order evolution equations.
        
        For second-order equations, it also computes the backward-in-time value `u_prev2`
        needed by the Leap-Frog method. The acceleration at t = 0 is computed from:
            ∂ₜ²u = L(u) + N(u) + f(x, t=0)
        where L is the linear operator, N is the nonlinear term, and f is the source term.
    
        Parameters
        ----------
        initial_condition : callable
            Function returning the initial condition u(x, 0) or u(x, y, 0).
        initial_velocity : callable or None
            Function returning the initial velocity ∂ₜu(x, 0) or ∂ₜu(x, y, 0). Required for
            second-order equations; ignored otherwise.
    
        Raises
        ------
        ValueError
            If `initial_velocity` is not provided for second-order equations.
    
        Notes
        -----
        - Applies periodic boundary conditions after setting initial data.
        - Stores a copy of the initial state in `self.frames` for visualization/output.
        - In second-order systems, initializes `self.u_prev2` using a Taylor expansion:
          u_prev2 = u_prev - dt * v_prev + 0.5 * dt² * (∂ₜ²u)
    
        See Also
        --------
        apply_boundary : Enforces periodic boundary conditions on the solution field.
        psiOp_apply : Computes pseudo-differential operator action for acceleration.
        linear_rhs : Evaluates linear part of the equation in Fourier space.
        apply_nonlinear : Handles nonlinear terms with spectral differentiation.
        evaluate_source_at_t0 : Evaluates source terms at the initial time.
        """
        # Initial condition
        if self.dim == 1:
            self.u_prev = initial_condition(self.X)
        else:
            self.u_prev = initial_condition(self.X, self.Y)
        self.apply_boundary(self.u_prev)
    
        # Initial velocity (second order)
        if self.temporal_order == 2:
            if initial_velocity is None:
                raise ValueError("Initial velocity is required for second-order equations.")
            if self.dim == 1:
                self.v_prev = initial_velocity(self.X)
            else:
                self.v_prev = initial_velocity(self.X, self.Y)
            self.u0 = np.copy(self.u_prev)
            self.v0 = np.copy(self.v_prev)
    
            # Calculation of u_prev2 (initial acceleration)
            if not hasattr(self, 'u_prev2'):
                if self.has_psi:
                    acc0 = self.apply_psiOp(self.u_prev)
                else:
                    acc0 = self.linear_rhs(self.u_prev, is_v=False)
                rhs_nl = self.apply_nonlinear(self.u_prev, is_v=False)
                acc0 += rhs_nl
                if hasattr(self, 'source_terms') and self.source_terms:
                    acc0 += self.evaluate_source_at_t0()
                self.u_prev2 = self.u_prev - self.dt * self.v_prev + 0.5 * self.dt**2 * acc0
    
        self.frames = [self.u_prev.copy()]
           
    def apply_boundary(self, u):
        """
        Apply boundary conditions to the solution array based on the specified type.
    
        This method supports two types of boundary conditions:
        
        - 'periodic': Enforces periodicity by copying opposite boundary values.
        - 'dirichlet': Sets all boundary values to zero (homogeneous Dirichlet condition).
    
        Parameters
        ----------
        u : np.ndarray
            The solution array representing the field values on a spatial grid.
            In 1D, shape must be (Nx,). In 2D, shape must be (Nx, Ny).
    
        Raises
        ------
        ValueError
            If `self.boundary_condition` is not one of {'periodic', 'dirichlet'}.
    
        Notes
        -----
        - For 'periodic':
            * In 1D: u[0] = u[-2], u[-1] = u[1]
            * In 2D: First and last rows/columns are set equal to their neighbors.
        - For 'dirichlet':
            * All boundary points are explicitly set to zero.
        """
    
        if self.boundary_condition == 'periodic':
            if self.dim == 1:
                u[0] = u[-2]
                u[-1] = u[1]
            elif self.dim == 2:
                u[0, :] = u[-2, :]
                u[-1, :] = u[1, :]
                u[:, 0] = u[:, -2]
                u[:, -1] = u[:, 1]
    
        elif self.boundary_condition == 'dirichlet':
            if self.dim == 1:
                u[0] = 0
                u[-1] = 0
            elif self.dim == 2:
                u[0, :] = 0
                u[-1, :] = 0
                u[:, 0] = 0
                u[:, -1] = 0
    
        else:
            raise ValueError(
                f"Invalid boundary condition '{self.boundary_condition}'. "
                "Supported types are 'periodic' and 'dirichlet'."
            )

    def apply_nonlinear(self, u, is_v=False):
        """
        Apply nonlinear terms to the solution using spectral differentiation with dealiasing.

        This method evaluates all nonlinear terms present in the PDE by substituting spatial 
        derivatives with their spectral approximations computed via FFT. The dealiasing mask 
        ensures numerical stability by removing high-frequency components that could lead 
        to aliasing errors.

        Parameters
        ----------
        u : numpy.ndarray
            Current solution array on the spatial grid.
        is_v : bool
            If True, evaluates nonlinear terms for the velocity field v instead of u.

        Returns:
            numpy.ndarray: Array representing the contribution of nonlinear terms multiplied by dt.

        Notes:
        
        - In 1D, computes ∂ₓu via FFT and substitutes any derivative term in the nonlinear expressions.
        - In 2D, computes ∂ₓu and ∂ᵧu via FFT and performs similar substitutions.
        - Uses lambdify to evaluate symbolic nonlinear expressions numerically.
        - Derivatives are replaced symbolically with 'u_x' and 'u_y' before evaluation.
        """
        if not self.nonlinear_terms:
            return np.zeros_like(u, dtype=np.complex128)
        
        nonlinear_term = np.zeros_like(u, dtype=np.complex128)
    
        if self.dim == 1:
            u_hat = self.fft(u)
            u_hat *= self.dealiasing_mask
            u = self.ifft(u_hat)
    
            u_x_hat = (1j * self.KX) * u_hat
            u_x = self.ifft(u_x_hat)
    
            for term in self.nonlinear_terms:
                term_replaced = term
                if term.has(Derivative):
                    for deriv in term.atoms(Derivative):
                        if deriv.args[1][0] == self.x:
                            term_replaced = term_replaced.subs(deriv, symbols('u_x'))            
                term_func = lambdify((self.t, self.x, self.u_eq, 'u_x'), term_replaced, 'numpy')
                if is_v:
                    nonlinear_term += term_func(0, self.X, self.v_prev, u_x)
                else:
                    nonlinear_term += term_func(0, self.X, u, u_x)
    
        elif self.dim == 2:
            u_hat = self.fft(u)
            u_hat *= self.dealiasing_mask
            u = self.ifft(u_hat)
    
            u_x_hat = (1j * self.KX) * u_hat
            u_y_hat = (1j * self.KY) * u_hat
            u_x = self.ifft(u_x_hat)
            u_y = self.ifft(u_y_hat)
    
            for term in self.nonlinear_terms:
                term_replaced = term
                if term.has(Derivative):
                    for deriv in term.atoms(Derivative):
                        if deriv.args[1][0] == self.x:
                            term_replaced = term_replaced.subs(deriv, symbols('u_x'))
                        elif deriv.args[1][0] == self.y:
                            term_replaced = term_replaced.subs(deriv, symbols('u_y'))
                term_func = lambdify((self.t, self.x, self.y, self.u_eq, 'u_x', 'u_y'), term_replaced, 'numpy')
                if is_v:
                    nonlinear_term += term_func(0, self.X, self.Y, self.v_prev, u_x, u_y)
                else:
                    nonlinear_term += term_func(0, self.X, self.Y, u, u_x, u_y)
        else:
            raise ValueError("Unsupported spatial dimension.")
        
        return nonlinear_term * self.dt

    def prepare_symbol_tables(self):
        """
        Precompute and store evaluated pseudo-differential operator symbols for spectral methods.

        This method evaluates all pseudo-differential operators (ψOp) present in the PDE
        over the spatial and frequency grids, scales them by their respective coefficients,
        and combines them into a single composite symbol used in time-stepping and inversion.

        The evaluation is performed via the `evaluate` method of each PseudoDifferentialOperator,
        which computes p(x, ξ) or p(x, y, ξ, η) numerically over the current grid configuration.

        Side Effects:
            self.precomputed_symbols : list of (coeff, symbol_array)
                Each tuple contains a coefficient and its evaluated symbol on the grid.
            self.combined_symbol : np.ndarray
                Sum of all scaled symbol arrays: ∑(coeffₖ * ψₖ(x, ξ))

        Raises:
            ValueError: If the spatial dimension is not 1D or 2D.
        """
        self.precomputed_symbols = []
        self.combined_symbol = 0
        for coeff, psi in self.psi_ops:
            if self.dim == 1:
                raw = psi.evaluate(self.X, None, self.KX, None)
            elif self.dim == 2:
                raw = psi.evaluate(self.X, self.Y, self.KX, self.KY)
            else:
                raise ValueError('Unsupported spatial dimension.')
            raw_flat = raw.flatten()
            converted = np.array([complex(N(val)) for val in raw_flat], dtype=np.complex128)
            raw_eval = converted.reshape(raw.shape)
            self.precomputed_symbols.append((coeff, raw_eval))
        self.combined_symbol = sum((coeff * sym for coeff, sym in self.precomputed_symbols))
        self.combined_symbol = np.array(self.combined_symbol, dtype=np.complex128)

    def total_symbol_expr(self):
        """
        Compute the total pseudo-differential symbol expression from all pseudo_terms.

        This method constructs the full symbol of the pseudo-differential operator
        by summing up all coefficient-weighted symbolic expressions.

        The result is cached in self.symbol_expr to avoid recomputation.

        Returns:
            sympy.Expr: The combined symbol expression, representing the full
                        pseudo-differential operator in symbolic form.

        Example:
            Given pseudo_terms = [(2, ξ²), (1, x·ξ)], this returns 2·ξ² + x·ξ.
        """
        if not hasattr(self, '_symbol_expr'):
            self.symbol_expr = sum(coeff * expr for coeff, expr in self.pseudo_terms)
        return self.symbol_expr

    def build_symbol_func(self, expr):
        """
        Build a numerical evaluation function from a symbolic pseudo-differential operator expression.
    
        This method converts a symbolic expression representing a pseudo-differential operator into
        a callable NumPy-compatible function. The function accepts spatial and frequency variables
        depending on the dimensionality of the problem.
    
        Parameters
        ----------
        expr : sympy expression
            A SymPy expression representing the symbol of the pseudo-differential operator. It may depend on spatial variables (x, y) and frequency variables (xi, eta).
    
        Returns:
            function : A lambdified function that takes:
            
                - In 1D: `(x, xi)` — spatial coordinate and frequency.
                - In 2D: `(x, y, xi, eta)` — spatial coordinates and frequencies.
                
              Returns a NumPy array of evaluated symbol values over input grids.
    
        Notes:
            - Uses `lambdify` from SymPy with the `'numpy'` backend for efficient vectorized evaluation.
            - Real variable assumptions are enforced to ensure proper behavior in numerical contexts.
            - Used internally by methods like `apply_psiOp`, `evaluate`, and visualization tools.
        """
        if self.dim == 1:
            x, xi = symbols('x xi', real=True)
            return lambdify((x, xi), expr, 'numpy')
        else:
            x, y, xi, eta = symbols('x y xi eta', real=True)
            return lambdify((x, y, xi, eta), expr, 'numpy')

    def apply_psiOp(self, u):
        """
        Apply the pseudo-differential operator to the input field u.
    
        This method dispatches the application of the pseudo-differential operator based on:
        
        - Whether the symbol is spatially dependent (x/y)
        - The boundary condition in use (periodic or dirichlet)
    
        Supported operations:
        
        - Constant-coefficient symbols: applied via Fourier multiplication.
        - Spatially varying symbols: applied via Kohn–Nirenberg quantization.
        - Dirichlet boundary conditions: handled with non-periodic convolution-like quantization.
    
        Dispatch Logic:\n
        if not self.is_spatial: u ↦ Op(p)(D) ⋅ u = 𝓕⁻¹[ p(ξ) ⋅ 𝓕(u) ]\n
        elif periodic: u ↦ Op(p)(x,D) ⋅ u ≈ ∫ eᶦˣᶿ p(x, ξ) 𝓕(u)(ξ) dξ based of FFT (quicker)\n
        elif dirichlet: u ↦ Op(p)(x,D) ⋅ u ≈ u ≈ ∫ eᶦˣᶿ p(x, ξ) 𝓕(u)(ξ) dξ (slower)\n
    
        Parameters
        ----------
        u : np.ndarray 
            Input field to which the operator is applied.
            Should be 1D or 2D depending on the problem dimension.
    
        Returns:
            np.ndarray: Result of applying the pseudo-differential operator to u.
    
        Raises:
            ValueError: If an unsupported boundary condition is specified.
        """
        if not self.is_spatial:
            return self.apply_psiOp_constant(u)
        elif self.boundary_condition == 'periodic':
            return self.apply_psiOp_kohn_nirenberg_fft(u)
        elif self.boundary_condition == 'dirichlet':
            return self.apply_psiOp_kohn_nirenberg_nonperiodic(u)
        else:
            raise ValueError(f"Invalid boundary condition '{self.boundary_condition}'")

    def apply_psiOp_constant(self, u):
        """
        Apply a constant-coefficient pseudo-differential operator in Fourier space.

        This method assumes the symbol is diagonal in the Fourier basis and acts as a 
        multiplication operator. It performs the operation:
        
            (ψu)(x) = 𝓕⁻¹[ -σ(k) · 𝓕[u](k) ]

        where:
        - σ(k) is the combined pseudo-differential operator symbol
        - 𝓕 denotes the forward Fourier transform
        - 𝓕⁻¹ denotes the inverse Fourier transform

        The dealiasing mask is applied before returning to physical space.
        
        Parameters
        ----------
        u : np.ndarray
            Input function in physical space (real-valued or complex-valued)

        Returns:
            np.ndarray : Result of applying the pseudo-differential operator to u, same shape as input
        """
        u_hat = self.fft(u)
        u_hat *= -self.combined_symbol
        u_hat *= self.dealiasing_mask
        return self.ifft(u_hat)

    def apply_psiOp_kohn_nirenberg_fft(self, u):
        """
        Apply a pseudo-differential operator using the Kohn–Nirenberg quantization in Fourier space.
    
        This method evaluates the action of a pseudo-differential operator defined by the total symbol,
        computed from all psiOp terms in the equation. It uses the fast Fourier transform (FFT) for
        efficiency in periodic domains.
    
        Parameters
        ----------
        u : np.ndarray
            Input function in real space to which the operator is applied.
    
        Returns:
            np.ndarray: Resulting function after applying the pseudo-differential operator.
    
        Process:
            1. Compute the total symbolic expression of the pseudo-differential operator.
            2. Build a callable numerical function from the symbol.
            3. Evaluate Op(p)(u) via the Kohn–Nirenberg quantization using FFT.
    
        Note:
            - Assumes periodic boundary conditions.
            - The returned result is the negative of the standard definition due to PDE sign conventions.
        """
        total_symbol = self.total_symbol_expr()
        symbol_func = self.build_symbol_func(total_symbol)
        return -self.kohn_nirenberg_fft(u_vals=u, symbol_func=symbol_func)

    def apply_psiOp_kohn_nirenberg_nonperiodic(self, u):
        """
        Apply a pseudo-differential operator using the Kohn–Nirenberg quantization on non-periodic domains.
    
        This method evaluates the action of a pseudo-differential operator Op(p) on a function u 
        via the Kohn–Nirenberg representation. It supports both 1D and 2D cases and uses spatial 
        and frequency grids to evaluate the operator symbol p(x, ξ).
    
        The operator symbol p(x, ξ) is extracted from the PDE and evaluated numerically using 
        `_total_symbol_expr` and `_build_symbol_func`.
    
        Parameters
        ----------
        u : np.ndarray
            Input function (real space) to which the operator is applied.
    
        Returns:
            np.ndarray: Result of applying Op(p) to u in real space.
    
        Notes:
            - For 1D: p(x, ξ) is evaluated over x_grid and xi_grid.
            - For 2D: p(x, y, ξ, η) is evaluated over (x_grid, y_grid) and (xi_grid, eta_grid).
            - The result is computed using `kohn_nirenberg_nonperiodic`, which handles non-periodic boundary conditions.
        """
        total_symbol = self.total_symbol_expr()
        symbol_func = self.build_symbol_func(total_symbol)
        if self.dim == 1:
            return -self.kohn_nirenberg_nonperiodic(u_vals=u, x_grid=self.x_grid, xi_grid=self.kx, symbol_func=symbol_func)
        else:
            return -self.kohn_nirenberg_nonperiodic(u_vals=u, x_grid=(self.x_grid, self.y_grid), xi_grid=(self.kx, self.ky), symbol_func=symbol_func)
     
    def step_order1_with_psi(self, source_contribution):
        """
        Perform one time step of a first-order evolution using a pseudo-differential operator.
    
        This method updates the solution field using an exponential integrator or explicit Euler scheme,
        depending on boundary conditions and the structure of the pseudo-differential symbol.
        It supports:
        - Linear dynamics via pseudo-differential operator L (possibly nonlocal)
        - Nonlinear terms computed via spectral differentiation
        - External source contributions
    
        The update follows **three distinct computational paths**:
    
        1. **Periodic boundaries + diagonalizable symbol**  
           Symbol is constant in space → use direct Fourier-based exponential integrator:  
               uₙ₊₁ = e⁻ᴸΔᵗ ⋅ uₙ + Δt ⋅ φ₁(−LΔt) ⋅ (N(uₙ) + F)
    
        2. **Non-diagonalizable but spatially uniform symbol**  
           General exponential time differencing of order 1:  
               uₙ₊₁ = eᴸΔᵗ ⋅ uₙ + Δt ⋅ φ₁(LΔt) ⋅ (N(uₙ) + F)
    
        3. **Spatially varying symbol**  
           No frequency diagonalization available → use explicit Euler:  
               uₙ₊₁ = uₙ + Δt ⋅ (L(uₙ) + N(uₙ) + F)
    
        where:
            L(uₙ) = linear part via pseudo-differential operator
            N(uₙ) = nonlinear contribution at current time step
            F     = external source term
            Δt    = time step size
            φ₁(z) = (eᶻ − 1)/z (with safe handling near z=0)
    
        Boundary conditions are applied after each update to ensure consistency.
    
        Parameters
            source_contribution (np.ndarray): Array representing the external source term at current time step.
                                              Must match the spatial dimensions of self.u_prev.
    
        Returns:
            np.ndarray: Updated solution array after one time step.
        """
        # Handling null source
        if np.isscalar(source_contribution):
            source = np.zeros_like(self.u_prev)
        else:
            source = source_contribution

        def spectral_filter(u, cutoff=0.8):
            if u.ndim == 1:
                u_hat = self.fft(u)
                N = len(u)
                k = fftfreq(N)
                mask = np.exp(-(k / cutoff)**8)
                return self.ifft(u_hat * mask).real
            elif u.ndim == 2:
                u_hat = self.fft(u)
                Ny, Nx = u.shape
                ky = fftfreq(Ny)[:, None]
                kx = fftfreq(Nx)[None, :]
                k_squared = kx**2 + ky**2
                mask = np.exp(-(np.sqrt(k_squared) / cutoff)**8)
                return self.ifft(u_hat * mask).real
            else:
                raise ValueError("Only 1D and 2D arrays are supported.")

        # Recalculate symbol if necessary
        if self.is_spatial:
            self.prepare_symbol_tables()  # Recalculates self.combined_symbol
    
        # Case with FFT (symbol diagonalizable in Fourier space)
        if self.boundary_condition == 'periodic' and not self.is_spatial:
            u_hat = self.fft(self.u_prev)
            u_hat *= np.exp(-self.dt * self.combined_symbol)
            u_hat *= self.dealiasing_mask
            u_symb = self.ifft(u_hat)
            u_nl = self.apply_nonlinear(self.u_prev)
            u_new = u_symb + u_nl + source
        else:
            if not self.is_spatial:
                # General case with ETD1
                u_nl = self.apply_nonlinear(self.u_prev)
    
                # Calculation of exp(dt * L) and phi1(dt * L)
                L_vals = self.combined_symbol  # Uses the updated symbol
                exp_L = np.exp(-self.dt * L_vals)
                phi1_L = (exp_L - 1.0) / (self.dt * L_vals)
                phi1_L[np.isnan(phi1_L)] = 1.0  # Handling division by zero
    
                # Fourier transform
                u_hat = self.fft(self.u_prev)
                u_nl_hat = self.fft(u_nl)
                source_hat = self.fft(source)
    
                # Assembling the solution in Fourier space
                u_hat_new = exp_L * u_hat + self.dt * phi1_L * (u_nl_hat + source_hat)
                u_new = self.ifft(u_hat_new)
            else:
                # if the symbol depends on spatial variables : Euler method
                Lu_prev = self.apply_psiOp(self.u_prev)
                u_nl = self.apply_nonlinear(self.u_prev)
                u_new = self.u_prev + self.dt * (Lu_prev + u_nl + source)
                u_new = spectral_filter(u_new, cutoff=self.dealiasing_ratio)
        # Applying boundary conditions
        self.apply_boundary(u_new)
        return u_new

    def step_order2_with_psi(self, source_contribution):
        """
        Perform one time step of a second-order time evolution using a pseudo-differential operator.
    
        This method updates the solution field using a second-order accurate scheme suitable for wave-like equations.
        The update includes contributions from:
        - Linear dynamics via a pseudo-differential operator (e.g., dispersion or stiffness)
        - Nonlinear terms computed via spectral differentiation
        - External source contributions
    
        Discretization follows a leapfrog-style finite difference in time:
        
            uₙ₊₁ = 2uₙ − uₙ₋₁ + Δt² ⋅ (L(uₙ) + N(uₙ) + F)
    
        where:
            L(uₙ) = linear part evaluated via pseudo-differential operator
            N(uₙ) = nonlinear contribution at current time step
            F     = external source term at current time step
            Δt    = time step size
    
        Boundary conditions are applied after each update to ensure consistency.
    
        Parameters
            source_contribution (np.ndarray): Array representing the external source term at current time step.
                                              Must match the spatial dimensions of self.u_prev.
    
        Returns:
            np.ndarray: Updated solution array after one time step.
        """
        Lu_prev = self.apply_psiOp(self.u_prev)
        rhs_nl = self.apply_nonlinear(self.u_prev, is_v=False)
        u_new = 2 * self.u_prev - self.u_prev2 + self.dt ** 2 * (Lu_prev + rhs_nl + source_contribution)
        self.apply_boundary(u_new)
        self.u_prev2 = self.u_prev
        self.u_prev = u_new
        self.u = u_new
        return u_new

    def solve(self):
        """
        Solve the partial differential equation numerically using spectral methods.
        
        This method evolves the solution in time using a combination of:
        - Fourier-based linear evolution (with dealiasing)
        - Nonlinear term handling via pseudo-spectral evaluation
        - Support for pseudo-differential operators (psiOp)
        - Source terms and boundary conditions
        
        The solver supports:
        - 1D and 2D spatial domains
        - First and second-order time evolution
        - Periodic and Dirichlet boundary conditions
        - Time-stepping schemes: default, ETD-RK4
        
        Returns:
            list[np.ndarray]: A list of solution arrays at each saved time frame.
        
        Side Effects:
            - Updates self.frames: stores solution snapshots
            - Updates self.energy_history: records total energy if enabled
            
        Algorithm Overview:
            For each time step:
                1. Evaluate source contributions (if any)
                2. Apply time evolution:
                    - Order 1:
                        - With psiOp: uses step_order1_with_psi
                        - With ETD-RK4: exponential time differencing
                        - Default: linear + nonlinear update
                    - Order 2:
                        - With psiOp: uses step_order2_with_psi
                        - With ETD-RK4: second-order exponential scheme
                        - Default: second-order leapfrog-style update
                3. Enforce boundary conditions
                4. Save solution snapshot periodically
                5. Record energy (for second-order systems without psiOp)
        """
        print('\n*******************')
        print('* Solving the PDE *')
        print('*******************\n')
        save_interval = max(1, self.Nt // self.n_frames)
        self.energy_history = []
        for step in range(self.Nt):
            if hasattr(self, 'source_terms') and self.source_terms:
                source_contribution = np.zeros_like(self.X, dtype=np.float64)
                for term in self.source_terms:
                    try:
                        if self.dim == 1:
                            source_func = lambdify((self.t, self.x), term, 'numpy')
                            source_contribution += source_func(step * self.dt, self.X)
                        elif self.dim == 2:
                            source_func = lambdify((self.t, self.x, self.y), term, 'numpy')
                            source_contribution += source_func(step * self.dt, self.X, self.Y)
                    except Exception as e:
                        print(f'Error evaluating source term {term}: {e}')
            else:
                source_contribution = 0

            if self.temporal_order == 1:
                if self.has_psi:
                    u_new = self.step_order1_with_psi(source_contribution)
                elif hasattr(self, 'time_scheme') and self.time_scheme == 'ETD-RK4':
                    u_new = self.step_ETD_RK4(self.u_prev)
                else:
                    u_hat = self.fft(self.u_prev)
                    u_hat *= self.exp_L
                    u_hat *= self.dealiasing_mask
                    u_lin = self.ifft(u_hat)
                    u_nl = self.apply_nonlinear(u_lin)
                    u_new = u_lin + u_nl + source_contribution
                self.apply_boundary(u_new)
                self.u_prev = u_new

            elif self.temporal_order == 2:
                if self.has_psi:
                    u_new = self.step_order2_with_psi(source_contribution)
                else:
                    if hasattr(self, 'time_scheme') and self.time_scheme == 'ETD-RK4':
                        u_new, v_new = self.step_ETD_RK4_order2(self.u_prev, self.v_prev)
                    else:
                        u_hat = self.fft(self.u_prev)
                        v_hat = self.fft(self.v_prev)
                        u_new_hat = self.cos_omega_dt * u_hat + self.sin_omega_dt * self.inv_omega * v_hat
                        v_new_hat = -self.omega_val * self.sin_omega_dt * u_hat + self.cos_omega_dt * v_hat
                        u_new = self.ifft(u_new_hat)
                        v_new = self.ifft(v_new_hat)
                        u_nl = self.apply_nonlinear(self.u_prev, is_v=False)
                        v_nl = self.apply_nonlinear(self.v_prev, is_v=True)
                        u_new += (u_nl + source_contribution) * self.dt ** 2 / 2
                        v_new += (u_nl + source_contribution) * self.dt
                    self.apply_boundary(u_new)
                    self.apply_boundary(v_new)
                    self.u_prev = u_new
                    self.v_prev = v_new

            if step % save_interval == 0:
                self.frames.append(self.u_prev.copy())

            if self.temporal_order == 2 and (not self.has_psi):
                E = self.compute_energy()
                self.energy_history.append(E)

        return self.frames  
                
    def solve_stationary_psiOp(self, order=3):
        """
        Solve stationary pseudo-differential equations of the form P[u] = f(x) or P[u] = f(x,y) using asymptotic inversion.
    
        This method computes the solution to a stationary (time-independent) pseudo-differential equation
        where the operator P is defined via symbolic expressions (psiOp). It constructs an asymptotic right inverse R 
        such that P∘R ≈ Id, then applies it to the source term f using either direct Fourier multiplication 
        (when the symbol is spatially independent) or Kohn–Nirenberg quantization (when spatial dependence is present).
    
        The inversion is based on the principal symbol of the operator and its asymptotic expansion up to the given order.
        Ellipticity of the symbol is checked numerically before inversion to ensure well-posedness.
    
        Parameters
        ----------
        order : int, default=3
            Order of the asymptotic expansion used to construct the right inverse of the pseudo-differential operator.
        method : str, optional
            Inversion strategy:
            - 'diagonal' (default): Fast approximate inversion using diagonal operators in frequency space.
            - 'full'                : Pointwise exact inversion (slower but more accurate).
    
        Returns
        -------
        ndarray
            The computed solution u(x) in 1D or u(x, y) in 2D as a NumPy array over the spatial grid.
    
        Raises
        ------
        ValueError
            If no pseudo-differential operator (psiOp) is defined.
            If linear or nonlinear terms other than psiOp are present.
            If the symbol is not elliptic on the grid.
            If no source term is provided for the right-hand side.
    
        Notes
        -----
        - The method assumes the problem is fully stationary: time derivatives must be absent.
        - Requires the equation to be purely pseudo-differential (no Op, Derivative, or nonlinear terms).
        - Symbol evaluation and inversion are dimension-aware (supports both 1D and 2D problems).
        - Supports optimization paths when the symbol does not depend on spatial variables.
    
        See Also
        --------
        right_inverse_asymptotic : Constructs the asymptotic inverse of the pseudo-differential operator.
        kohn_nirenberg           : Numerical implementation of general pseudo-differential operators.
        is_elliptic_numerically  : Verifies numerical ellipticity of the symbol.
        """

        print("\n*******************************")
        print("* Solving the stationnary PDE *")
        print("*******************************\n")
        print("boundary condition: ",self.boundary_condition)
        

        if not self.has_psi:
            raise ValueError("Only supports problems with psiOp.")
    
        if self.linear_terms or self.nonlinear_terms:
            raise ValueError("Stationary psiOp problems must be linear and purely pseudo-differential.")

        if self.boundary_condition not in ('periodic', 'dirichlet'):
            raise ValueError(
                "For stationary PDEs, boundary conditions must be explicitly defined. "
                "Supported types are 'periodic' and 'dirichlet'."
            )    
            
        if self.dim == 1:
            x = self.x
            xi = symbols('xi', real=True)
            spatial_vars = (x,)
            freq_vars = (xi,)
            X, KX = self.X, self.KX
        elif self.dim == 2:
            x, y = self.x, self.y
            xi, eta = symbols('xi eta', real=True)
            spatial_vars = (x, y)
            freq_vars = (xi, eta)
            X, Y, KX, KY = self.X, self.Y, self.KX, self.KY
        else:
            raise ValueError("Unsupported spatial dimension.")
    
        total_symbol = sum(coeff * psi.expr for coeff, psi in self.psi_ops)
        psi_total = PseudoDifferentialOperator(total_symbol, spatial_vars, mode='symbol')
    
        # Check ellipticity
        if self.dim == 1:
            is_elliptic = psi_total.is_elliptic_numerically(X, KX)
        else:
            is_elliptic = psi_total.is_elliptic_numerically((X[:, 0], Y[0, :]), (KX[:, 0], KY[0, :]))
        if not is_elliptic:
            raise ValueError("❌ The pseudo-differential symbol is not numerically elliptic on the grid.")
        print("✅ Elliptic pseudo-differential symbol: inversion allowed.")
    
        R_symbol = psi_total.right_inverse_asymptotic(order=order)
        print("Right inverse asymptotic symbol:")
        pprint(R_symbol, num_columns=NUM_COLS)

        if self.dim == 1:
            if R_symbol.has(x):
                R_func = lambdify((x, xi), R_symbol, modules='numpy')
            else:
                R_func = lambdify((xi,), R_symbol, modules='numpy')
        else:
            if R_symbol.has(x) or R_symbol.has(y):
                R_func = lambdify((x, y, xi, eta), R_symbol, modules='numpy')
            else:
                R_func = lambdify((xi, eta), R_symbol, modules='numpy')
    
        # Build rhs
        if self.source_terms:
            f_expr = sum(self.source_terms)
            used_vars = [v for v in spatial_vars if f_expr.has(v)]
            f_func = lambdify(used_vars, -f_expr, modules='numpy')
            if self.dim == 1:
                rhs = f_func(self.x_grid) if used_vars else np.zeros_like(self.x_grid)
            else:
                rhs = f_func(self.X, self.Y) if used_vars else np.zeros_like(self.X)
        elif self.initial_condition:
            raise ValueError("Initial condition should be None for stationnary equation.")
        else:
            raise ValueError("No source term provided to construct the right-hand side.")
    
        f_hat = self.fft(rhs)
    
        if self.boundary_condition == 'periodic':
            if self.dim == 1:
                if not R_symbol.has(x):
                    print("⚡ Optimization: symbol independent of x — direct product in Fourier.")
                    R_vals = R_func(self.KX)
                    u_hat = R_vals * f_hat
                    u = self.ifft(u_hat)
                else:
                    print("⚙️ 1D Kohn-Nirenberg Quantification")
                    x, xi = symbols('x xi', real=True)
                    R_func = lambdify((x, xi), R_symbol, 'numpy')  # Still 2 args for uniformity
                    u = self.kohn_nirenberg_fft(u_vals=rhs, symbol_func=R_func)
                    
            elif self.dim == 2:
                if not R_symbol.has(x) and not R_symbol.has(y):
                    print("⚡ Optimization: Symbol independent of x and y — direct product in 2D Fourier.")
                    R_vals = np.vectorize(R_func)(self.KX, self.KY)
                    u_hat = R_vals * f_hat
                    u = self.ifft(u_hat)
                else:
                    print("⚙️ 2D Kohn-Nirenberg Quantification")
                    x, xi, y, eta = symbols('x xi y eta', real=True)
                    R_func = lambdify((x, y, xi, eta), R_symbol, 'numpy')  # Still 2 args for uniformity
                    u = self.kohn_nirenberg_fft(u_vals=rhs, symbol_func=R_func)
            self.u = u
            return u
        elif self.boundary_condition == 'dirichlet':
            if self.dim == 1:
                x, xi = symbols('x xi', real=True)
                R_func = lambdify((x, xi), R_symbol, 'numpy')
                u = self.kohn_nirenberg_nonperiodic(u_vals=rhs, x_grid=X, xi_grid=KX, symbol_func=R_func)
            elif self.dim == 2:
                x, xi, y, eta = symbols('x xi y eta', real=True)
                R_func = lambdify((x, y, xi, eta), R_symbol, 'numpy')
                u = self.kohn_nirenberg_nonperiodic(u_vals=rhs, x_grid=(self.x_grid, self.y_grid), xi_grid=(self.kx, self.ky), symbol_func=R_func)
            self.u = u
            return u   
        else:
            raise ValueError(
                f"Invalid boundary condition '{self.boundary_condition}'. "
                "Supported types are 'periodic' and 'dirichlet'."
            )

    def kohn_nirenberg_fft(self, u_vals, symbol_func,
                           freq_window='gaussian', clamp=1e6,
                           space_window=False):
        """
        Numerically stable Kohn–Nirenberg quantization of a pseudo-differential operator.
        
        Applies the pseudo-differential operator Op(p) to the function f via the Kohn–Nirenberg quantization:
        
            [Op(p)f](x) = (1/(2π)^d) ∫ p(x, ξ) e^{ix·ξ} ℱ[f](ξ) dξ
        
        where p(x, ξ) is a symbol that may depend on both spatial variables x and frequency variables ξ.
        
        This method supports both 1D and 2D cases and includes optional smoothing techniques to improve numerical stability.
    
        Parameters
        ----------
        u_vals : np.ndarray
            Spatial samples of the input function f(x) or f(x, y), defined on a uniform grid.
        symbol_func : callable
            A function representing the full symbol p(x, ξ) in 1D or p(x, y, ξ, η) in 2D.
            Must accept NumPy-compatible array inputs and return a complex-valued array.
        freq_window : {'gaussian', 'hann', None}, optional
            Type of frequency-domain window to apply:
            - 'gaussian': smooth decay near high frequencies
            - 'hann': cosine-based tapering with hard cutoff
            - None: no frequency window applied
        clamp : float, optional
            Upper bound on the absolute value of the symbol. Prevents numerical blow-up from large values.
        space_window : bool, optional
            Whether to apply a spatial Gaussian window to suppress edge effects in physical space.
    
        Returns
        -------
        np.ndarray
            The result of applying the pseudo-differential operator to f, returned as a real or complex array
            of the same shape as u_vals.
    
        Notes
        -----
        - The implementation uses FFT-based quadrature of the inverse Fourier transform.
        - Symbol evaluation is vectorized over spatial and frequency grids.
        - Frequency and spatial windows help mitigate oscillatory behavior and aliasing.
        - In 2D, the integration is performed over a 4D tensor product grid (x, y, ξ, η).
        """
        # === Common setup ===
        xg = self.x_grid
        dx = xg[1] - xg[0]
    
        if self.dim == 1:
            # === 1D case ===
    
            # Frequency grid (shifted to center zero)
            Nx = self.Nx
            k = 2 * np.pi * fftshift(fftfreq(Nx, d=dx))
            dk = k[1] - k[0]
    
            # Centered FFT of input
            f_shift = fftshift(u_vals)
            f_hat = self.fft(f_shift) * dx
            f_hat = fftshift(f_hat)
    
            # Build meshgrid for (x, ξ)
            X, K = np.meshgrid(xg, k, indexing='ij')
    
            # Evaluate the symbol p(x, ξ)
            P = symbol_func(X, K)
    
            # Optional: clamp extreme values
            P = np.clip(P, -clamp, clamp)
    
            # === Frequency-domain window ===
            if freq_window == 'gaussian':
                sigma = 0.8 * np.max(np.abs(k))
                W = np.exp(-(K / sigma) ** 4)
                P *= W
            elif freq_window == 'hann':
                W = 0.5 * (1 + np.cos(np.pi * K / np.max(np.abs(K))))
                P *= W * (np.abs(K) < np.max(np.abs(K)))
    
            # === Optional spatial window ===
            if space_window:
                x0 = (xg[0] + xg[-1]) / 2
                L = (xg[-1] - xg[0]) / 2
                S = np.exp(-((X - x0) / L) ** 2)
                P *= S
    
            # === Oscillatory kernel and integration ===
            kernel = np.exp(1j * X * K)
            integrand = P * f_hat[None, :] * kernel
    
            # Approximate inverse Fourier integral
            u = np.sum(integrand, axis=1) * dk / (2 * np.pi)
            return u
    
        else:
            # === 2D case ===
    
            yg = self.y_grid
            dy = yg[1] - yg[0]
            Nx, Ny = self.Nx, self.Ny
    
            # Frequency grids
            kx = 2 * np.pi * fftshift(fftfreq(Nx, d=dx))
            ky = 2 * np.pi * fftshift(fftfreq(Ny, d=dy))
            dkx = kx[1] - kx[0]
            dky = ky[1] - ky[0]
    
            # 2D FFT of f(x, y)
            f_shift = fftshift(u_vals)
            f_hat = self.fft(f_shift) * dx * dy
            f_hat = fftshift(f_hat)
    
            # Create 4D grids for broadcasting
            X, Y = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            Xb = X[:, :, None, None]
            Yb = Y[:, :, None, None]
            KXb = KX[None, None, :, :]
            KYb = KY[None, None, :, :]
    
            # Evaluate p(x, y, ξ, η)
            P_vals = symbol_func(Xb, Yb, KXb, KYb)
            P_vals = np.clip(P_vals, -clamp, clamp)
    
            # === Frequency windowing ===
            if freq_window == 'gaussian':
                sigma_kx = 0.8 * np.max(np.abs(kx))
                sigma_ky = 0.8 * np.max(np.abs(ky))
                W_kx = np.exp(-(KXb / sigma_kx) ** 4)
                W_ky = np.exp(-(KYb / sigma_ky) ** 4)
                P_vals *= W_kx * W_ky
            elif freq_window == 'hann':
                Wx = 0.5 * (1 + np.cos(np.pi * KXb / np.max(np.abs(kx))))
                Wy = 0.5 * (1 + np.cos(np.pi * KYb / np.max(np.abs(ky))))
                mask_x = np.abs(KXb) < np.max(np.abs(kx))
                mask_y = np.abs(KYb) < np.max(np.abs(ky))
                P_vals *= Wx * Wy * mask_x * mask_y
    
            # === Optional spatial tapering ===
            if space_window:
                x0 = (self.x_grid[0] + self.x_grid[-1]) / 2
                y0 = (self.y_grid[0] + self.y_grid[-1]) / 2
                Lx = (self.x_grid[-1] - self.x_grid[0]) / 2
                Ly = (self.y_grid[-1] - self.y_grid[0]) / 2
                S = np.exp(-((Xb - x0) / Lx) ** 2 - ((Yb - y0) / Ly) ** 2)
                P_vals *= S
    
            # === Oscillatory kernel and integration ===
            phase = np.exp(1j * (Xb * KXb + Yb * KYb))
            integrand = P_vals * phase * f_hat[None, None, :, :]
    
            # 2D Fourier inversion (numerical integration)
            u = np.sum(integrand, axis=(2, 3)) * dkx * dky / (2 * np.pi) ** 2
            return u
        
    def kohn_nirenberg_nonperiodic(self, u_vals, x_grid, xi_grid, symbol_func,
                                   freq_window='gaussian', clamp=1e6, space_window=False):
        """
        Numerically applies the Kohn–Nirenberg quantization of a pseudo-differential operator 
        in a non-periodic setting.
    
        This method computes:
        
        [Op(p)u](x) = (1/(2π)^d) ∫ p(x, ξ) e^{i x·ξ} ℱ[u](ξ) dξ
        
        where p(x, ξ) is a general symbol that may depend on both spatial and frequency variables.
        It supports both 1D and 2D inputs and includes optional numerical smoothing techniques 
        to enhance stability for non-smooth or oscillatory symbols.
    
        Parameters
        ----------
        u_vals : np.ndarray
            Input function values defined on a uniform spatial grid. Can be 1D (Nx,) or 2D (Nx, Ny).
        x_grid : np.ndarray
            Spatial grid points along each axis. In 1D: shape (Nx,). In 2D: tuple of two arrays (X, Y)
            or list of coordinate arrays.
        xi_grid : np.ndarray
            Frequency grid points. In 1D: shape (Nxi,). In 2D: tuple of two arrays (Xi, Eta)
            or list of frequency arrays.
        symbol_func : callable
            A function representing the full symbol p(x, ξ) in 1D or p(x, y, ξ, η) in 2D.
            Must accept NumPy-compatible array inputs and return a complex-valued array.
        freq_window : {'gaussian', 'hann', None}, optional
            Type of frequency-domain window to apply for regularization:
            
            - 'gaussian': Smooth exponential decay near high frequencies.
            - 'hann': Cosine-based tapering with hard cutoff.
            - None: No frequency window applied.
        clamp : float, optional
            Maximum absolute value allowed for the symbol to prevent numerical overflow.
            Default is 1e6.
        space_window : bool, optional
            If True, applies a smooth spatial Gaussian window centered in the domain to reduce
            boundary artifacts. Default is False.
    
        Returns
        -------
        np.ndarray
            The result of applying the pseudo-differential operator Op(p) to u. Shape matches u_vals.
        
        Notes
        -----
        - This version does not assume periodicity and is suitable for Dirichlet or Neumann boundary conditions.
        - In 1D, the integral is evaluated as a sum over (x, ξ), using matrix exponentials.
        - In 2D, the integration is performed over a 4D tensor product grid (x, y, ξ, η), which can be computationally intensive.
        - Symbol evaluation should be vectorized for performance.
        - For large grids, consider reducing resolution via resampling before calling this function.
    
        See Also
        --------
        kohn_nirenberg_fft : Faster implementation for periodic domains using FFT.
        PseudoDifferentialOperator : Class for symbolic manipulation of pseudo-differential operators.
        """
        if u_vals.ndim == 1:
            # === 1D case ===
            x = x_grid
            xi = xi_grid
            dx = x[1] - x[0]
            dxi = xi[1] - xi[0]
    
            phase_ft = np.exp(-1j * np.outer(xi, x))  # (Nxi, Nx)
            u_hat = dx * np.dot(phase_ft, u_vals)     # (Nxi,)
    
            X, XI = np.meshgrid(x, xi, indexing='ij')  # (Nx, Nxi)
            sigma_vals = symbol_func(X, XI)
    
            # Clamp values
            sigma_vals = np.clip(sigma_vals, -clamp, clamp)
    
            # Frequency window
            if freq_window == 'gaussian':
                sigma = 0.8 * np.max(np.abs(XI))
                window = np.exp(-(XI / sigma)**4)
                sigma_vals *= window
            elif freq_window == 'hann':
                window = 0.5 * (1 + np.cos(np.pi * XI / np.max(np.abs(XI))))
                sigma_vals *= window * (np.abs(XI) < np.max(np.abs(XI)))
    
            # Spatial window
            if space_window:
                x_center = (x[0] + x[-1]) / 2
                L = (x[-1] - x[0]) / 2
                window = np.exp(-((X - x_center)/L)**2)
                sigma_vals *= window
    
            exp_matrix = np.exp(1j * np.outer(x, xi))  # (Nx, Nxi)
            integrand = sigma_vals * u_hat[np.newaxis, :] * exp_matrix
            result = dxi * np.sum(integrand, axis=1) / (2 * np.pi)
            return result
    
        elif u_vals.ndim == 2:
            # === 2D case ===
            x1, x2 = x_grid
            xi1, xi2 = xi_grid
            dx1 = x1[1] - x1[0]
            dx2 = x2[1] - x2[0]
            dxi1 = xi1[1] - xi1[0]
            dxi2 = xi2[1] - xi2[0]
    
            X1, X2 = np.meshgrid(x1, x2, indexing='ij')
            XI1, XI2 = np.meshgrid(xi1, xi2, indexing='ij')
    
            # Fourier transform of u(x1, x2)
            phase_ft = np.exp(-1j * (np.tensordot(x1, xi1, axes=0)[:, None, :, None] +
                                     np.tensordot(x2, xi2, axes=0)[None, :, None, :]))
            u_hat = np.tensordot(u_vals, phase_ft, axes=([0,1], [0,1])) * dx1 * dx2
    
            # Symbol evaluation
            sigma_vals = symbol_func(X1[:, :, None, None], X2[:, :, None, None],
                                     XI1[None, None, :, :], XI2[None, None, :, :])
    
            # Clamp values
            sigma_vals = np.clip(sigma_vals, -clamp, clamp)
    
            # Frequency window
            if freq_window == 'gaussian':
                sigma_xi1 = 0.8 * np.max(np.abs(XI1))
                sigma_xi2 = 0.8 * np.max(np.abs(XI2))
                window = np.exp(-(XI1[None, None, :, :] / sigma_xi1)**4 -
                                (XI2[None, None, :, :] / sigma_xi2)**4)
                sigma_vals *= window
            elif freq_window == 'hann':
                # Frequency window - Hanning
                wx = 0.5 * (1 + np.cos(np.pi * XI1 / np.max(np.abs(XI1))))
                wy = 0.5 * (1 + np.cos(np.pi * XI2 / np.max(np.abs(XI2))))
                
                # Mask to zero outside max frequency
                mask_x = (np.abs(XI1) < np.max(np.abs(XI1)))
                mask_y = (np.abs(XI2) < np.max(np.abs(XI2)))
                
                # Expand wx and wy to match sigma_vals shape: (64, 64, 64, 64)
                sigma_vals *= wx[:, :, None, None] * wy[:, :, None, None]
                sigma_vals *= mask_x[:, :, None, None] * mask_y[:, :, None, None]
    
            # Spatial window
            if space_window:
                x_center = (x1[0] + x1[-1])/2
                y_center = (x2[0] + x2[-1])/2
                Lx = (x1[-1] - x1[0])/2
                Ly = (x2[-1] - x2[0])/2
                window = np.exp(-((X1 - x_center)/Lx)**2 - ((X2 - y_center)/Ly)**2)
                sigma_vals *= window[:, :, None, None]
    
            # Oscillatory phase
            phase = np.exp(1j * (X1[:, :, None, None] * XI1[None, None, :, :] +
                                 X2[:, :, None, None] * XI2[None, None, :, :]))

            integrand = sigma_vals * u_hat[None, None, :, :] * phase
            result = dxi1 * dxi2 * np.sum(integrand, axis=(2, 3)) / (2 * np.pi)**2
            return result
    
        else:
            raise NotImplementedError("Only 1D and 2D supported")

    def step_ETD_RK4(self, u):
        """
        Perform one Exponential Time Differencing Runge-Kutta of 4th order (ETD-RK4) time step 
        for first-order in time PDEs of the form:
        
            ∂ₜu = L u + N(u)
        
        where L is a linear operator (possibly nonlocal or pseudo-differential), and N is a 
        nonlinear term treated via pseudo-spectral methods. This method evaluates the 
        exponential integrator up to fourth-order accuracy in time.
    
        The ETD-RK4 scheme uses four stages to approximate the integral of the variation-of-constants formula:
        
            uⁿ⁺¹ = e^(L Δt) uⁿ + Δt ∫₀¹ e^(L Δt (1 - τ)) φ(N(u(τ))) dτ
        
        where φ denotes the nonlinear contributions evaluated at intermediate stages.
    
        Parameters
            u (np.ndarray): Current solution in real space (physical grid values).
    
        Returns:
            np.ndarray: Updated solution in real space after one ETD-RK4 time step.
    
        Notes:
        - The linear part L is diagonal in Fourier space and precomputed as self.L(k).
        - Nonlinear terms are evaluated in physical space and transformed via FFT.
        - The functions φ₁(z) and φ₂(z) are entire functions arising from the ETD scheme:
          
              φ₁(z) = (eᶻ - 1)/z   if z ≠ 0
                     = 1            if z = 0
    
              φ₂(z) = (eᶻ - 1 - z)/z²   if z ≠ 0
                     = ½              if z = 0
    
        - This implementation assumes periodic boundary conditions and uses spectral differentiation via FFT.
        - See Hochbruck & Ostermann (2010) for theoretical background on exponential integrators.
    
        See Also:
            step_ETD_RK4_order2 : For second-order in time equations.
            psiOp_apply           : For applying pseudo-differential operators.
            apply_nonlinear      : For handling nonlinear terms in the PDE.
        """
        dt = self.dt
        L_fft = self.L(self.KX) if self.dim == 1 else self.L(self.KX, self.KY)
    
        E  = np.exp(dt * L_fft)
        E2 = np.exp(dt * L_fft / 2)
    
        def phi1(z):
            return np.where(np.abs(z) > 1e-12, (np.exp(z) - 1) / z, 1.0)
    
        def phi2(z):
            return np.where(np.abs(z) > 1e-12, (np.exp(z) - 1 - z) / z**2, 0.5)
    
        phi1_dtL = phi1(dt * L_fft)
        phi2_dtL = phi2(dt * L_fft)
    
        fft = self.fft
        ifft = self.ifft
    
        u_hat = fft(u)
        N1 = fft(self.apply_nonlinear(u))
    
        a = ifft(E2 * (u_hat + 0.5 * dt * N1 * phi1_dtL))
        N2 = fft(self.apply_nonlinear(a))
    
        b = ifft(E2 * (u_hat + 0.5 * dt * N2 * phi1_dtL))
        N3 = fft(self.apply_nonlinear(b))
    
        c = ifft(E * (u_hat + dt * N3 * phi1_dtL))
        N4 = fft(self.apply_nonlinear(c))
    
        u_new_hat = E * u_hat + dt * (
            N1 * phi1_dtL + 2 * (N2 + N3) * phi2_dtL + N4 * phi1_dtL
        ) / 6
    
        return ifft(u_new_hat)

    def step_ETD_RK4_order2(self, u, v):
        """
        Perform one time step of the Exponential Time Differencing Runge-Kutta 4th-order (ETD-RK4) scheme for second-order PDEs.
    
        This method evolves the solution u and its time derivative v forward in time by one step using the ETD-RK4 integrator. 
        It is designed for systems of the form:
        
            ∂ₜ²u = L u + N(u)
            
        where L is a linear operator and N is a nonlinear term computed via self.apply_nonlinear.
        
        The exponential integrator handles the linear part exactly in Fourier space, while the nonlinear terms are integrated 
        using a fourth-order Runge-Kutta-like approach. This ensures high accuracy and stability for stiff systems.
    
        Parameters:
            u (np.ndarray): Current solution array in real space.
            v (np.ndarray): Current time derivative of the solution (∂ₜu) in real space.
    
        Returns:
            tuple: (u_new, v_new), updated solution and its time derivative after one time step.
    
        Notes:
            - Assumes periodic boundary conditions and uses FFT-based spectral methods.
            - Handles both 1D and 2D problems seamlessly.
            - Uses phi functions to compute exponential integrators efficiently.
            - Suitable for wave equations and other second-order evolution equations with stiffness.
        """
        dt = self.dt
    
        L_fft = self.L(self.KX) if self.dim == 1 else self.L(self.KX, self.KY)
        fft = self.fft
        ifft = self.ifft
    
        def rhs(u_val):
            return ifft(L_fft * fft(u_val)) + self.apply_nonlinear(u_val, is_v=False)
    
        # Stage A
        A = rhs(u)
        ua = u + 0.5 * dt * v
        va = v + 0.5 * dt * A
    
        # Stage B
        B = rhs(ua)
        ub = u + 0.5 * dt * va
        vb = v + 0.5 * dt * B
    
        # Stage C
        C = rhs(ub)
        uc = u + dt * vb
    
        # Stage D
        D = rhs(uc)
    
        # Final update
        u_new = u + dt * v + (dt**2 / 6.0) * (A + 2*B + 2*C + D)
        v_new = v + (dt / 6.0) * (A + 2*B + 2*C + D)
    
        return u_new, v_new

    def check_cfl_condition(self):
        """
        Check the CFL (Courant–Friedrichs–Lewymann) condition based on group velocity 
        for second-order time-dependent PDEs.
    
        This method verifies whether the chosen time step dt satisfies the numerical stability 
        condition derived from the maximum wave propagation speed in the system. It supports both 
        1D and 2D problems, with or without a symbolic dispersion relation ω(k).
    
        The CFL condition ensures that information does not propagate further than one grid cell 
        per time step. A safety factor of 0.5 is applied by default to ensure robustness.
    
        Notes:
        
        - In 1D, the group velocity v₉(k) = dω/dk is used to compute the maximum wave speed.
        - In 2D, the x- and y-directional group velocities are evaluated independently.
        - If no dispersion relation is available, the imaginary part of the linear operator L(k) 
          is used as an approximation for wave speed.
    
        Raises:
        -------
        NotImplementedError: 
            If the spatial dimension is not 1D or 2D.
    
        Prints:
        -------
        Warning message if the current time step dt exceeds the CFL-stable limit.
        """
        print("\n*****************")
        print("* CFL condition *")
        print("*****************\n")

        cfl_factor = 0.5  # Safety factor
        
        if self.dim == 1:
            if self.temporal_order == 2 and hasattr(self, 'omega'):
                k_vals = self.kx
                omega_vals = np.real(self.omega(k_vals))
                with np.errstate(divide='ignore', invalid='ignore'):
                    v_group = np.gradient(omega_vals, k_vals)
                max_speed = np.max(np.abs(v_group))
            else:
                max_speed = np.max(np.abs(np.imag(self.L(self.kx))))
            
            dx = self.Lx / self.Nx
            cfl_limit = cfl_factor * dx / max_speed if max_speed != 0 else np.inf
            
            if self.dt > cfl_limit:
                print(f"CFL condition violated: dt = {self.dt}, max allowed dt = {cfl_limit}")
    
        elif self.dim == 2:
            if self.temporal_order == 2 and hasattr(self, 'omega'):
                k_vals = self.kx
                omega_x = np.real(self.omega(k_vals, 0))
                omega_y = np.real(self.omega(0, k_vals))
                with np.errstate(divide='ignore', invalid='ignore'):
                    v_group_x = np.gradient(omega_x, k_vals)
                    v_group_y = np.gradient(omega_y, k_vals)
                max_speed_x = np.max(np.abs(v_group_x))
                max_speed_y = np.max(np.abs(v_group_y))
            else:
                max_speed_x = np.max(np.abs(np.imag(self.L(self.kx, 0))))
                max_speed_y = np.max(np.abs(np.imag(self.L(0, self.ky))))
            
            dx = self.Lx / self.Nx
            dy = self.Ly / self.Ny
            cfl_limit = cfl_factor / (max_speed_x / dx + max_speed_y / dy) if (max_speed_x + max_speed_y) != 0 else np.inf
            
            if self.dt > cfl_limit:
                print(f"CFL condition violated: dt = {self.dt}, max allowed dt = {cfl_limit}")
    
        else:
            raise NotImplementedError("Only 1D and 2D problems are supported.")

    def check_symbol_conditions(self, k_range=None, verbose=True):
        """
        Check strict analytic conditions on the linear symbol self.L_symbolic:
            This method evaluates three key properties of the Fourier multiplier 
            symbol a(k) = self.L(k), which are crucial for well-posedness, stability,
            and numerical efficiency. The checks apply to both 1D and 2D cases.
        
        Conditions checked:
        ------------------
        1. **Stability condition**: Re(a(k)) ≤ 0 for all k ≠ 0
           Ensures that the system does not exhibit exponential growth in time.
    
        2. **Dissipation condition**: Re(a(k)) ≤ -δ |k|² for large |k|
           Ensures sufficient damping at high frequencies to avoid oscillatory instability.
    
        3. **Growth condition**: |a(k)| ≤ C (1 + |k|)^m with m ≤ 4
           Ensures that the symbol does not grow too rapidly with frequency, 
           which would otherwise cause numerical instability or unphysical amplification.
    
        Parameters
        ----------
        k_range : tuple or None, optional
            Specifies the range of frequencies to test in the form (k_min, k_max, N).
            If None, defaults are used: [-10, 10] with 500 points in 1D, or [-10, 10] 
            with 100 points per axis in 2D.
    
        verbose : bool, default=True
            If True, prints detailed results of each condition check.
    
        Returns:
        --------
        None
            Output is printed directly to the console for interpretability.
    
        Notes:
        ------
        - In 2D, the radial frequency |k| = √(kx² + ky²) is used for comparisons.
        - The dissipation threshold assumes δ = 0.01 and p = 2 by default.
        - The growth ratio is compared against |k|⁴; values above 100 indicate rapid growth.
        - This function is typically called during solver setup or analysis phase.
    
        See Also:
        ---------
        analyze_wave_propagation : For further symbolic and numerical analysis of dispersion.
        plot_symbol : Visualizes the symbol's behavior over the frequency domain.
        """
        print("\n********************")
        print("* Symbol condition *")
        print("********************\n")

    
        if self.dim == 1:    
            if k_range is None:
                k_vals = np.linspace(-10, 10, 500)
            else:
                k_min, k_max, N = k_range
                k_vals = np.linspace(k_min, k_max, N)
    
            L_vals = self.L(k_vals)
            k_abs = np.abs(k_vals)
    
        elif self.dim == 2:
            if k_range is None:
                k_vals = np.linspace(-10, 10, 100)
            else:
                k_min, k_max, N = k_range
                k_vals = np.linspace(k_min, k_max, N)
    
            KX, KY = np.meshgrid(k_vals, k_vals)
            L_vals = self.L(KX, KY)
            k_abs = np.sqrt(KX**2 + KY**2)
    
        else:
            raise ValueError("Only 1D and 2D dimensions are supported.")

    
        re_vals = np.real(L_vals)
        abs_vals = np.abs(L_vals)
    
        # === Condition 1: Stability
        if np.any(re_vals > 1e-12):
            max_pos = np.max(re_vals)
            if verbose:
                print(f"❌ Stability violated: max Re(a(k)) = {max_pos}")
            print("Unstable symbol: Re(a(k)) > 0")
        elif verbose:
            print("✅ Spectral stability satisfied: Re(a(k)) ≤ 0")
    
        # === Condition 2: Dissipation
        mask = k_abs > 2
        if np.any(mask):
            re_decay = re_vals[mask]
            expected_decay = -0.01 * k_abs[mask]**2
            if np.any(re_decay > expected_decay + 1e-6):
                if verbose:
                    print("⚠️ Insufficient high-frequency dissipation")
            else:
                if verbose:
                    print("✅ Proper high-frequency dissipation")
    
        # === Condition 3: Growth
        growth_ratio = abs_vals / (1 + k_abs)**4
        if np.max(growth_ratio) > 100:
            if verbose:
                print("⚠️ Symbol grows rapidly: |a(k)| ≳ |k|^4")
        else:
            if verbose:
                print("✅ Reasonable spectral growth")
    
        if verbose:
            print("✔ Symbol analysis completed.")

    def analyze_wave_propagation(self):
        """
        Perform a detailed analysis of wave propagation characteristics based on the dispersion relation ω(k).
    
        This method visualizes key wave properties in both 1D and 2D settings:
        
        - Dispersion relation: ω(k)
        - Phase velocity: v_p(k) = ω(k)/|k|
        - Group velocity: v_g(k) = ∇ₖ ω(k)
        - Anisotropy in 2D (via magnitude of group velocity)
    
        The symbolic dispersion relation 'omega_symbolic' must be defined beforehand.
        This is typically available only for second-order-in-time equations.
    
        In 1D:
            Plots ω(k), v_p(k), and v_g(k) over a range of k values.
    
        In 2D:
            Displays heatmaps of ω(kx, ky), v_p(kx, ky), and |v_g(kx, ky)| over a 2D wavenumber grid.
    
        Raises:
            AttributeError: If 'omega_symbolic' is not defined, the method exits gracefully with a message.
    
        Side Effects:
            Generates and displays matplotlib plots.
        """
        print("\n*****************************")
        print("* Wave propagation analysis *")
        print("*****************************\n")
        if not hasattr(self, 'omega_symbolic'):
            print("❌ omega_symbolic not defined. Only available for 2nd order in time.")
            return
    
        if self.dim == 1:
            k = self.k_symbols[0]
            omega_func = lambdify(k, self.omega_symbolic, 'numpy')
    
            k_vals = np.linspace(-10, 10, 1000)
            omega_vals = omega_func(k_vals)
    
            with np.errstate(divide='ignore', invalid='ignore'):
                v_phase = np.where(k_vals != 0, omega_vals / k_vals, 0.0)
    
            dk = k_vals[1] - k_vals[0]
            v_group = np.gradient(omega_vals, dk)
    
            plt.figure(figsize=(10, 6))
            plt.plot(k_vals, omega_vals, label=r'$\omega(k)$')
            plt.plot(k_vals, v_phase, label=r'$v_p(k)$')
            plt.plot(k_vals, v_group, label=r'$v_g(k)$')
            plt.title("1D Wave Propagation Analysis")
            plt.xlabel("k")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()
    
        elif self.dim == 2:
            kx, ky = self.k_symbols
            omega_func = lambdify((kx, ky), self.omega_symbolic, 'numpy')
    
            k_vals = np.linspace(-10, 10, 200)
            KX, KY = np.meshgrid(k_vals, k_vals)
            K_mag = np.sqrt(KX**2 + KY**2)
            K_mag[K_mag == 0] = 1e-8  # Avoid division by 0
    
            omega_vals = omega_func(KX, KY)
            v_phase = np.real(omega_vals) / K_mag
    
            dk = k_vals[1] - k_vals[0]
            domega_dx = np.gradient(omega_vals, dk, axis=0)
            domega_dy = np.gradient(omega_vals, dk, axis=1)
            v_group_norm = np.sqrt(np.abs(domega_dx)**2 + np.abs(domega_dy)**2)
    
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
            im0 = axs[0].imshow(np.real(omega_vals), extent=[-10, 10, -10, 10],
                                origin='lower', cmap='viridis')
            axs[0].set_title(r'$\omega(k_x, k_y)$')
            plt.colorbar(im0, ax=axs[0])
    
            im1 = axs[1].imshow(v_phase, extent=[-10, 10, -10, 10],
                                origin='lower', cmap='plasma')
            axs[1].set_title(r'$v_p(k_x, k_y)$')
            plt.colorbar(im1, ax=axs[1])
    
            im2 = axs[2].imshow(v_group_norm, extent=[-10, 10, -10, 10],
                                origin='lower', cmap='inferno')
            axs[2].set_title(r'$|v_g(k_x, k_y)|$')
            plt.colorbar(im2, ax=axs[2])
    
            for ax in axs:
                ax.set_xlabel(r'$k_x$')
                ax.set_ylabel(r'$k_y$')
                ax.set_aspect('equal')
    
            plt.tight_layout()
            plt.show()
    
        else:
            print("❌ Only 1D and 2D wave analysis supported.")
        
    def plot_symbol(self, component="abs", k_range=None, cmap="viridis"):
        """
        Visualize the spectral symbol L(k) or L(kx, ky) in 1D or 2D.
    
        This method plots the linear operator's symbolic Fourier representation 
        either as a function of a single wavenumber k (1D), or two wavenumbers 
        kx and ky (2D). The user can choose to display the real part, imaginary part, 
        or absolute value of the symbol.
    
        Parameters
        ----------
        component : str {'abs', 're', 'im'}
            Component of the symbol to visualize:
            
                - 'abs' : absolute value |a(k)|
                - 're'  : real part Re[a(k)]
                - 'im'  : imaginary part Im[a(k)]
                
        k_range : tuple (kmin, kmax, N), optional
            Wavenumber range for evaluation:
            
                - kmin: minimum wavenumber
                - kmax: maximum wavenumber
                - N: number of sampling points
                
            If None, defaults to [-10, 10] with high resolution.
        cmap : str, optional
            Colormap used for 2D surface plots. Default is 'viridis'.
    
        Raises
        ------
            ValueError: If the spatial dimension is not 1D or 2D.
    
        Notes:
            - In 1D, the symbol is plotted using a standard 2D line plot.
            - In 2D, a 3D surface plot is generated with color-mapped height.
            - Symbol evaluation uses self.L(k), which must be defined and callable.
        """
        print("\n*******************")
        print("* Symbol plotting *")
        print("*******************\n")
        
        assert component in ("abs", "re", "im"), "component must be 'abs', 're' or 'im'"
        
    
        if self.dim == 1:
            if k_range is None:
                k_vals = np.linspace(-10, 10, 1000)
            else:
                kmin, kmax, N = k_range
                k_vals = np.linspace(kmin, kmax, N)
            L_vals = self.L(k_vals)
    
            if component == "re":
                vals = np.real(L_vals)
                label = "Re[a(k)]"
            elif component == "im":
                vals = np.imag(L_vals)
                label = "Im[a(k)]"
            else:
                vals = np.abs(L_vals)
                label = "|a(k)|"
    
            plt.plot(k_vals, vals)
            plt.xlabel("k")
            plt.ylabel(label)
            plt.title(f"Spectral symbol: {label}")
            plt.grid(True)
            plt.show()
    
        elif self.dim == 2:
            if k_range is None:
                k_vals = np.linspace(-10, 10, 300)
            else:
                kmin, kmax, N = k_range
                k_vals = np.linspace(kmin, kmax, N)
    
            KX, KY = np.meshgrid(k_vals, k_vals)
            L_vals = self.L(KX, KY)
    
            if component == "re":
                Z = np.real(L_vals)
                title = "Re[a(kx, ky)]"
            elif component == "im":
                Z = np.imag(L_vals)
                title = "Im[a(kx, ky)]"
            else:
                Z = np.abs(L_vals)
                title = "|a(kx, ky)|"
    
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
        
            surf = ax.plot_surface(KX, KY, Z, cmap=cmap, edgecolor='none', antialiased=True)
            fig.colorbar(surf, ax=ax, shrink=0.6)
        
            ax.set_xlabel("kx")
            ax.set_ylabel("ky")
            ax.set_zlabel(title)
            ax.set_title(f"2D spectral symbol: {title}")
            plt.tight_layout()
            plt.show()
    
        else:
            raise ValueError("Only 1D and 2D supported.")

    def compute_energy(self):
        """
        Compute the total energy of the wave equation solution for second-order temporal PDEs. 
        The energy is defined as:
            E(t) = 1/2 ∫ [ (∂ₜu)² + |L¹ᐟ²u|² ] dx
        where L is the linear operator associated with the spatial part of the PDE,
        and L¹ᐟ² denotes its square root in Fourier space.
    
        This method supports both 1D and 2D problems and is only meaningful when 
        self.temporal_order == 2 (second-order time derivative).
    
        Returns
        -------
        float or None: 
            Total energy at current time step. Returns None if the temporal order is not 2 or if no valid velocity data (v_prev) is available.
    
        Notes
        -----
        - Uses FFT-based spectral differentiation to compute the spatial contributions.
        - Assumes periodic boundary conditions.
        - Handles both real and complex-valued solutions.
        """
        if self.temporal_order != 2 or self.v_prev is None:
            return None
    
        u = self.u_prev
        v = self.v_prev
    
        # Fourier transform of u
        u_hat = self.fft(u)
    
        if self.dim == 1:
            # 1D case
            L_vals = self.L(self.KX)
            sqrt_L = np.sqrt(np.abs(L_vals))
            Lu_hat = sqrt_L * u_hat  # Apply sqrt(|L(k)|) in Fourier space
            Lu = self.ifft(Lu_hat)
    
            dx = self.Lx / self.Nx
            energy_density = 0.5 * (np.abs(v)**2 + np.abs(Lu)**2)
            total_energy = np.sum(energy_density) * dx
    
        elif self.dim == 2:
            # 2D case
            L_vals = self.L(self.KX, self.KY)
            sqrt_L = np.sqrt(np.abs(L_vals))
            Lu_hat = sqrt_L * u_hat
            Lu = self.ifft(Lu_hat)
    
            dx = self.Lx / self.Nx
            dy = self.Ly / self.Ny
            energy_density = 0.5 * (np.abs(v)**2 + np.abs(Lu)**2)
            total_energy = np.sum(energy_density) * dx * dy
    
        else:
            raise ValueError("Unsupported dimension for u.")
    
        return total_energy

    def plot_energy(self, log=False):
        """
        Plot the time evolution of the total energy for wave equations. 
        Visualizes the energy computed during simulation for both 1D and 2D cases. 
        Requires temporal_order=2 and prior execution of compute_energy() during solve().
        
        Parameters:
            log : bool
                If True, displays energy on a logarithmic scale to highlight exponential decay/growth.
        
        Notes:
            - Energy is defined as E(t) = 1/2 ∫ [ (∂ₜu)² + |L¹⸍²u|² ] dx
            - Only available if energy monitoring was activated in solve()
            - Automatically skips plotting if no energy data is available
        
        Displays:
            - Time vs. Total Energy plot with grid and legend
            - Appropriate axis labels and dimensional context (1D/2D)
            - Logarithmic or linear scaling based on input parameter
        """
        if not hasattr(self, 'energy_history') or not self.energy_history:
            print("No energy data recorded. Call compute_energy() within solve().")
            return
    
        # Time vector for plotting
        t = np.linspace(0, self.Lt, len(self.energy_history))
    
        # Create the figure
        plt.figure(figsize=(6, 4))
        if log:
            plt.semilogy(t, self.energy_history, label="Energy (log scale)")
        else:
            plt.plot(t, self.energy_history, label="Energy")
    
        # Axis labels and title
        plt.xlabel("Time")
        plt.ylabel("Total energy")
        plt.title("Energy evolution ({}D)".format(self.dim))
    
        # Display options
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def show_stationary_solution(self, u=None, component='abs', cmap='viridis'):
        """
        Display the stationary solution computed by solve_stationary_psiOp.

        This method visualizes the solution of a pseudo-differential equation 
        solved in stationary mode. It supports both 1D and 2D spatial domains, 
        with options to display different components of the solution (real, 
        imaginary, absolute value, or phase).

        Parameters
        ----------
        u : ndarray, optional
            Precomputed solution array. If None, calls solve_stationary_psiOp() 
            to compute the solution.
        component : str, optional {'real', 'imag', 'abs', 'angle'}
            Component of the complex-valued solution to display:
            - 'real': Real part
            - 'imag': Imaginary part
            - 'abs' : Absolute value (modulus)
            - 'angle' : Phase (argument)
        cmap : str, optional
            Colormap used for 2D visualization (default: 'viridis').

        Raises
        ------
        ValueError
            If an invalid component is specified or if the spatial dimension 
            is not supported (only 1D and 2D are implemented).

        Notes
        -----
        - In 1D, the solution is displayed using a standard line plot.
        - In 2D, the solution is visualized as a 3D surface plot.
        """
        def get_component(u):
            if component == 'real':
                return np.real(u)
            elif component == 'imag':
                return np.imag(u)
            elif component == 'abs':
                return np.abs(u)
            elif component == 'angle':
                return np.angle(u)
            else:
                raise ValueError("Invalid component")
                
        if u is None:
            u = self.solve_stationary_psiOp()

        if self.dim == 1:
            # Plot the solution in 1D
            plt.figure(figsize=(8, 4))
            plt.plot(self.x_grid, get_component(u), label=f'{component} of u')
            plt.xlabel('x')
            plt.ylabel(f'{component} of u')
            plt.title('Stationary solution (1D)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
    
        elif self.dim == 2:
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel(f'{component.title()} of u')
            plt.title('Stationary solution (2D)')    
            data0 = get_component(u)
            ax.plot_surface(self.X, self.Y, data0, cmap='viridis')
            plt.tight_layout()
            plt.show()
    
        else:
            raise ValueError("Only 1D and 2D display are supported.")

    def animate(self, component='abs', overlay='contour', mode='surface'):
        """
        Create an animated plot of the solution evolution over time.
    
        This method generates a dynamic visualization of the stored solution frames
        `self.frames`. It supports:
          - 1D line animation (unchanged),
          - 2D surface animation (original behavior, 'surface'),
          - 2D image animation using imshow (new, 'imshow') which is faster and
            often clearer for large grids.
    
        Parameters
        ----------
        component : str, optional, one of {'real', 'imag', 'abs', 'angle'}
            Which component of the complex field to visualize:
              - 'real'  : Re(u)
              - 'imag'  : Im(u)
              - 'abs'   : |u|
              - 'angle' : arg(u)
            Default is 'abs'.
    
        overlay : str or None, optional, one of {'contour', 'front', None}
            For 2D modes only. If None, no overlay is drawn.
              - 'contour' : draw contour lines on top (or beneath for 3D surface)
              - 'front'   : detect and mark wavefronts using gradient maxima
            Default is 'contour'.
    
        mode : str, optional, one of {'surface', 'imshow'}
            2D rendering mode. 'surface' keeps the original 3D surface plot.
            'imshow' draws a 2D raster (faster, often more readable).
            Default is 'surface' for backward compatibility.
    
        Returns
        -------
        FuncAnimation
            A Matplotlib `FuncAnimation` instance (you can display it in a notebook
            or save it to file).
    
        Notes
        -----
        - The method uses the same time-mapping logic as before (linear sampling of
          stored frames to animation frames).
        - For 'angle' the color scale is fixed between -π and π.
        - For other components, color scaling is by default dynamically adapted per
          frame in 'imshow' mode (this avoids extreme clipping if amplitudes vary).
        - Overlays are updated cleanly: previous contour/scatter artists are removed
          before drawing the next frame to avoid memory/visual accumulation.
        - Animation interval is 50 ms per frame (unchanged).
        """
        def get_component(u):
            if component == 'real':
                return np.real(u)
            elif component == 'imag':
                return np.imag(u)
            elif component == 'abs':
                return np.abs(u)
            elif component == 'angle':
                return np.angle(u)
            else:
                raise ValueError("Invalid component: choose 'real','imag','abs' or 'angle'")
    
        print("\n*********************")
        print("* Solution plotting *")
        print("*********************\n")
    
        # === Calculate time vector of stored frames ===
        save_interval = max(1, self.Nt // self.n_frames)
        frame_times = np.arange(0, self.Lt + self.dt, save_interval * self.dt)
    
        # === Target times for animation ===
        target_times = np.linspace(0, self.Lt, self.n_frames // 2)
    
        # Map target times to nearest frame indices
        frame_indices = [np.argmin(np.abs(frame_times - t)) for t in target_times]
    
        # -------------------------
        # 1D case (unchanged logic)
        # -------------------------
        if self.dim == 1:
            fig, ax = plt.subplots()
            initial = get_component(self.frames[0])
            line, = ax.plot(self.X, np.real(initial) if np.iscomplexobj(initial) else initial)
            ax.set_ylim(np.min(initial), np.max(initial))
            ax.set_xlabel('x')
            ax.set_ylabel(f'{component} of u')
            ax.set_title('Initial condition')
            plt.tight_layout()
    
            def update_1d(frame_number):
                frame = frame_indices[frame_number]
                ydata = get_component(self.frames[frame])
                ydata_real = np.real(ydata) if np.iscomplexobj(ydata) else ydata
                line.set_ydata(ydata_real)
                ax.set_ylim(np.min(ydata_real), np.max(ydata_real))
                current_time = target_times[frame_number]
                ax.set_title(f't = {current_time:.2f}')
                return (line,)
    
            ani = FuncAnimation(fig, update_1d, frames=len(target_times), interval=50)
            return ani
    
        # -------------------------
        # 2D case
        # -------------------------
        # Validate mode
        if mode not in ('surface', 'imshow'):
            raise ValueError("Invalid mode: choose 'surface' or 'imshow'")
    
        # Common data
        data0 = get_component(self.frames[0])
    
        if mode == 'surface':
            # original surface behavior, but ensure clean updates
            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel(f'{component.title()} of u')
            ax.zaxis.labelpad = 0
            ax.set_title('Initial condition')
    
            surf = ax.plot_surface(self.X, self.Y, data0, cmap='viridis')
            plt.tight_layout()
    
            def update_surface(frame_number):
                frame = frame_indices[frame_number]
                current_data = get_component(self.frames[frame])
                z_offset = np.max(current_data) + 0.05 * (np.max(current_data) - np.min(current_data))
    
                ax.clear()
                surf_obj = ax.plot_surface(self.X, self.Y, current_data,
                                           cmap='viridis',
                                           vmin=(-np.pi if component == 'angle' else None),
                                           vmax=(np.pi if component == 'angle' else None))
                # overlays
                if overlay == 'contour':
                    # place contours slightly below the surface (use offset)
                    try:
                        ax.contour(self.X, self.Y, current_data, levels=10, cmap='cool', offset=z_offset)
                    except Exception:
                        # fallback: simple contour without offset if not supported
                        ax.contour(self.X, self.Y, current_data, levels=10, cmap='cool')
    
                elif overlay == 'front':
                    dx = self.x_grid[1] - self.x_grid[0]
                    dy = self.y_grid[1] - self.y_grid[0]
                    # numpy.gradient: axis0 -> y spacing, axis1 -> x spacing
                    du_dy, du_dx = np.gradient(current_data, dy, dx)
                    grad_norm = np.sqrt(du_dx**2 + du_dy**2)
                    local_max = (grad_norm == maximum_filter(grad_norm, size=5))
                    if np.max(grad_norm) > 0:
                        normalized = grad_norm[local_max] / np.max(grad_norm)
                    else:
                        normalized = np.zeros(np.count_nonzero(local_max))
                    colors = cm.plasma(normalized)
                    ax.scatter(self.X[local_max], self.Y[local_max],
                               z_offset * np.ones_like(self.X[local_max]),
                               color=colors, s=10, alpha=0.8)
    
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel(f'{component.title()} of u')
                current_time = target_times[frame_number]
                ax.set_title(f'Solution at t = {current_time:.2f}')
                return (surf_obj,)
    
            ani = FuncAnimation(fig, update_surface, frames=len(target_times), interval=50)
            return ani
    
        else:  # mode == 'imshow'
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Initial condition')
    
            # extent uses physical coordinates so axes show real x/y values
            extent = [self.x_grid[0], self.x_grid[-1], self.y_grid[0], self.y_grid[-1]]
    
            if component == 'angle':
                vmin, vmax = -np.pi, np.pi
                cmap = 'twilight'
            else:
                vmin, vmax = np.min(data0), np.max(data0)
                cmap = 'viridis'
    
            im = ax.imshow(data0, extent=extent, origin='lower', cmap=cmap,
                           vmin=vmin, vmax=vmax, aspect='auto')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(f"{component} of u")
            plt.tight_layout()
    
            # containers for dynamic overlay artists (stored on function object)
            # update_im.contour_art and update_im.scatter_art will be created dynamically
    
            def update_im(frame_number):
                frame = frame_indices[frame_number]
                current_data = get_component(self.frames[frame])
    
                # update raster
                im.set_data(current_data)
                if component != 'angle':
                    # dynamic per-frame scaling (keeps contrast when amplitude varies)
                    cmin = np.nanmin(current_data)
                    cmax = np.nanmax(current_data)
                    # avoid identical vmin==vmax
                    if cmax > cmin:
                        im.set_clim(cmin, cmax)
    
                # remove previous contour if exists
                if overlay == 'contour':
                    if hasattr(update_im, 'contour_art') and update_im.contour_art is not None:
                        for coll in update_im.contour_art.collections:
                            try:
                                coll.remove()
                            except Exception:
                                pass
                        update_im.contour_art = None
                    # draw new contours (use meshgrid coords)
                    try:
                        update_im.contour_art = ax.contour(self.X, self.Y, current_data, levels=10, cmap='cool')
                    except Exception:
                        # fallback: contour with axis coordinates (x_grid, y_grid)
                        Xc, Yc = np.meshgrid(self.x_grid, self.y_grid)
                        update_im.contour_art = ax.contour(Xc, Yc, current_data, levels=10, cmap='cool')
    
                # remove previous scatter if exists
                if overlay == 'front':
                    if hasattr(update_im, 'scatter_art') and update_im.scatter_art is not None:
                        try:
                            update_im.scatter_art.remove()
                        except Exception:
                            pass
                        update_im.scatter_art = None
    
                    dx = self.x_grid[1] - self.x_grid[0]
                    dy = self.y_grid[1] - self.y_grid[0]
                    du_dy, du_dx = np.gradient(current_data, dy, dx)
                    grad_norm = np.sqrt(du_dx**2 + du_dy**2)
                    local_max = (grad_norm == maximum_filter(grad_norm, size=5))
                    if np.max(grad_norm) > 0:
                        normalized = grad_norm[local_max] / np.max(grad_norm)
                    else:
                        normalized = np.zeros(np.count_nonzero(local_max))
                    colors = cm.plasma(normalized)
                    update_im.scatter_art = ax.scatter(self.X[local_max], self.Y[local_max],
                                                       c=colors, s=10, alpha=0.8)
    
                current_time = target_times[frame_number]
                ax.set_title(f'Solution at t = {current_time:.2f}')
                # return main image plus any overlay artists present so Matplotlib can redraw them
                artists = [im]
                if overlay == 'contour' and hasattr(update_im, 'contour_art') and update_im.contour_art is not None:
                    artists.extend(update_im.contour_art.collections)
                if overlay == 'front' and hasattr(update_im, 'scatter_art') and update_im.scatter_art is not None:
                    artists.append(update_im.scatter_art)
                return tuple(artists)
    
            ani = FuncAnimation(fig, update_im, frames=len(target_times), interval=50)
            return ani

    def test(self, u_exact, t_eval=None, norm='relative', threshold=1e-2, component='real'):
        """
        Test the solver against an exact solution.

        This method quantitatively compares the numerical solution with a provided exact solution 
        at a specified time using either relative or absolute error norms. It supports both 
        stationary and time-dependent problems in 1D and 2D. If enabled, it also generates plots 
        of the solution, exact solution, and pointwise error.

        Parameters
        ----------
        u_exact : callable
            Exact solution function taking spatial coordinates and optionally time as arguments.
        t_eval : float, optional
            Time at which to compare solutions. For non-stationary problems, defaults to final time Lt.
            Ignored for stationary problems.
        norm : str {'relative', 'absolute'}
            Type of error norm used in comparison.
        threshold : float
            Acceptable error threshold; raises an assertion if exceeded.
        plot : bool
            Whether to display visual comparison plots (default: True).
        component : str {'real', 'imag', 'abs'}
            Component of the solution to compare and visualize.

        Raises
        ------
        ValueError
            If unsupported dimension is encountered or requested evaluation time exceeds simulation duration.
        AssertionError
            If computed error exceeds the given threshold.

        Prints
        ------
        - Information about the closest available frame to the requested evaluation time.
        - Computed error value and comparison to threshold.

        Notes
        -----
        - For time-dependent problems, the solution is extracted from precomputed frames.
        - Plots are adapted to spatial dimension: line plots for 1D, image plots for 2D.
        - The method ensures consistent handling of real, imaginary, and magnitude components.
        """
        if self.is_stationary:
            print("Testing a stationary solution.")
            u_num = self.u
    
            # Compute exact solution
            if self.dim == 1:
                u_ex = u_exact(self.X)
            elif self.dim == 2:
                u_ex = u_exact(self.X, self.Y)
            else:
                raise ValueError("Unsupported dimension.")
            actual_t = None
        else:
            if t_eval is None:
                t_eval = self.Lt
    
            save_interval = max(1, self.Nt // self.n_frames)
            frame_times = np.arange(0, self.Lt + self.dt, save_interval * self.dt)
            frame_index = np.argmin(np.abs(frame_times - t_eval))
            actual_t = frame_times[frame_index]
            print(f"Closest available time to t_eval={t_eval}: {actual_t}")
    
            if frame_index >= len(self.frames):
                raise ValueError(f"Time t = {t_eval} exceeds simulation duration.")
    
            u_num = self.frames[frame_index]
    
            # Compute exact solution at the actual time
            if self.dim == 1:
                u_ex = u_exact(self.X, actual_t)
            elif self.dim == 2:
                u_ex = u_exact(self.X, self.Y, actual_t)
            else:
                raise ValueError("Unsupported dimension.")
    
        # Select component
        if component == 'real':
            diff = np.real(u_num) - np.real(u_ex)
            ref = np.real(u_ex)
        elif component == 'imag':
            diff = np.imag(u_num) - np.imag(u_ex)
            ref = np.imag(u_ex)
        elif component == 'abs':
            diff = np.abs(u_num) - np.abs(u_ex)
            ref = np.abs(u_ex)
        else:
            raise ValueError("Invalid component.")
    
        # Compute error
        if norm == 'relative':
            error = np.linalg.norm(diff) / np.linalg.norm(ref)
        elif norm == 'absolute':
            error = np.linalg.norm(diff)
        else:
            raise ValueError("Unknown norm type.")
    
        label_time = f"t = {actual_t}" if actual_t is not None else ""
        print(f"Test error {label_time}: {error:.3e}")
        assert error < threshold, f"Error too large {label_time}: {error:.3e}"
    
        # Plot
        if self.plot:
            if self.dim == 1:
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 1, 1)
                plt.plot(self.X, np.real(u_num), label='Numerical')
                plt.plot(self.X, np.real(u_ex), '--', label='Exact')
                plt.title(f'Solution {label_time}, error = {error:.2e}')
                plt.legend()
                plt.grid()
    
                plt.subplot(2, 1, 2)
                plt.plot(self.X, np.abs(diff), color='red')
                plt.title('Absolute Error')
                plt.grid()
                plt.tight_layout()
                plt.show()
            else:
                extent = [-self.Lx/2, self.Lx/2, -self.Ly/2, self.Ly/2]
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.title("Numerical Solution")
                plt.imshow(np.abs(u_num), origin='lower', extent=extent, cmap='viridis')
                plt.colorbar()
    
                plt.subplot(1, 3, 2)
                plt.title("Exact Solution")
                plt.imshow(np.abs(u_ex), origin='lower', extent=extent, cmap='viridis')
                plt.colorbar()
    
                plt.subplot(1, 3, 3)
                plt.title(f"Error (Norm = {error:.2e})")
                plt.imshow(np.abs(diff), origin='lower', extent=extent, cmap='inferno')
                plt.colorbar()
                plt.tight_layout()
                plt.show()
