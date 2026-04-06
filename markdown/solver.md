# Underlying Theory of the PDESolver (including nonlinear aspects)

[Back to psipy main page](./psipy.md)

## 1. Spectral Discretisation

The solver uses a **Fourier spectral method** for spatial discretisation. The unknown function $u(\mathbf{x}, t)$ is approximated by its truncated Fourier series on a uniform grid with periodic boundary conditions (non‑periodic conditions are handled via a pseudo‑differential reformulation, see §7).

- In 1D, with $x \in [-L_x/2, L_x/2]$ and $N_x$ points, the discrete Fourier transform pair is  
  $$
  \hat{u}(k) = \frac{1}{N_x}\sum_{j=0}^{N_x-1} u(x_j)\, e^{-ikx_j}, \qquad
  u(x_j) = \sum_{k} \hat{u}(k)\, e^{ikx_j},
  $$
  where the wavenumbers are $k = 2\pi n / L_x,\; n = -N_x/2,\dots,N_x/2-1$.

- Spatial derivatives become multiplication by the wavenumber in Fourier space:
  $$
  \frac{\partial}{\partial x} \;\longrightarrow\; i k,\qquad
  \frac{\partial^2}{\partial x^2} \;\longrightarrow\; -k^2,\qquad
  \text{etc.}
  $$

- **Dealiasing**: To avoid aliasing errors from quadratic (or higher) nonlinear terms, the highest one‑third of the Fourier modes are zeroed out (the 2/3‑rule). A sharp spectral cut‑off mask is applied after each nonlinear evaluation.

## 2. Equation Parsing and Linear Operator

The user provides a PDE as a SymPy `Eq`. The solver:

- Identifies the unknown function $u(\mathbf{x},t)$ and its spatial and temporal dependencies.
- Separates the equation into **linear**, **nonlinear**, **source**, and **pseudo‑differential** terms.
- For the linear part, a plane wave ansatz $u \sim e^{i(\mathbf{k}\cdot\mathbf{x} - \omega t)}$ is substituted. Each derivative is replaced by its Fourier symbol:
  $$
  \partial_t \to -i\omega,\qquad \partial_{x_i} \to i k_i,\qquad
  \partial_{x_i}^2 \to -k_i^2,\quad \text{etc.}
  $$
  The resulting algebraic equation is solved for $\omega$ (or for the linear operator $L(\mathbf{k})$ when the equation is first order in time).

- For a first‑order‑in‑time equation $\partial_t u = L u + N(u) + f$, the Fourier symbol of the linear operator is  
  $$
  L(\mathbf{k}) = \frac{\text{linear RHS in Fourier space}}{\hat{u}}.
  $$
  For a second‑order equation $\partial_t^2 u = L u + \dots$, the dispersion relation $\omega(\mathbf{k})$ is extracted, and the linear evolution is governed by $\omega^2 = -L(\mathbf{k})$.

## 3. Nonlinear Term Handling

### 3.1 Detection and Classification

During parsing, a term is considered **nonlinear** if it contains:

- Any function applied to $u$ (e.g., $\sin(u)$, $\exp(u)$, $\sqrt{u}$) – this includes `Abs`, `sin`, `exp`, etc.
- A power of $u$ with exponent $\neq 1$ (e.g., $u^2$, $u^3$).
- A product containing both $u$ and a derivative of $u$ (e.g., $u\,\partial_x u$).  
Pure derivatives of $u$ are linear; products of derivatives without $u$ (e.g., $(\partial_x u)^2$) are also nonlinear because they are not linear in $u$.

All such terms are collected in the list `self.nonlinear_terms` and are never absorbed into the linear operator.

### 3.2 Pseudo‑spectral Evaluation

Nonlinear terms are evaluated in **physical space** using the pseudo‑spectral method:

1. **Forward FFT**: Compute $\hat{u} = \mathcal{F}(u)$.
2. **Dealiasing**: Multiply $\hat{u}$ by the dealiasing mask (zero out high wavenumbers).
3. **Inverse FFT**: Obtain $u_{\text{filtered}} = \mathcal{F}^{-1}(\hat{u}_{\text{filtered}})$.
4. **Derivative computation**: For each derivative appearing in the nonlinear expression (e.g., $\partial_x u$), compute it spectrally:
   $$
   \widehat{\partial_{x_i}u} = i k_i \,\hat{u},\qquad
   \widehat{\partial_{x_i}^2 u} = -k_i^2 \,\hat{u},
   $$
   then transform back to physical space.
5. **Symbolic substitution**: Replace each derivative in the SymPy expression by a temporary symbol (`'u_x'`, `'u_y'`) to avoid re‑differentiation.
6. **Lambdification**: Compile the substituted expression into a fast NumPy function using `lambdify`.
7. **Evaluation**: Evaluate the lambdified function at the current grid points, passing the filtered $u$ and its derivatives.
8. **Summation**: Sum all nonlinear term contributions.

For 2D problems, if there are multiple independent nonlinear terms, the evaluation is parallelised using `ThreadPoolExecutor` (one thread per term) to reduce overhead.

### 3.3 Time‑stepping with Nonlinear Terms

The nonlinear contribution $N(u)$ is included in all time integration schemes:

- **Exponential Euler (first order)**  
  $$
  u^{n+1} = e^{L\Delta t} u^n + \Delta t\,\varphi_1(L\Delta t)\, \bigl(N(u^n) + f(t_n)\bigr)
  $$

- **ETD‑RK4 (fourth order)**  
  The scheme evaluates $N$ at four intermediate states (stages $a, b, c$) and combines them with the exponential functions $\varphi_1$ and $\varphi_2$.

- **Second‑order (wave) leapfrog**  
  $$
  u^{n+1} = 2u^n - u^{n-1} + \Delta t^2 \bigl(L u^n + N(u^n) + f(t_n)\bigr)
  $$

- **ETD‑RK4 for second order**  
  A similar multi‑stage approach but applied to the first‑order system $(u, v = \partial_t u)$.

In all cases, the nonlinear term is evaluated at the **current** (or intermediate) state and then multiplied by the appropriate factor (e.g., $\Delta t$ or $\Delta t^2$). The solver does **not** treat nonlinear terms implicitly; they are always explicit.

## 4. Time Integration Schemes

### 4.1 Exponential Euler (first order)

For $\partial_t u = L u + R(u,t)$ with $R = N + f$, the variation‑of‑constants formula gives
$$
u(t+\Delta t) = e^{L\Delta t} u(t) + \int_0^{\Delta t} e^{L(\Delta t-\tau)} R(u(t+\tau), t+\tau)\,d\tau.
$$
The exponential Euler method approximates the integral by the rectangle rule at $\tau=0$:
$$
u^{n+1} = e^{L\Delta t} u^n + \Delta t\, \varphi_1(L\Delta t)\, R(u^n,t_n),
$$
where $\varphi_1(z) = (e^z-1)/z$ (with $\varphi_1(0)=1$). In Fourier space, $L(\mathbf{k})$ is diagonal, so $e^{L\Delta t}$ and $\varphi_1(L\Delta t)$ are pointwise multipliers.

### 4.2 ETD‑RK4 (fourth order)

For higher accuracy, the solver implements the **Exponential Time Differencing Runge‑Kutta scheme of order 4** (Kassam & Trefethen, 2005). It uses four stages to approximate the integral of the nonlinear term:
$$
\begin{aligned}
a_n &= e^{L\Delta t/2} u^n + \frac{\Delta t}{2} \varphi_1(L\Delta t/2)\, N(u^n),\\
b_n &= e^{L\Delta t/2} u^n + \frac{\Delta t}{2} \varphi_1(L\Delta t/2)\, N(a_n),\\
c_n &= e^{L\Delta t} u^n + \Delta t\, \varphi_1(L\Delta t)\, N(b_n),\\
u^{n+1} &= e^{L\Delta t} u^n + \Delta t\Bigl[ \varphi_1(L\Delta t) N(u^n) + 2\varphi_2(L\Delta t)\bigl(N(a_n)+N(b_n)\bigr) + \varphi_1(L\Delta t)N(c_n)\Bigr]/6,
\end{aligned}
$$
with $\varphi_2(z) = (e^z-1-z)/z^2$. The functions $\varphi_1,\varphi_2$ are precomputed for each wavenumber.

### 4.3 Second‑order equations (wave‑like)

For $\partial_t^2 u = L u + N(u) + f$, the system is reduced to first order by introducing $v = \partial_t u$. In the simplest “leapfrog” spectral scheme (when $L$ is diagonalisable):
$$
\begin{aligned}
\hat{u}^{n+1} &= \cos(\omega\Delta t)\,\hat{u}^n + \frac{\sin(\omega\Delta t)}{\omega}\,\hat{v}^n,\\
\hat{v}^{n+1} &= -\omega\sin(\omega\Delta t)\,\hat{u}^n + \cos(\omega\Delta t)\,\hat{v}^n,
\end{aligned}
$$
where $\omega(\mathbf{k}) = \sqrt{-L(\mathbf{k})}$ (taking the principal branch). Nonlinear and source terms are added as corrections (using a Strang‑like splitting or ETD‑RK4 for second order).

## 5. Pseudo‑Differential Operators (ψOp)

A pseudo‑differential operator with symbol $p(\mathbf{x},\boldsymbol{\xi})$ is defined by
$$
\bigl(\psi\text{Op}(p)\,u\bigr)(\mathbf{x}) = \frac{1}{(2\pi)^d}\int_{\mathbb{R}^d} e^{i\mathbf{x}\cdot\boldsymbol{\xi}}\, p(\mathbf{x},\boldsymbol{\xi})\, \hat{u}(\boldsymbol{\xi})\, d\boldsymbol{\xi},
$$
where $\hat{u}$ is the Fourier transform of $u$. The solver supports such operators through the `psiOp` construct.

### 5.1 Constant‑coefficient symbols

If $p$ does not depend on $\mathbf{x}$, the operator reduces to a Fourier multiplier:
$$
\widehat{\psi\text{Op}(p)u}(\boldsymbol{\xi}) = p(\boldsymbol{\xi})\,\hat{u}(\boldsymbol{\xi}),
$$
which is applied efficiently by an FFT.

### 5.2 Spatially varying symbols (Kohn‑Nirenberg quantisation)

For $p(\mathbf{x},\boldsymbol{\xi})$ that depends on $\mathbf{x}$, the operator is evaluated numerically using the Kohn‑Nirenberg formula:
$$
\bigl(\psi\text{Op}(p)u\bigr)(\mathbf{x}) \approx \frac{1}{(2\pi)^d}\sum_{\boldsymbol{\xi}} e^{i\mathbf{x}\cdot\boldsymbol{\xi}}\, p(\mathbf{x},\boldsymbol{\xi})\, \hat{u}(\boldsymbol{\xi})\,\Delta\boldsymbol{\xi},
$$
where the sum is over the discrete wavenumbers. This is implemented via:
- A loop over all grid points (or blocks of points) to evaluate $p(\mathbf{x},\boldsymbol{\xi})$.
- An inverse FFT for each $\mathbf{x}$? In practice, for efficiency the solver uses a block‑parallel approach: the $\mathbf{x}$‑axis is partitioned, and for each block the symbol is evaluated on the block’s spatial grid and multiplied by the FFT of $u$, then summed over $\boldsymbol{\xi}$.

### 5.3 Non‑periodic boundary conditions (Dirichlet / Neumann)

When the boundary condition is not periodic, the Fourier basis is no longer appropriate. The solver instead uses a **cosine/sine expansion** implicitly via the Kohn‑Nirenberg quantisation with a non‑periodic kernel. The operator application becomes a dense matrix‑vector product (or a fast summation if the symbol has a special structure). For stationary problems, an **asymptotic right inverse** is constructed symbolically and then applied via a similar quadrature.

## 6. Stationary Pseudo‑Differential Problems

For an equation of the form
$$
\psi\text{Op}(p)\,u = f(\mathbf{x}),
$$
with $p$ elliptic (i.e. $|p(\mathbf{x},\boldsymbol{\xi})| \ge C(1+|\boldsymbol{\xi}|)^m$ for large $|\boldsymbol{\xi}|$), the solver constructs an **asymptotic right inverse** $R$ of order $r$ such that
$$
\psi\text{Op}(p)\circ R \approx I + \text{smoother}.
$$
The symbol of $R$ is obtained by a symbolic series expansion:
$$
\sigma(R) \sim \frac{1}{p(\mathbf{x},\boldsymbol{\xi})} + \text{lower order terms in } \boldsymbol{\xi}^{-1}.
$$
Up to a user‑specified order (typically 0,1,2,3), the solver computes $R$ symbolically, lambdifies it, and applies it to $f$ using the same Kohn‑Nirenberg quantisation (or direct Fourier multiplication if the symbol is constant). The result $u = Rf$ approximates the solution.

## 7. Stability and Diagnostics

- **CFL condition**: The solver estimates the maximum group velocity $v_g = \nabla_{\mathbf{k}}\omega(\mathbf{k})$ and ensures
  $$
  \Delta t \le \frac{0.5}{\max |v_g| / \Delta x}\quad\text{(1D)},\qquad
  \Delta t \le \frac{0.5}{ \frac{|v_{g,x}|}{\Delta x} + \frac{|v_{g,y}|}{\Delta y} }\quad\text{(2D)}.
  $$
- **Symbol conditions**: The solver checks that $\operatorname{Re} L(\mathbf{k}) \le 0$ (stability), that there is sufficient high‑frequency dissipation ($\operatorname{Re} L \le -\delta |\mathbf{k}|^2$), and that the symbol grows no faster than $|\mathbf{k}|^4$ (to avoid severe stiffness).
- **Energy monitoring**: For second‑order conservative systems, the total energy
  $$
  E(t) = \frac12\int \bigl[(\partial_t u)^2 + |L^{1/2}u|^2\bigr] d\mathbf{x}
  $$
  is computed and can be plotted to check conservation/dissipation.

## 8. Boundary Conditions

| Condition | Implementation |
|-----------|----------------|
| **Periodic** | Native to the Fourier basis. The spatial grid is taken on a periodic interval, and the FFT assumes periodicity. |
| **Dirichlet** | $u=0$ at boundaries. Not directly representable by Fourier series. The solver **requires** the PDE to be written using `psiOp`. The pseudo‑differential operator’s application routine (`kohn_nirenberg_nonperiodic`) enforces the zero boundary condition by using a basis of sine functions (implicitly) and by setting the grid points at the boundaries to zero after each step. |
| **Neumann** | $\partial u/\partial n = 0$ at boundaries. Similar to Dirichlet, but the boundary values are set equal to their nearest neighbours (reflective condition) and the ψOp quantisation uses a cosine basis. |

In all non‑periodic cases, the solver falls back to an explicit Euler time step (or an exponential integrator with a spatially varying symbol that is treated as a full matrix), because the Fourier diagonalisation is lost.

## References

1. Canuto, C., Hussaini, M. Y., Quarteroni, A., & Zang, T. A. *Spectral Methods: Fundamentals in Single Domains*, Springer, 2006.
2. Trefethen, L. N. *Spectral Methods in MATLAB*, SIAM, 2000.
3. Hochbruck, M., & Ostermann, A. “Exponential integrators”, *Acta Numerica* **19**, 209–286, 2010.
4. Kassam, A.-K., & Trefethen, L. N. “Fourth‑order time‑stepping for stiff PDEs”, *SIAM J. Sci. Comput.* **26**(4), 1214–1233, 2005.
5. Kohn, J. J., & Nirenberg, L. “An algebra of pseudo‑differential operators”, *Comm. Pure Appl. Math.* **18**, 269–305, 1965.