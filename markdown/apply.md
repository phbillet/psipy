# Algorithms in `psiop.py`

This note presents the core algorithms of the Peetre-based application
pipeline using **mathematical equations** rather than pseudo-code.

---

## 1. `apply(u, x_grid, kx, …)` — operator application

### 1.1 Kohn–Nirenberg quantization

The fundamental operation applied by every backend is the
Kohn–Nirenberg integral. For a symbol $p(x,\xi)$ acting on a field
$u(x)$ in one dimension:

$$
(\mathrm{Op}(p)\,u)(x)
= \frac{1}{2\pi}\int_{\mathbb{R}} e^{i x \xi}\, p(x,\xi)\,
  \widehat{u}(\xi)\,d\xi,
\qquad
\widehat{u}(\xi) = \int_{\mathbb{R}} e^{-i y \xi}\,u(y)\,dy.
$$

In two dimensions $(x,y,\xi,\eta)$ the prefactor becomes $(2\pi)^{-2}$
and the integral extends over both frequency variables.

### 1.2 Constant-coefficient fast path

When $p$ is independent of $x$, the operator reduces to a **Fourier
multiplier**:

$$
\mathrm{Op}(p)\,u = \mathcal{F}^{-1}\!\bigl[p(\xi)\;\mathcal{F}[u](\xi)\bigr],
$$

computed in $O(N\log N)$ via the FFT. The discrete implementation is:

$$
U_m = \mathrm{FFT}(u)_m,\qquad
P_m = p(k_m),\qquad
(\mathrm{Op}(p)\,u)_j = \mathrm{IFFT}(P\cdot U)_j,
$$

where $k_m = 2\pi\,\mathrm{fftfreq}(N,\Delta x)_m$ is the angular
frequency grid in FFT ordering.

### 1.3 Weyl → Kohn–Nirenberg conversion

When the operator is Weyl-quantized, the symbol is first converted to
its Kohn–Nirenberg equivalent via the asymptotic series:

$$
p_{\mathrm{KN}}(x,\xi)
= \exp\!\Bigl(\tfrac{i}{2}\,\partial_x\partial_\xi\Bigr)\,
  p_{\mathrm{Weyl}}(x,\xi)
= \sum_{k=0}^{\mathrm{order}}
  \frac{1}{k!}\Bigl(\tfrac{i}{2}\Bigr)^{k}
  \partial_x^k \partial_\xi^k\, p_{\mathrm{Weyl}}(x,\xi).
$$

In 2D the cross-derivative operator becomes
$\partial_x\partial_\xi + \partial_y\partial_\eta$, and the
multinomial expansion distributes the $k$ differentiations between the
two pairs:

$$
\bigl(\partial_x\partial_\xi + \partial_y\partial_\eta\bigr)^k
= \sum_{j=0}^{k}\binom{k}{j}
  (\partial_x\partial_\xi)^{j}\,(\partial_y\partial_\eta)^{k-j}.
$$

### 1.4 Dispatch summary

| Condition | Path | Complexity |
|---|---|---|
| $p$ independent of $x$, periodic BC | FFT multiplier | $O(N\log N)$ |
| $p$ depends on $x$, periodic BC | `kohn_nirenberg_fft` (chunked) | $O(N^2)$ in 1D |
| $p$ depends on $x$, Dirichlet/Neumann | `kohn_nirenberg_nonperiodic` | $O(N^2)$ in 1D |
| `backend = "peetre"` | Peetre decomposition pipeline | see §2 |

---

## 2. Peetre decomposition

### 2.1 Formal decomposition

The Peetre-type decomposition splits a symbol into three structurally
distinct parts:

$$
p(x,\xi) \;=\; p_{\mathrm{local}}(x,\xi)
             \;+\; p_{\mathrm{sep}}(x,\xi)
             \;+\; p_{\mathrm{joint}}(x,\xi).
$$

**Local part** (polynomial in $\xi$, differential-type):

$$
p_{\mathrm{local}}(x,\xi)
= \sum_{\alpha} a_\alpha(x)\,\xi^{\alpha},
\qquad
a_\alpha(x) \in C^\infty.
$$

In 1D this reads
$p_{\mathrm{local}} = \sum_{n=0}^{d} a_n(x)\,\xi^n$.

**Separable part** (Fourier multipliers modulated by spatial amplitude):

$$
p_{\mathrm{sep}}(x,\xi)
= \sum_{k} a_k(x)\,q_k(\xi),
\qquad
q_k \text{ non-polynomial in } \xi.
$$

Each term is applied as
$u \mapsto a_k(x)\;\mathrm{Op}(q_k)\,u$.

**Joint residual** (genuinely entangled, not expressible as a finite
sum of $a(x)\,q(\xi)$ terms):

$$
p_{\mathrm{joint}}(x,\xi)
= p(x,\xi) - p_{\mathrm{local}}(x,\xi) - p_{\mathrm{sep}}(x,\xi).
$$

### 2.2 Classification criteria

For each additive term $t$ of the expanded symbol, the classification
proceeds by extracting the frequency-independent factor:

$$
t = a(x)\cdot q(x,\xi)
\quad\text{via}\quad
(a,\,q) = t.\texttt{as\_independent}(\xi).
$$

The decision rule is:

$$
\boxed{
\begin{aligned}
q \text{ still contains } x &\;\Longrightarrow\; t \in \texttt{joint},\\[4pt]
q \text{ is a polynomial in } \xi &\;\Longrightarrow\; t \in \texttt{local},\\[4pt]
\text{otherwise} &\;\Longrightarrow\; t \in \texttt{separable}.
\end{aligned}
}
$$

### 2.3 Application of the decomposition

The application of the Peetre decomposition to a field $u$ is:

$$
\mathrm{Op}(p)\,u
= \underbrace{\sum_{\alpha} a_\alpha(x)\;\mathrm{Op}(\xi^\alpha)\,u}_{\text{local}}
+ \underbrace{\sum_{k} a_k(x)\;\mathrm{Op}(q_k)\,u}_{\text{separable}}
+ \underbrace{\mathrm{Op}(p_{\mathrm{joint}})\,u}_{\text{joint residual}}.
$$

The joint residual is handled by one of the backends described in §3.

### 2.4 `classify_joint` option

When `peetre_decomposition(classify_joint=True)` is requested, the
auto-selector (§3.5) is run at decomposition time and the recommended
backend is stored:

$$
\texttt{result}[\texttt{"joint\_backend"}]
= \texttt{\_auto\_select\_joint\_backend}(p_{\mathrm{joint}}).
$$

This is purely symbolic — no grid, no bounds required.

---

## 3. Joint-residual backends

### 3.1 Direct backend

The joint residual is applied via the full Kohn–Nirenberg quadrature
(§1.1), with no approximation:

$$
(\mathrm{Op}(p_{\mathrm{joint}})\,u)(x)
= \frac{1}{2\pi}\int e^{ix\xi}\,p_{\mathrm{joint}}(x,\xi)\,
  \widehat{u}(\xi)\,d\xi.
$$

This is exact but costs $O(N^2)$ in 1D and $O(N^4)$ in 2D.

### 3.2 Low-rank backend (`factorize_symbolic`)

#### 3.2.1 Chebyshev grid construction

For each variable $s \in \{x_1,\ldots,\xi_1,\ldots\}$ with bounds
$[s_{\min}, s_{\max}]$, the Chebyshev–Gauss–Lobatto nodes on $[-1,1]$
are:

$$
\tau_k = \cos\!\Bigl(\frac{\pi k}{d}\Bigr),
\qquad k = 0, 1, \ldots, d,
$$

mapped affinely to the physical interval:

$$
s_k = \frac{s_{\min}+s_{\max}}{2}
    + \frac{s_{\max}-s_{\min}}{2}\,\tau_k.
$$

#### 3.2.2 Coefficient extraction

The symbol is evaluated on the tensor-product grid to form a
$d_x$-dimensional tensor $P_{\mathrm{eval}}$, then converted to
Chebyshev coefficients by inverting the Vandermonde matrix along each
axis:

$$
V_{ij} = T_j(\tau_i),
\qquad
C = V^{-1}\,P_{\mathrm{eval}}\quad\text{(applied along each axis)}.
$$

#### 3.2.3 SVD low-rank truncation

The coefficient tensor is reshaped into a matrix
$C \in \mathbb{C}^{N_x \times N_\xi}$ where
$N_x = (d+1)^{d_x}$ and $N_\xi = (d+1)^{d_\xi}$, then decomposed:

$$
C = U\,\Sigma\,V^*,
\qquad
\Sigma = \mathrm{diag}(\sigma_1, \sigma_2, \ldots).
$$

The truncation retains indices $k$ satisfying:

$$
\sigma_k > \texttt{tol}\cdot\sigma_1.
$$

The retained SVD energy fraction is:

$$
E_{\mathrm{retained}}
= 100\cdot\frac{\sum_{k\in\texttt{keep}}\sigma_k^2}
                {\sum_{k}\sigma_k^2}.
$$

#### 3.2.4 Symbolic reconstruction

Each retained singular triplet $(\sigma_k, u_k, v_k)$ produces a
separable pair:

$$
a_k(x) = \sum_{\mathbf{m}}
  \sqrt{\sigma_k}\;u_k[\mathbf{m}]\;
  \prod_{j} T_{m_j}\!\Bigl(\frac{2x_j - (x_j^{\min}+x_j^{\max})}
                                   {x_j^{\max}-x_j^{\min}}\Bigr),
$$

$$
q_k(\xi) = \sum_{\mathbf{n}}
  \sqrt{\sigma_k}\;v_k[\mathbf{n}]\;
  \prod_{j} T_{n_j}\!\Bigl(\frac{2\xi_j - (\xi_j^{\min}+\xi_j^{\max})}
                                   {\xi_j^{\max}-\xi_j^{\min}}\Bigr),
$$

where terms with $|\sqrt{\sigma_k}\,u_k[\mathbf{m}]| \le \texttt{tol}$
are dropped. The joint residual is then approximated as:

$$
p_{\mathrm{joint}}(x,\xi)
\;\approx\; \sum_{k=1}^{r} a_k(x)\,q_k(\xi).
$$

### 3.3 NUFFT backend

#### 3.3.1 Target structure

The NUFFT backend targets joint residuals of the form:

$$
p_{\mathrm{joint}}(x,\xi)
= \sum_{k} c_k(x)\;g_k(\xi)\;
  e^{\,i\,\Lambda_k(x)\,M_k(\xi)}.
$$

#### 3.3.2 Phase extraction

Each additive term is rewritten via Euler's formula
($\sin\theta = \frac{e^{i\theta}-e^{-i\theta}}{2i}$), expanded, and
factored. For a single exponential factor $e^{f}$, the exponent is
split:

$$
f = i\,\underbrace{\Lambda(x)}_{\text{spatial}}
    \cdot\underbrace{M(\xi)}_{\text{spectral}}
  + \underbrace{r(x,\xi)}_{\text{real envelope}}.
$$

The term is NUFFT-representable if and only if:

$$
\Lambda(x)\cdot M(\xi) = \mathrm{phase}(f),
\qquad
\Lambda \text{ independent of } \xi,
\qquad
M \text{ independent of } x.
$$

#### 3.3.3 Application via type-3 NUFFT

For each term, the application reduces to a type-3 non-uniform FFT:

$$
f_j = \sum_{m} w_m\; e^{\,i\,(x_j\,s_m + \Lambda(x_j)\,M(k_m))},
\qquad
w_m = g(k_m)\,\widehat{u}(k_m)\,\frac{\Delta\xi}{2\pi},
$$

where $s_m = k_m$ are the source frequencies and
$(x_j, \Lambda(x_j))$ are the target points. The result is modulated
by the spatial amplitude:

$$
(\mathrm{Op}(p_{\mathrm{joint}})\,u)(x_j)
= c(x_j)\;f_j.
$$

This is $O(N\log N)$ when `finufft` is available, or $O(N\cdot M)$
via direct summation as a fallback.

#### 3.3.4 Periodicity requirement

The NUFFT backend requires `boundary_condition = 'periodic'`. For
non-periodic boundary conditions, the executor falls back to direct
quadrature.

### 3.4 AAA backend

#### 3.4.1 Target structure

The AAA backend targets joint residuals that are **rational** or have
**algebraic decay / poles** in the frequency variable, with no
oscillatory phase:

$$
p_{\mathrm{joint}}(x,\xi)
= \frac{N(x,\xi)}{D(x,\xi)},
\qquad
\text{or}\qquad
p_{\mathrm{joint}}(x,\xi) \sim |\xi|^{-\alpha}.
$$

#### 3.4.2 Vector-valued AAA approximation

The symbol is sampled at Chebyshev nodes $\{x_j\}$ in the spatial
variable and at uniform points $\{\xi_m\}$ in the frequency variable,
producing a matrix:

$$
F_{m,j} = p(x_j,\xi_m).
$$

A barycentric rational approximation with **shared poles** across all
spatial nodes is constructed:

$$
r(\xi) = \frac{\displaystyle\sum_{\ell}
  \frac{w_\ell\,f_\ell}{\xi - z_\ell}}
  {\displaystyle\sum_{\ell}
  \frac{w_\ell}{\xi - z_\ell}},
$$

where $z_\ell$ are the support points (selected greedily by maximum
residual), $f_\ell = F_{\ell,:}$ are the corresponding row vectors,
and $w_\ell$ are the barycentric weights obtained from the null space
of the Loewner matrix.

#### 3.4.3 Quality gate

The fit is accepted if the relative $L^2$ error over a validation grid
satisfies:

$$
\frac{\|F_{\mathrm{val}} - r(\xi_{\mathrm{val}})\|_2}
     {\|F_{\mathrm{val}}\|_2}
\;\le\; 10\cdot\texttt{rtol}.
$$

If the gate fails (e.g. for **moving poles** where $z_\ell$ depends on
$x$), the executor falls back to direct quadrature.

#### 3.4.4 Application

The AAA plan is wrapped as a fast numpy callable
$\tilde{p}(x,\xi)$ and passed to the existing Kohn–Nirenberg
quadrature:

$$
(\mathrm{Op}(p_{\mathrm{joint}})\,u)(x)
= \frac{1}{2\pi}\int e^{ix\xi}\,
  \tilde{p}(x,\xi)\,\widehat{u}(\xi)\,d\xi.
$$

This supports **both** periodic and non-periodic boundary conditions.

### 3.5 Auto-selection (`_auto_select_joint_backend`)

The auto-selector applies three symbolic tests in sequence:

$$
\boxed{
\begin{aligned}
&\textbf{Test 1 (NUFFT):}\quad
\text{try\_nufft\_decomposition}(p_{\mathrm{joint}}) \neq \text{None}
\;\Longrightarrow\; \texttt{'nufft'}.\\[6pt]
&\textbf{Test 2 (AAA):}\quad
p_{\mathrm{joint}}\text{ is rational}
\;\lor\;
\exists\, (b,e)\in p_{\mathrm{joint}}.\texttt{atoms(Pow)}
\text{ with } e<0,\; b\text{ polynomial}
\;\Longrightarrow\; \texttt{'aaa'}.\\[6pt]
&\textbf{Test 3 (Lowrank):}\quad
\text{otherwise}
\;\Longrightarrow\; \texttt{'lowrank'}.
\end{aligned}
}
$$

### 3.6 Hybrid routing (`apply_hybrid`)

For symbols containing a **mix** of structural classes, monolithic
`auto` may fail to find a global pattern. The hybrid method exploits
the **linearity** of the operator:

$$
\mathrm{Op}(A + B + C)\,u
= \mathrm{Op}(A)\,u + \mathrm{Op}(B)\,u + \mathrm{Op}(C)\,u.
$$

The joint residual is split into its additive terms:

$$
p_{\mathrm{joint}} = \sum_{k} t_k,
$$

and each term $t_k$ is routed independently via `auto`:

$$
\mathrm{Op}(p_{\mathrm{joint}})\,u
= \sum_{k} \mathrm{Op}(t_k)\,u,
\qquad
\text{each } \mathrm{Op}(t_k) \text{ dispatched by }
\texttt{\_auto\_select\_joint\_backend}(t_k).
$$

This preserves $O(N\log N)$ per term, recovering the fast path across
the board even for mixed symbols.

---

## 4. Quality metrics (`evaluate_decomposition_quality`)

### 4.1 Monte Carlo sampling

Given the original symbol $p(x,\xi)$ and a candidate approximation
$\tilde{p}(x,\xi) = \sum_k a_k(x)\,q_k(\xi)$, the quality is assessed
by drawing $M$ random points uniformly over the bounding box:

$$
(x^{(j)},\xi^{(j)}) \sim \mathcal{U}(\texttt{bounds}),
\qquad j = 1,\ldots,M.
$$

### 4.2 Error metrics

$$
\texttt{rel\_l2\_error}
= \frac{\bigl\|p(\mathbf{x}^{(j)},\boldsymbol{\xi}^{(j)})
       - \tilde{p}(\mathbf{x}^{(j)},\boldsymbol{\xi}^{(j)})\bigr\|_2}
       {\bigl\|p(\mathbf{x}^{(j)},\boldsymbol{\xi}^{(j)})\bigr\|_2},
$$

$$
\texttt{max\_abs\_error}
= \max_j \bigl|p^{(j)} - \tilde{p}^{(j)}\bigr|,
$$

$$
\texttt{mean\_abs\_error}
= \frac{1}{M}\sum_{j=1}^{M}
  \bigl|p^{(j)} - \tilde{p}^{(j)}\bigr|.
$$

### 4.3 Fallback criterion

The low-rank or AAA approximation is **accepted** if:

$$
\texttt{rel\_l2\_error} \le \texttt{joint\_max\_rel\_error}.
$$

Otherwise the executor falls back to direct quadrature (§3.1).

---

## 5. Unified representation layer

### 5.1 `_resolve_joint_representation`

This method normalizes the joint residual into a **typed dict** that is
consumed identically by `apply_peetre`, `apply_hybrid`, and
`print_peetre_decomposition`:

| `rep["type"]` | backend | content | grid needed? |
|---|---|---|---|
| `"zero"` | — | (empty) | no |
| `"direct"` | direct | raw symbol | no |
| `"separable_pairs"` | lowrank | `pairs`, `metrics` | yes (bounds) |
| `"nufft_plan"` | nufft | `plan_info` | no (symbolic) |
| `"nufft_unrepresentable"` | nufft | raw symbol | no |
| `"aaa_callable"` | aaa | `symbol_func`, `metrics` | yes (bounds) |
| `"aaa_unfit"` | aaa | raw symbol | no |

### 5.2 `_apply_joint_residual`

The executor applies the representation with quality gates and
fallbacks. The decision tree is:

$$
\boxed{
\begin{aligned}
&\texttt{type} = \texttt{"zero"}
  \;\Longrightarrow\; \mathbf{0}.\\[4pt]
&\texttt{type} = \texttt{"direct"}
  \;\Longrightarrow\; \text{direct KN quadrature}.\\[4pt]
&\texttt{type} = \texttt{"separable\_pairs"}:\\
&\quad
  \text{if } \texttt{rel\_l2\_error} > \texttt{joint\_max\_rel\_error}
  \;\Longrightarrow\; \text{fallback to direct},\\
&\quad
  \text{else } \sum_k a_k(x)\;\mathrm{Op}(q_k)\,u.\\[4pt]
&\texttt{type} = \texttt{"nufft\_plan"}:\\
&\quad
  \text{if BC} \neq \texttt{periodic}
  \;\Longrightarrow\; \text{fallback to direct},\\
&\quad
  \text{else apply via type-3 NUFFT}.\\[4pt]
&\texttt{type} = \texttt{"aaa\_callable"}:\\
&\quad
  \text{if } \texttt{rel\_l2\_error} > \texttt{joint\_max\_rel\_error}
  \;\Longrightarrow\; \text{fallback to direct},\\
&\quad
  \text{else apply via KN quadrature with } \tilde{p}(x,\xi).\\[4pt]
&\texttt{type} \in \{\texttt{"nufft\_unrepresentable"},
  \texttt{"aaa\_unfit"}\}
  \;\Longrightarrow\; \text{warn, fallback to direct}.
\end{aligned}
}
$$

---

## 6. How the pieces fit together

```
apply(u, ...)
 └─ backend='peetre' ─► apply_peetre(u, ...)
                          ├─ peetre_decomposition(classify_joint=...)
                          │    └─ classify_terms → local / separable / joint
                          │    └─ (optional) auto_select → joint_backend
                          │
                          ├─ apply local & separable terms
                          │    └─ a(x) · Op(q)(u) for each (a, q) pair
                          │
                          └─ apply joint residual via _apply_joint_residual:
                               ├─ _resolve_joint_representation(...)
                               │    ├─ 'direct'  → direct type
                               │    ├─ 'lowrank' → separable_pairs type
                               │    ├─ 'nufft'   → nufft_plan / unrepresentable
                               │    └─ 'aaa'     → aaa_callable / unfit
                               │
                               └─ execute based on rep["type"]:
                                    ├─ separable_pairs → Σ a_k · Op(q_k) u
                                    ├─ nufft_plan     → type-3 NUFFT
                                    ├─ aaa_callable   → KN quadrature
                                    └─ fallback       → direct KN quadrature


apply_hybrid(u, ...)
 ├─ apply_peetre(u, ..., apply_joint=False)
 └─ for each joint term t_k:
      └─ sub_op.apply_peetre(u, ..., joint_backend='auto')
           └─ auto_select routes to optimal backend
```

---

## 7. Addendum: The Class of Approximately Separable Functions

### 7.1 Definition

A function $p(x,\xi)$ is **approximately separable** on a bounded
domain $\Omega$ if it admits a low-rank representation:

$$
p(x,\xi) \approx \sum_{k=1}^{r} a_k(x)\,q_k(\xi),
\qquad r \ll \min(N_x, N_\xi),
$$

with relative $L^2$ error at most `joint_max_rel_error`.

### 7.2 SVD energy condition

The approximation succeeds when the singular values of the Chebyshev
coefficient matrix decay rapidly:

$$
\frac{\bigl(\sum_{k>r}\sigma_k^2\bigr)^{1/2}}
     {\bigl(\sum_{k}\sigma_k^2\bigr)^{1/2}}
\;\lesssim\; \texttt{joint\_max\_rel\_error}.
$$

### 7.3 Typology

| Category | Example | Numerical rank | Error expectation | Action |
|---|---|---|---|---|
| Algebraically obfuscated | $\cos(x+\xi)$ | Exact, small | $\approx 0$ | Accept low-rank |
| Weakly coupled / smooth | $e^{-(x-\xi)^2}$ | Small | $\le$ tolerance | Accept low-rank |
| Oscillatory phase | $\sin(x\cdot\xi)$ | Very high | $>$ tolerance | Use NUFFT |
| Rational / poles | $\frac{1}{1+(x-\xi)^2}$ | Very high | $>$ tolerance | Use AAA |
| Strongly entangled | $e^{ix\xi}$ on large domains | Unbounded | $>$ tolerance | Direct fallback |

### 7.4 Backend suitability summary

| Symbol class | Best backend | Why others fail |
|---|---|---|
| Smooth Gaussian $e^{-(x\xi)^2/w}$ | `lowrank` | NUFFT: no phase; AAA: no poles |
| Rational $\frac{1}{1+x^2+\xi^2}$ | `aaa` | NUFFT: no phase; lowrank: slow convergence on peaks |
| Oscillatory $\sin(x\xi)$ | `nufft` | AAA: cannot fit oscillations; lowrank: too many terms |
| Chirp $x^2 e^{ix\xi}\cos\xi$ | `nufft` | Same as oscillatory |
| Mixed (all three) | `hybrid` | Monolithic `auto`: no global pattern → direct |
```