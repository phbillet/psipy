# Algorithms in `psiop.py`

This note extracts, as pseudocode, the four algorithms requested:

1. `PseudoDifferentialOperator.apply`
2. Peetre-style symbolic decomposition (`peetre_decomposition` + `apply_peetre`)
3. `factorize_symbolic`
4. `evaluate_decomposition_quality`

---

## 1. `apply(u, x_grid, kx, ...)` — operator application

**Goal.** Compute `(Op(p) u)` for the symbol `p(x, ξ)` of the operator, dispatching
to the cheapest numerically valid path.

```
ALGORITHM apply(u, x_grid, kx, boundary_condition, y_grid, ky,
                backend, weyl_order, apply_joint, joint_backend, ...):

    backend ← backend or self.apply_backend        # 'direct' | 'peetre'

    IF backend == "peetre":
        RETURN apply_peetre(u, x_grid, kx, ...)      # see Algorithm 2

    is_spatial ← symbol depends on x (and/or y)?

    # --- Case 1: constant-coefficient symbol on a periodic grid ---
    IF NOT is_spatial AND boundary_condition == "periodic":
        RETURN _apply_constant_fft(u, x_grid, kx, y_grid, ky, ...)
        #   U   ← FFT(u)
        #   P   ← p(ξ)  evaluated once on the FFT frequency grid
        #   P   ← clip(P, clamp); apply freq_window (gaussian/hann)
        #   RETURN IFFT(P · U)                        # O(N log N)

    # --- Effective symbol (handles Weyl → Kohn-Nirenberg correction) ---
    IF self.quantization == "weyl":
        p_eff ← weyl_to_kn_symbol(order = weyl_order)
        #   p_KN = exp(+i/2 ∂x ∂ξ) p_Weyl   (series truncated at weyl_order)
    ELSE:
        p_eff ← self.symbol
    symbol_func ← lambdify(p_eff)                    # numeric callable

    # --- Case 2: spatially varying symbol, periodic boundary ---
    IF boundary_condition == "periodic":
        RETURN kohn_nirenberg_fft(u, symbol_func, x_grid, kx, fft, ifft, ...)
        #   Fast path (symbol numerically x-independent despite is_spatial flag):
        #       U ← FFT(u);  P ← p(x0, ξ);  RETURN IFFT(P·U)
        #   Slow / memory-bounded path (genuine x-dependence):
        #       f̂(ξ) ← FFT(u) · dx                       (shifted, windowed)
        #       for each x (chunked to stay under a fixed memory budget):
        #           (Op p u)(x) = (1/2π) Σ_ξ p(x, ξ) f̂(ξ) e^{i x ξ} dξ
        #       1D: chunk over x;  2D: chunk over x AND ξ, using
        #           multi-threaded row blocks + np.einsum for the ξ-sum.
        #       clamp |p| ≤ clamp; apply freq_window; optional Gaussian
        #       space_window taper on u before transforming.

    # --- Case 3: Dirichlet / Neumann boundary (non-periodic) ---
    IF boundary_condition in {"dirichlet", "neumann"}:
        RETURN kohn_nirenberg_nonperiodic(u, x_grid, kx, symbol_func, ...)
        #   Same Kohn-Nirenberg integral as above but evaluated with a
        #   non-periodic (e.g. sine/cosine or direct quadrature) transform
        #   pair instead of the FFT, since periodic wrap-around is invalid.

    ELSE:
        RAISE ValueError("invalid boundary_condition")
```

**Key ideas**

- `apply` is a *dispatcher*: it never computes anything numerically heavy
  itself; it decides which of `_apply_constant_fft`, `kohn_nirenberg_fft`,
  `kohn_nirenberg_nonperiodic`, or `apply_peetre` should do the work.
- The choice hinges on three independent axes:
  - **backend**: `direct` (apply the full symbol) vs `peetre` (decompose first).
  - **spatial dependence**: constant symbols get the O(N log N) FFT-multiplier
    fast path; x-dependent symbols fall back to the O(N²)-type Kohn–Nirenberg
    quadrature (with a fast-path re-check and a chunked slow path to bound memory).
  - **boundary condition**: periodic uses FFTs; Dirichlet/Neumann uses a
    non-periodic quadrature variant of the same Kohn–Nirenberg integral.
- Weyl-quantized operators are always converted to an equivalent
  Kohn–Nirenberg symbol (`p_KN = exp(i/2 ∂x∂ξ) p_Weyl`, truncated at
  `weyl_order`) before any numerical kernel runs, so only one numerical
  backend needs to be maintained.

---

## 2. Peetre decomposition

Mathematically, the Peetre-type decomposition splits a symbol `p(x, ξ)` into

```
p(x, ξ) = p_local(x, ξ) + p_separable(x, ξ) + p_joint(x, ξ)
```

where

- `p_local` is **polynomial in ξ** with x-dependent coefficients
  (a genuine differential part, e.g. `(1+x²) ξ² + x ξ + V(x)`),
- `p_separable` is a finite sum `Σ a_k(x) q_k(ξ)` with `q_k` **not**
  polynomial in ξ but depending only on ξ (Fourier multipliers modulated
  by a spatial amplitude),
- `p_joint` is whatever remains genuinely entangled (not expressible as a
  finite sum of `a(x) q(ξ)` terms by simple term inspection).

### 2a. Classification (`_peetre_classify_terms`)

```
ALGORITHM classify_terms(expr):
    xi_vars ← frequency symbols in use (ξ, or ξ,η in 2D)
    expr ← expand(expr)                     # sum of monomial-like terms

    local_terms, separable, joint ← [], [], []

    FOR each additive term t in expr:
        (a, q) ← as_independent(t, xi_vars)   # t = a(x) · q(x, ξ)
        IF q still contains an x-variable:
            joint.append(t)                    # truly entangled
        ELIF q is a polynomial in xi_vars:
            local_terms.append(t)              # differential-type term
        ELSE:
            separable.append((a, q))           # Fourier-multiplier term

    # Collapse local_terms into a coefficient dictionary keyed by ξ-multi-index
    poly ← Poly(Σ local_terms, xi_vars)
    local_coeffs ← { multi_index: simplify(coeff) for each poly term }
    #  (falls back to keeping the terms as "joint" if Poly parsing fails,
    #   to avoid silently mis-attributing coefficients)

    RETURN local_coeffs, separable, joint
```

### 2b. Assembling the decomposition (`peetre_decomposition`)

```
ALGORITHM peetre_decomposition(use_cache, separable_local):
    IF cached result matches (self.symbol, separable_local): RETURN it

    local_coeffs, separable, joint ← classify_terms(self.symbol)

    separable ← merge_separable(separable)      # combine terms sharing q(ξ)
    joint     ← [expand(Σ joint)]  if nonzero else []
    drop all-zero entries from local_coeffs / separable / joint

    # Re-express the local polynomial part operationally as a(x)·q(ξ) pairs,
    # merging terms that share the same spatial coefficient a(x):
    local_terms ← local_as_separable(local_coeffs)

    IF separable_local AND local_terms:
        # legacy mode: fold local terms into "separable" and empty "local"
        separable ← merge_separable(local_terms + separable)
        local_coeffs, local_terms, local_symbol ← {}, [], 0
    ELSE:
        local_symbol ← rebuild_polynomial(local_coeffs)   # Σ coeff(x)·ξ^α

    separable_symbol ← Σ a·q over separable
    joint_symbol     ← Σ joint  (0 if empty)

    RETURN {local, local_terms, separable, joint_residual,
            local_symbol, separable_symbol, joint_symbol, separable_local}
```

`merge_separable` groups `(a, q)` pairs that share the *same* `q(ξ)` and
sums their spatial amplitudes; `local_as_separable`/`local_to_separable`
do the converse — turn each `coeff(x)·ξ^α` term into an `(a, q)` pair and
merge pairs that share the same spatial coefficient `a(x)`. Both directions
exist purely to give a single uniform operational form `a(x) · Op(q) u` for
everything that isn't in the irreducible joint residual.

### 2c. Applying the decomposition (`apply_peetre`)

```
ALGORITHM apply_peetre(u, x_grid, kx, ..., apply_joint, joint_backend):
    IF self.quantization == "weyl":
        p_eff ← weyl_to_kn_symbol(weyl_order)      # convert once
        decomposition ← peetre_decomposition(of a temporary KN operator on p_eff)
    ELSE:
        decomposition ← self.peetre_decomposition()

    result ← 0

    # 1. Local polynomial part: apply each a(x)·Op(ξ^α) as
    #        a(x) · (direct apply of the monomial suboperator to u)
    FOR (a, q) in decomposition.local_terms:
        result += a(x_grid) * PseudoDifferentialOperator(q).apply(u, ..., backend='direct')

    # 2. Separable non-local part: identical pattern
    FOR (a, q) in decomposition.separable:
        result += a(x_grid) * PseudoDifferentialOperator(q).apply(u, ..., backend='direct')

    # 3. Joint residual (irreducible x–ξ coupling)
    IF joint_symbol is nonzero:
        IF NOT apply_joint:
            warn "joint residual ignored — approximate result"
        ELIF joint_backend == "direct":
            result += PseudoDifferentialOperator(joint_symbol).apply(u, ..., backend='direct')
        ELIF joint_backend == "lowrank":
            bounds ← joint_bounds or inferred from (x_grid, kx[, y_grid, ky])
            pairs, metrics ← factorize_symbolic(joint_symbol, bounds, ...)  # Algorithm 3
            IF metrics.rel_l2_error > joint_max_rel_error:
                warn; result += direct application of joint_symbol      # fallback
            ELSE:
                FOR (a_k, q_k) in pairs:
                    result += a_k(x_grid) * PseudoDifferentialOperator(q_k).apply(u, ..., backend='direct')

    RETURN result
```

**Key idea.** Every local or separable term reduces application to
"Fourier-multiply, then pointwise-multiply by a spatial amplitude" —
cheap and exact. Only the leftover joint residual (if any) needs either
a full direct Kohn–Nirenberg quadrature, or — optionally — a further
*numerical* low-rank separable approximation via `factorize_symbolic`,
trading a controlled `rel_l2_error` for speed.

---

## 3. `factorize_symbolic` — low-rank Chebyshev/SVD decomposition of the joint residual

**Goal.** Given a symbol `p(x, ξ)` that resisted the algebraic Peetre split
(the joint residual), produce a numerically-fitted separable approximation

```
p(x, ξ) ≈ Σ_{k=1}^{r} a_k(x) · q_k(ξ)
```

valid on a bounded rectangle, by treating `p` as a matrix over a
tensor-product Chebyshev grid and truncating its SVD.

```
ALGORITHM factorize_symbolic(expr, x_syms, xi_syms, bounds,
                              degree, tol, num_samples, seed, digits):

    all_syms ← x_syms + xi_syms

    # 1. Chebyshev–Gauss–Lobatto nodes on [-1, 1] for each variable
    nodes_1d[i] ← cos(π · k / degree),  k = 0..degree      for each variable

    # 2. Affine map each variable's physical bounds [s_min, s_max] to [-1, 1]
    #    and build the corresponding physical grid points from the nodes.

    # 3. Evaluate expr on the full tensor-product grid
    mesh ← meshgrid(physical nodes for every variable)
    P_eval ← lambdify(expr)(mesh)                # (degree+1)^d tensor
    IF P_eval ≈ 0 everywhere: RETURN [], empty_metrics

    # 4. Chebyshev coefficients via per-axis Vandermonde inversion
    FOR each variable axis i:
        V_i ← chebvander(nodes_1d[i], degree)          # (degree+1)×(degree+1)
        C_tensor ← apply  inv(V_i)  along axis i        # coefficient tensor

    # 5. Reshape the coefficient tensor into a matrix:
    #        rows    = all spatial multi-indices   (size (degree+1)^{d_x})
    #        columns = all frequency multi-indices (size (degree+1)^{d_ξ})
    C_matrix ← reshape(C_tensor, (N_x_total, N_xi_total))

    # 6. SVD low-rank truncation
    U, S, Vt ← SVD(C_matrix)
    keep ← { k : S[k] > tol · S[0] }                # relative cutoff
    IF keep is empty: keep ← {0}                    # always keep leading mode
    svd_energy_retained_pct ← 100 · Σ_{k∈keep} S[k]² / Σ_k S[k]²

    # 7. Reconstruct each retained mode as a pair of symbolic polynomials
    FOR k in keep (ordered by descending singular value):
        a_k(x)  ← Σ over spatial multi-indices of
                     (√S[k] · U[:,k][idx]) · Π_m T_{deg_m}(normalized x_m)
                     — terms with |coeff| ≤ tol are dropped
        q_k(ξ)  ← same construction using V[k,:] and Chebyshev polys in ξ
        append (expand(a_k), expand(q_k)) to symbolic_pairs

    # 8. Quality diagnostics via Monte Carlo (Algorithm 4)
    metrics ← evaluate_decomposition_quality(expr, symbolic_pairs,
                                              x_syms, xi_syms, bounds,
                                              num_samples, seed)
    metrics.svd_energy_retained_pct ← svd_energy_retained_pct
    metrics.singular_values ← S[keep]

    RETURN symbolic_pairs, metrics
```

**Key idea.** This is a discrete separation-of-variables procedure:
sampling `p` on a Chebyshev tensor grid turns "is `p` (approximately)
separable?" into "does the coefficient matrix `C` (approximately) have low
rank?", answered by SVD. Each retained singular triplet `(σ_k, u_k, v_k)`
becomes one separable term, with `u_k`/`v_k` re-expanded from
Chebyshev-coefficient space back into explicit polynomials in `x` and `ξ`
respectively (weighted by `√σ_k` so that `a_k · q_k` carries the correct
magnitude). The Chebyshev basis is used (rather than monomials) for its
favorable conditioning/interpolation accuracy on `[-1, 1]`.

---

## 4. `evaluate_decomposition_quality` — Monte Carlo symbol-level error

**Goal.** Quantify how well `Σ a_k(x) q_k(ξ)` reproduces the original
expression `p(x, ξ)`, off the interpolation grid (to catch overfitting to
the Chebyshev nodes), using random sampling.

```
ALGORITHM evaluate_decomposition_quality(orig_expr, symbolic_pairs,
                                          x_syms, xi_syms, bounds,
                                          num_samples, seed):

    rng ← seeded random generator
    FOR each variable s in x_syms ∪ xi_syms:
        sample s ~ Uniform(bounds[s].min, bounds[s].max), num_samples draws

    y_orig  ← lambdify(orig_expr)(samples)                     # ground truth
    y_approx ← Σ_k  lambdify(a_k)(x-samples) · lambdify(q_k)(ξ-samples)

    diff ← y_orig - y_approx

    rel_l2_error   ← ‖diff‖₂ / ‖y_orig‖₂          (or ‖diff‖₂ if y_orig ≡ 0)
    max_abs_error  ← max |diff|
    mean_abs_error ← mean |diff|

    RETURN {rel_l2_error, max_abs_error, mean_abs_error}
```

**Key idea.** A simple, cheap statistical check: draw `num_samples` random
points uniformly over the bounding box used for the fit, evaluate both the
exact symbol and the candidate low-rank approximation there, and report
relative-L2 / max / mean absolute error. This is what `factorize_symbolic`
uses (and what `apply_peetre`'s `joint_backend="lowrank"` path checks
against `joint_max_rel_error`) to decide whether the separable
approximation is trustworthy enough to use instead of falling back to a
direct (exact) application of the joint residual.

---

## How the four pieces fit together

```
apply(u, ...)
 └─ backend='peetre' ─► apply_peetre(u, ...)
                          ├─ peetre_decomposition()      [Algorithm 2a/2b]
                          │    └─ classify_terms → local / separable / joint
                          ├─ apply local & separable terms directly (FFT-based)
                          └─ apply joint residual:
                               ├─ 'direct'  → full Kohn–Nirenberg apply()
                               └─ 'lowrank' → factorize_symbolic()   [Algorithm 3]
                                               └─ evaluate_decomposition_quality()  [Algorithm 4]
                                                  → accept fit, or fall back to 'direct'
```