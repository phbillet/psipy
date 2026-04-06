# Underlying Theory of the `riemannian` Module

[Back to psipy main page](./psipy.md)

This document collects the mathematical definitions, identities, and algorithms that form the foundation of the `riemannian.py` toolkit.  
All notation follows standard Riemannian geometry references (do Carmo, Jost, Lee, Petersen).

---

## 1. Riemannian metrics

A **Riemannian metric** $g$ on an $n$-dimensional manifold $M$ assigns to each point $x\in M$ an inner product $\langle\cdot,\cdot\rangle_g$ on the tangent space $T_xM$. In local coordinates $(x^1,\dots,x^n)$ it is written as  

$$
g = g_{ij}(x) \, dx^i \otimes dx^j,\qquad 
ds^2 = g_{ij}(x)\, dx^i dx^j,
$$

where $g_{ij}(x)=\langle\partial_i,\partial_j\rangle_g$ are the covariant components.  
The **inverse metric** $g^{ij}$ satisfies $g^{ik}g_{kj}=\delta^i_{\,j}$.

For a 1‑dimensional manifold $g = g_{11}(x)\,dx\otimes dx$ (a single component).  
For a 2‑dimensional manifold $g$ is a $2\times 2$ symmetric positive‑definite matrix.

---

## 2. Christoffel symbols (second kind)

The Levi‑Civita connection is described by the Christoffel symbols  

$$
\Gamma^i_{jk} = \frac12 g^{i\ell}\bigl(\partial_j g_{k\ell} + \partial_k g_{j\ell} - \partial_\ell g_{jk}\bigr).
$$

They govern the covariant derivative of vector fields  

$$
\nabla_{\partial_j} V = \bigl(\partial_j V^i + \Gamma^i_{jk} V^k\bigr)\partial_i,
$$

and the geodesic equation (see below).

---

## 3. Geodesic equation and Hamiltonian formulation

A geodesic $\gamma(t)$ is a curve with zero covariant acceleration:

$$
\ddot\gamma^i + \Gamma^i_{jk}(\gamma)\,\dot\gamma^j\dot\gamma^k = 0.
$$

Equivalently, geodesics are the projections of Hamiltonian flow of  

$$
H(q,p) = \frac12 g^{ij}(q) p_i p_j,
$$

where $p_i = g_{ij}(q)\dot q^j$ are the canonical momenta. Hamilton’s equations  

$$
\dot q^i = \frac{\partial H}{\partial p_i},\qquad 
\dot p_i = -\frac{\partial H}{\partial q^i}
$$

preserve the symplectic form $dq^i\wedge dp_i$ and the energy $H$.

---

## 4. Curvature

### 4.1 Riemann curvature tensor

$$
R^i_{\,jkl} = \partial_k\Gamma^i_{jl} - \partial_l\Gamma^i_{jk} + \Gamma^i_{mk}\Gamma^m_{jl} - \Gamma^i_{ml}\Gamma^m_{jk}.
$$

It measures the non‑commutativity of covariant derivatives:  
$(\nabla_k\nabla_l - \nabla_l\nabla_k)V = R(\partial_k,\partial_l)V$, with $R(\partial_k,\partial_l)V = R^i_{\,jkl} V^j\partial_i$.

### 4.2 Ricci tensor and scalar curvature

$$
R_{ij} = R^k_{\,ikj},\qquad 
R = g^{ij}R_{ij}.
$$

### 4.3 Gaussian curvature (2D)

In two dimensions the Riemann tensor has only one independent component. The **Gaussian curvature** $K$ is defined by  

$$
R_{1212} = K\, \det(g),\qquad\text{so}\qquad 
K = \frac{R_{1212}}{\det(g)}.
$$

The scalar curvature $R = 2K$.

#### Brioschi formula for $K$

Using $E=g_{11},\;F=g_{12},\;G=g_{22}$,

$$
K = \frac{B-A}{(EG-F^2)^2},
$$

where  

$$
\begin{aligned}
A &= \begin{vmatrix}
0 & \frac12 E_v & \frac12 G_u \\[2pt]
\frac12 E_v & E & F \\[2pt]
\frac12 G_u & F & G
\end{vmatrix},\\[6pt]
B &= \begin{vmatrix}
-\frac12 E_{vv} + F_{uv} - \frac12 G_{uu} & \frac12 E_u & F_u - \frac12 E_v \\[2pt]
\frac12 E_v & E & F \\[2pt]
F_u - \frac12 G_u & F & G
\end{vmatrix}.
\end{aligned}
$$

All derivatives are with respect to the coordinates $(u,v)$. This formula expresses $K$ purely through the metric components and their first and second partial derivatives, avoiding Christoffel symbols.

---

## 5. Laplace–Beltrami operator

On scalar functions $f$,

$$
\Delta_g f = \frac{1}{\sqrt{|\det g|}}\; \partial_i\!\left(\sqrt{|\det g|}\; g^{ij}\,\partial_j f\right).
$$

Its **principal symbol** (in cotangent variables $\xi_i$) is  

$$
\sigma_2(\Delta_g)(x,\xi) = g^{ij}(x)\,\xi_i\xi_j,
$$

and the **subprincipal symbol** (the first‑order transport term) is  

$$
\sigma_1(\Delta_g) = \frac{1}{\sqrt{|\det g|}}\Bigl(\partial_i\bigl(\sqrt{|\det g|}\,g^{ij}\bigr)\Bigr)\xi_j.
$$

The full microlocal symbol is $\sigma_2 + i\sigma_1$.

---

## 6. Hodge theory on differential forms (2D)

### 6.1 Hodge star operator

For an oriented 2‑manifold with volume form $dV = \sqrt{|\det g|}\,dx\wedge dy$:

- On a 0‑form (function) $f$: $\star f = f\,dV$.
- On a 1‑form $\alpha = \alpha_x dx + \alpha_y dy$:  

  $$
  \star\alpha = \sqrt{|\det g|}\bigl(-g_{12}\alpha_x - g_{22}\alpha_y\bigr)dx
              + \sqrt{|\det g|}\bigl( g_{11}\alpha_x + g_{12}\alpha_y\bigr)dy.
  $$

- On a 2‑form $f\,dx\wedge dy$: $\star(f\,dx\wedge dy) = \dfrac{f}{\sqrt{|\det g|}}$.

Properties: $\star\star = (-1)^{k(2-k)}\operatorname{id}$; for $k=1$ in 2D, $\star\star = \operatorname{id}$.

### 6.2 Codifferential

For a $k$-form $\omega$, the codifferential $\delta$ is  

$$
\delta\omega = (-1)^{n(k-1)+1}\star d\star\omega,
$$

where $d$ is the exterior derivative. For a 1‑form $\alpha$ in 2D:

$$
\delta\alpha = -\frac{1}{\sqrt{|\det g|}}\;\partial_i\!\left(\sqrt{|\det g|}\;g^{ij}\alpha_j\right).
$$

### 6.3 de Rham–Hodge Laplacian

$$
\Delta = d\delta + \delta d.
$$

On **0‑forms** $\Delta = \Delta_g$ (Laplace‑Beltrami).  
On **1‑forms** the **Weitzenböck identity** holds:

$$
\Delta\alpha = \nabla^*\nabla\alpha + \operatorname{Ric}(\alpha^\sharp)^\flat,
$$

where $\nabla^*\nabla$ is the rough (connection) Laplacian acting component‑wise. In 2D, $\operatorname{Ric}(\alpha^\sharp)^\flat = K\,\alpha$ (multiplication by the Gaussian curvature). Hence  

$$
\Delta\alpha = \nabla^*\nabla\alpha + K\alpha.
$$

On **2‑forms** $\Delta(f\,dx\wedge dy) = \bigl(\Delta_g(f/\sqrt{|\det g|})\bigr)\,\sqrt{|\det g|}\,dx\wedge dy$.

### 6.4 Hodge decomposition

On a compact Riemannian manifold with (or without) boundary, every $k$-form decomposes orthogonally as  

$$
\omega = d\varphi + \delta\psi + h,
$$

where $\varphi$ is a $(k-1)$-form (exact part), $\psi$ is a $(k+1)$-form (co‑exact part), and $h$ is harmonic ($\Delta h = 0$). For the special cases in 2D:

- **0‑form** $f$: $f = \Delta u + h_0$, where $h_0$ is constant (the weighted mean of $f$).
- **1‑form** $\alpha$: $\alpha = d\varphi + \star d\psi + h$, with $\varphi,\psi$ scalar potentials solving Poisson equations.
- **2‑form** $\omega = f\,dx\wedge dy$: $\omega = d(\star d\varphi) + h$ (no co‑exact part because no 3‑forms exist).

The decomposition is unique once boundary conditions are fixed (e.g., Dirichlet for $\varphi$, Neumann for $\psi$, and a gauge‑fixing for the constant mode).

---

## 7. Geodesic deviation (Jacobi fields)

Let $\gamma(t)$ be a geodesic with tangent $v=\dot\gamma$. A Jacobi field $J(t)$ satisfies  

$$
\frac{D^2J}{dt^2} + R(J,v)v = 0,
$$

where $D/dt$ is the covariant derivative along $\gamma$. In coordinates this becomes a system of second‑order ODEs. Jacobi fields describe the linearised behaviour of nearby geodesics; vanishing of $J$ indicates a conjugate point.

---

## 8. Parallel transport

A vector $V(t)$ is **parallel** along a curve $\gamma(t)$ if  

$$
\frac{DV}{dt} = 0 \quad\Longleftrightarrow\quad 
\dot V^i + \Gamma^i_{jk}(\gamma)\,\dot\gamma^j V^k = 0.
$$

Parallel transport preserves the inner product: $\langle V(t),W(t)\rangle_g$ is constant.

---

## 9. Exponential map

For $p\in M$ and $v\in T_pM$, let $\gamma_v(t)$ be the unique geodesic with $\gamma_v(0)=p$, $\dot\gamma_v(0)=v$. The exponential map is  

$$
\exp_p(v) = \gamma_v(1).
$$

It provides a diffeomorphism from a neighbourhood of $0\in T_pM$ onto a neighbourhood of $p\in M$.

---

## 10. Gauss–Bonnet theorem

For a compact oriented 2‑manifold $M$ without boundary,

$$
\int_M K \, dA = 2\pi\,\chi(M),
$$

where $\chi(M)$ is the Euler characteristic. For a topological disk ($\chi=1$) the right‑hand side is $2\pi$.

---

## 11. Nash–Kuiper corrugation (approximate isometric embedding)

The Nash–Kuiper theorem states that any short $C^1$ embedding of a Riemannian $n$-manifold into $\mathbb{R}^{n+1}$ can be approximated arbitrarily closely by a $C^1$ isometric embedding. The construction uses **corrugations** – high‑frequency normal oscillations that fill the metric deficit.

Given a symmetric deficit tensor $\Delta g = g_{\text{target}} - g_{\text{induced}}$, one decomposes it pointwise into rank‑1 terms $\lambda\, w\otimes w$ (with $\lambda>0$). A sinusoidal normal bump  

$$
R \leftarrow R + \rho \sin\!\bigl(2\pi \nu\,\langle w,\text{coord}\rangle + \phi\bigr)\, \mathbf{n},
\qquad
\rho = \frac{\sqrt{\lambda}}{\sqrt{2}\,\pi\nu},
$$

adds, in the phase‑averaged sense, exactly $\lambda\, w\otimes w$ to the induced metric. Iterating with geometrically doubling frequencies $\nu_k = \nu_0\cdot 2^k$ converges to an isometric embedding.

---

## References

- do Carmo, M. P. *Riemannian Geometry*. Birkhäuser, 1992.  
- Jost, J. *Riemannian Geometry and Geometric Analysis*. Springer, 2011.  
- Lee, J. M. *Riemannian Manifolds: An Introduction to Curvature*. Springer, 1997.  
- Petersen, P. *Riemannian Geometry*. Springer, 2016.  
- Frankel, T. *The Geometry of Physics*. Cambridge University Press, 2011.  
- Warner, F. W. *Foundations of Differentiable Manifolds and Lie Groups*. Springer, 1983.