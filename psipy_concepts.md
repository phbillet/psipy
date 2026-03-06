# Mathematical Concepts in `caustics.py`

This document collects all mathematical definitions, conditions, and formulas used in the `caustics.py` module.

---

## 1. Caustics and Ray Families

In semiclassical analysis, a family of rays is parameterised by an initial‑condition parameter $q \in \mathbb{R}^n$.  
The **stability matrix**

$$
J(t) = \frac{\partial x(t)}{\partial q}
$$

satisfies the variational equation

$$
\frac{dJ}{dt} = H_{px}(x(t),\xi(t)) \cdot J, \qquad J(0) = I,
$$

where $H_{px} = \frac{\partial^2 H}{\partial \xi \partial x}$ is the mixed Hessian of the Hamiltonian.

A **caustic** occurs at a time $t^*$ when

$$
\det\bigl(J(t^*)\bigr) = 0.
$$

At a caustic the ray density becomes infinite and the standard WKB approximation breaks down.

---

## 2. Maslov Index

The **Maslov index** $\mu$ is the signed number of caustic crossings along a ray:

$$
\mu = \sum_{k} \operatorname{sign}\left(\frac{d\det J}{dt}\Big|_{t_k^*}\right).
$$

Each crossing contributes a phase shift of $\pi/2$ to the wave function.  
The total Maslov phase correction factor is

$$
e^{i\mu\pi/2}.
$$

---

## 3. Arnold’s Classification of Catastrophes

Catastrophes classify the singularities of gradient maps $\nabla H = 0$.  
The classification depends on the rank of the Hessian and higher derivatives.

### 3.1 One‑dimensional case

For a function $f(\xi)$ with a critical point at $\xi_0$ (i.e. $f'(\xi_0)=0$), the type is determined by the first non‑vanishing derivative:

$$
\begin{array}{ccl}
f^{(2)}(\xi_0)\neq0 &:& \text{A1 (Morse / non‑degenerate)}\\
f^{(3)}(\xi_0)\neq0 &:& \text{A2 (Fold)}\\
f^{(4)}(\xi_0)\neq0 &:& \text{A3 (Cusp)}\\
f^{(5)}(\xi_0)\neq0 &:& \text{A4 (Swallowtail)}\\
f^{(6)}(\xi_0)\neq0 &:& \text{A5 (Butterfly)}\\
\vdots
\end{array}
$$

**Normal forms** (up to diffeomorphism):
$$
\begin{aligned}
\text{A2 (Fold)} &: \xi^3,\\
\text{A3 (Cusp)} &: \xi^4,\\
\text{A4 (Swallowtail)} &: \xi^5.
\end{aligned}
$$

### 3.2 Two‑dimensional case

Let $H(\xi,\eta)$ have a critical point at $(\xi_0,\eta_0)$.  
Let $\mathbf{H}$ be the Hessian matrix and denote its rank $r$.

#### Rank 2 – Morse
$$
\text{type} = \text{Morse (non‑degenerate)}.
$$

#### Rank 1 – $A_k$ series
Let $\mathbf{v}$ be the null direction (eigenvector of the zero eigenvalue).  
Define the directional derivative operator  

$$
D = v_x\frac{\partial}{\partial\xi} + v_y\frac{\partial}{\partial\eta}.
$$

Compute the successive derivatives $D^3H, D^4H, \dots$ at the critical point.  
The first non‑zero one determines the type:

$$
\begin{array}{ccl}
D^3H \neq 0 &:& A_2\ (\text{Fold})\\
D^4H \neq 0 &:& A_3\ (\text{Cusp})\\
D^5H \neq 0 &:& A_4\ (\text{Swallowtail})\\
D^6H \neq 0 &:& A_5\ (\text{Butterfly})\\
\vdots
\end{array}
$$

#### Rank 0 – Umbilics $D_4^\pm$
The cubic terms form a homogeneous polynomial of degree three:

$$
C(v,w) = a v^3 + 3b v^2 w + 3c v w^2 + d w^3,
$$
with  
$a = H_{\xi\xi\xi},\; b = H_{\xi\xi\eta},\; c = H_{\xi\eta\eta},\; d = H_{\eta\eta\eta}$.

The **discriminant** (or cubic invariant) is  

$$
I = 18abcd - 4b^3d + b^2c^2 - 4ac^3 - 27a^2d^2.
$$

- $I > 0$ : three distinct real roots → **D4– (Elliptic umbilic)**, normal form $\xi^3 - 3\xi\eta^2$.
- $I < 0$ : one real root → **D4+ (Hyperbolic umbilic)**, normal form $\xi^3 + \eta^3$.
- $I = 0$ : degenerate case (possibly E6 or higher).

---

## 4. Uniform Asymptotic Corrections

Near a caustic the oscillatory integral is replaced by a special function that provides a uniform approximation.

### 4.1 Fold caustic (A2) – Airy function

The Airy function is defined by the integral

$$
\operatorname{Ai}(z) = \frac{1}{2\pi} \int_{-\infty}^{\infty} e^{i(t^3/3 + z t)} dt.
$$

Uniform approximation for the wave function near a fold:

$$
u(x) \approx 2\sqrt{\pi}\; \varepsilon^{1/6}\; a_c\; |\partial_s J|^{-1/2}\;
\operatorname{Ai}\!\bigl(-\varepsilon^{-2/3}\zeta(x)\bigr)\; e^{i S_c/\varepsilon},
$$

where  
- $\varepsilon$ is the small parameter (e.g. $\hbar$),  
- $x_c$ is the caustic position,  
- $\zeta(x) = x - x_c$ measures distance to the caustic,  
- $a_c$ is the WKB amplitude at the caustic,  
- $S_c$ is the phase at the caustic,  
- $\partial_s J$ is the derivative of $\det J$ with respect to the ray parameter.

### 4.2 Cusp caustic (A3) – Pearcey integral

The Pearcey integral is

$$
P(x,y) = \int_{-\infty}^{\infty} e^{i(t^4 + x t^2 + y t)} dt.
$$

Uniform approximation near a cusp:

$$
u(x,y) \approx \varepsilon^{1/4}\; a_c\;
P\!\bigl(\varepsilon^{-1/2}(x-x_c),\; \varepsilon^{-3/4}(y-y_c)\bigr)\;
e^{i S_c/\varepsilon}.
$$

### 4.3 Swallowtail caustic (A4) – Swallowtail integral

$$
\operatorname{SW}(x,y,z) = \int_{-\infty}^{\infty} e^{i(t^5 + x t^3 + y t^2 + z t)} dt.
$$

(No explicit uniform formula is given, but the integral defines the special function for the A4 catastrophe.)

---

## 5. Phase Shift from Maslov Index

The total phase correction due to caustic crossings is

$$
\exp\!\left(i\,\frac{\mu\pi}{2}\right),
$$

where $\mu$ is the Maslov index (signed sum of zero crossings of $\det J$).

# Mathematical Concepts in `wkb.py`

This document collects all mathematical definitions, equations, and derivations used in the `wkb.py` module for multidimensional WKB approximations with caustic corrections.

---

## 1. WKB Ansatz

The Wentzel–Kramers–Brillouin (WKB) method seeks an asymptotic solution to a linear partial differential equation

$$
P(x, -i\varepsilon \nabla) u(x) = 0, \qquad \varepsilon \to 0,
$$

where $P(x,\xi)$ is the principal symbol (a smooth function of position $x$ and momentum $\xi$).  
The solution is represented as a sum over rays (bicharacteristics):

$$
u(x) \approx \sum_j A_j(x) e^{i S_j(x)/\varepsilon}.
$$

For a single ray, we write the expansion

$$
u(x) = e^{iS(x)/\varepsilon} \bigl( a_0(x) + \varepsilon a_1(x) + \varepsilon^2 a_2(x) + \cdots \bigr).
$$

---

## 2. Eikonal Equation (Order $\varepsilon^0$)

Inserting the ansatz into the equation and collecting terms of order $\varepsilon^0$ yields the **eikonal equation**

$$
P\bigl(x, \nabla S(x)\bigr) = 0.
$$

This is a Hamilton–Jacobi equation for the phase $S(x)$. It is solved by the method of characteristics (rays).

---

## 3. Hamilton’s Equations for Rays

The rays are the characteristics of the eikonal equation. They satisfy Hamilton’s equations with Hamiltonian $P$:

$$
\begin{aligned}
\frac{dx}{dt} &= \frac{\partial P}{\partial \xi},\$$4pt]
\frac{d\xi}{dt} &= -\frac{\partial P}{\partial x},\$$4pt]
\frac{dS}{dt} &= \xi \cdot \frac{\partial P}{\partial \xi} - P.
\end{aligned}
$$

Here $t$ is a parameter along the ray (often chosen as the time along the bicharacteristic).  
The initial data for the rays are given on a codimension‑1 surface (the “initial curve” or “initial surface”).

---

## 4. Transport Equations (Amplitudes)

Collecting higher orders in $\varepsilon$ gives transport equations for the amplitudes $a_k$ along each ray.

### 4.1 Leading order amplitude $a_0$

$$
\frac{\partial P}{\partial \xi} \cdot \nabla a_0 \;+\; \frac12 \bigl( \nabla_\xi\!\cdot\!\nabla_x P \bigr) a_0 = 0.
$$

Along a ray this becomes an ordinary differential equation:

$$
\frac{d a_0}{dt} + \frac12 \bigl( \nabla_\xi\!\cdot\!\nabla_x P \bigr) a_0 = 0,
$$
where the divergence term is evaluated on the ray.

### 4.2 First correction $a_1$

For order $\varepsilon^1$, the transport equation includes an additional term involving $a_0$ and the mixed derivatives of $P$:

$$
\frac{d a_1}{dt} + \frac12 (\nabla_\xi\!\cdot\!\nabla_x P) a_1 \;=\; -\frac12 \bigl( \nabla_x a_0 \cdot \nabla_\xi P + \nabla_\xi a_0 \cdot \nabla_x P \bigr) + \text{higher terms}.
$$

In the code, simplified forms are used (see the ODE implementation).

### 4.3 Higher orders $a_2, a_3, \dots$

The ODEs become progressively more complicated, involving third and higher derivatives of $P$. For example, for order $a_2$ (in 1D) the code uses

$$
\frac{d a_2}{dt} = -\frac12 (\partial_\xi^2 P)\, a_2
               -\frac18 (\partial_\xi^3 P)\, \frac{d\xi}{dt}\, a_0
               -\frac14 (\nabla_\xi\!\cdot\!\nabla_x P)\, a_1.
$$

Similar expressions are implemented for 2D and for order $a_3$.

---

## 5. Stability Matrix and Caustics

The **stability matrix** $J(t) = \frac{\partial x(t)}{\partial q}$ measures how the ray position changes with respect to the initial parameter $q$ (the label of the ray along the initial curve).  
It satisfies the variational equation obtained by differentiating Hamilton’s equations with respect to $q$:

$$
\frac{dJ}{dt} = H_{px}\bigl(x(t),\xi(t)\bigr) \cdot J, \qquad J(0) = I,
$$
where $H_{px} = \frac{\partial^2 P}{\partial \xi \partial x}$ is the mixed Hessian (a $d\times d$ matrix in $d$ dimensions).

A **caustic** occurs at a time $t^*$ when the ray density becomes infinite, i.e. when the mapping from initial parameters to positions becomes singular:

$$
\det\!\bigl(J(t^*)\bigr) = 0.
$$

The code integrates $J$ alongside each ray (in 1D a scalar $J$; in 2D a $2\times2$ matrix) so that caustics can be detected by monitoring zero crossings of $\det J$.

---

## 6. Maslov Index

The **Maslov index** $\mu$ is the signed count of caustic crossings along a ray:

$$
\mu = \sum_{k} \operatorname{sign}\!\left( \frac{d}{dt}\det J(t)\Big|_{t_k^*} \right),
$$
where $t_k^*$ are the caustic times. Each crossing contributes a phase shift of $\pi/2$ to the wave function. The total Maslov phase correction is

$$
e^{i\mu\pi/2}.
$$

The module computes $\mu$ automatically from the detected zero crossings of $\det J$.

---

## 7. Caustic Corrections (Uniform Asymptotics)

Near a caustic the standard WKB amplitude diverges. Uniform approximations replace the oscillatory exponential by a special function that remains finite and correctly describes the wave field.

### 7.1 Fold caustic (A₂) – Airy function

For a fold caustic, the uniform approximation involves the Airy function $\operatorname{Ai}$:

$$
u(x) \approx 2\sqrt{\pi}\; \varepsilon^{1/6}\; a_c\; |\partial_s J|^{-1/2}\;
\operatorname{Ai}\!\bigl(-\varepsilon^{-2/3}\zeta(x)\bigr)\; e^{i S_c/\varepsilon},
$$

where  
- $\zeta(x) = x - x_c$ measures the distance to the caustic,  
- $a_c$ and $S_c$ are the amplitude and phase at the caustic,  
- $\partial_s J$ is the derivative of $\det J$ with respect to the ray parameter (related to the rate of focusing).

In the code, a simplified version is implemented:

$$
u_{\text{corr}}(x) = a_c\; \pi\; \operatorname{Ai}\!\bigl(\varepsilon^{-2/3}(x_c - x)\bigr)\; e^{i S_c/\varepsilon}.
$$

### 7.2 Cusp caustic (A₃) – Pearcey integral

For a cusp, the uniform approximation uses the Pearcey integral

$$
P(x,y) = \int_{-\infty}^{\infty} e^{i(t^4 + x t^2 + y t)} dt.
$$

The corrected wave function near the cusp is

$$
u(x,y) \approx \varepsilon^{1/4}\; a_c\;
P\!\bigl(\varepsilon^{-1/2}(x-x_c),\; \varepsilon^{-3/4}(y-y_c)\bigr)\;
e^{i S_c/\varepsilon}.
$$

The module provides a numerical quadrature for $P(x,y)$.

### 7.3 Swallowtail caustic (A₄)

For completeness, the swallowtail integral

$$
\operatorname{SW}(x,y,z) = \int_{-\infty}^{\infty} e^{i(t^5 + x t^3 + y t^2 + z t)} dt
$$

is also defined, though not used in the current uniform approximations.

---

## 8. Numerical Implementation Details

### 8.1 State vector for ray tracing

To integrate all rays simultaneously, the ODE solver uses a flat state vector that packs the variables for each ray. The layout depends on the dimension:

**1D**  
$[x,\; \xi,\; S,\; J,\; K,\; a_0,\; a_1,\; a_2,\; \dots]$  
where $J = dx/dq$ and $K = d\xi/dq$ are coupled through the variational equations.

**2D**  
$[x,\; y,\; \xi,\; \eta,\; S,\; J_{11},J_{12},J_{21},J_{22},\; K_{11},K_{12},K_{21},K_{22},\; a_0,\; a_1,\; \dots]$  
Here $J_{ij} = \partial x_i / \partial q_j$ and $K_{ij} = \partial \xi_i / \partial q_j$ satisfy the linearised system.

### 8.2 ODE for the stability matrix (2D)

The code implements the full system:

$$
\begin{aligned}
\frac{d}{dt} J &= H_{px} \cdot J,\\
\frac{d}{dt} K &= -H_{xx} \cdot J - H_{x\xi} \cdot K,
\end{aligned}
$$

where $H_{px}$ is the mixed Hessian, $H_{xx}$ the spatial Hessian, and $H_{x\xi} = (H_{\xi x})^T$.  
The components are computed from the symbol $P$ via symbolic differentiation and lambdified for fast evaluation.

---

## 9. References

- [1] Maslov, V. P. & Fedoriuk, M. V.  *Semi‑Classical Approximation in Quantum Mechanics*, Reidel, 1981.
- [2] Duistermaat, J. J.  “Oscillatory integrals, Lagrange immersions and unfolding of singularities”, *Comm. Pure Appl. Math.* **27**, 207–281, 1974.
- [3] Berry, M. V. & Howls, C. J.  “High orders of the Weyl expansion for quantum billiards”, *Phys. Rev. E* **50**(5), 3577–3595, 1994.
- [4] Ludwig, D.  “Uniform asymptotic expansions at a caustic”, *Comm. Pure Appl. Math.* **19**, 215–250, 1966.
- [5] Kravtsov, Yu. A. & Orlov, Yu. I.  *Caustics, Catastrophes and Wave Fields*, Springer, 1999.

# Mathematical Concepts in `symplectic.py`

This document collects all mathematical definitions, equations, and algorithms used in the `symplectic.py` module for Hamiltonian mechanics and symplectic geometry.

---

## 1. Symplectic Geometry

A Hamiltonian system with $n$ degrees of freedom is defined on a **phase space** of dimension $2n$ with canonical coordinates  

$$
(z_1, \dots, z_{2n}) = (x_1, p_1, x_2, p_2, \dots, x_n, p_n).
$$

### 1.1 Symplectic Form

The canonical **symplectic 2‑form** is  

$$
\omega = \sum_{i=1}^{n} dx_i \wedge dp_i .
$$

In matrix form, with the ordering $(x_1, p_1, x_2, p_2, \dots)$, the constant matrix representing $\omega$ is block‑diagonal with $2\times2$ blocks  

$$
\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}.
$$

### 1.2 Poisson Bracket

For two functions $f, g$ on phase space, the **Poisson bracket** is  

$$
\{f, g\} = \sum_{i=1}^{n} \left( \frac{\partial f}{\partial x_i} \frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i} \frac{\partial g}{\partial x_i} \right) = (\nabla f)^T \, \omega^{-1} \, \nabla g .
$$

It satisfies antisymmetry, the Leibniz rule, and the Jacobi identity.  
For canonical variables:  

$$
\{x_i, p_j\} = \delta_{ij}, \quad \{x_i, x_j\} = 0, \quad \{p_i, p_j\} = 0.
$$

### 1.3 Hamiltonian Vector Field

Given a Hamiltonian $H(z)$, the **Hamiltonian vector field** $X_H$ is defined by  

$$
\iota_{X_H} \omega = dH.
$$

In coordinates, Hamilton’s equations are  

$$
\dot{x}_i = \frac{\partial H}{\partial p_i}, \qquad \dot{p}_i = -\frac{\partial H}{\partial x_i}.
$$

---

## 2. Hamilton’s Equations – Integration

The module provides three integration methods:

### 2.1 Symplectic Euler (first‑order)

$$
\begin{aligned}
p_i^{n+1} &= p_i^n + \Delta t \left( -\frac{\partial H}{\partial x_i}(x^n, p^n) \right),\\
x_i^{n+1} &= x_i^n + \Delta t \frac{\partial H}{\partial p_i}(x^n, p^{n+1}).
\end{aligned}
$$

(Note: the actual implementation may update momenta first, then positions using the new momenta.)

### 2.2 Velocity Verlet (second‑order)

$$
\begin{aligned}
p_i^{n+1/2} &= p_i^n + \frac{\Delta t}{2} \left( -\frac{\partial H}{\partial x_i}(x^n, p^n) \right),\\
x_i^{n+1}   &= x_i^n + \Delta t \frac{\partial H}{\partial p_i}(x^n, p^{n+1/2}),\\
p_i^{n+1}   &= p_i^{n+1/2} + \frac{\Delta t}{2} \left( -\frac{\partial H}{\partial x_i}(x^{n+1}, p^{n+1/2}) \right).
\end{aligned}
$$

### 2.3 Runge–Kutta 45 (RK45) – non‑symplectic

Standard adaptive Runge‑Kutta from `scipy.integrate.solve_ivp` for comparison.

---

## 3. Fixed Points and Linear Stability

### 3.1 Finding Fixed Points

Fixed points satisfy  

$$
\frac{\partial H}{\partial z_i} = 0 \quad \forall i.
$$

Both symbolic solving (via `sympy.solve`) and numerical root‑finding (via `scipy.optimize.fsolve`) are used.

### 3.2 Linearization

Let $z_0$ be a fixed point. The linearised equations are  

$$
\delta \dot{z} = J \, \text{Hess}(H)(z_0) \, \delta z,
$$

where $J$ is the symplectic matrix  

$$
J = \begin{pmatrix} 0 & -I \\ I & 0 \end{pmatrix} \quad \text{(in block form)},
$$

and $\text{Hess}(H)$ is the Hessian matrix of second derivatives.  
The **stability matrix** $M = J \cdot \text{Hess}(H)$ is returned; its eigenvalues determine the type:

- **Elliptic**: all eigenvalues purely imaginary (centre).
- **Hyperbolic**: all eigenvalues real (or come in opposite real pairs, saddle).
- **Mixed**: otherwise (partially hyperbolic/elliptic).

---

## 4. 1‑Degree‑of‑Freedom Systems ($n=1$)

For a 1‑DOF Hamiltonian $H(x, p)$, the phase space is two‑dimensional.

### 4.1 Action Integral

The **action variable** (adiabatic invariant) is  

$$
I(E) = \frac{1}{2\pi} \oint p \, dx,
$$

where the integral is taken over a closed orbit at energy $E$.  
For a periodic orbit, $p$ is obtained from $H(x,p)=E$ as $p(x,E)$.  
The integral is computed numerically or symbolically (if possible).

### 4.2 Action‑Angle Variables

For an integrable system, one can transform to action‑angle coordinates $(I, \theta)$ such that  

$$
H = H(I), \qquad \dot{\theta} = \omega(I) = \frac{dH}{dI}.
$$

The frequency $\omega(I)$ is computed via differentiation of $H(I)$.

### 4.3 Phase Portrait

The level sets of $H$ are plotted as contours. The vector field $( \dot{x}, \dot{p} ) = (\partial H/\partial p, -\partial H/\partial x)$ is overlaid.

### 4.4 Separatrix Analysis

Near a hyperbolic fixed point (saddle), the **stable and unstable manifolds** are computed by integrating a small perturbation along the eigenvectors of the linearised system forward/backward in time.

---

## 5. 2‑Degree‑of‑Freedom Systems ($n=2$)

For $n=2$, the phase space is four‑dimensional. Several specialised tools are provided.

### 5.1 Poincaré Section

A **Poincaré section** is a lower‑dimensional surface transverse to the flow, defined by a condition  

$$
\Sigma = \{ z \mid \phi(z) = \text{const.} \},
$$

e.g., $x_2 = 0$ with a chosen direction (positive crossing).  
The section points are obtained by interpolating the trajectory when the condition is met.

### 5.2 First Return Map

Given a sequence of section points $\{z_k\}$, the **first return map** (or Poincaré map) is  

$$
\mathcal{P}(z_k) = z_{k+1}.
$$

Often one plots e.g. $x_1$ vs. $p_1$ of successive intersections to reveal regular or chaotic dynamics.

### 5.3 Monodromy Matrix (Floquet Multipliers)

For a periodic orbit of period $T$, the **monodromy matrix** $M$ is the linearisation of the return map after one period:

$$
M = \frac{\partial z(T)}{\partial z(0)}.
$$

Its eigenvalues (Floquet multipliers) determine the stability of the orbit:

- All eigenvalues on the unit circle → stable (elliptic).
- Any eigenvalue outside → unstable (hyperbolic).

The module computes $M$ via finite‑difference perturbations.

### 5.4 Lyapunov Exponents

The (maximal) **Lyapunov exponent** quantifies the average exponential divergence of nearby trajectories:

$$
\lambda = \lim_{t\to\infty} \frac{1}{t} \ln \frac{\|\delta z(t)\|}{\|\delta z(0)\|}.
$$

A simplified estimation is performed by evolving a set of perturbation vectors with periodic Gram–Schmidt orthonormalisation.

---

## 6. Projections of 4D Trajectories

For visualisation, a 4D trajectory can be projected onto a 2D plane, e.g.:

- **Configuration space** $(x_1, x_2)$
- **Momentum plane** $(p_1, p_2)$
- **Mixed planes** $(x_1, p_1)$ or $(x_2, p_2)$ etc.

---

## 7. Poisson Bracket Computation

The module computes $\{f, g\}$ symbolically using  

$$
\{f, g\} = \sum_{i=1}^{n} \left( \frac{\partial f}{\partial x_i} \frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i} \frac{\partial g}{\partial x_i} \right).
$$

---

## 8. Variable Inference

The module attempts to infer phase space variables from the Hamiltonian expression by looking for symbols named `x`, `p`, `x1`, `p1`, etc. If ambiguous, the user must provide the variable list explicitly.

---

## 9. References

- [1] Arnold, V. I. *Mathematical Methods of Classical Mechanics*, Springer‑Verlag, 1989.
- [2] Goldstein, H., Poole, C., & Safko, J. *Classical Mechanics*, Addison‑Wesley, 2002.
- [3] Hairer, E., Lubich, C., & Wanner, G. *Geometric Numerical Integration*, Springer, 2006.
- [4] Lichtenberg, A. J., & Lieberman, M. A. *Regular and Chaotic Dynamics*, Springer, 1992.
- [5] Meyer, K. R., Hall, G. R., & Offin, D. *Introduction to Hamiltonian Dynamical Systems and the N‑Body Problem*, Springer, 2009.

# Mathematical Concepts in `riemannian.py`

This document collects all mathematical definitions, equations, and algorithms used in the `riemannian.py` module for Riemannian geometry in one and two dimensions.

---

## 1. Riemannian Metric

A **Riemannian metric** on an $n$-dimensional manifold assigns an inner product to each tangent space. In local coordinates $(x^1,\dots,x^n)$ the metric is written as

$$
ds^2 = g_{ij}(x) \, dx^i dx^j,
$$

where $g_{ij}$ is a symmetric positive‑definite matrix (the metric tensor). The inverse metric is denoted $g^{ij}$ with $g^{ik}g_{kj} = \delta^i_j$.

The module supports two dimensions:

- **1D**: $g_{11}(x)$ (a scalar function).
- **2D**: a $2\times2$ matrix $g_{ij}(x,y)$.

The **determinant** is $|g| = \det(g_{ij})$ and its square root $\sqrt{|g|}$ gives the Riemannian volume element.

---

## 2. Christoffel Symbols

The **Christoffel symbols** of the Levi‑Civita connection are

$$
\Gamma^i_{\,jk} = \frac{1}{2} g^{i\ell}\left( \partial_j g_{k\ell} + \partial_k g_{j\ell} - \partial_\ell g_{jk} \right).
$$

In **1D** this simplifies to

$$
\Gamma^1_{\,11} = \frac{1}{2} (\log g_{11})'.
$$

The module computes them symbolically and provides numerical evaluation functions.

---

## 3. Geodesic Equation

A curve $\gamma(t)$ is a geodesic if it satisfies

$$
\ddot{x}^i + \Gamma^i_{\,jk} \dot{x}^j \dot{x}^k = 0.
$$

In **1D** this becomes

$$
\ddot{x} + \Gamma^1_{\,11}(x) \dot{x}^2 = 0.
$$

For numerical integration the module offers several methods:

- Standard ODE solvers (`rk4`, `rk45`, `adaptive`) applied to the second‑order system.
- **Symplectic integrators** (Verlet, symplectic Euler) for the Hamiltonian formulation of geodesic flow (see below).

### 3.1 Hamiltonian formulation

Geodesics can also be obtained from the Hamiltonian

$$
H(x,p) = \frac{1}{2} g^{ij}(x) \, p_i p_j,
$$

where $p_i = g_{ij} \dot{x}^j$ are the conjugate momenta. Hamilton’s equations are

$$
\dot{x}^i = \frac{\partial H}{\partial p_i} = g^{ij} p_j,\qquad
\dot{p}_i = -\frac{\partial H}{\partial x^i} = -\frac12 \frac{\partial g^{jk}}{\partial x^i} p_j p_k.
$$

This formulation is used by `geodesic_hamiltonian_flow`, which calls the symplectic integrators from the companion `symplectic` module.

### 3.2 Arc‑length parametrisation

For a geodesic, the arc length $s(t)$ is obtained from

$$
\frac{ds}{dt} = \sqrt{g_{ij}\dot{x}^i\dot{x}^j}.
$$

If `reparametrize=True`, the output includes an array `arc_length` computed by cumulative trapezoidal integration.

---

## 4. Curvature

### 4.1 Riemann curvature tensor

For a 2D manifold, the Riemann tensor $R^i_{\,jkl}$ is computed from the Christoffel symbols:

$$
R^i_{\,jkl} = \partial_k \Gamma^i_{\,jl} - \partial_l \Gamma^i_{\,jk} + \Gamma^i_{\,mk} \Gamma^m_{\,jl} - \Gamma^i_{\,ml} \Gamma^m_{\,jk}.
$$

The module stores it as a nested dictionary `R[i][j][k][l]`. (In 1D the tensor is identically zero.)

### 4.2 Ricci tensor

$$
R_{ij} = R^k_{\,ikj}.
$$

### 4.3 Scalar curvature

$$
R = g^{ij} R_{ij}.
$$

### 4.4 Gaussian curvature (2D only)

For a surface, the Gaussian curvature $K$ is related to the Riemann tensor by

$$
R_{1212} = K \, |g|,
$$

and the module computes $K = R_{1212} / |g|$.

---

## 5. Laplace–Beltrami Operator

The Laplace–Beltrami operator acting on functions is

$$
\Delta = \frac{1}{\sqrt{|g|}} \partial_i \left( \sqrt{|g|}\, g^{ij} \partial_j \right).
$$

Its **symbol** (for microlocal analysis) is given in the module as

$$
\sigma(\Delta) = \underbrace{g^{ij} \xi_i \xi_j}_{\text{principal}} \;+\; \underbrace{i\, \frac{1}{\sqrt{|g|}} \partial_i\!\left( \sqrt{|g|}\, g^{ij} \right) \xi_j}_{\text{subprincipal}}.
$$

In 1D this simplifies to

$$
\sigma(\Delta) = g^{11} \xi^2 \;+\; i\, \frac{(\sqrt{g_{11}})'}{\sqrt{g_{11}}} g^{11} \xi.
$$

The method `laplace_beltrami_symbol()` returns a dictionary with the principal, subprincipal, and full (complex) symbols.

---

## 6. Riemannian Volume and Arc Length

The **Riemannian volume** of a domain is

$$
\text{Vol} = \int \sqrt{|g|}\, dx^1\cdots dx^n.
$$

In 1D this is the arc length: $\int \sqrt{g_{11}}\, dx$.

The module provides both symbolic integration (using SymPy) and numerical integration (using `scipy.integrate.quad` or `dblquad`).

---

## 7. 1D‑Specific Tools

### 7.1 Sturm–Liouville reduction

The eigenvalue problem $-\Delta u + V u = \lambda u$ can be written in Sturm–Liouville form

$$
-(p\, u')' + q u = \lambda w u,
$$

with

$$
p = \sqrt{g_{11}}\, g^{11},\qquad
w = \sqrt{g_{11}},\qquad
q = V\sqrt{g_{11}}.
$$

The function `sturm_liouville_reduce()` returns the symbolic expressions and corresponding numerical functions.

---

## 8. 2D‑Specific Tools

### 8.1 Exponential map

The **exponential map** $\exp_p(tv)$ sends a tangent vector $v$ at $p$ to the point reached after time $t$ along the geodesic starting at $p$ with velocity $v$. It is computed by integrating the geodesic equation.

### 8.2 Geodesic distance

The distance between two points $p,q$ is obtained by solving for the velocity $v$ such that $\exp_p(v) = q$. Two methods are implemented:

- **Shooting**: iteratively adjust $v$ to minimise the error in the target point.
- **Optimisation**: minimise the energy $\frac12 g_{ij}(p) v^i v^j$ with a penalty for the target error.

### 8.3 Jacobi equation (geodesic deviation)

The Jacobi field $J(t)$ measures the linearised deviation between nearby geodesics. It satisfies

$$
\frac{D^2 J^i}{dt^2} + R^i_{\,jkl} \dot{\gamma}^j J^k \dot{\gamma}^l = 0.
$$

The module solves this equation numerically for given initial $J(0)$ and $\frac{DJ}{dt}(0)$ along a previously computed geodesic.

### 8.4 Hodge star operator

On a 2D oriented Riemannian manifold, the Hodge star $\star$ maps $k$-forms to $(2-k)$-forms:

- On a 0‑form $f$: $\star f = f \sqrt{|g|}\, dx^1\wedge dx^2$.
- On a 1‑form $\alpha = \alpha_1 dx^1 + \alpha_2 dx^2$:
  $$
  \star \alpha = \sqrt{|g|}\, g^{ij} \alpha_j \, \varepsilon_{ik} dx^k,
  $$
  where $\varepsilon$ is the Levi‑Civita symbol. Explicitly,
  $$
  \star\alpha = \sqrt{|g|}\bigl( g^{11}\alpha_2 - g^{12}\alpha_1 \bigr) dx^1
                + \sqrt{|g|}\bigl( -g^{12}\alpha_2 + g^{22}\alpha_1 \bigr) dx^2.
  $$
- On a 2‑form $\omega = f \, dx^1\wedge dx^2$: $\star\omega = f / \sqrt{|g|}$.

The function `hodge_star(metric, form_degree)` returns a callable implementing the map.

### 8.5 de Rham Laplacian on forms

The Hodge Laplacian on $k$-forms is $\Delta = d\delta + \delta d$. For 0‑forms it coincides with the Laplace–Beltrami operator. For 1‑forms on a 2D manifold, the principal symbol is the same as for 0‑forms (i.e. $g^{ij}\xi_i\xi_j$), but the subprincipal term may differ. The module provides a placeholder for future extensions.

### 8.6 Gauss–Bonnet theorem verification

For a compact oriented 2‑manifold without boundary,

$$
\int_M K \, dA = 2\pi \chi(M),
$$

where $\chi(M)$ is the Euler characteristic. The function `verify_gauss_bonnet()` numerically integrates $K \sqrt{|g|}$ over a given domain and compares with $2\pi$ (assuming the domain is topologically a sphere or a region that should yield this value).

---

## 9. Visualisation Helpers

The module includes functions to plot:

- **1D**: metric component, Christoffel symbol, and geodesic trajectories (with optional colouring by speed, time, or curvature).
- **2D**: geodesic trajectories overlaid on a colour map of Gaussian or scalar curvature.

These are intended for quick inspection and do not contain additional mathematical content beyond the data already described.

---

## 10. References

- [1] do Carmo, M. P. *Riemannian Geometry*, Birkhäuser, 1992.
- [2] Jost, J. *Riemannian Geometry and Geometric Analysis*, Springer, 2011.
- [3] Lee, J. M. *Riemannian Manifolds: An Introduction to Curvature*, Springer, 1997.
- [4] Petersen, P. *Riemannian Geometry*, Springer, 2016.
- [5] Frankel, T. *The Geometry of Physics*, Cambridge University Press, 2011.

# Mathematical Concepts in `psiop.py`

This document collects all mathematical definitions, equations, and algorithms used in the `psiop.py` module for pseudo‑differential operators in one and two dimensions.

---

## 1. Pseudo‑differential Operators – Definition

A pseudo‑differential operator `P` acting on functions of `x ∈ ℝⁿ` is defined by its **symbol** `p(x,ξ)`, a function on phase space `T*ℝⁿ`. The action on a function `u` is given by the **Kohn‑Nirenberg quantisation**

$$
(P u)(x) = (2\pi)^{-n} \int_{\mathbb{R}^n} e^{i x \cdot \xi} \, p(x,\xi) \, \hat{u}(\xi) \, d\xi,
$$

where $\hat{u}(\xi) = \int e^{-i x\cdot\xi} u(x) dx$ is the Fourier transform of `u`. If the symbol does not depend on `x`, the operator reduces to a Fourier multiplier.

For a differential operator, the symbol is obtained by replacing derivatives with frequencies: e.g., for $P = a(x,D) = \sum_{|\alpha|\le m} a_\alpha(x) D^\alpha$, the symbol is $p(x,\xi) = \sum a_\alpha(x) \xi^\alpha$ with $D^\alpha = (-i\partial)^\alpha$.

In the module, the operator can be constructed in two modes:

- **`mode='symbol'`**: directly from an explicit symbol expression.
- **`mode='auto'`**: automatically extracts the symbol by applying the differential expression to the complex exponential $e^{i x\cdot\xi}$ and dividing by it:

$$
p(x,\xi) = \frac{P(e^{i x\cdot\xi})}{e^{i x\cdot\xi}}.
$$

---

## 2. Asymptotic Analysis

### 2.1 Principal Symbol

The **principal symbol** is the leading homogeneous component of the symbol as $|\xi|\to\infty$. For a symbol of order $m$,

$$
p(x,\xi) = p_m(x,\xi) + \text{lower order terms},
$$

where $p_m(x,\xi)$ is homogeneous of degree $m$ in $\xi$. The method `principal_symbol(order)` returns the leading term of the expansion in $1/|\xi|$ up to the specified order. In 2D, the expansion is performed in polar coordinates $(\rho,\theta)$ with $\rho = |(\xi,\eta)|$ and then converted back.

### 2.2 Symbol Order

The **order** of the operator is the degree of growth of the symbol as $|\xi|\to\infty$. It is estimated by examining the asymptotic behaviour of the symbol. The method `symbol_order()` attempts to determine this order using series expansions in $1/|\xi|$ (or in $1/z$ after substituting $\xi = 1/z$).

### 2.3 Asymptotic Expansion

The full asymptotic expansion of the symbol at infinity is given by

$$
p(x,\xi) \sim \sum_{j=0}^{\infty} p_{m-j}(x,\xi),
$$

where each $p_{m-j}$ is homogeneous of degree $m-j$. The method `asymptotic_expansion(order)` computes this series up to a given order.

---

## 3. Symbolic Calculus – Composition

### 3.1 Kohn‑Nirenberg Composition

For two symbols $p$ and $q$, the symbol of the composition $P \circ Q$ in Kohn‑Nirenberg quantisation is given by the asymptotic series

$$
(p \circ q)(x,\xi) \sim \sum_{\alpha} \frac{i^{-|\alpha|}}{\alpha!} \,\partial_\xi^\alpha p(x,\xi) \,\partial_x^\alpha q(x,\xi).
$$

In 1D:

$$
(p \circ q)(x,\xi) = \sum_{n=0}^\infty \frac{i^{-n}}{n!} \,\partial_\xi^n p(x,\xi) \,\partial_x^n q(x,\xi).
$$

In 2D, with multi‑index $\alpha = (\alpha_1,\alpha_2)$:

$$
(p \circ q)(x,y,\xi,\eta) = \sum_{\alpha_1,\alpha_2=0}^\infty \frac{i^{-(\alpha_1+\alpha_2)}}{\alpha_1!\,\alpha_2!}
\,\partial_\xi^{\alpha_1}\partial_\eta^{\alpha_2} p \;\partial_x^{\alpha_1}\partial_y^{\alpha_2} q.
$$

The method `compose_asymptotic(other, order, mode='kn')` truncates this series at a given order.

### 3.2 Weyl Star Product

In Weyl quantisation, the composition symbol is given by the **Moyal star product**:

$$
(p \# q)(x,\xi) = \exp\left( \frac{i}{2} \big( \partial_\xi \partial_y - \partial_x \partial_\eta \big) \right) p(x,\xi) q(y,\eta) \big|_{y=x,\eta=\xi}.
$$

The series expansion is

$$
(p \# q)(x,\xi) = \sum_{n=0}^\infty \frac{1}{n!}\left(\frac{i}{2}\right)^n \big( \partial_\xi \partial_y - \partial_x \partial_\eta \big)^n p(x,\xi) q(y,\eta) \big|_{y=x,\eta=\xi}.
$$

In 1D, with $\Lambda = \partial_\xi \partial_y - \partial_x \partial_\eta$,

$$
(p \# q) = \sum_{n=0}^\infty \frac{1}{n!}\left(\frac{i}{2}\right)^n \Lambda^n(p\,q).
$$

In 2D, the analogous expansion involves mixed derivatives.

The method `compose_asymptotic(other, order, mode='weyl')` implements this truncated series.

### 3.3 Commutator

The symbol of the commutator $[P,Q] = P Q - Q P$ is obtained by subtracting the compositions:

$$
\sigma_{[P,Q]}(x,\xi) = (p \circ q)(x,\xi) - (q \circ p)(x,\xi).
$$

The leading term is proportional to the Poisson bracket:

$$
\sigma_{[P,Q]}(x,\xi) = i \{p,q\}(x,\xi) + \text{lower order},
$$

where $\{p,q\} = \sum \big( \partial_\xi p \,\partial_x q - \partial_x p \,\partial_\xi q \big)$.

The method `commutator_symbolic(other, order)` computes the asymptotic expansion of the commutator symbol.

---

## 4. Formal Inverses

### 4.1 Right Inverse

A formal right inverse $R$ satisfies $P \circ R = I$ modulo a smoothing operator. It is constructed recursively:

$$
R_0 = \frac{1}{p},\qquad
R = R_0 - R_0 \sum_{k\ge 1} \frac{i^{-k}}{k!} \big( \partial_\xi^k p \,\partial_x^k R \big).
$$

The method `right_inverse_asymptotic(order)` implements this recursion.

### 4.2 Left Inverse

Similarly, a left inverse $L$ satisfies $L \circ P = I$. The recursion is

$$
L_0 = \frac{1}{p},\qquad
L = L_0 - \sum_{k\ge 1} \frac{i^{-k}}{k!} \big( \partial_\xi^k L \,\partial_x^k p \big) L_0.
$$

Implemented in `left_inverse_asymptotic(order)`.

---

## 5. Formal Adjoint

The symbol of the formal adjoint $P^*$ (with respect to the $L^2$ inner product) is given by an asymptotic expansion:

$$
p^*(x,\xi) \sim \sum_{\alpha} \frac{i^{-|\alpha|}}{\alpha!} \,\partial_\xi^\alpha \overline{\partial_x^\alpha p}(x,\xi).
$$

In 1D, to leading order:

$$
p^*(x,\xi) = \overline{p}(x,\xi) + \text{lower order terms}.
$$

The method `formal_adjoint()` returns the adjoint symbol after a high‑frequency expansion.

---

## 6. Exponential of an Operator

The symbol of $\exp(t P)$ can be expanded asymptotically for small $t$ using the series

$$
\exp(t P) \sim \sum_{n=0}^\infty \frac{t^n}{n!} P^n.
$$

Each power $P^n$ is approximated by successive compositions. The method `exponential_symbol(t, order)` returns the truncated series symbol.

---

## 7. Semiclassical Trace Formula

The **semiclassical trace** of a pseudo‑differential operator is given by the phase‑space integral of its symbol:

$$
\operatorname{Tr}(P) = \frac{1}{(2\pi)^n} \int\!\!\int p(x,\xi) \, dx \, d\xi.
$$

This formula is exact for trace‑class operators and provides an asymptotic approximation in general. The method `trace_formula(volume_element, numerical, x_bounds, xi_bounds)` computes this integral either symbolically or numerically.

---

## 8. Hamiltonian Flow from the Symbol

The **Hamiltonian vector field** associated with the principal symbol $p(x,\xi)$ generates the bicharacteristic flow:

$$
\frac{dx}{dt} = \frac{\partial p}{\partial \xi},\qquad
\frac{d\xi}{dt} = -\frac{\partial p}{\partial x}.
$$

In 2D:

$$
\frac{dx}{dt} = \frac{\partial p}{\partial \xi},\quad
\frac{dy}{dt} = \frac{\partial p}{\partial \eta},\quad
\frac{d\xi}{dt} = -\frac{\partial p}{\partial x},\quad
\frac{d\eta}{dt} = -\frac{\partial p}{\partial y}.
$$

The method `symplectic_flow()` returns these expressions as a dictionary. The method `plot_hamiltonian_flow()` integrates these equations and visualises the trajectories.

---

## 9. Pseudospectrum

For an operator $P$, the **ε‑pseudospectrum** is the set

$$
\sigma_\varepsilon(P) = \big\{ \lambda \in \mathbb{C} \mid \|(P-\lambda I)^{-1}\| \ge \varepsilon^{-1} \big\}.
$$

Equivalently, it can be characterised via the smallest singular value:

$$
\|(P-\lambda I)^{-1}\| = \frac{1}{\sigma_{\min}(P-\lambda I)}.
$$

The method `pseudospectrum_analysis()` discretises the operator on a spatial grid, builds a matrix $H$, and computes $\sigma_{\min}(H-\lambda I)$ over a grid of $\lambda$. The resolvent norm is then $\|(H-\lambda I)^{-1}\| = 1/\sigma_{\min}$. Contours of constant $\varepsilon$ are plotted.

---

## 10. Visualisation Concepts

Several visualisation methods are based on geometric properties of the symbol:

- **Cotangent fiber**: $p(x_0,\xi)$ as a function of $\xi$ at fixed $x_0$.
- **Characteristic set**: $\{ (x,\xi) \mid p(x,\xi) \approx 0 \}$, approximated by contour lines of $|p|$.
- **Micro‑support**: estimate of the region where the symbol is not elliptic, often shown as $1/|p|$.
- **Group velocity field**: $\nabla_\xi p(x,\xi)$.
- **Symplectic vector field**: $(\nabla_\xi p, -\nabla_x p)$.

These are primarily numerical and visual, but they illustrate the underlying microlocal structure.

---

## 11. Kohn‑Nirenberg Quantisation (Numerical Implementation)

The module provides two numerical implementations of the Kohn‑Nirenberg quantisation:

### 11.1 Periodic case – FFT based

For periodic boundary conditions, the operator is applied via

$$
(P u)(x) = \frac{1}{(2\pi)^n} \int e^{i x\cdot\xi} p(x,\xi) \hat{u}(\xi) d\xi,
$$

where $\hat{u}$ is the FFT of $u$. The integral is discretised and computed using FFTs. The symbol is evaluated on the full spatial and frequency grids; the 4‑D tensor in 2D is handled block‑wise to save memory.

### 11.2 Non‑periodic case – matrix multiplication

For non‑periodic domains, the Fourier transform is approximated by a discrete Fourier transform on a bounded interval (non‑periodic). The operator is applied as

$$
(P u)(x) \approx \frac{\Delta\xi}{2\pi} \sum_{\xi} e^{i x\xi} p(x,\xi) \hat{u}(\xi),
$$

where $\hat{u}(\xi) = \Delta x \sum_x e^{-i\xi x} u(x)$. This is implemented as matrix‑vector products. A cache stores pre‑computed phase matrices to speed up repeated applications.

---

## 12. References

- [1] Hörmander, L. *The Analysis of Linear Partial Differential Operators III*, Springer, 1985.
- [2] Taylor, M. E. *Pseudo Differential Operators*, Princeton University Press, 1981.
- [3] Zworski, M. *Semiclassical Analysis*, American Mathematical Society, 2012.
- [4] Martinez, A. *An Introduction to Semiclassical and Microlocal Analysis*, Springer, 2002.
- [5] Trefethen, L. N. & Embree, M. *Spectra and Pseudospectra*, Princeton University Press, 2005.

# Mathematical Concepts in `asymptotic.py`

This document collects all mathematical definitions, formulas, and algorithms used in the `asymptotic.py` module for large‑parameter asymptotics of oscillatory and Laplace‑type integrals.

---

## 1. Asymptotic Integrals – Overview

The module handles integrals of the form

$$
I(\lambda) = \int a(x)\, e^{i\lambda\varphi(x)}\, dx,\qquad \lambda\to +\infty,
$$

where $x\in\mathbb{R}^n$, $a(x)$ is an amplitude (complex‑valued) and $\varphi(x)$ is a phase function. Depending on the nature of $\varphi$, three classical methods apply:

| $\varphi$          | Integral type           | Method               |
|----------------------|-------------------------|----------------------|
| Purely real          | oscillatory             | **Stationary phase** |
| Purely imaginary     | exponentially damped    | **Laplace**          |
| Genuinely complex    | oscillatory + damped    | **Saddle‑point** (steepest descent) |

The module automatically detects the method from the symbolic expression of $\varphi$ (or falls back to a numerical test).

---

## 2. Classification of Critical Points

A critical point $x_c$ satisfies $\nabla\varphi(x_c)=0$. Its type determines the asymptotic formula.

### 2.1 Morse (non‑degenerate)

$$
\det\nabla^2\varphi(x_c) \neq 0.
$$

The Hessian matrix $H = \nabla^2\varphi(x_c)$ is invertible. The **signature** $\sigma$ (number of negative eigenvalues) defines the Maslov index $\mu = n - 2\sigma$.

### 2.2 Degenerate (corank 1)

In one dimension, if $\varphi''(x_c)=0$:

- If $\varphi'''(x_c)\neq 0$ → **Airy (A₂)** singularity.
- If $\varphi'''(x_c)=0$ and $\varphi^{(4)}(x_c)\neq 0$ → **Pearcey (A₃)** singularity.

In two dimensions, rank‑1 Hessian (one zero eigenvalue). The classification uses directional derivatives along the null eigenvector:

- If cubic term ≠ 0 → **Airy‑2D**.
- If cubic = 0, quartic ≠ 0 → **Pearcey**.

Higher‑order degeneracies are marked as `HIGHER_ORDER` and not implemented.

---

## 3. Stationary Phase (Oscillatory Integrals, $\varphi$ real)

### 3.1 Leading term – Morse point

$$
I_0(\lambda) = \left(\frac{2\pi}{\lambda}\right)^{n/2}
\frac{a(x_c)\, e^{i\lambda\varphi(x_c)}\, e^{i\pi\mu/4}}
{\sqrt{|\det H|}},
$$

where $\mu = n - 2\sigma$ is the Morse index.

### 3.2 Second‑order correction – Morse point

$$
I_1(\lambda) = I_0(\lambda)\,\frac{1}{i\lambda}\, C,
$$

with

$$
C = \frac12\operatorname{tr}\!\big(H^{-1}\nabla^2 a\big)
   - \frac12 \langle H^{-1}\nabla a,\; V\rangle
   + \frac{a(x_c)}{24}\big(5 S_3 - 3 S_4\big),
$$

where  
$V_k = \sum_{i,j} (H^{-1})_{ij}\,\partial_{ijk}\varphi$,  
$S_4 = \sum_{i,j,k,l} (H^{-1})_{ij}(H^{-1})_{kl}\,\partial_{ijkl}\varphi$,  
$S_3 = \sum_{i,j,k,l,m,n} (H^{-1})_{ij}(H^{-1})_{kl}(H^{-1})_{mn}\,
       \partial_{ikm}\varphi\;\partial_{jln}\varphi$.

### 3.3 Airy (1D) – degenerate cubic

Normal form: $\varphi \sim \alpha\, u^3/3$ with $\alpha = \frac12\varphi'''(x_c)$.  
Asymptotic contribution (exact in the canonical case):

$$
I(\lambda) = 2\pi\,\operatorname{Ai}(0)\; ( \lambda|\alpha| )^{-1/3}\; a(x_c)\, e^{i\lambda\varphi(x_c)},
$$
where $\operatorname{Ai}(0) = 3^{-2/3}/\Gamma(2/3) \approx 0.355028$.

The integral is purely real because $\sin(\lambda\alpha u^3/3)$ is odd.

### 3.4 Airy (2D) – corank‑1 cubic

Let the degenerate direction have cubic coefficient $\alpha$, and the transverse non‑degenerate direction have quadratic coefficient $\beta$. Then

$$
I(\lambda) = a(x_c)\, e^{i\lambda\varphi(x_c)}
            \times \underbrace{2\pi\operatorname{Ai}(0)(\lambda|\alpha|)^{-1/3}}_{\text{Airy part}}
            \times \underbrace{\sqrt{\frac{2\pi}{\lambda|\beta|}}\; e^{i\pi\operatorname{sign}(\beta)/4}}_{\text{transverse Gaussian}}.
$$

Total scaling: $\lambda^{-5/6}$.

### 3.5 Pearcey – corank‑1 quartic

Let the quartic coefficient in the degenerate direction be $\gamma$ (from normal form $\gamma u^4/4$), and transverse quadratic coefficient $\beta$. Then

$$
I(\lambda) = a(x_c)\, e^{i\lambda\varphi(x_c)}
            \times \underbrace{\left(\frac{4}{\lambda|\gamma|}\right)^{1/4}\!
               \frac12\Gamma\!\left(\frac14\right)
               e^{i\pi\operatorname{sign}(\gamma)/8}}_{\text{Pearcey part}}
            \times \underbrace{\sqrt{\frac{2\pi}{\lambda|\beta|}}\;
               e^{i\pi\operatorname{sign}(\beta)/4}}_{\text{transverse Gaussian}}.
$$

Total scaling: $\lambda^{-3/4}$.  
(Here $\Gamma(1/4)$ is the gamma function.)

---

## 4. Laplace’s Method (Exponentially Damped, $\varphi = i\psi$, $\psi$ real)

For $I(\lambda) = \int a(x)\, e^{-\lambda\psi(x)}\,dx$ with a strict minimum of $\psi$ at $x_c$.

### 4.1 Leading term

$$
I_0(\lambda) = a(x_c)\, e^{-\lambda\psi(x_c)}\;
               \left(\frac{2\pi}{\lambda}\right)^{n/2}
               \frac{1}{\sqrt{\det H}},
$$
where $H = \nabla^2\psi(x_c)$ (positive definite).

### 4.2 Second‑order correction (real)

$$
I_1(\lambda) = I_0(\lambda)\,\frac{1}{\lambda}\,C_{\text{real}},
$$
with

$$
\begin{aligned}
C_{\text{real}} = &\;\frac12\operatorname{tr}(H^{-1}\nabla^2 a)
                 - \frac12\sum_{i,j,k} (\nabla a)_i\,(H^{-1})_{jk}(H^{-1})_{kl}\,
                   \partial_{jkl}\psi \\
                 &\;-\frac18\sum_{i,j,k,l} (H^{-1})_{ij}(H^{-1})_{kl}\,
                   \partial_{ijkl}\psi \\
                 &\;+\frac{5}{24}\sum_{i,j,k,l,m,n} (H^{-1})_{il}(H^{-1})_{jm}(H^{-1})_{kn}\,
                   \partial_{ijk}\psi\;\partial_{lmn}\psi .
\end{aligned}
$$

All quantities are real; the correction is real.

---

## 5. Saddle‑Point Method (Complex Phase)

For $\varphi$ genuinely complex, the integral is analytically continued to $\mathbb{C}^n$. A **saddle point** $z_c\in\mathbb{C}^n$ satisfies $\nabla\varphi(z_c)=0$. The leading contribution (non‑degenerate case) is

$$
I(\lambda) \approx \left(\frac{2\pi}{\lambda}\right)^{n/2}
               a(z_c)\, e^{i\lambda\varphi(z_c)}
               \frac{1}{\sqrt{\det\nabla^2\varphi(z_c)}},
$$

where the complex square root is taken on the principal branch.  
**Important:** This formula is valid only if the original integration contour can be deformed to pass through $z_c$ along a steepest‑descent path; the module does **not** verify this (a warning is issued).

Degenerate saddles are not implemented.

---

## 6. Unified AsymptoticEvaluator

The class `AsymptoticEvaluator` dispatches evaluation to the correct specialised evaluator based on the method stored in the `CriticalPoint` object (`cp.method`).

- `STATIONARY_PHASE` → `StationaryPhaseEvaluator` (handles Morse, Airy, Pearcey).
- `LAPLACE` → `LaplaceEvaluator`.
- `SADDLE_POINT` → `SaddlePointEvaluator`.

---

## 7. Visualisation Concepts

### 7.1 Phase landscape

Plots the phase function $\varphi$ (real part, imaginary part, modulus, or argument) in 2D, with critical points overlaid.

### 7.2 Integrand structure

Visualises the integrand $f(x)=a(x)e^{i\lambda\varphi(x)}$ at a chosen $\lambda$:

- For stationary phase: real part (oscillations).
- For Laplace: real value (exponential peak).
- For saddle point: both real part and modulus (damping envelope).

### 7.3 Asymptotic convergence

Log‑log plot of $|I_0(\lambda)|$ and $|I_1(\lambda)|$ vs $\lambda$ together with the theoretical decay slope $-\!p$, where

| Method / type        | Exponent $p$            |
|----------------------|---------------------------|
| Morse (any $n$)    | $n/2$                   |
| Airy 1D              | $1/3$                   |
| Airy 2D              | $5/6$                   |
| Pearcey              | $3/4$                   |
| Laplace (any $n$)  | $n/2$                   |
| Saddle‑point (any $n$) | $n/2$               |

---

## 8. References

- [1] Hörmander, L. *The Analysis of Linear Partial Differential Operators I*, Springer, 1983.
- [2] Olver, F. W. J. *Asymptotics and Special Functions*, Academic Press, 1974.
- [3] Wong, R. *Asymptotic Approximations of Integrals*, Academic Press, 1989.
- [4] Bleistein, N. & Handelsman, R. *Asymptotic Expansions of Integrals*, Holt, Rinehart & Winston, 1975.
- [5] Berry, M. V. & Howls, C. J. “High orders of the Weyl expansion for quantum billiards”, *Phys. Rev. E* **50**(5), 3577–3595, 1994.
- [6] Delabaere, E. & Howls, C. J. “Global asymptotics for multiple integrals with boundaries”, *Duke Math. J.* **112**(2), 199–264, 2002.

# Mathematical Concepts in `microlocal.py`

This document collects all mathematical definitions, equations, and algorithms used in the `microlocal.py` module for unified microlocal analysis in 1D and 2D.

---

## 1. Microlocal Analysis – Core Concepts

The module provides a high‑level interface to study the propagation of singularities for linear partial differential operators using microlocal methods. It builds upon the companion modules `wkb` (WKB approximations) and `caustics` (catastrophe classification and ray caustic detection). The central objects are:

- **Principal symbol** $p(x,\xi)$ of a pseudo‑differential operator $P$.
- **Characteristic variety** $\operatorname{Char}(P) = \{(x,\xi)\in T^*\mathbb{R}^n \setminus \{0\} : p(x,\xi)=0\}$.
- **Bicharacteristic flow** – Hamiltonian flow generated by $p$ on the cotangent bundle.
- **Wavefront set** $\operatorname{WF}(u)$ – a refined notion of singular support that also records the directions (frequencies) in which the singularity occurs.
- **WKB approximation** – asymptotic solutions of the form $u(x) \approx A(x)e^{iS(x)/\varepsilon}$.
- **Caustics and Maslov index** – where rays focus, causing the standard WKB amplitude to diverge; the Maslov index provides a phase correction.

All functions automatically detect the spatial dimension (1 or 2) from the input data.

---

## 2. Characteristic Variety

The **characteristic variety** of an operator with principal symbol $p$ is the zero set in phase space:

$$
\operatorname{Char}(P) = \big\{ (x,\xi) \in T^*\mathbb{R}^n\setminus\{0\} \;\big|\; p(x,\xi)=0 \big\}.
$$

It captures the points where the operator fails to be elliptic, and it is the locus where singularities can propagate.

The module provides a function `characteristic_variety` that returns:

- the implicit equation $p(x,\xi)=0$,
- explicit solutions $\xi(x)$ (in 1D, if solvable),
- a callable to evaluate $p$ numerically.

---

## 3. Bicharacteristic Flow

The **bicharacteristic flow** is the Hamiltonian flow generated by the principal symbol $p$ on the cotangent bundle. It governs the propagation of singularities: if $(x_0,\xi_0)\in\operatorname{WF}(u)$, then the whole bicharacteristic through $(x_0,\xi_0)$ lies in $\operatorname{WF}(u)$ (up to the possible addition of lower‑order terms).

Hamilton’s equations:

**1D**  
$$
\frac{dx}{dt} = \frac{\partial p}{\partial \xi}, \qquad
\frac{d\xi}{dt} = -\frac{\partial p}{\partial x}.
$$

**2D**  
$$
\frac{dx}{dt} = \frac{\partial p}{\partial \xi},\quad
\frac{dy}{dt} = \frac{\partial p}{\partial \eta},\quad
\frac{d\xi}{dt} = -\frac{\partial p}{\partial x},\quad
\frac{d\eta}{dt} = -\frac{\partial p}{\partial y}.
$$

The function `bicharacteristic_flow` integrates these equations with various numerical methods (RK45, symplectic Euler, Verlet). It also propagates the **stability matrix** $J(t)=\partial(x,y)/\partial(x_0,y_0)$ for 2D, satisfying

$$
\frac{dJ}{dt} = H_{px}\big(x(t),y(t),\xi(t),\eta(t)\big)\cdot J,\qquad J(0)=I_2,
$$

where $H_{px}$ is the matrix of mixed second derivatives:

$$
H_{px} = \begin{pmatrix}
\dfrac{\partial^2 p}{\partial\xi\partial x} & \dfrac{\partial^2 p}{\partial\xi\partial y}\$$6pt]
\dfrac{\partial^2 p}{\partial\eta\partial x} & \dfrac{\partial^2 p}{\partial\eta\partial y}
\end{pmatrix}.
$$

The stability matrix is essential for detecting caustics (see Section 6).

---

## 4. Wavefront Set

The **wavefront set** $\operatorname{WF}(u)$ is a closed conic subset of $T^*\mathbb{R}^n\setminus\{0\}$ that refines the singular support: a point $(x_0,\xi_0)$ is not in $\operatorname{WF}(u)$ if there exists a cutoff function $\phi$ with $\phi(x_0)\neq0$ such that the Fourier transform $\widehat{\phi u}(\xi)$ decays rapidly in a conic neighbourhood of $\xi_0$. Intuitively, it records both where $u$ is singular and in which directions the singularity occurs.

The function `plot_wavefront_set` visualises the propagation of an initial wavefront set by seeding bicharacteristics from a set of initial phase‑space points $(x_0,\xi_0)$ and integrating them over a time interval. The union of the resulting curves in phase space approximates $\operatorname{WF}(u)$ at later times. Different projections (position space, frequency space, mixed coordinates) are available.

---

## 5. Bohr–Sommerfeld Quantisation (1D)

For a 1D bound state problem with Hamiltonian $H(x,p)$, the Bohr–Sommerfeld quantisation condition determines semiclassical energy levels:

$$
\frac{1}{2\pi}\oint p(x,E)\,dx = \hbar\Bigl(n + \frac{1}{2}\Bigr),\qquad n=0,1,2,\dots,
$$

where the integral is taken over a full period of the classical motion at energy $E$, and the Maslov index $\alpha=1/2$ accounts for the phase shift at turning points. The function `bohr_sommerfeld_quantization` solves this condition numerically, returning the quantised energies $E_n$ and corresponding actions.

---

## 6. Caustics and Maslov Index

A **caustic** occurs where neighbouring rays focus, i.e. where the stability matrix $J(t)$ becomes singular:

$$
\det J(t^*) = 0.
$$

At such points the standard WKB amplitude diverges, and a phase correction – the **Maslov index** $\mu$ – must be applied. The Maslov index is the signed number of caustic crossings along a ray, each contributing a phase shift of $\pi/2$. The total Maslov phase is $e^{i\mu\pi/2}$.

The module provides:

- `compute_maslov_index(traj)` – given a trajectory (with stored stability matrix), computes the Maslov index using `RayCausticDetector` from the `caustics` module.
- `compute_caustics_2d(p, initial_curve, tmax)` – for a 2D Hamiltonian, integrates a bundle of rays from an initial curve and detects caustic events via $\det J=0$.

For 1D, a simplified (and less rigorous) caustic condition is offered: $\partial^2 p/\partial\xi^2 \approx 0$, which identifies turning points in frequency space. The function `find_caustics_1d` plots this indicator.

---

## 7. Visualisation Functions

- `plot_characteristic_set` – contour plot of $p(x,\xi)=0$ (1D) or a slice with fixed $(\xi,\eta)$ (2D).
- `plot_bicharacteristics` – draws bicharacteristic curves in the chosen projection (position, frequency, or mixed).
- `plot_wavefront_set` – as described in Section 4, with options for colouring, endpoints, and various projections including a 2×2 full cotangent‑bundle view.

---

## 8. References

- [1] Hörmander, L. *The Analysis of Linear Partial Differential Operators I*, Springer, 1983. Chapter 8: Wave Front Sets.
- [2] Duistermaat, J. J. *Fourier Integral Operators*, Courant Institute Lecture Notes, 1996.
- [3] Maslov, V. P. & Fedoriuk, M. V. *Semi‑Classical Approximation in Quantum Mechanics*, Reidel, 1981.
- [4] Zworski, M. *Semiclassical Analysis*, American Mathematical Society, 2012. Chapter 3: Propagation of Singularities.
- [5] Taylor, M. E. *Partial Differential Equations II*, Springer, 2011. Chapter 8: Microlocal Analysis.

# Mathematical Concepts in `solver.py`

This document collects all mathematical definitions, equations, and algorithms used in the `solver.py` module for spectral PDE solving with pseudo‑differential operators.

---

## 1. Spectral Discretisation

The solution $u(x,t)$ (or $u(x,y,t)$) is approximated by a truncated Fourier series.  
Spatial derivatives are replaced by multiplication with the corresponding wavenumber in Fourier space:

$$
\frac{\partial}{\partial x} \;\longleftrightarrow\; i k,\qquad
\frac{\partial^2}{\partial x^2} \;\longleftrightarrow\; -k^2,
$$

and in 2D

$$
\frac{\partial}{\partial x} \leftrightarrow i k_x,\quad
\frac{\partial}{\partial y} \leftrightarrow i k_y,\quad
\nabla^2 \leftrightarrow -(k_x^2 + k_y^2).
$$

Nonlinear terms are evaluated in physical space and transformed back (pseudo‑spectral approach).  
**Dealiasing** is applied by zeroing out the highest one‑third of the Fourier modes (the 2/3‑rule).

---

## 2. Linear Operator Symbol

From the PDE, after parsing linear terms, a Fourier multiplier $L(k)$ (or $L(k_x,k_y)$) is derived.  
For a first‑order‑in‑time equation

$$
\partial_t u = L u + N(u) + f(x,t),
$$

the linear part acts in Fourier space as

$$
\widehat{L u}(k) = L(k)\,\hat u(k).
$$

For a second‑order equation

$$
\partial_t^2 u = L u + N(u) + f(x,t),
$$

the dispersion relation $\omega(k)$ satisfies $\omega^2 = -L(k)$ (with appropriate sign conventions) and the linear propagator involves $\cos(\omega t)$ and $\sin(\omega t)$.

---

## 3. Exponential Time Integrators

### 3.1 Exponential Euler (first order)

$$
u^{n+1} = e^{L\Delta t} u^n + \Delta t\,\varphi_1(L\Delta t)\bigl(N(u^n)+f^n\bigr),
$$

where the $\varphi_1$ function is defined as

$$
\varphi_1(z) = \frac{e^z - 1}{z},\qquad \varphi_1(0)=1.
$$

### 3.2 ETD‑RK4 (fourth order)

The Exponential Time Differencing Runge‑Kutta 4 scheme uses four stages and the functions $\varphi_1$ and $\varphi_2$:

$$
\varphi_2(z) = \frac{e^z - 1 - z}{z^2},\qquad \varphi_2(0)=\frac12.
$$

Stages (with $E = e^{L\Delta t}$, $E_2 = e^{L\Delta t/2}$):

$$
\begin{aligned}
a   &= \text{ifft}\!\left(E_2\bigl(\hat u^n + \tfrac{\Delta t}{2} \varphi_1(L\Delta t)\,\hat N_1\bigr)\right),\\
b   &= \text{ifft}\!\left(E_2\bigl(\hat u^n + \tfrac{\Delta t}{2} \varphi_1(L\Delta t)\,\hat N_2\bigr)\right),\\
c   &= \text{ifft}\!\left(E\bigl(\hat u^n + \Delta t\,\varphi_1(L\Delta t)\,\hat N_3\bigr)\right),\\
\hat N_1 &= \text{fft}(N(a)),\;\; \hat N_2 = \text{fft}(N(b)),\;\; \hat N_3 = \text{fft}(N(b)),\;\; \hat N_4 = \text{fft}(N(c)),\\
\hat u^{n+1} &= E\,\hat u^n + \Delta t\Bigl(\hat N_1\varphi_1 + 2(\hat N_2+\hat N_3)\varphi_2 + \hat N_4\varphi_1\Bigr)/6.
\end{aligned}
$$

---

## 4. Pseudo‑differential Operators

When the equation contains a term $\texttt{psiOp}(p(x,\xi), u)$, the operator is defined via Kohn‑Nirenberg quantisation:

$$
(Pu)(x) = \frac{1}{(2\pi)^n}\int e^{i x\cdot\xi}\, p(x,\xi)\,\hat u(\xi)\,d\xi.
$$

If the symbol is spatially independent, this reduces to a Fourier multiplier: $\widehat{Pu}(k) = p(k)\,\hat u(k)$.  
For spatially varying symbols, a numerical quadrature (block‑parallel in 2D) is used.

---

## 5. Stationary Problems and Asymptotic Inversion

For a stationary pseudo‑differential equation $P u = f$, a formal right inverse is constructed via an asymptotic series (Kohn‑Nirenberg symbol calculus).  
The leading term is $1/p(x,\xi)$; higher orders involve derivatives of $p$ and of the inverse.  
The module computes the inverse symbol $r(x,\xi)$ up to a given order, then applies it to the source term $f$ using the same quantisation (FFT‑based for constant‑coefficient symbols, otherwise direct quadrature).

---

## 6. CFL Condition

The Courant‑Friedrichs‑Lewy condition ensures that the time step $\Delta t$ is small enough to resolve the fastest wave.  
Based on the group velocity $v_g = d\omega/dk$ (for dispersive waves) or on the imaginary part of $L(k)$ for non‑dispersive problems, the allowed time step is estimated as

$$
\Delta t \le C\,\frac{\Delta x}{\max |v_g|},
$$

with a safety factor $C \approx 0.5$. In 2D the condition is

$$
\Delta t \le \frac{C}{\displaystyle\frac{\max |v_{g,x}|}{\Delta x} + \frac{\max |v_{g,y}|}{\Delta y}}.
$$

---

## 7. Symbol Conditions for Well‑posedness

The linear symbol $L(k)$ is checked against three criteria:

1. **Stability**: $\operatorname{Re} L(k) \le 0$ for all $k$ (no exponential growth).
2. **High‑frequency dissipation**: $\operatorname{Re} L(k) \le -\delta |k|^2$ for large $|k|$ (sufficient damping).
3. **Growth bound**: $|L(k)| \le C(1+|k|)^m$ with $m \le 4$ (reasonable growth for numerical stability).

---

## 8. Energy for Second‑Order Wave Equations

For an equation of the form $\partial_t^2 u = L u$, the conserved energy (in the linear case) is

$$
E(t) = \frac12 \int \bigl( (\partial_t u)^2 + |L^{1/2}u|^2 \bigr)\,dx,
$$

where $L^{1/2}$ is defined spectrally: $\widehat{L^{1/2}u}(k) = \sqrt{|L(k)|}\,\hat u(k)$ (with a suitable branch).  
The method `_compute_energy()` evaluates this expression at each time step when energy monitoring is active.

---

## 9. Dispersion Relation, Phase and Group Velocities

For second‑order equations, the dispersion relation $\omega(k)$ is obtained from the linear symbol via $-\omega^2 = L(k)$.  
The **phase velocity** and **group velocity** are

$$
v_p(k) = \frac{\omega(k)}{|k|},\qquad
v_g(k) = \nabla_k\omega(k).
$$

These are visualised in `_analyze_wave_propagation()`.

---

## 10. References

- [1] Canuto, C., Hussaini, M. Y., Quarteroni, A., & Zang, T. A. *Spectral Methods: Fundamentals in Single Domains*, Springer, 2006.
- [2] Trefethen, L. N. *Spectral Methods in MATLAB*, SIAM, 2000.
- [3] Hochbruck, M., & Ostermann, A. “Exponential integrators”, *Acta Numerica* **19**, 209–286, 2010.
- [4] Kassam, A.-K., & Trefethen, L. N. “Fourth‑order time‑stepping for stiff PDEs”, *SIAM J. Sci. Comput.* **26**(4), 1213–1233, 2005.

# Mathematical Concepts in `fio_bridge.py`

This document collects all mathematical definitions, equations, and algorithms used in the `fio_bridge.py` module, which connects pseudo‑differential operators (`psiop`) with asymptotic evaluation (`asymptotic`) via Fourier Integral Operators (FIOs).

---

## 1. Fourier Integral Operators – Definition

A Fourier Integral Operator (FIO) with phase $\phi(x,y,\theta)$ and amplitude $a(x,y,\theta)$ acts on a function $u(y)$ as

$$
(Fu)(x) = \frac{1}{(2\pi)^n} \int_{\mathbb{R}^n} \int_{\mathbb{R}^N} e^{i\lambda \phi(x,y,\theta)}\, a(x,y,\theta)\, u(y)\, dy\, d\theta,
$$

where $\lambda$ is a large parameter, $x$ are observation variables, $y$ source variables, and $\theta$ are auxiliary (phase) variables.  
In the standard semi‑classical setting $n = \dim x = \dim y$ and $\dim\theta = n$ (equal dimensions). The phase must be **non‑degenerate** in the sense of Hörmander (see Section 3).

---

## 2. Action of a Pseudo‑differential Operator on a WKB State

For a pseudo‑differential operator with symbol $p(x,\xi)$ and a WKB input state

$$
u(y) = a_u(y)\, e^{i\lambda S_u(y)},
$$

the action $(Pu)(x)$ can be written as an FIO with the specific phase

$$
\phi(x,y,\theta) = (x - y)\cdot\theta + S_u(y),
$$

and amplitude

$$
a(x,y,\theta) = p(y,\theta)\, a_u(y).
$$

The large parameter $\lambda$ appears in both the WKB phase and the FIO exponential; the total phase used in the asymptotic analysis is $\Phi(y,\theta) = \phi(x,y,\theta)/\lambda + S_u(y)$.

The stationary conditions for the phase are:

$$
\frac{\partial\phi}{\partial\theta} = 0 \;\Longrightarrow\; y = x,\qquad
\frac{\partial\phi}{\partial y} = 0 \;\Longrightarrow\; \theta = \nabla S_u(y).
$$

Thus, at the critical point $y_c = x$ and $\theta_c = \nabla S_u(x)$, the contribution simplifies to

$$
(Pu)(x) \approx p\bigl(x, \nabla S_u(x)\bigr)\, a_u(x)\, e^{i\lambda S_u(x)},
$$

which is the leading‑order semi‑classical approximation.

---

## 3. Canonical Relation and Non‑degeneracy

For a general FIO, the **canonical relation** is the set

$$
C = \bigl\{ (x, \nabla_x\phi,\; y, -\nabla_y\phi) \;\big|\; \nabla_\theta\phi = 0 \bigr\} \subset T^*\mathbb{R}^n \times T^*\mathbb{R}^n.
$$

The phase is **non‑degenerate** if the mixed Hessian matrix

$$
\frac{\partial^2\phi}{\partial\theta_i\partial x_j}
$$

has maximal rank (i.e. $\min(\dim\theta,\dim x)$). This condition guarantees that the canonical relation is a smooth Lagrangian submanifold and that the FIO belongs to Hörmander’s class $I^m$.

The method `is_non_degenerate()` checks this condition symbolically by evaluating the determinant (for square matrices) or the rank.

---

## 4. Asymptotic Evaluation – Delegation to `asymptotic.py`

All critical‑point analysis and asymptotic formula are **delegated** to the `asymptotic` module. The FIO kernel is passed to an `Analyzer`, which determines the method (stationary phase, Laplace, or saddle point) and computes contributions from critical (or saddle) points.

For a Morse (non‑degenerate) critical point $(y_c,\theta_c)$ with Hessian $H = \nabla^2\Phi$, the `asymptotic` module returns

$$
I_{\text{asym}} = \left(\frac{2\pi}{\lambda}\right)^{n/2}
\frac{a(x,y_c,\theta_c)\, e^{i\lambda\Phi_c}}
{\sqrt{|\det H|}}\; e^{i\pi\mu/4},
$$

where $\mu$ is the Maslov index (signature of the Hessian).  
The FIO definition carries an extra factor $1/(2\pi)^n$; therefore the **total prefactor** applied after summation is

$$
\text{prefactor} = \frac{\lambda}{(2\pi)^{\dim\theta}}.
$$

This factor arises because the `asymptotic` result already contains $(2\pi/\lambda)^{(\dim y+\dim\theta)/2}$; dividing by $(2\pi)^{\dim\theta}$ leaves one power of $\lambda$.

---

## 5. Performance Optimisation – Precomputation with a Placeholder

A key optimisation in `PsiOpFIOBridge` avoids repeated symbolic work for each observation point $x$.  
The phase for a psiOp is linear in $x$:

$$
\phi(x,y,\theta) = (x - y)\cdot\theta + S_u(y).
$$

All symbolic derivatives (gradient, Hessian, etc.) are computed **once** using a placeholder symbol `_xp` (and `_yp` in 2D).  
The resulting expressions are lambdified into NumPy functions that accept the integration variables and the placeholder.  
When evaluating at a specific $x$, the placeholder is supplied as an extra argument; no SymPy re‑evaluation occurs.

This reduces the symbolic cost from $O(N)$ to $O(1)$ per grid point.

---

## 6. PropagatorBridge – Exponential of an Operator

For a time‑dependent problem, the semi‑classical propagator $e^{itP}$ is constructed using `PseudoDifferentialOperator.exponential_symbol()`, which returns the symbol of $e^{itP}$ as an asymptotic series:

$$
e^{itP}(x,\xi) \sim 1 + it\,p(x,\xi) + \frac{(it)^2}{2!}\,(p\circ p)(x,\xi) + \cdots .
$$

The result is a new pseudo‑differential operator whose action on a WKB state is then evaluated by `PsiOpFIOBridge`. The parameter $t$ enters the series as a multiplicative factor; the method uses $t = i$ times the physical time to match the oscillatory factor $e^{i t p}$.

---

## 7. CompositionBridge – Asymptotic Composition

The composition of two pseudo‑differential operators $P$ and $Q$ is approximated by their **symbolic composition**:

$$
(p\circ q)(x,\xi) \sim \sum_{\alpha} \frac{i^{-|\alpha|}}{\alpha!}
\,\partial_\xi^\alpha p(x,\xi)\,\partial_x^\alpha q(x,\xi)
$$

(Kohn‑Nirenberg quantisation). The `CompositionBridge` constructs the composed symbol using `P.compose_asymptotic(Q, order)`, wraps it in a new `PseudoDifferentialOperator`, and then evaluates it via `PsiOpFIOBridge`.

---

## 8. WKBState – Carrying the WKB Ansatz

A `WKBState` encapsulates the WKB ansatz

$$
u(x) = a(x)\, e^{i\lambda S(x)}.
$$

It provides:

- `to_array(x_grid)` – evaluates the complex array.
- `as_callable()` – returns a function suitable as an initial condition for `PDESolver`.
- `wkb_phase_gradient(x_grid)` – computes the local wavenumber $k(x) = \lambda S'(x)$.
- `dominant_wavenumber(x_grid)` – median of $|k(x)|$, useful for spectral splitting.

---

## 9. SpectralSplitter – Low‑/High‑Frequency Decomposition

Given a uniform spatial grid, a `SpectralSplitter` with cutoff wavenumber $k_{\text{cut}}$ performs a sharp spectral split:

$$
u_{\text{low}} = \text{IFFT}\big[ \hat u(k)\, \mathbf{1}_{|k|\le k_{\text{cut}}} \big],\qquad
u_{\text{high}} = \text{IFFT}\big[ \hat u(k)\, \mathbf{1}_{|k|> k_{\text{cut}}} \big].
$$

The split is lossless: $u = u_{\text{low}} + u_{\text{high}}$ up to machine precision.  
The splitter also provides energy ratios and can suggest a cutoff that isolates a desired fraction of energy in the high‑frequency band.

---

## 10. SemiclassicalCorrector – Refining a Spectral Solution

The `SemiclassicalCorrector` replaces the high‑frequency part of a spectral solver’s solution with the asymptotically more accurate WKB estimate:

1. Split the solver solution $u_{\text{solver}}$ into $u_{\text{low}}$ and $u_{\text{high}}$.
2. Use `PsiOpFIOBridge` to compute the operator’s action on the WKB state, keeping only its high‑frequency content.
3. Recombine: $u_{\text{corrected}} = u_{\text{low}} + u_{\text{high}}^{\text{(bridge)}}$.

The magnitude of the correction $\|u_{\text{corrected}} - u_{\text{solver}}\|/\|u_{\text{solver}}\|$ indicates whether the solver’s high‑frequency component has significant WKB error.

---

## 11. CrossValidator – Comparing Asymptotic and Spectral Solvers

The `CrossValidator` runs both an asymptotic bridge and a spectral solver (via `PDESolver`) on the same problem and produces a `ValidationReport`. The report includes:

- Point‑wise absolute and relative errors.
- Error spectrum $\big|\widehat{u_{\text{solver}} - u_{\text{bridge}}}\big|$.
- A validity flag based on $\text{max rel error} < 3/\lambda$ (theoretical $O(\lambda^{-1})$ accuracy).

A $\lambda$-sweep method helps determine the threshold below which the WKB approximation becomes reliable.

---

## 12. References

- [1] Hörmander, L. “Fourier Integral Operators I”, *Acta Math.* **127** (1971).
- [2] Duistermaat, J.J. *Fourier Integral Operators*, Birkhäuser, 1996.
- [3] Zworski, M. *Semiclassical Analysis*, AMS Graduate Studies, 2012.

# Mathematical Concepts in `physics.py`

This document collects all mathematical definitions, equations, and algorithms used in the `physics.py` module for Lagrangian–Hamiltonian transformations and symbolic PDE generation.

---

## 1. Legendre Transform – Classical

Given a Lagrangian $L(x,u,p)$ depending on position $x$, field $u$ (optional), and generalised velocities $p$ (often denoted $\dot{x}$), the conjugate momenta are defined as

$$
\xi_i = \frac{\partial L}{\partial p_i}.
$$

If the Hessian matrix $\frac{\partial^2 L}{\partial p_i \partial p_j}$ is invertible (non‑singular), the velocities can be expressed as functions of $\xi$ and the Hamiltonian is obtained via the **Legendre transform**:

$$
H(x,u,\xi) = \sum_i \xi_i \, p_i(\xi) \;-\; L\bigl(x,u,p(\xi)\bigr).
$$

The module implements this symbolically using `sympy.solve` to invert the relation $\xi = \partial L/\partial p$. For quadratic Lagrangians of the form

$$
L = \frac12 p^T A p + b^T p + c,
$$

an explicit formula is used:

$$
p = A^{-1}(\xi - b),\qquad
H = \frac12 (\xi - b)^T A^{-1} (\xi - b) - c,
$$

where $A$ is the Hessian matrix (constant in $p$).

---

## 2. Legendre–Fenchel Transform (Convex Conjugate)

When the Lagrangian is not strictly convex, or when the Hessian is singular, the classical Legendre transform is replaced by the **Legendre–Fenchel conjugate**:

$$
H(x,u,\xi) = \sup_{p} \bigl( \xi\cdot p - L(x,u,p) \bigr).
$$

This always yields a convex (in $\xi$) function and is well‑defined even for non‑smooth or non‑convex Lagrangians (the supremum may be $+\infty$ in the non‑convex case).

The module provides three ways to compute it:

* **`fenchel_symbolic`**: Attempts to solve $\partial L/\partial p = \xi$ symbolically and then takes the maximum over multiple solutions (if several branches exist). This works only if $L$ is smooth and the equation can be solved.
* **`fenchel_numeric` (1D)**: For a given $\xi$, the supremum is approximated by evaluating $\xi p - L(p)$ on a dense grid in $p$ or by a multi‑start SciPy optimiser.
* **`fenchel_numeric` (2D)**: Similar grid‑based or multi‑start optimisation over a rectangle in $(p_1,p_2)$.

For 1D the numeric method returns a callable $H_{\text{num}}(\xi)$; for 2D a callable $H_{\text{num}}(\xi,\eta)$.

---

## 3. Inverse Legendre Transform

Given a Hamiltonian $H(x,u,\xi)$, the Lagrangian can be recovered (when the map $\xi \mapsto \partial H/\partial\xi$ is invertible) by solving

$$
p_i = \frac{\partial H}{\partial \xi_i}
$$

for $\xi$ as functions of $p$, and then

$$
L(x,u,p) = \sum_i \xi_i(p) \, p_i \;-\; H\bigl(x,u,\xi(p)\bigr).
$$

The method `H_to_L` implements this symbolically.

---

## 4. Hamiltonian Decomposition

For a Hamiltonian expressed in terms of momenta $\xi_i$ (and possibly coordinates $x,u$), the module decomposes it into a **polynomial (local) part** and a **non‑polynomial (non‑local) part** using a heuristic:

* Terms containing $\sqrt{\;\;}$, $\operatorname{Abs}$, $\operatorname{sign}$, or any other non‑polynomial function are flagged as non‑local.
* Terms that are polynomial in each $\xi_i$ (according to `sympy`’s `is_polynomial`) are kept as local.
* All remaining terms are also added to the non‑local part.

This decomposition helps identify the principal symbol and lower‑order terms when constructing pseudo‑differential operators.

---

## 5. Symbolic PDE Generation

Using the decomposed Hamiltonian, the module generates formal PDEs where the action of the pseudo‑differential operator with symbol $H$ is represented by the placeholder $\psi \mathrm{Op}(H,u)$. Three types are supported:

**Stationary (eigenvalue) equation**

$$
\psi \mathrm{Op}(H,u) = E\, u,
$$

with $E$ a real parameter.

**Schrödinger‑type equation**

$$
i\,\partial_t u = \psi \mathrm{Op}(H,u).
$$

**Wave equation**

$$
\partial_{tt} u + \psi \mathrm{Op}(H,u) = 0.
$$

The result is a dictionary containing the SymPy equation, the polynomial and non‑local parts, a formatted string, and the chosen mode.

---

## 6. References

- [1] Arnold, V. I. *Mathematical Methods of Classical Mechanics*, Springer‑Verlag, 1989 (2nd ed.). §14: Legendre Transform.
- [2] Rockafellar, R. T. *Convex Analysis*, Princeton University Press, 1970. Chapter 12: Conjugate Functions.
- [3] Evans, L. C. *Partial Differential Equations*, American Mathematical Society, 2010 (2nd ed.). §4.3: Hamilton–Jacobi Equations.
- [4] Folland, G. B. *Quantum Field Theory: A Tourist Guide for Mathematicians*, American Mathematical Society, 2008. §1: Legendre Transform and Quantisation.

# Mathematical Concepts in `geometry.py`

This document collects all mathematical definitions, equations, and algorithms used in the `geometry.py` module for geometric and semiclassical analysis of symbols (Hamiltonians) in one and two dimensions.

---

## 1. Hamiltonian Dynamics

The module studies a **Hamiltonian** $H$ on phase space $T^*\mathbb{R}^n$ ($n=1,2$) with canonical coordinates $(x,p)$ in 1D, $(x,y,\xi,\eta)$ in 2D.

### 1.1 Hamilton’s equations

**1D**  
$$
\frac{dx}{dt} = \frac{\partial H}{\partial \xi},\qquad
\frac{d\xi}{dt} = -\frac{\partial H}{\partial x}.
$$

**2D**  
$$
\frac{dx}{dt} = \frac{\partial H}{\partial \xi},\quad
\frac{dy}{dt} = \frac{\partial H}{\partial \eta},\quad
\frac{d\xi}{dt} = -\frac{\partial H}{\partial x},\quad
\frac{d\eta}{dt} = -\frac{\partial H}{\partial y}.
$$

---

## 2. Variational (Jacobi) Equations

To detect caustics and compute stability, the module integrates the linearised flow (Jacobian matrix) alongside the Hamiltonian flow.

### 2.1 1D variational system

Let $J = \frac{\partial x}{\partial \xi_0}$ and $K = \frac{\partial \xi}{\partial \xi_0}$ (derivatives with respect to initial momentum). They satisfy:

$$
\frac{dJ}{dt} = \frac{\partial^2 H}{\partial \xi^2}\,J + \frac{\partial^2 H}{\partial x\partial\xi}\,K,
$$
$$
\frac{dK}{dt} = -\frac{\partial^2 H}{\partial x\partial\xi}\,J - \frac{\partial^2 H}{\partial x^2}\,K.
$$

Initial conditions: $J(0)=0,\; K(0)=1$.

### 2.2 2D variational system

In 2D, the full $4\times4$ Jacobian matrix

$$
M(t) = \frac{\partial (x,y,\xi,\eta)}{\partial (x_0,y_0,\xi_0,\eta_0)}
$$

satisfies

$$
\frac{dM}{dt} = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix} \cdot \nabla^2 H \cdot M,
$$

where $\nabla^2 H$ is the $4\times4$ Hessian matrix of $H$. The symplectic matrix

$$
J_0 = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ -1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \end{pmatrix}
$$

encodes the canonical structure. The evolution of $M$ is:

$$
\frac{dM}{dt} = J_0 \, \nabla^2 H \; M.
$$

The **caustic condition** is $\det\big( \partial(x,y)/\partial(\xi_0,\eta_0) \big) = 0$, i.e. the upper‑right $2\times2$ block of $M$ becomes singular.

---

## 3. Caustics and Maslov Index

A **caustic** occurs when nearby rays converge, i.e. when the Jacobian determinant (or the relevant block) vanishes. In 1D, this is simply $J(t)=0$. In 2D, it is $\det(\partial(x,y)/\partial(\xi_0,\eta_0)) = 0$.

Each caustic contributes a phase shift of $-\frac{\pi}{2}$ times the **Maslov index** $\mu$ (the number of caustics crossed). The module classifies caustics as **fold** ($\mu=1$) or **cusp** ($\mu=2$) based on local curvature analysis.

---

## 4. Lyapunov Exponent (Stability)

For a periodic orbit, the (largest) Lyapunov exponent $\lambda$ is computed by integrating a small perturbation $\delta$ and measuring its growth:

$$
\lambda = \frac{1}{T} \ln \frac{\|\delta(T)\|}{\|\delta(0)\|}.
$$

In 2D, two exponents may be computed.

---

## 5. Periodic Orbits and EBK Quantisation

A periodic orbit of period $T$ has **action**

$$
S = \oint p\,dq = \int_0^T \big( \xi(t)\dot{x}(t) + \eta(t)\dot{y}(t) \big)\,dt.
$$

The **Einstein–Brillouin–Keller (EBK) quantisation** for an integrable system (1D) reads

$$
S = 2\pi\hbar\left(n + \frac{\alpha}{4}\right),
$$

where $\alpha$ is the Maslov index (typically $\alpha=2$ for two turning points). In 2D, analogous conditions hold for each action variable.

---

## 6. Gutzwiller Trace Formula

The **Gutzwiller trace formula** expresses the oscillatory part of the density of states as a sum over periodic orbits $\gamma$:

$$
\rho_{\text{osc}}(E) \approx \frac{1}{\pi\hbar} \sum_{\gamma} \frac{T_\gamma}{\sqrt{|\det(M_\gamma - I)|}} 
\cos\left( \frac{S_\gamma}{\hbar} - \frac{\pi\mu_\gamma}{2} \right).
$$

In the module, this is used to construct the Fourier transform of the trace $\operatorname{Tr} e^{-iHt/\hbar}$:

$$
\operatorname{Tr} e^{-iHt/\hbar} \approx \sum_{\gamma} A_\gamma(t)\, e^{iS_\gamma/\hbar},
$$

with amplitudes $A_\gamma(t)$ derived from the orbit’s period and stability.

---

## 7. Semiclassical Spectrum

The **semiclassical spectrum** is obtained by Fourier transforming the trace:

$$
\rho(E) \propto \left| \int e^{iEt/\hbar} \operatorname{Tr} e^{-iHt/\hbar}\, dt \right|.
$$

Peaks in this Fourier spectrum correspond to approximate energy levels.

---

## 8. Weyl’s Law

The **Weyl law** gives the asymptotic number of states below energy $E$:

$$
N(E) \sim \frac{1}{(2\pi\hbar)^d} \operatorname{Vol}\{ H \le E \},
$$

where $d$ is the number of degrees of freedom (1 or 2). For a simple power‑law Hamiltonian $H \sim p^2 + V(x)$ this becomes $N(E) \propto E^d$.

---

## 9. Level Spacing Distributions

For an integrable system, the normalised level spacings $s$ follow the **Poisson distribution**

$$
P(s) = e^{-s}.
$$

For a chaotic system, the **Wigner surmise** (GOE) applies:

$$
P(s) = \frac{\pi s}{2}\, e^{-\pi s^2/4}.
$$

The module computes the ratio $\langle s^2\rangle / \langle s\rangle^2$ to classify the system.

---

## 10. Berry–Tabor Formula (1D)

The smoothed density of states from periodic orbits (Berry–Tabor) is

$$
\rho(E) \approx \frac{1}{2\pi} \sum_{\gamma} T_\gamma \, \frac{1}{\sqrt{2\pi\sigma^2}} 
\exp\!\left( -\frac{(E_\gamma - E)^2}{2\sigma^2} \right),
$$

with a Gaussian window of width $\sigma$.

---

## 11. Phase Space Volume (Monte Carlo)

The volume of phase space below energy $E$ is estimated by Monte Carlo:

$$
\operatorname{Vol}\{H \le E\} = \int \mathbf{1}_{H\le E}\, dx\,dy\,d\xi\,d\eta
\approx \frac{N_{\text{hits}}}{N_{\text{total}}} \times \text{total box volume}.
$$

---

## 12. KAM Tori Detection (2D)

Periodic orbits are clustered by action proximity to identify approximate **KAM tori**. The hierarchical clustering (Ward’s method) groups orbits with similar actions, indicating tori on which they lie.

---

## 13. References

- [1] Arnold, V. I. *Mathematical Methods of Classical Mechanics*, Springer‑Verlag, 1989.
- [2] Gutzwiller, M. C. *Chaos in Classical and Quantum Mechanics*, Springer‑Verlag, 1990.
- [3] Maslov, V. P. & Fedoriuk, M. V. *Semi‑Classical Approximation in Quantum Mechanics*, Reidel, 1981.
- [4] Berry, M. V. & Tabor, M. “Level clustering in the regular spectrum”, *Proc. R. Soc. Lond. A* 356, 375–394, 1977.
- [5] Bohigas, O., Giannoni, M. J., & Schmit, C. “Characterization of chaotic quantum spectra”, *Phys. Rev. Lett.* 52, 1–4, 1984.
- [6] Kravtsov, Yu. A. & Orlov, Yu. I. *Caustics, Catastrophes and Wave Fields*, Springer, 1999.
