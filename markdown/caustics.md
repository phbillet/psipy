# Underlying Theory of the `caustics` Module

[Back to psipy main page](./psipy.md)

This document presents the mathematical foundations implemented in `caustics.py`, covering catastrophe classification, ray‑based caustic detection, Maslov phase shifts, and uniform asymptotic corrections. No code or examples are included – only the essential theory.

---

## 1. Caustics in Semiclassical Analysis

In a Hamiltonian system with $n$ degrees of freedom, a family of rays is parameterised by an initial‑condition vector $q \in \mathbb{R}^n$. Let $x(t;q)$ be the position at time $t$ for a ray starting with parameters $q$. The **stability matrix**

$$
J(t) = \frac{\partial x(t)}{\partial q}
$$

measures how neighbouring rays diverge. It satisfies the variational equation

$$
\frac{dJ}{dt} = H_{px}\bigl(x(t),\xi(t)\bigr)\; J(t), \qquad J(0)=I_n,
$$

where $H_{px} = \frac{\partial^2 H}{\partial \xi\,\partial x}$ is the mixed Hessian of the Hamiltonian $H(x,\xi)$.  
A **caustic** occurs at time $t^*$ when

$$
\det J(t^*) = 0.
$$

At that instant the ray density becomes infinite, and the standard WKB approximation breaks down.

---

## 2. Arnold’s Classification of Catastrophes

Catastrophe theory classifies critical points of a smooth function $H(\xi_1,\dots,\xi_n)$ where $\nabla H = 0$. The classification is based on the rank of the Hessian and higher derivatives.

### 2.1 One‑dimensional case

For $H(\xi)$, expand around a critical point $\xi_0$:

$$
H(\xi) = \text{const} + \frac{H^{(k)}(\xi_0)}{k!}\,(\xi-\xi_0)^k + \cdots,
$$

with $k \ge 2$ the first order where $H^{(k)}(\xi_0) \neq 0$. The singularity is of type **A$_{k-1}$**:

- $k=2$: Morse (non‑degenerate)
- $k=3$: A₂ (fold)
- $k=4$: A₃ (cusp)
- $k=5$: A₄ (swallowtail)
- $k=6$: A₅ (butterfly)

### 2.2 Two‑dimensional case

For $H(\xi,\eta)$, let the Hessian at the critical point be $\mathbf{Hess}$. Three cases arise:

#### Rank 2 – Morse (non‑degenerate)
The point is a regular extremum or saddle.

#### Rank 1 – Aₖ series
The Hessian has a one‑dimensional null space. Let $v = (v_x, v_y)$ be the null direction. Define the directional derivative operator

$$
D = v_x\frac{\partial}{\partial\xi} + v_y\frac{\partial}{\partial\eta}.
$$

The first non‑zero derivative $D^k H$ (with $k \ge 3$) determines the type:

- $D^3 H \neq 0$ → A₂ (fold)
- $D^4 H \neq 0$ → A₃ (cusp)
- $D^5 H \neq 0$ → A₄ (swallowtail)
- $D^6 H \neq 0$ → A₅ (butterfly)

#### Rank 0 – D₄ umbilics
The Hessian vanishes identically; the leading term is a binary cubic form

$$
C(\xi,\eta) = a\xi^3 + 3b\xi^2\eta + 3c\xi\eta^2 + d\eta^3.
$$

The **cubic invariant**

$$
I = 18abcd - 4b^3d + b^2c^2 - 4ac^3 - 27a^2d^2
$$

is the discriminant of the cubic. Its sign distinguishes:

- $I < 0$ → D₄⁺ (hyperbolic umbilic) – normal form $\xi^3 + \eta^3$
- $I > 0$ → D₄⁻ (elliptic umbilic) – normal form $\xi^3 - 3\xi\eta^2$
- $I = 0$ → higher degeneracy (E₆, etc.)

---

## 3. Maslov Index and Phase Shift

Each time a ray crosses a caustic ($\det J = 0$), the Maslov index $\mu$ changes by $\pm 1$, where the sign is the sign of $d(\det J)/dt$ at the crossing. The total Maslov index for a ray is the signed sum of such crossings.

The semiclassical wavefunction acquires a phase shift

$$
e^{i\mu\pi/2}
$$

for each ray. This factor ensures that the wavefunction remains single‑valued when crossing caustics.

---

## 4. Uniform Special Functions

Near a caustic the WKB amplitude diverges. Uniform approximations replace the oscillatory exponential by special functions that remain finite.

### 4.1 Fold (A₂) – Airy function

The uniform approximation for a fold caustic is

$$
u(x) \approx 2\sqrt{\pi}\; \varepsilon^{1/6}\; a_c\; |dJ/ds|^{-1/2}\; \operatorname{Ai}\!\bigl(-\varepsilon^{-2/3}(x-x_c)\bigr)\; e^{iS_c/\varepsilon},
$$

where $\varepsilon$ is the small parameter, $x_c$ the caustic position, $a_c$ and $S_c$ the amplitude and phase at the caustic, and $dJ/ds$ the slope of $\det J$ with respect to the parameter along the ray.

The Airy function is defined by

$$
\operatorname{Ai}(z) = \frac{1}{\pi}\int_0^\infty \cos\!\left(\frac{t^3}{3}+zt\right)dt.
$$

### 4.2 Cusp (A₃) – Pearcey integral

The Pearcey integral

$$
P(x,y) = \int_{-\infty}^{\infty} \exp\!\bigl(i(t^4 + x t^2 + y t)\bigr)\,dt
$$

describes the wave field near a cusp. The uniform approximation is

$$
u(x,y) \approx \varepsilon^{1/4}\; a_c\; P\!\bigl(\varepsilon^{-1/2}(x-x_c),\; \varepsilon^{-3/4}(y-y_c)\bigr)\; e^{iS_c/\varepsilon}.
$$

### 4.3 Swallowtail (A₄) – Swallowtail integral

The three‑parameter integral

$$
SW(x,y,z) = \int_{-\infty}^{\infty} \exp\!\bigl(i(t^5 + x t^3 + y t^2 + z t)\bigr)\,dt
$$

governs the A₄ swallowtail caustic. Its scaling for uniform approximation follows the same pattern as the cusp, with appropriate powers of $\varepsilon$.

---

## 5. References

1. Arnold, V. I. *Catastrophe Theory*, Springer‑Verlag, 1986.
2. Duistermaat, J. J. “Oscillatory integrals, Lagrange immersions and unfolding of singularities”, *Comm. Pure Appl. Math.* **27**, 207–281, 1974.
3. Maslov, V. P. & Fedoriuk, M. V. *Semi‑Classical Approximation in Quantum Mechanics*, Reidel, 1981.
4. Kravtsov, Yu. A. & Orlov, Yu. I. *Caustics, Catastrophes and Wave Fields*, Springer, 1999.
5. Berry, M. V. & Howls, C. J. “High orders of the Weyl expansion for quantum billiards”, *Phys. Rev. E* **50**(5), 3577–3595, 1994.
6. Connor, J. N. L. “Practical methods for the uniform asymptotic evaluation of oscillatory integrals”, *Mol. Phys.* **31**(1), 33–55, 1976.