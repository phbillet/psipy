# Underlying Theory of Fourier Integral Operators and Semiclassical Analysis

[Back to psipy main page](./psipy.md)

This document presents the mathematical foundations of the `fio_bridge` module, covering Fourier integral operators (FIOs), pseudodifferential operators ($\psi$DOs), the WKB method, stationary phase asymptotics, caustics, Maslov index, Egorov’s theorem, and selected applications (quantum tunneling, gravitational lensing). The presentation follows Hörmander [1], Duistermaat [2], and Zworski [3].

---

## 1. Pseudodifferential Operators and their Symbolic Calculus

A **pseudodifferential operator** $P$ acting on functions $u(x)$ on $\mathbb{R}^n$ is defined by

$$
(Pu)(x) = \frac{1}{(2\pi)^n}\int_{\mathbb{R}^n}\int_{\mathbb{R}^n} e^{i\langle x-y,\xi\rangle}\, p(x,\xi)\, u(y)\, dy\,d\xi,
$$

where $p(x,\xi)$ is the **symbol** of $P$. For semiclassical problems one introduces a large parameter $\lambda$ (often $\lambda = 1/\hbar$) and writes

$$
(P_\lambda u)(x) = \frac{1}{(2\pi)^n}\iint e^{i\lambda\langle x-y,\xi\rangle}\, p(x,\xi)\, u(y)\, dy\,d\xi.
$$

The symbol is assumed to have an asymptotic expansion

$$
p(x,\xi) \sim p_0(x,\xi) + \lambda^{-1}p_1(x,\xi) + \cdots,
$$

where $p_0$ is the **principal symbol**. Composition of two $\psi$DOs is given by the **Moyal product** (or **twisted product**):

$$
p\,\#\, q \;\sim\; \sum_{|\alpha|\ge 0} \frac{1}{\alpha!}\left(\frac{i}{2}\right)^{|\alpha|}
\bigl(\partial_\xi^\alpha p\bigr)\bigl(\partial_x^\alpha q\bigr)
$$

for the Weyl quantisation, and by a similar formula for the Kohn–Nirenberg quantisation. In the semiclassical limit $\lambda\to\infty$, the principal symbol behaves like a classical observable, and the commutator $[P,Q]$ corresponds to the Poisson bracket $\{p_0,q_0\}$.

---

## 2. Fourier Integral Operators (Hörmander’s Theory)

A **Fourier integral operator** $F$ is a generalisation of a $\psi$DO that allows **canonical transformations** in phase space. It is written as an oscillatory integral

$$
(Fu)(x) = \frac{1}{(2\pi)^{n_\theta}}\iint e^{i\lambda\,\phi(x,y,\theta)}\, a(x,y,\theta)\, u(y)\, dy\,d\theta,
$$

where:

- $x\in\mathbb{R}^{n_x}$ (observation coordinates),
- $y\in\mathbb{R}^{n_y}$ (source coordinates),
- $\theta\in\mathbb{R}^{n_\theta}$ (additional phase variables),
- $\phi$ is the **phase function**, homogeneous of degree 1 in $\theta$ (or real‑valued),
- $a$ is the **amplitude** (smooth, compactly supported or slowly varying).

The canonical relation of $F$ is the set

$$
C = \bigl\{\, (x, \nabla_x\phi,\; y, -\nabla_y\phi)\;:\; \nabla_\theta\phi = 0 \,\bigr\}
\;\subset\; T^*\mathbb{R}^{n_x}\times T^*\mathbb{R}^{n_y}.
$$

$C$ is a Lagrangian submanifold (with respect to the symplectic form $\omega = d\xi\wedge dx - d\eta\wedge dy$). The **non‑degeneracy condition** of Hörmander requires that the mixed Hessian

$$
\det\Bigl(\frac{\partial^2\phi}{\partial x_i\,\partial\theta_j}\Bigr) \neq 0,
$$

i.e., the map $(x,\theta)\mapsto \nabla_\theta\phi$ is a local diffeomorphism. This guarantees that $C$ is a smooth Lagrangian manifold.

For a $\psi$DO $P$, the natural phase is $\phi(x,y,\theta) = \langle x-y,\theta\rangle$, giving the canonical relation

$$
C = \{(x,\theta, y,-\theta) \;:\; x=y\},
$$

which is the diagonal. Hence $\psi$DOs are a special class of FIOs.

---

## 3. WKB Method and Asymptotic Evaluation

The **WKB (Wentzel–Kramers–Brillouin) ansatz** for a solution of a semiclassical equation is

$$
u_\lambda(y) \sim a(y)\, e^{i\lambda S(y)},
$$

where $a(y)$ is the amplitude and $S(y)$ the phase (real‑valued for oscillatory states). The action of a $\psi$DO $P$ on such a state yields an oscillatory integral

$$
(Pu_\lambda)(x) = \frac{1}{(2\pi)^{n}}\iint e^{i\lambda\bigl[(x-y)\cdot\xi + S(y)\bigr]}\, p(y,\xi)\, a(y)\, dy\,d\xi.
$$

The total phase is $\Phi(y,\xi;x) = (x-y)\cdot\xi + S(y)$. The stationary points are given by

$$
\frac{\partial\Phi}{\partial\xi} = x - y = 0 \quad\Longrightarrow\quad y_c = x,
$$
$$
\frac{\partial\Phi}{\partial y} = -\xi + S'(y) = 0 \quad\Longrightarrow\quad \xi_c = S'(x).
$$

Thus the only critical point is $(y_c,\xi_c)=(x, S'(x))$, provided $S$ is smooth. The Hessian matrix in the $(y,\xi)$ variables is

$$
H = \begin{pmatrix}
S''(x) & -I\\
-I & 0
\end{pmatrix},
\qquad \det H = (-1)^n,\quad |\det H| = 1.
$$

Applying the **method of stationary phase** gives the leading asymptotic

$$
(Pu_\lambda)(x) \sim p\bigl(x, S'(x)\bigr)\, a(x)\, e^{i\lambda S(x)} + O(\lambda^{-1}).
$$

In other words, to leading order the operator $P$ simply multiplies the WKB state by the symbol evaluated at the **local wavenumber** $\xi = S'(x)$. This is the **WKB approximation**.

For a more general FIO with phase $\phi(x,y,\theta)$ and WKB input $u(y)=a_u(y)e^{i\lambda S_u(y)}$, the total phase becomes

$$
\Psi(y,\theta;x) = \phi(x,y,\theta) + S_u(y).
$$

The stationary conditions are

$$
\nabla_\theta\Psi = \nabla_\theta\phi = 0,\qquad
\nabla_y\Psi = \nabla_y\phi + \nabla S_u(y) = 0.
$$

These equations define the **critical manifold** that projects to the Lagrangian relation $C$ shifted by the differential $dS_u$.

---

## 4. Stationary Phase and Laplace Method

The **method of stationary phase** evaluates integrals of the form

$$
I(\lambda) = \int_{\mathbb{R}^n} e^{i\lambda\Phi(u)}\, A(u)\, du
$$

as $\lambda\to\infty$. Assume $\Phi$ is real‑valued and has an isolated non‑degenerate critical point at $u_0$ ($\nabla\Phi(u_0)=0$, $\det\Phi''(u_0)\neq0$). Then

$$
I(\lambda) \sim e^{i\lambda\Phi(u_0)} \left(\frac{2\pi}{\lambda}\right)^{n/2}
\frac{A(u_0)}{\sqrt{|\det\Phi''(u_0)|}}\,
\exp\!\left(i\frac{\pi}{4}\,\operatorname{sgn}\Phi''(u_0)\right),
$$

where $\operatorname{sgn}$ denotes the signature (number of positive eigenvalues minus negative eigenvalues) of the Hessian. The factor $\exp(i\pi\,\operatorname{sgn}/4)$ is the **Maslov index** contribution.

If $\Phi$ is complex (e.g., in tunnelling problems) and $\operatorname{Im}\Phi\ge0$, one uses the **Laplace method** (or saddle‑point method) which deforms the contour into the complex domain to pass through a complex saddle point where $\nabla\Phi=0$ and $\operatorname{Im}\Phi$ is minimal.

For degenerate critical points (where $\det\Phi''=0$), the asymptotic involves special functions: Airy functions for fold caustics, Pearcey functions for cusps, etc. These are the **catastrophe integrals** that describe diffraction patterns.

---

## 5. Caustics and Maslov Index

A **caustic** is the projection of the critical set of the Lagrangian manifold $C$ onto configuration space. At a caustic, the Hessian $\partial^2\Phi/\partial y\partial y$ becomes singular, and the stationary phase formula breaks down. The correct asymptotic involves Airy functions (for a fold) or higher‑order special functions.

The **Maslov index** $\mu$ is a topological invariant that counts the number of times a curve in the Lagrangian Grassmannian crosses the “Maslov cycle”. In the stationary phase formula it contributes a phase factor $e^{i\pi\mu/2}$. For a Morse critical point, $\mu = \operatorname{sgn}(\Phi'')/2$. For a fold caustic, the Airy function $Ai$ appears with a characteristic phase shift of $\pi/2$.

In the harmonic oscillator example (see notebook), the exact Mehler kernel exhibits a caustic at $t=\pi/4$ where the prefactor $\sqrt{\lambda/(2\pi\sin2t)}$ diverges, and the bridge correctly reproduces the Airy‑type enhancement.

---

## 6. Egorov’s Theorem

Let $P$ be a $\psi$DO with real principal symbol $p(x,\xi)$. Denote by $\Phi_t$ the Hamiltonian flow generated by $p$, i.e.,

$$
\dot{x} = \frac{\partial p}{\partial\xi},\qquad
\dot{\xi} = -\frac{\partial p}{\partial x}.
$$

Let $Q$ be another $\psi$DO with symbol $q(x,\xi)$. Then **Egorov’s theorem** states that, for any finite time $t$,

$$
e^{-itP/\lambda}\, Q\, e^{itP/\lambda} \quad\text{is a }\psi\text{DO with principal symbol}\quad q\circ\Phi_t + O(\lambda^{-1}).
$$

In other words, the quantum evolution conjugates observables by the classical flow. For quadratic Hamiltonians (e.g., $p=\xi^2$), the conjugation is exact (no $O(\lambda^{-1})$ remainder). This is verified in the notebook for the symbol $q(x,\xi)=\sin x\cdot\xi$, where $q_t(x,\xi)=\sin(x-2\xi t)\,\xi$.

The **Kohn–Nirenberg asymptotic composition** $e^{-itp}\,\#\, q \,\#\, e^{itp}$ yields a power series in $(t/\lambda)$ that converges only when $2k_0t\ll1$; beyond that, the series diverges, illustrating the limitation of the semiclassical expansion for large times.

---

## 7. Quantum Tunnelling and Complex Saddles

When the classical energy $E$ is less than the potential barrier $V(x)$, the WKB wavenumber $\xi(x)=\sqrt{2(E-V(x))}$ becomes imaginary in the classically forbidden region. The phase $\Phi$ in the oscillatory integral becomes complex, and the stationary point moves into the complex plane. The **saddle‑point method** is applied by deforming the contour to pass through a complex saddle where $\partial\Phi/\partial y = 0$ and $\operatorname{Im}\Phi$ is stationary.

For a one‑dimensional barrier, the transmission amplitude is exponentially small:

$$
T \sim e^{-\lambda\Gamma},\qquad \Gamma = \int_{x_-}^{x_+} \sqrt{V(x)-E}\;dx,
$$

where $x_\pm$ are the classical turning points. This is the **WKB tunnelling formula**. The bridge automatically switches to the saddle‑point branch when the total phase is complex, as demonstrated for the Gaussian barrier in the notebook.

---

## 8. Gravitational Lensing as an FIO

In the thin‑lens approximation, a point mass $M$ acts as a gravitational lens. The lensing potential in the source plane (impact parameter $y$) is

$$
\psi(y) = -4M \ln|y|,
$$

so that the FIO phase for the lensed field observed at $b_{\text{obs}}$ is

$$
\Phi(y,\xi; b_{\text{obs}}) = (b_{\text{obs}} - y)\,\xi + k\,y - \theta_E^2 \ln|y|,
$$

where $k$ is the wavenumber and $\theta_E = \sqrt{4M D_{LS}/(D_L D_S)}$ is the Einstein radius. The stationary conditions give the **lens equation**:

$$
\frac{\partial\Phi}{\partial\xi}=0 \;\Longrightarrow\; y_c = b_{\text{obs}},\qquad
\frac{\partial\Phi}{\partial y}=0 \;\Longrightarrow\; \xi_c = k - \frac{\theta_E^2}{y_c}.
$$

Substituting the second into the first yields

$$
b_{\text{obs}} = y_c - \frac{\theta_E^2}{y_c},
$$

which has two solutions $y_\pm$ (two images) for any $b_{\text{obs}}\neq0$. At $b_{\text{obs}}=0$, the two images coalesce into an **Einstein ring**. The Hessian of $\Phi$ at that point is degenerate, leading to an Airy‑type caustic and a bright ring.

The bridge reproduces the interference fringes between the two images, the magnification $\mu = \frac{y}{b}\bigl(1-(\theta_E/y)^2\bigr)^{-1}$, and the Einstein ring enhancement.

---

## 9. Schwarzschild Light Ring and Maslov Phase

For a Schwarzschild black hole ($G=c=1$, mass $M$), the radial effective potential for null geodesics is

$$
V_{\text{eff}}(r) = \left(1-\frac{2M}{r}\right)\frac{L^2}{r^2},
$$

where $L$ is the angular momentum. The Hamiltonian flow of $p_r = (1-2M/r)^2\xi_r^2 + V_{\text{eff}}(r)$ has a circular orbit at $r=3M$ (the **light ring**) where $dV_{\text{eff}}/dr=0$. At this radius the Hessian of the FIO phase vanishes, producing a caustic. The bridge detects this as a rank‑deficient critical point and applies the appropriate Airy‑type asymptotics, leading to an enhancement of the wave amplitude (the “Maslov phase” jump of $\pi/2$).

---

## 10. Cross‑Validation and Semiclassical Corrector

The module provides a **cross‑validation** framework that compares a spectral numerical solver (global, all‑frequency) with the asymptotic bridge (WKB, high‑frequency). The **SpectralSplitter** performs a sharp Fourier cutoff at a wavenumber $k_{\text{cut}}$, decomposing a field into low‑ and high‑frequency parts:

$$
u = u_{\text{low}} + u_{\text{high}},\qquad
\hat u_{\text{low}}(k) = \hat u(k)\,\mathbf{1}_{|k|\le k_{\text{cut}}}.
$$

The **SemiclassicalCorrector** replaces $u_{\text{high}}$ from the solver by the bridge‑computed $u_{\text{high}}$ from the WKB initial condition. This corrects the phase error accumulated by the spectral method at high wavenumbers. The **CrossValidator** then computes

$$
\text{error}(x) = |u_{\text{solver}}(x) - u_{\text{bridge}}(x)|,
$$

and declares the WKB regime **valid** when $\max_x \text{relative error} < 3/\lambda$ (theoretical $O(\lambda^{-1})$ bound). This provides a practical criterion for when the semiclassical approximation is reliable.

---

## References

[1] L. Hörmander, *Fourier Integral Operators I*, Acta Math. 127 (1971).  
[2] J.J. Duistermaat, *Fourier Integral Operators*, Birkhäuser (1996).  
[3] M. Zworski, *Semiclassical Analysis*, AMS Graduate Studies in Mathematics, Vol. 138 (2012).