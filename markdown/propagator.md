# Underlying Theory of the Semiclassical Propagator

[Back to psipy main page](./psipy.md)

This document presents the mathematical and physical foundations of the semiclassical (WKB / Van Vleck–Pauli–Morette) wavefunction propagator implemented in the `propagator` module. It covers the classical Hamiltonian flow, the Van Vleck determinant, the Maslov index, uniform caustic corrections, and the extension to parabolic (heat) and wave equations.

---

## 1. Classical Hamiltonian Mechanics

We consider a system with $d$ degrees of freedom ($d=1,2$ in the implementation). The classical dynamics is governed by a Hamiltonian $H(q,p)$, where $q = (q^1,\dots,q^d)$ are coordinates and $p = (p_1,\dots,p_d)$ are canonical momenta. Hamilton’s equations are

$$
\dot{q}^i = \frac{\partial H}{\partial p_i}, \qquad
\dot{p}_i = -\frac{\partial H}{\partial q^i}.
$$

For a point source located at $q_0$ at time $t=0$, we launch a **fan of rays** with different initial momenta $p_0$. Each ray $(q(t),p(t))$ solves the equations of motion and carries a **classical action**

$$
S(t) = \int_0^t \bigl[ p(\tau)\cdot \dot{q}(\tau) - H(q(\tau),p(\tau)) \bigr] d\tau
      = \int_0^t p(\tau)\cdot \dot{q}(\tau) \, d\tau,
$$

where the last equality holds because $H$ is conserved (the integrand is the Lagrangian). In the special case of a kinetic Hamiltonian $H = \frac12 g^{ij}(q) p_i p_j$ (geodesic motion on a Riemannian manifold), the action simplifies to

$$
S(t) = \int_0^t g_{ij}(q(\tau))\,\dot{q}^i(\tau)\dot{q}^j(\tau)\,d\tau.
$$

---

## 2. Van Vleck–Pauli–Morette Propagator

The **semiclassical approximation** to the quantum propagator $K(q,q_0;t)$ for the Schrödinger equation

$$
i\hbar\,\frac{\partial\psi}{\partial t} = \hat H \psi
$$

is given by the Van Vleck formula [1,2,3]:

$$
\psi(q,t) \approx \sum_{\text{rays }k} 
\frac{1}{\sqrt{|\det J_k(q,t)|}} \;
\exp\!\left( \frac{i}{\hbar} S_k(q,t) - i\frac{\pi}{2}\mu_k \right),
$$

where:

- The sum runs over all classical rays that start at $q_0$ and reach $q$ at time $t$.
- $S_k(q,t)$ is the Hamilton principal function (action) along that ray.
- $J_k(q,t)$ is the **Jacobi matrix** $J_{ij} = \frac{\partial q^i}{\partial p_0^j}$, whose determinant measures how a small spread in initial momenta translates into a spread in final positions. The **Van Vleck amplitude** is $1/\sqrt{|\det J_k|}$.
- $\mu_k$ is the **Maslov index** – an integer counting how many times the ray has passed through a caustic (where $\det J_k = 0$). Each crossing contributes a phase factor $\exp(-i\pi/2) = -i$.

For the **parabolic (heat) equation** $\partial u/\partial t = \hat H u$, the same ray data are used but the ansatz becomes real‑exponential:

$$
u(q,t) \approx \sum_k \frac{1}{\sqrt{|\det J_k|}} \;
\exp\!\left( \frac{S_k(q,t)}{\hbar} \right),
$$

and caustics are corrected with the **parabolic cylinder function** $D_{-1/2}$ instead of the Airy function.

For the **wave equation** $\partial^2 u/\partial t^2 = \hat H u$, the dispersion relation $\omega^2 = H$ splits into two branches: $\omega = +\sqrt{H}$ and $\omega = -\sqrt{H}$. The wavefunction is the coherent sum over both families of rays:

$$
u(q,t) \approx \sum_{k,\pm} \frac{1}{\sqrt{|\det J_k^\pm|}} \;
\exp\!\left( \frac{i}{\hbar} S_k^\pm(q,t) - i\frac{\pi}{2}\mu_k^\pm \right).
$$

---

## 3. Jacobi Determinant and Caustics

The Jacobi matrix evolves according to the **geodesic deviation equation** (for a kinetic Hamiltonian) or, more generally, the linearised Hamilton equations. In 1D, it is a scalar $J(t) = \partial q(t)/\partial p_0$ satisfying

$$
\frac{dJ}{dt} = \frac{\partial^2 H}{\partial p^2}\, K, \qquad
\frac{dK}{dt} = -\frac{\partial^2 H}{\partial q^2}\, J,
$$

with initial conditions $J(0)=0$, $K(0)=1$. The Van Vleck amplitude $1/\sqrt{|J|}$ diverges when $J=0$ – these are **caustics** (focal points). Each time $J$ changes sign, the Maslov index $\mu$ increases by 1.

In 2D, two independent Jacobi fields are integrated (one for each initial momentum direction), and the determinant $\det J = J_{11}J_{22} - J_{12}J_{21}$ is formed.

---

## 4. Uniform Caustic Corrections

The WKB amplitude breaks down at caustics. The correct behaviour is obtained by replacing the singular expression with a uniform asymptotic approximation involving special functions.

### 4.1 Fold Caustic – Airy Function (Schrödinger)

Near a simple fold caustic at $x = x_c$, the wavefunction behaves as

$$
\psi(x) \approx 2\pi\, a_c\, \hbar^{1/6} |\alpha|^{-1/3}\,
\operatorname{Ai}\!\bigl(\xi(x)\bigr)\;
\exp\!\bigl(i S_c/\hbar\bigr),
$$

where:

- $\alpha = d(\det J)/dx$ evaluated at the caustic,
- $\xi(x) = \bigl(\alpha/(2\hbar)\bigr)^{1/3} (x - x_c)$,
- $a_c$ is the “physical” amplitude obtained by undoing the $1/\sqrt{|\det J|}$ regularisation.

### 4.2 Fold Caustic – Parabolic Cylinder Function (Heat equation)

For the heat‑type equation, the uniform approximation uses the parabolic cylinder function of order $-1/2$:

$$
u(x) \approx a_c\, \hbar^{1/4} |\alpha|^{-1/4}\,
D_{-1/2}\!\bigl(\zeta(x)\bigr)\;
\exp\!\bigl(S_c/\hbar\bigr),
$$

with $\zeta(x) = \bigl(\alpha/\hbar\bigr)^{1/4} (x - x_c)$.

### 4.3 Cusp Caustic – Pearcey Integral (Schrödinger, 2D)

When both partial derivatives of $\det J$ vanish simultaneously, the caustic is a cusp. The normal form is the Pearcey integral

$$
\Psi(x,y) = \int_{-\infty}^{\infty} \exp\!\bigl(i(t^4 + x t^2 + y t)\bigr)\, dt,
$$

which exhibits the characteristic “swallowtail” interference pattern. The implementation uses the `asymptotic` module to evaluate the integral and blends the result with a Gaussian taper.

---

## 5. Maslov Index – Counting Caustic Crossings

The Maslov index $\mu$ is an integer that accumulates each time the ray passes through a caustic. In 1D, it is simply the number of sign changes of $J(t)$ (ignoring exact zeros). In 2D, the situation is more subtle because $\det J$ may stay positive even when individual Jacobi field components cross zero (e.g., isotropic harmonic oscillator). The implementation therefore monitors the four scalar Jacobi field components $J_{1x}, J_{1y}, J_{2x}, J_{2y}$ and counts their zero crossings, then divides by 2 to obtain the correct $\mu$.

---

## 6. Riemannian Metrics and Geodesic Motion

When the Hamiltonian is purely kinetic, $H = \frac12 g^{ij}(q) p_i p_j$, the geometry is encoded in the **metric tensor** $g_{ij}(q)$. The code uses a `Metric` class that provides:

- The metric components $g_{ij}$ and their inverses $g^{ij}$ as SymPy expressions,
- Conversion between velocities $v^i = \dot{q}^i$ and momenta $p_i = g_{ij} v^j$,
- The geodesic equations $\ddot{q}^i + \Gamma^i_{jk}\dot{q}^j\dot{q}^k = 0$ and the Jacobi equation.

This allows the propagator to work on curved manifolds (sphere, paraboloid, etc.) without any modification.

---

## 7. Numerical Implementation Overview

The actual computation follows these steps:

1. **Build the Hamiltonian** – from a `Metric` (kinetic) or a user‑supplied SymPy expression.
2. **Launch the ray fan** – convert initial velocities to momenta (if needed) and integrate Hamilton’s equations using a symplectic integrator (Verlet or RK45).
3. **Compute the Jacobi determinant** – solve the variational ODE (1D) or the Jacobi equation (2D) along each ray.
4. **Accumulate the action** – integrate $p\cdot\dot{q}$ along the ray (exact for metric‑based momenta).
5. **Count caustic crossings** – determine $\mu$ from sign changes / zero crossings.
6. **Assemble the wavefunction on a grid** – interpolate the scattered ray data (actions, determinants, Maslov indices) onto a regular grid using `scipy.interpolate.griddata`.
7. **Apply uniform caustic corrections** – detect points where $|\det J|$ is small, estimate $\alpha = \nabla\det J$, and replace the WKB value by the Airy/Pearcey/parabolic‑cylinder patch in a small window around each caustic.
8. **For the wave equation** – repeat the procedure for the $+ \sqrt{H}$ and $- \sqrt{H}$ branches and sum the results.
9. **Return** a `WKBResult` object containing the gridded wavefunction, the full ray data, and all intermediate quantities for visualisation.

---

## 8. References

1. Van Vleck, J.H. (1928). "The correspondence principle in the statistical interpretation of quantum mechanics". *Proc. Natl. Acad. Sci.* **14**, 178.
2. Morette, C. (1951). "On the definition and approximation of Feynman's path integrals". *Phys. Rev.* **81**, 848.
3. Gutzwiller, M.C. (1990). *Chaos in Classical and Quantum Mechanics*. Springer, New York. (Chapter 12)
4. Maslov, V.P. & Fedoriuk, M.V. (1981). *Semi-Classical Approximation in Quantum Mechanics*. Reidel, Dordrecht.
5. Berry, M.V. & Mount, K.E. (1972). "Semiclassical approximations in wave mechanics". *Rep. Prog. Phys.* **35**, 315.

---

*This document reflects the theory as implemented in the `propagator.py` module, which supports Schrödinger, parabolic (heat), and wave equations on flat or curved manifolds, with uniform caustic corrections up to cusp singularities.*