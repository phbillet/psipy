# Underlying Theory of the Multidimensional WKB Method

[Back to psipy main page](./psipy.md)

## 1. The WKB Ansatz

We consider a linear partial differential equation of the form  

$$
P(x, -i\varepsilon\nabla)\, u(x) = 0, \qquad \varepsilon \to 0^+,
$$

where $P(x,\xi)$ is the (pseudo‑differential) symbol of the operator.  
The Wentzel–Kramers–Brillouin (WKB) method seeks an asymptotic solution as a sum over “rays” (bicharacteristics):

$$
u(x) \;\approx\; \sum_j A_j(x)\, e^{i S_j(x)/\varepsilon},
$$

with a small parameter $\varepsilon$ representing the inverse wavelength.  
The phase $S(x)$ satisfies the **eikonal equation**, and the amplitude $A(x)$ is expanded in powers of $\varepsilon$:

$$
A(x) = a_0(x) + \varepsilon\, a_1(x) + \varepsilon^2 a_2(x) + \cdots .
$$

Thus the full WKB ansatz is  

$$
u(x) \;\sim\; e^{i S(x)/\varepsilon} \sum_{k=0}^{\infty} \varepsilon^{k} a_k(x).
$$

---

## 2. Eikonal Equation (Hamilton–Jacobi)

Inserting the ansatz into $P(x,-i\varepsilon\nabla)u=0$ and collecting the leading order in $\varepsilon$ gives the **eikonal equation**:

$$
P\bigl(x,\nabla S(x)\bigr) = 0.
$$

This is a first‑order non‑linear PDE. It is solved by the method of characteristics, which yields **Hamilton’s equations** for the rays $(x(t),\xi(t))$ in phase space:

$$
\begin{aligned}
\frac{dx}{dt} &= \frac{\partial P}{\partial \xi}(x,\xi), \\[4pt]
\frac{d\xi}{dt} &= -\frac{\partial P}{\partial x}(x,\xi).
\end{aligned}
$$

Along each ray the phase evolves as  

$$
\frac{dS}{dt} = \xi\cdot\frac{\partial P}{\partial\xi} - P.
$$

---

## 3. Transport Equations for the Amplitudes

The next orders in $\varepsilon$ give a hierarchy of linear transport equations for the amplitudes $a_k$.  

**Order $\varepsilon^0$ (leading amplitude):**

$$
\frac{\partial P}{\partial\xi}\cdot\nabla a_0 \;+\; \frac12\Bigl(\nabla_\xi\cdot\nabla_x P\Bigr) a_0 = 0.
$$

Along a ray this becomes an ordinary differential equation:

$$
\frac{d a_0}{dt} = -\frac12\; \bigl(\nabla_\xi\cdot\nabla_x P\bigr)\; a_0,
$$

where $\nabla_\xi\cdot\nabla_x P = \sum_{i,j} \frac{\partial^2 P}{\partial\xi_i\partial x_j}$ is the **geometric spreading** term.

**Higher orders ($k\ge 1$):**

$$
\frac{d a_k}{dt} = -\frac12\bigl(\nabla_\xi\cdot\nabla_x P\bigr)a_k \;+\; \mathcal{F}_k\bigl(a_{k-1}, a_{k-2},\dots\bigr),
$$

where $\mathcal{F}_k$ involves higher derivatives of $P$ (up to order $k+1$) and couples lower‑order amplitudes. In particular:

- For $k=1$, the source term contains $\frac{\partial^2 P}{\partial\xi_i\partial x_j}$ (the mixed Hessian).
- For $k=2$, third derivatives of $P$ appear, etc.

---

## 4. Stability Matrix and Caustics

To monitor the focusing of nearby rays, one integrates the **variational equations** along each ray. Let $q$ be a parameter along the initial curve (e.g., the initial position $x_0$). Define the **stability matrix**  

$$
J(t) = \frac{\partial x(t)}{\partial q}.
$$

Its evolution follows from differentiating Hamilton’s equations:

$$
\frac{dJ}{dt} = \frac{\partial^2 P}{\partial\xi\partial x}\, J.
$$

For two dimensions, $J$ is a $2\times2$ matrix, and the variational equations also involve the momentum derivatives $K = \partial\xi/\partial q$.

A **caustic** occurs when the ray family focuses, i.e. when $\det J(t) = 0$ (in 1D, when $J_{11}=0$). At such points the standard WKB amplitude diverges because the geometric spreading factor vanishes.

---

## 5. Caustic Corrections

### 5.1 Maslov Index

When a ray passes through a caustic, the phase $S$ acquires an additional shift of $\pi/2$ (in 1D) or $\pi/2$ times the **Maslov index** (the number of negative eigenvalues of the Hessian of the phase). The corrected phase becomes  

$$
S_{\text{corr}} = S + \frac{\pi}{2}\, \nu,
$$

where $\nu$ counts how many caustics have been crossed.

### 5.2 Uniform Approximation for a Fold Caustic (Airy Function)

Near a simple (fold) caustic the ray family forms a cusp in physical space. The uniform asymptotic approximation replaces the diverging WKB expression by an Airy function:

$$
u(x) \;\approx\; C\; \operatorname{Ai}\!\left(\frac{z}{\varepsilon^{2/3}}\right) e^{i S_c/\varepsilon},
$$

where $z$ is a scaled distance from the caustic, $S_c$ is the phase at the caustic, and $C$ is a constant determined by matching. The Airy function $\operatorname{Ai}$ remains finite and oscillatory on one side while decaying exponentially on the other.

### 5.3 Uniform Approximation for a Cusp Caustic (Pearcey Function)

For a cusp caustic (the simplest higher catastrophe), the uniform approximation involves the **Pearcey integral**:

$$
\Psi(X,Y) = \int_{-\infty}^{\infty} \exp\!\bigl(i(t^4 + X t^2 + Y t)\bigr)\, dt.
$$

The solution is expressed as  

$$
u(x,y) \;\approx\; D\; \Psi\!\left(\frac{x}{\varepsilon^{1/2}}, \frac{y}{\varepsilon^{1/4}}\right) e^{i S_c/\varepsilon},
$$

with appropriate scaling of the coordinates $(x,y)$ relative to the cusp point.

---

## 6. Summary of the Algorithm

1. **Symbolic setup** – define the symbol $P(x,\xi)$, compute its derivatives, and lambdify them for numerical evaluation.
2. **Initial data** – specify initial positions, momenta, phase, and amplitudes on a set of rays.
3. **Ray tracing** – integrate the coupled ODEs (Hamilton, phase, stability, amplitudes) for all rays simultaneously using a vectorised ODE solver.
4. **Caustic detection** – monitor $\det J(t)$; when it crosses a threshold, record a caustic event with its type (fold, cusp, etc.).
5. **Interpolation** – map the ray‑based data (phase, amplitudes) onto a regular spatial grid using linear interpolation (1D) or scattered‑data interpolation (2D).
6. **Caustic correction** – if desired, apply Maslov phase shifts, Airy (fold) or Pearcey (cusp) uniform approximations near the detected caustics.
7. **Output** – return the complex WKB field $u(x)$ on the grid, together with diagnostic data (rays, caustics, amplitude components).

---

## References

The mathematical foundations are drawn from:

- Maslov, V. P. & Fedoriuk, M. V. (1981). *Semi‑Classical Approximation in Quantum Mechanics*. Reidel.
- Duistermaat, J. J. (1974). “Oscillatory integrals, Lagrange immersions and unfolding of singularities”. *Comm. Pure Appl. Math.* **27**, 207–281.
- Kravtsov, Yu. A. & Orlov, Yu. I. (1999). *Caustics, Catastrophes and Wave Fields*. Springer.
- Berry, M. V. & Howls, C. J. (1994). “High orders of the Weyl expansion for quantum billiards”. *Phys. Rev. E* **50**(5), 3577–3595.