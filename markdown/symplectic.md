# Underlying Theory of Symplectic Geometry and Hamiltonian Mechanics

[Back to psipy main page](./psipy.md)

This document summarises the mathematical foundations of the `symplectic` module and the physical concepts illustrated in the accompanying notebooks.

---

## 1. Symplectic Structure

A **symplectic manifold** is a pair $(M, \omega)$ where $M$ is a smooth manifold of even dimension $2n$ and $\omega$ is a closed, non‑degenerate 2‑form.  
In canonical coordinates $(x_1, p_1, \dots, x_n, p_n)$ on $\mathbb{R}^{2n}$ (Darboux coordinates) the symplectic form takes the standard expression  

$$
\omega = \sum_{i=1}^{n} dx_i \wedge dp_i .
$$

The **symplectic matrix** (or Poisson tensor) $J$ satisfies $J^T = -J$ and $J^2 = -I_{2n}$. In canonical coordinates  

$$
J = \begin{pmatrix}
0 & -I_n \\ I_n & 0
\end{pmatrix},\qquad
\omega(u,v) = u^T J^{-1} v .
$$

---

## 2. Hamiltonian Mechanics

A **Hamiltonian** $H: M \to \mathbb{R}$ defines a vector field $X_H$ (the Hamiltonian vector field) via  

$$
\iota_{X_H} \omega = dH .
$$

In canonical coordinates this gives Hamilton’s equations  

$$
\dot{x}_i = \frac{\partial H}{\partial p_i},\qquad
\dot{p}_i = -\frac{\partial H}{\partial x_i}.
$$

The flow of $X_H$ is called the **Hamiltonian flow**; it preserves the symplectic form (i.e. it is a symplectomorphism).  

The **symplectic gradient** $X_f$ of a function $f$ is defined analogously and satisfies $X_f = J^{-1}\nabla f$.

---

## 3. Poisson Bracket

For two observables $f,g: M \to \mathbb{R}$ the Poisson bracket is  

$$
\{f,g\} = \omega(X_f, X_g) = \sum_{i=1}^{n}
\left( \frac{\partial f}{\partial x_i}\frac{\partial g}{\partial p_i}
- \frac{\partial f}{\partial p_i}\frac{\partial g}{\partial x_i} \right).
$$

It is bilinear, antisymmetric, satisfies the Jacobi identity, and acts as a derivation.  
The time evolution of any observable is $\dot{f} = \{f, H\}$.  
A function $L$ is a **first integral** (conserved quantity) iff $\{L, H\} = 0$.

---

## 4. Liouville’s Theorem

The Hamiltonian flow preserves the phase‑space volume (Liouville’s theorem). In canonical coordinates  

$$
\frac{d}{dt} \int_{\Omega_t} dx_1 dp_1 \cdots dx_n dp_n = 0 .
$$

Equivalently, the divergence of the Hamiltonian vector field vanishes:  
$\nabla \cdot X_H = 0$. The volume form $\omega^{\wedge n} = n! \, dx_1\wedge dp_1\wedge\cdots\wedge dx_n\wedge dp_n$ is invariant under the flow.

---

## 5. Darboux Theorem

Any symplectic manifold is locally symplectomorphic to $(\mathbb{R}^{2n}, \omega_0)$ with the canonical symplectic form.  
Thus there always exist local coordinates in which Hamilton’s equations take the standard form. This is the symplectic analogue of the flatness of Riemannian manifolds.

---

## 6. Gromov Non‑Squeezing Theorem

A symplectic ball $B^{2n}(r)$ can be symplectically embedded into a cylinder  
$Z^{2n}(R) = B^2(R) \times \mathbb{R}^{2n-2}$ **if and only if** $r \le R$.  
This is a rigidity result: volume‑preserving embeddings can squeeze a ball arbitrarily thin, but symplectic maps cannot reduce the area of the projection onto any canonical plane below $\pi r^2$.

---

## 7. Action‑Angle Variables (1‑DOF)

For a 1‑degree‑of‑freedom system with bounded motion, the **action variable** is  

$$
I(E) = \frac{1}{2\pi}\oint p\,dx ,
$$

the area enclosed by the orbit in phase space divided by $2\pi$.  
In action‑angle coordinates $(I, \theta)$ the Hamiltonian becomes $H = H(I)$ and the angle evolves as  

$$
\dot{\theta} = \omega(I) = \frac{dH}{dI}.
$$

The action is an adiabatic invariant and is the classical precursor of the quantum number.

---

## 8. Integrability and KAM Theory

**Liouville integrability**: A Hamiltonian system with $n$ degrees of freedom is integrable if there exist $n$ functionally independent first integrals $F_1=H, F_2,\dots,F_n$ that are in involution ($\{F_i,F_j\}=0$).  
The joint level sets are Lagrangian tori; the motion is confined to these tori and is quasi‑periodic.

The **Kolmogorov–Arnold–Moser (KAM) theorem** states that under small perturbations of an integrable system, most non‑resonant tori survive (they are only slightly deformed), while resonant tori break up, giving rise to chaotic motion. The phase space becomes a mixture of regular (KAM) tori and chaotic regions.

---

## 9. Spectral Statistics and Integrability

**Berry–Tabor conjecture** (1977): The quantum energy level spacings of an integrable system follow a **Poisson distribution**  
$P(s) = e^{-s}$, where $s$ is the normalised nearest‑neighbour spacing.

**Bohigas–Giannoni–Schmit (BGS) conjecture** (1984): For a classically chaotic system with time‑reversal symmetry, the level spacings follow the **Wigner (GOE) surmise**  
$P(s) = \frac{\pi}{2}s\,e^{-\pi s^2/4}$.

The **Brody distribution** interpolates between the two extremes:

$$
P(s;\beta) = (\beta+1)\,b\,s^{\beta}\,e^{-b s^{\beta+1}},\qquad
b = \left[\Gamma\!\left(\frac{\beta+2}{\beta+1}\right)\right]^{\beta+1},
$$

with $\beta=0$ (Poisson) and $\beta=1$ (Wigner).  
Fitting $\beta$ to a spectrum provides a quantitative measure of integrability vs chaos.

---

## 10. Topological Monodromy

For a 2‑DOF integrable system, the energy‑momentum map $F = (H, L): M \to \mathbb{R}^2$ has fibres that are Liouville tori. When a loop in the image plane encircles a critical value (a focus‑focus singularity), the action lattice undergoes a linear transformation  

$$
\begin{pmatrix}I_1\\ I_2\end{pmatrix} \mapsto M \begin{pmatrix}I_1\\ I_2\end{pmatrix},\qquad M \in GL(2,\mathbb{Z}).
$$

If $M \neq I$, the system exhibits **non‑trivial monodromy** – a global obstruction to smooth action‑angle coordinates.  
The spherical pendulum is a classic example with $M = \begin{pmatrix}1&1\\0&1\end{pmatrix}$.

---

## 11. Floer Homology

Floer homology is an infinite‑dimensional Morse theory for the **symplectic action functional** on the loop space of a symplectic manifold. Its critical points correspond to periodic orbits of a Hamiltonian flow. The differential counts “pseudo‑holomorphic cylinders” connecting these orbits.

Floer homology is invariant under Hamiltonian isotopies and is a powerful tool for proving the **Arnold conjecture**:  
For a non‑degenerate Hamiltonian diffeomorphism on a closed symplectic manifold, the number of fixed points is at least the sum of the Betti numbers of the manifold.

---

## 12. Bohr–Sommerfeld Quantization

The old quantum theory (Bohr–Sommerfeld) postulates that the action integral of a periodic orbit is quantised:

$$
I(E_n) = \frac{1}{2\pi}\oint p\,dx = \hbar\left(n + \frac{\mu}{4}\right),
$$

where $\mu$ is the Maslov index (number of caustics along the orbit). For a 1‑DOF bounded system $\mu = 2$, giving the familiar $n + \frac12$ correction.  
This rule provides a semiclassical bridge between classical symplectic geometry and the quantum energy spectrum.

---

## 13. Summary of Key Mathematical Objects

| Object | Symbol | Role |
|--------|--------|------|
| Symplectic form | $\omega$ | Defines geometry of phase space |
| Poisson bracket | $\{\cdot,\cdot\}$ | Algebra of observables |
| Hamiltonian vector field | $X_H$ | Generates time evolution |
| Action variable | $I$ | Adiabatic invariant; quantisation |
| Liouville torus | – | Invariant manifold of integrable system |
| Brody parameter | $\beta$ | Chaos indicator from level spacings |
| Monodromy matrix | $M$ | Topological obstruction to action‑angle coordinates |
| Maslov index | $\mu$ | Caustic count in Bohr–Sommerfeld rule |

---

These concepts form the theoretical backbone of the `symplectic` module and are demonstrated numerically in the accompanying notebooks.