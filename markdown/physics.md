# Underlying Theory of the `physics` Module

[Back to psipy main page](./psipy.md)

This document describes the mathematical foundations used in the `physics` module for Lagrangian–Hamiltonian transformations and pseudo‑differential operator formalisms.

## 1. Legendre Transform (Classical)

In classical mechanics, a system is described by a Lagrangian  

$$
L(x, u, p)
$$

where  
- $x$ denotes position (or a set of coordinates),  
- $u$ is a field variable (optional),  
- $p$ represents generalised velocities (often denoted $\dot{x}$).

The conjugate momenta are defined as  

$$
\xi = \frac{\partial L}{\partial p}.
$$

When the Hessian matrix $\frac{\partial^2 L}{\partial p^2}$ is non‑singular (strict convexity in $p$), the relation $\xi = \partial L/\partial p$ can be inverted to express $p$ as a function of $\xi$. The **Hamiltonian** is then obtained via the Legendre transform:

$$
H(x, u, \xi) = \xi \cdot p - L(x, u, p),
$$

where $p$ is replaced by its expression in terms of $\xi$.

The inverse Legendre transform (from Hamiltonian back to Lagrangian) is given by

$$
L(x, u, p) = \xi \cdot p - H(x, u, \xi),
$$

with $\xi$ determined from $p = \partial H/\partial \xi$.

## 2. Legendre–Fenchel Transform (Convex Conjugate)

If the Lagrangian is not strictly convex, or if the Hessian is singular, the classical Legendre transform is not well‑defined. In such cases one uses the **Legendre–Fenchel transform** (or convex conjugate), which always yields a convex Hamiltonian (in $\xi$):

$$
H(x, u, \xi) = \sup_{p}\; \bigl( \xi \cdot p - L(x, u, p) \bigr).
$$

This definition works for any Lagrangian, even those that are non‑differentiable or non‑convex. The supremum is taken over all admissible velocities $p$. When $L$ is convex and differentiable, the supremum is attained at the unique point where $\partial L/\partial p = \xi$, and the Legendre–Fenchel transform reduces to the classical Legendre transform.

The Legendre–Fenchel transform is an involution on the class of convex, lower‑semicontinuous functions (Fenchel–Moreau theorem).

## 3. Hamiltonian Decomposition into Local and Non‑Local Parts

In the context of pseudo‑differential operators, the Hamiltonian $H(x, \xi)$ is interpreted as the symbol of an operator. A **local operator** (differential operator) corresponds to a symbol that is a polynomial in $\xi$. A **non‑local operator** (pseudo‑differential operator) arises when the symbol contains non‑polynomial expressions such as $\sqrt{1+\xi^2}$, $|\xi|$, or $\operatorname{sgn}(\xi)$.

The module decomposes a given Hamiltonian symbol as:

$$
H(x,\xi) = H_{\text{poly}}(x,\xi) + H_{\text{nonlocal}}(x,\xi),
$$

where  
- $H_{\text{poly}}$ collects all terms that are polynomial in $\xi$ (and do not contain functions like $\sqrt{\;}$, $|\cdot|$, or $\operatorname{sgn}$),  
- $H_{\text{nonlocal}}$ collects the remaining terms.

This decomposition is heuristic but useful for identifying the principal symbol and lower‑order terms of a pseudo‑differential operator.

## 4. Formal PDEs from a Hamiltonian Symbol

Let $\psi\,\text{Op}(H, u)$ denote the pseudo‑differential operator whose symbol is $H(x, \xi)$. The module generates formal partial differential equations by replacing the symbol with this operator placeholder. Three standard types are supported:

### 4.1 Stationary (eigenvalue) equation

$$
\psi\,\text{Op}(H, u) = E\,u,
$$

where $E$ is a real constant (energy eigenvalue).

### 4.2 Schrödinger equation

$$
i\,\frac{\partial u}{\partial t} = \psi\,\text{Op}(H, u).
$$

Here $i$ is the imaginary unit, and $t$ is time.

### 4.3 Wave equation

$$
\frac{\partial^2 u}{\partial t^2} + \psi\,\text{Op}(H, u) = 0.
$$

These equations are “formal” in the sense that $\psi\,\text{Op}(H, u)$ is left as an unevaluated symbolic operator; the actual expansion into derivatives (for polynomial $H$) or into integral operators (for non‑polynomial $H$) is not performed by the module.

## 5. Supported Dimensions

The module currently implements the transforms for **one‑dimensional** and **two‑dimensional** coordinate spaces only. In 1D the conjugate momentum variable is denoted $\xi$; in 2D the pair $(\xi, \eta)$ is used. Extension to higher dimensions is conceptually straightforward but not implemented in the provided code.

## References

1. Arnold, V. I. *Mathematical Methods of Classical Mechanics*, Springer‑Verlag, 1989 (2nd ed.). §14: Legendre Transform.
2. Rockafellar, R. T. *Convex Analysis*, Princeton University Press, 1970. Chapter 12: Conjugate Functions.
3. Evans, L. C. *Partial Differential Equations*, American Mathematical Society, 2010 (2nd ed.). §4.3: Hamilton–Jacobi Equations.
4. Folland, G. B. *Quantum Field Theory: A Tourist Guide for Mathematicians*, American Mathematical Society, 2008. §1: Legendre Transform and Quantisation.