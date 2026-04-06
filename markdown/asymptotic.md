# Underlying Theory of Large‑Parameter Asymptotics

[Back to psipy main page](./psipy.md)

This document summarises the mathematical foundations of the asymptotic approximation methods implemented in the code.  
We consider integrals of the form  

$$
I(\lambda)=\int_{\mathbb{R}^n} a(\mathbf{x})\,e^{i\lambda\phi(\mathbf{x})}\,d\mathbf{x},\qquad \lambda\to+\infty,
$$

where the **phase** $\phi(\mathbf{x})$ and **amplitude** $a(\mathbf{x})$ are smooth functions, and $\lambda$ is a large real parameter.  
The behaviour of $I(\lambda)$ is dominated by the neighbourhood of **critical points** of $\phi$ – points where $\nabla\phi(\mathbf{x}_c)=0$.  
Depending on the nature of $\phi$ (real, purely imaginary, or genuinely complex), three different methods apply.

---

## 1. Stationary Phase (Real Phase)

When $\phi(\mathbf{x})$ is **real**, the integrand oscillates rapidly for large $\lambda$.  
The main contribution comes from points where the phase is stationary, i.e. $\nabla\phi(\mathbf{x}_c)=0$.  
The classical stationary‑phase formula for a **non‑degenerate** (Morse) critical point reads  

$$
I_0(\lambda)=\left(\frac{2\pi}{\lambda}\right)^{\!n/2}\frac{a(\mathbf{x}_c)\,e^{i\lambda\phi(\mathbf{x}_c)}\,e^{i\pi\mu/4}}{\sqrt{|\det H|}},
$$

where  

* $H=\nabla^2\phi(\mathbf{x}_c)$ is the Hessian matrix,  
* $\mu=n-2\sigma$ is the **Maslov index**, $\sigma$ being the number of negative eigenvalues of $H$ (the signature),  
* the square root $\sqrt{|\det H|}$ is taken positive.  

This leading term is of order $O(\lambda^{-n/2})$.

### 1.1 Second‑order correction (Morse points)

The next term, $I_1(\lambda)=O(\lambda^{-n/2-1})$, is obtained by expanding the integrand to higher order and performing Gaussian integrals.  
It involves the amplitude derivatives and the cubic and quartic derivatives of $\phi$:

$$
I_1(\lambda)=\frac{I_0(\lambda)}{i\lambda}\left[
\frac12\operatorname{tr}(H^{-1}\nabla^2 a) \;-\; \frac12\langle H^{-1}\nabla a,\,V\rangle
\;+\; \frac{a(\mathbf{x}_c)}{24}\bigl(5S_3-3S_4\bigr)
\right],
$$

where  

* $V_k = (H^{-1})_{ij}\,\partial_{ijk}\phi$ (coupling of the inverse Hessian with the third derivatives),  
* $S_4 = (H^{-1})_{ij}(H^{-1})_{kl}\,\partial_{ijkl}\phi$ (contraction of the fourth derivatives),  
* $S_3 = (H^{-1})_{ij}(H^{-1})_{kl}(H^{-1})_{mn}\,\partial_{ikm}\phi\,\partial_{jln}\phi$ (the “theta‑graph” term).  

The correction is often essential for obtaining accurate approximations at moderate $\lambda$.

### 1.2 Degenerate stationary points

When the Hessian is singular, the critical point is **degenerate**. The asymptotic order changes, and special functions (Airy, Pearcey) appear.

#### 1.2.1 Airy singularity (1D)

If $n=1$ and the first non‑zero term in the Taylor expansion is cubic,  
$\phi(x)\sim \alpha\,x^3/3$ near $x_c=0$. Then  

$$
I(\lambda)= \int_{-\infty}^{\infty} e^{i\lambda\alpha x^3/3}\,dx
= 2\pi\,\operatorname{Ai}(0)\;(\lambda|\alpha|)^{-1/3},
$$

where $\operatorname{Ai}(0)=3^{-2/3}/\Gamma(2/3)\approx 0.355028$.  
The decay order is $O(\lambda^{-1/3})$.

#### 1.2.2 Airy singularity (2D, corank 1)

When one direction is degenerate (cubic) and the transverse direction is non‑degenerate (quadratic),  
$\phi(\mathbf{x})\sim \alpha\,u^3/3 + \beta\,v^2/2$. The asymptotic contribution factorises:

$$
I(\lambda)= \left[2\pi\,\operatorname{Ai}(0)\,(\lambda|\alpha|)^{-1/3}\right]
\cdot\left[\sqrt{\frac{2\pi}{\lambda|\beta|}}\,e^{i\pi\operatorname{sign}(\beta)/4}\right]
= O(\lambda^{-5/6}).
$$

#### 1.2.3 Pearcey singularity (cusp, 2D)

If the cubic coefficient vanishes but the quartic term is non‑zero,  
$\phi(\mathbf{x})\sim \gamma\,u^4/4 + \beta\,v^2/2$. Then  

$$
I(\lambda)= \left[\frac12\Gamma\!\left(\frac14\right)
\left(\frac{4}{\lambda|\gamma|}\right)^{\!1/4}
e^{i\pi\operatorname{sign}(\gamma)/8}\right]
\cdot\left[\sqrt{\frac{2\pi}{\lambda|\beta|}}\,
e^{i\pi\operatorname{sign}(\beta)/4}\right]
= O(\lambda^{-3/4}).
$$

---

## 2. Laplace’s Method (Purely Imaginary Phase)

If $\phi(\mathbf{x})=i\psi(\mathbf{x})$ with $\psi(\mathbf{x})$ **real**, the integral becomes  

$$
I(\lambda)=\int a(\mathbf{x})\,e^{-\lambda\psi(\mathbf{x})}\,d\mathbf{x}.
$$

The integrand is exponentially concentrated near the **global minimum** of $\psi$ (assuming $\psi$ is positive definite).  
For a non‑degenerate minimum at $\mathbf{x}_c$ ($\nabla\psi(\mathbf{x}_c)=0$, Hessian $H$ positive definite), the leading term is  

$$
I_0(\lambda)=a(\mathbf{x}_c)\,e^{-\lambda\psi(\mathbf{x}_c)}\left(\frac{2\pi}{\lambda}\right)^{\!n/2}\frac{1}{\sqrt{\det H}}.
$$

The first correction (order $\lambda^{-n/2-1}$) has a structure analogous to the stationary‑phase correction, but with real arithmetic and a factor $1/\lambda$ instead of $1/(i\lambda)$:

$$
I_1(\lambda)=I_0(\lambda)\,\frac{1}{\lambda}\left[
\frac12\operatorname{tr}(H^{-1}\nabla^2 a)
-\frac12\langle H^{-1}\nabla a,\,V\rangle
-\frac18 S_4 + \frac{5}{24}S_3
\right],
$$

where $V$, $S_3$, $S_4$ are the same contractions as before, evaluated with the real $\psi$.  
The total approximation is $I(\lambda)\approx I_0(\lambda)+I_1(\lambda)$.

---

## 3. Saddle‑Point Method (Complex Phase)

When $\phi(\mathbf{x})$ is **genuinely complex**, the integral is both oscillatory and exponentially damped.  
The dominant contribution comes from **saddle points** in the complex plane (or $\mathbb{C}^n$) satisfying $\nabla\phi(\mathbf{z}_c)=0$.  
The contour of integration is deformed to pass through these saddles along **steepest‑descent** paths.

For a non‑degenerate saddle $\mathbf{z}_c$, the leading term is formally identical to the Morse stationary‑phase formula, but with complex arguments:

$$
I_0(\lambda)=\left(\frac{2\pi}{\lambda}\right)^{\!n/2}
a(\mathbf{z}_c)\,e^{i\lambda\phi(\mathbf{z}_c)}\,
\frac{1}{\sqrt{\det H(\mathbf{z}_c)}},
$$

where the square root is taken on the principal branch.  
Unlike the real case, the exponential factor $e^{i\lambda\phi(\mathbf{z}_c)}$ may provide both oscillation and exponential growth/decay, and the phase of $\sqrt{\det H}$ contains the Maslov‑like index of the complex saddle.

**Important caveats**  
* The existence of a saddle point does **not** guarantee that the original real contour can be deformed to it without crossing other singularities (Picard‑Lefschetz theory).  
* The correct branch of the square root must be chosen according to the direction of steepest descent.  
* Higher‑order corrections are not implemented for complex saddles in the code; only the leading term is used.

---

## 4. Classification of Critical Points

The type of a critical point is determined by the rank of the Hessian and the lowest non‑zero term in the Taylor expansion of $\phi$ along the null direction(s).

| Hessian rank | Lowest non‑zero term | Singularity type | Decay order $O(\lambda^{-p})$ |
|--------------|----------------------|------------------|--------------------------------|
| $n$ (full) | – (Morse)            | Morse            | $n/2$                        |
| $n-1$      | cubic                | Airy (corank 1)  | $1/3$ (1D), $5/6$ (2D)     |
| $n-1$      | quartic              | Pearcey (cusp)   | $3/4$ (2D)                   |
| $< n-1$    | higher order         | higher‑order     | not implemented                |

The classification is performed by analysing the Hessian eigenvalues and, for rank deficiency, projecting the third‑ and fourth‑order derivatives onto the null eigenvector.

---

## References

1. **Hörmander, L.** *The Analysis of Linear Partial Differential Operators I*, Springer, 1983.  
2. **Olver, F. W. J.** *Asymptotics and Special Functions*, Academic Press, 1974 (reprinted 1997).  
3. **Wong, R.** *Asymptotic Approximations of Integrals*, Academic Press, 1989.  
4. **Bleistein, N. & Handelsman, R.** *Asymptotic Expansions of Integrals*, Holt, Rinehart & Winston, 1975.  
5. **Berry, M. V. & Howls, C. J.** “High orders of the Weyl expansion for quantum billiards”, *Physical Review E* 50(5), 3577–3595, 1994.  
6. **Delabaere, E. & Howls, C. J.** “Global asymptotics for multiple integrals with boundaries”, *Duke Mathematical Journal* 112(2), 199–264, 2002.