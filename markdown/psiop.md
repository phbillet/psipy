# Underlying Theory of Pseudo‑Differential Operators

[Back to psipy main page](./psipy.md)

This document summarises the mathematical foundations of pseudo‑differential operators (ΨDOs) as implemented in the `psiop` package.  
The presentation follows the standard framework of microlocal analysis and semiclassical calculus.

## 1. Definition and Quantisation

Let $ u(x) $ be a smooth function on $ \mathbb{R}^n $ (with $ n = 1 $ or $ 2 $ in this implementation).  
A **pseudo‑differential operator** $ P $ with symbol $ p(x,\xi) $ is formally defined by the **Kohn‑Nirenberg quantisation**

$$
(P u)(x) \;=\; \frac{1}{(2\pi)^n} \int_{\mathbb{R}^n} e^{i x\cdot\xi}\, p(x,\xi)\, \hat u(\xi)\, d\xi,
$$

where  
$ \hat u(\xi) = \int_{\mathbb{R}^n} e^{-i y\cdot\xi} u(y)\, dy $ is the Fourier transform.  
The symbol $ p(x,\xi) $ is a function on the phase space $ T^*\mathbb{R}^n \simeq \mathbb{R}^n_x \times \mathbb{R}^n_\xi $.

If $ p(x,\xi) $ does not depend on $ x $, the operator becomes a **Fourier multiplier**:

$$
(P u)(x) = \frac{1}{(2\pi)^n} \int e^{i x\cdot\xi}\, p(\xi)\, \hat u(\xi)\, d\xi.
$$

The **Weyl quantisation** provides a symmetric alternative:

$$
(P^{\rm Weyl} u)(x) \;=\; \frac{1}{(2\pi)^n} \iint e^{i (x-y)\cdot\xi}\, p\!\left(\frac{x+y}{2},\xi\right) u(y)\, dy\, d\xi.
$$

Both quantisations agree up to lower‑order terms.

## 2. Symbol Classes and Asymptotic Expansions

A symbol $ p(x,\xi) $ is said to belong to the class $ S^m $ if it satisfies

$$
|\partial_x^\alpha \partial_\xi^\beta p(x,\xi)| \;\le\; C_{\alpha\beta}\, \langle\xi\rangle^{m-|\beta|}, \qquad \langle\xi\rangle = (1+|\xi|^2)^{1/2}.
$$

The **asymptotic expansion** as $ |\xi|\to\infty $ is a formal series

$$
p(x,\xi) \;\sim\; \sum_{j=0}^\infty p_{m-j}(x,\xi),
$$

where each $ p_{m-j} $ is homogeneous of degree $ m-j $ in $ \xi $ for large $ |\xi| $.  
The leading term $ p_m $ is called the **principal symbol**.  
If $ p $ is exactly homogeneous (i.e. $ p(x,\lambda\xi) = \lambda^m p(x,\xi) $ for $ \lambda>0 $), the operator order is $ m $.

## 3. Symbolic Calculus (Composition)

For two ΨDOs $ P $ and $ Q $ with symbols $ p $ and $ q $, the symbol of the composition $ P\circ Q $ admits an asymptotic expansion.  
In the **Kohn‑Nirenberg** quantisation:

$$
(p\# q)(x,\xi) \;\sim\; \sum_{|\alpha|\ge 0} \frac{1}{\alpha!}\, (i)^{-|\alpha|}\, \partial_\xi^\alpha p(x,\xi)\; \partial_x^\alpha q(x,\xi).
$$

In the **Weyl** quantisation (Moyal star product):

$$
(p\#_{\rm Weyl} q)(x,\xi) \;=\; \exp\!\left(\frac{i}{2}\bigl(\partial_\xi^p \partial_x^q - \partial_x^p \partial_\xi^q\bigr)\right) p(x,\xi)\, q(x,\xi),
$$

where the exponential is understood as a formal series.

The **commutator** $ [P,Q] = PQ - QP $ has symbol

$$
[p,q]_{\#} \;\sim\; \sum_{|\alpha|\ge 1} \frac{1}{\alpha!}\, (i)^{-|\alpha|}\, \bigl( \partial_\xi^\alpha p\; \partial_x^\alpha q - \partial_\xi^\alpha q\; \partial_x^\alpha p \bigr).
$$

To leading order, $ [P,Q] \sim -i\,\{p,q\} $ where $ \{p,q\} = \partial_\xi p \cdot \partial_x q - \partial_x p \cdot \partial_\xi q $ is the Poisson bracket.

## 4. Formal Adjoint

The **formal adjoint** $ P^* $ satisfies $ \langle P u, v\rangle = \langle u, P^* v\rangle $ for test functions.  
Its symbol is given by

$$
p^*(x,\xi) \;\sim\; \sum_{|\alpha|\ge 0} \frac{1}{\alpha!}\, (i)^{-|\alpha|}\, \partial_\xi^\alpha \partial_x^\alpha \overline{p(x,\xi)}.
$$

If $ p = p^* $, the operator is **formally self‑adjoint** (Hermitian).

## 5. Asymptotic Inverses

Assume the principal symbol $ p_m $ never vanishes. Then $ P $ is **elliptic** and admits a **parametrix** – an approximate inverse.

A **right inverse** $ R $ satisfies $ P\circ R = I + \text{smoothing} $ (order $ -\infty $). Its symbol is constructed recursively:

$$
r_{-m}(x,\xi) = \frac{1}{p_m(x,\xi)},\qquad
r_{-m-k} = -\, \frac{1}{p_m} \sum_{\substack{|\alpha|+j<k \\ |\alpha|>0}} \frac{1}{\alpha!}\,(i)^{-|\alpha|}\, \partial_\xi^\alpha p_m\; \partial_x^\alpha r_{-m-j}.
$$

A **left inverse** $ L $ satisfies $ L\circ P = I + \text{smoothing} $ and is built by a similar recursive formula.

## 6. Exponential of an Operator

For a parameter $ t $ (often $ t = -i\tau $ for Schrödinger evolution), the symbol of $ e^{tP} $ is formally

$$
\exp(tP) \;\sim\; I + tP + \frac{t^2}{2!} P^2 + \frac{t^3}{3!} P^3 + \cdots,
$$

where each power is computed using the asymptotic composition rule.  
This expansion is valid for small $ |t| $ or for symbols with appropriate decay.

## 7. Semiclassical Trace Formula

For a trace‑class ΨDO, the trace is given by the phase‑space integral

$$
\operatorname{Tr}(P) \;=\; \frac{1}{(2\pi)^n} \iint p(x,\xi)\, dx\, d\xi.
$$

This is the semiclassical approximation of the quantum trace.

## 8. Hamiltonian Flow (Bicharacteristics)

The principal symbol $ p_m $ generates a Hamiltonian vector field

$$
\dot x = \frac{\partial p_m}{\partial \xi},\qquad
\dot \xi = -\frac{\partial p_m}{\partial x}.
$$

Its integral curves are the **bicharacteristics** of $ P $. They govern the propagation of singularities (microlocal Huygens principle).

## 9. Ellipticity

A ΨDO is **elliptic** if its principal symbol satisfies $ |p_m(x,\xi)| \ge c\,|\xi|^m $ for large $ |\xi| $, with $ c>0 $.  
Elliptic operators are invertible modulo smoothing operators and enjoy regularity properties (hypoellipticity).

## 10. Pseudospectrum

For a non‑normal operator $ P $ (discretised on a grid), the **ε‑pseudospectrum** is

$$
\sigma_\varepsilon(P) \;=\; \bigl\{ \lambda\in\mathbb{C} : \|(P-\lambda I)^{-1}\| \ge \varepsilon^{-1} \bigr\}.
$$

It is visualised by plotting the resolvent norm $ \|(P-\lambda I)^{-1}\| $ or the smallest singular value $ \sigma_{\min}(P-\lambda I) $.  
The pseudospectrum reveals spectral instability and transient growth phenomena.

## 11. Microlocal Concepts

- **Characteristic set**: $ \operatorname{Char}(P) = \{(x,\xi)\neq 0 : p_m(x,\xi)=0\} $.  
  Singularities of solutions propagate along bicharacteristics contained in $ \operatorname{Char}(P) $.

- **Micro‑support**: The complement of the largest open set where $ p $ is elliptic; it indicates where the operator “acts” in phase space.

- **Cotangent fibre**: The set of frequencies above a fixed spatial point $ x_0 $, i.e. $ \{ (x_0,\xi) : \xi\in\mathbb{R}^n \} $. Visualising $ |p(x_0,\xi)| $ reveals the operator’s frequency response.

- **Group velocity**: $ v_g = \nabla_\xi p(x,\xi) $. It describes the propagation speed of wave packets (stationary phase).

## References

1. Hörmander, L. *The Analysis of Linear Partial Differential Operators III*. Springer, 1985.  
2. Taylor, M. E. *Pseudo Differential Operators*. Princeton University Press, 1981.  
3. Zworski, M. *Semiclassical Analysis*. AMS, 2012.  
4. Martinez, A. *An Introduction to Semiclassical and Microlocal Analysis*. Springer, 2002.  
5. Trefethen, L. N. & Embree, M. *Spectra and Pseudospectra*. Princeton University Press, 2005.