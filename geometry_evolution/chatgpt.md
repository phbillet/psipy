

# 📦 1) Package : Géométrie Riemannienne

*(entrée : un Hamiltonien (H(x,p)) ou une métrique (g_{ij}(x)))*

## 🚩 **A. Commun 1D–2D (infrastructure générale)**

1. **Représentation de métrique**

   * 1D : (g_{11}(x))
   * 2D : matrice (g_{ij}(x))
   * stockage du déterminant (|g|), inverse (g^{ij})

2. **Calcul des dérivées de métrique**

   * (\partial_k g_{ij})
   * (\partial_k g^{ij})

3. **Calcul des symboles associés**

   * symbole principal : (g^{ij}(x)\xi_i\xi_j)
   * symbole sous-principal du Laplacien (optionnel)

4. **Extraction efficace de la métrique depuis un Hamiltonien quadratique**
   [
   g^{ij}(x)=\frac{\partial^2 H}{\partial p_i\partial p_j}
   ]

---

## 🚩 **B. Spécifique 1D**

En 1D, la géométrie est simple mais utile (milieux non uniformes).

1. **Christoffel 1D**
   [
   \Gamma^1_{11} = \frac12 (\log g_{11})'
   ]

2. **Équation géodésique**
   [
   \ddot x +\Gamma^1_{11}\dot x^2=0
   ]

3. **Laplacien-Beltrami 1D**
   [
   \Delta_g f = g^{11}\partial^2_x f +
   \left(\partial_x g^{11} + g^{11}\partial_x\log\sqrt{g_{11}}\right)\partial_x f
   ]

4. **Volume riemannien**
   [
   \mathrm{d}V_g = \sqrt{g_{11}(x)},\mathrm{d}x
   ]

5. **Transport géométrique pour PDE (ex: Schrödinger 1D sur métrique non triviale)**

---

## 🚩 **C. Spécifique 2D**

La 2D est déjà riche (courbure, géodésiques non triviales).

1. **Christoffel 2D**
   [
   \Gamma^i_{jk}=\frac12 g^{i\ell}
   (\partial_j g_{\ell k}+\partial_k g_{\ell j}-\partial_\ell g_{jk})
   ]

2. **Géodésiques 2D**
   [
   \ddot x^i + \Gamma^i_{jk}\dot x^j\dot x^k=0
   ]

3. **Courbure de Gauss**
   [
   K = \frac{R_{1212}}{\det(g)}
   ]
   ou formule explicite en coordonnées.

4. **Tenseur de Ricci**
   [
   R_{ij} = R^k{}_{ikj}
   ]

5. **Courbure scalaire**
   [
   R = g^{ij}R_{ij} = 2K
   ]

6. **Laplacien-Beltrami 2D**
   [
   \Delta_g f = |g|^{-1/2}\partial_i\left(|g|^{1/2}g^{ij}\partial_j f\right)
   ]

7. **Volume riemannien 2D**
   [
   \mathrm{d}V_g = \sqrt{|g|},\mathrm{d}x^1\wedge\mathrm{d}x^2
   ]

8. **Opérateurs pseudo-diff différentiels sur métriques**

   * symbole du Laplacien
   * transport du symbole par flot hamiltonien
   * corrections d’ordre 1 (optionnel)

9. **Visualisation 2D**

   * lignes géodésiques
   * champs de directions
   * carte de courbure (d’après `geometry_2d`)

---

# 📦 2) Package : Géométrie Symplectique

*(entrée : forme symplectique (\omega) et Hamiltonien (H))*

## 🚩 **A. Commun 1D–2D**

1. **Représentation de la forme symplectique**

   * Matrice (\omega_{ij})
   * Inverse (\omega^{ij})

2. **Champ hamiltonien**
   [
   X_H^i = \omega^{ij}\partial_j H
   ]

3. **Équations de Hamilton**
   [
   \dot z = X_H(z)
   ]

4. **Crochet de Poisson**
   [
   {f,g} = \omega^{ij},\partial_i f,\partial_j g
   ]

5. **Flot hamiltonien numérique**

   * intégrateur symplectique (leapfrog, Verlet, symplectic Euler)

6. **Transport des symboles (Egorov classique)**
   [
   a_t = a\circ \phi_t
   ]

7. **Dérivée temporelle des symboles**
   [
   \frac{\mathrm{d}}{\mathrm{d}t}a_t = {H,a_t}
   ]

8. **Quantification semi-classique, commutateur**
   [
   \frac{1}{ih}[\mathrm{Op}(a),\mathrm{Op}(b)] \approx \mathrm{Op}({a,b})
   ]

9. **Gradient symplectique**
   [
   X_H = J\nabla H
   ]

---

## 🚩 **B. Cas 1D (espace de phase 2D)**

1. **Forme symplectique standard**
   [
   \omega = \mathrm{d}x\wedge \mathrm{d}p
   ]

2. **Équations de Hamilton**
   [
   \dot x = \partial_p H,\qquad \dot p = -\partial_x H
   ]

3. **Crochet de Poisson standard**
   [
   {f,g} = \partial_x f,\partial_p g - \partial_p f,\partial_x g
   ]

4. **Trajectoires hamiltoniennes 2D — visualisables avec `geometry_2d`**

5. **Action-angle pour systèmes intégrables (optionnel)**

6. **Visualisation du portrait de phase (poincaré, courbes d’énergie)**

---

## 🚩 **C. Cas 2D (espace de phase 4D)**

Même si la visualisation est plus difficile, les opérations symboliques restent simples.

1. **Forme symplectique standard**
   [
   \omega = \mathrm{d}x_1\wedge\mathrm{d}p_1 + \mathrm{d}x_2\wedge\mathrm{d}p_2
   ]

2. **Équations de Hamilton**
   [
   \dot x_i = \partial_{p_i}H,\qquad
   \dot p_i = -\partial_{x_i}H
   ]

3. **Crochet de Poisson 4D**
   [
   {f,g} =
   \sum_{i=1}^{2} \left(
   \partial_{x_i} f,\partial_{p_i} g

   * \partial_{p_i} f,\partial_{x_i} g
     \right)
     ]

4. **Flot hamiltonien 4D**

   * intégrateurs symplectiques
   * projection pour visualisation (ex: projection sur plans ((x_1,x_2)), ((x_1,p_1)), etc.)

5. **Linearisation du flot (stabilité)**
   [
   \dot{\delta z} = D X_H(z),\delta z
   ]

6. **Transformations canoniques**
   [
   \phi^\ast\omega=\omega
   ]

7. **Hamilton-Jacobi**
   [
   H\big(x,\partial_x S\big) = E
   ]

8. **Symplectic reduction (optionnel)**

---

# 🧱 Résumé synthétique (pour structurer les modules)

## 📦 `riemannian_1d/`

* métrique 1D
* Christoffel
* géodésiques
* Laplacien-Beltrami
* volume
* symbole du Laplacien
* outils PDE 1D métrique

## 📦 `riemannian_2d/`

* métrique 2D
* inverse, dérivées
* Christoffel
* géodésiques
* courbure
* Laplacien-Beltrami
* opérateurs pseudo-diff sur surfaces
* visualisations (courbure, géodésiques)

## 📦 `symplectic_1d/`

* forme symplectique standard
* Hamilton eq.
* Poisson
* flots 2D
* action-angle
* portraits de phase

## 📦 `symplectic_2d/`

* forme symplectique standard 4D
* Poisson 4D
* flots hamiltoniens 4D
* transformation canoniques
* Egorov + transport
* linéarisation du flot + stabilité


