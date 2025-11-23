

# 1. Géométrie riemannienne — 1D

**Résumé des fonctionnalités clés (mathématiques)**

* Représentation de la métrique (g_{11}(x)), son inverse (g^{11}), (|g|) et (\sqrt{|g|}). 
* Christoffel (\Gamma^1_{11}=\tfrac12(\log g_{11})'). 
* Équation des géodésiques (\ddot x + \Gamma^1_{11}\dot x^2=0) (formes accélération et hamiltonienne). 
* Laplacien-Beltrami 1D (formule explicite avec terme de transport d'ordre 1). 
* Volume riemannien (dV_g=\sqrt{g_{11}},dx). 
* Lien avec opérateurs de Sturm–Liouville et potentiel effectif par réduction géodésique. 

**API recommandée (module `riemannian_1d`)**

* `class Metric1D`: champs `.g(x)`, `.g_inv(x)`, `.sqrt_det(x)`. Méthodes `from_hamiltonian(H)` (extraction (g^{11}\leftarrow\partial^2_{p}H)), `eval(x)`. 
* `christoffel(metric) -> callable(x) -> float`
* `geodesic_integrator(metric, x0, v0, tspan, method='rk4'|'symplectic') -> trajectory` (option hamiltonienne). 
* `laplace_beltrami(metric) -> OperatorSymbol` (retourne symbole principal + sous-principal). 
* `sturm_liouville_reduce(metric, separation_params) -> (p(x), q(x))` (pour utilisation avec solvers). 

**Numérique & tests**

* Intégrateurs explicites (RK4) et adaptatifs ; option symplectique pour formulation hamiltonienne (utile si on intègre couples ((x,\xi))). 
* Tests unitaires : métrique plate (analytique), métrique de courbure connue (cas modèle) — vérifier longueur d'arc, laplacien appliqué sur fonctions tests, conservation d'invariants. 

**Visualisation**

* Tracé de géodésiques sur la droite avec colorisation selon courbure effective / vitesse paramétrique. Export facile vers `geometry_1d` UI. 

**Points d'intégration avec `psiop.py`**

* `Metric1D.from_hamiltonian(H)` → permet d'extraire le symbole principal de l'opérateur (utile pour construire le symbole du Laplacien et quantifier). 

---

# 2. Géométrie riemannienne — 2D

**Résumé des fonctionnalités clés (mathématiques)**

* Représentation matricielle (g_{ij}(x,y)), inverse (g^{ij}), (|g|) et (\sqrt{|g|}). 
* Christoffel (\Gamma^i_{jk}), géodésiques (\ddot x^i + \Gamma^i_{jk}\dot x^j\dot x^k=0). 
* Courbure de Gauss (K), tenseur de Riemann (une composante indépendante en 2D), tenseur de Ricci, courbure scalaire (R=2K). 
* Exponentielle (\exp_p(v)), coordonnées normales, distances géodésiques, équation de Jacobi (déviation). 
* Laplacien-Beltrami 2D, opérateurs pseudo-différentiels sur surfaces et symbole du Laplacien. 

**API recommandée (module `riemannian_2d`)**

* `class Metric2D`: `.g(x,y)`, `.g_inv(x,y)`, `.det(x,y)`, `.sqrt_det(x,y)` ; init depuis (g_{ij}) ou depuis Hamiltonien quadratique via `from_hamiltonian(H)`. 
* `christoffel(metric) -> Gamma(x,y)` (retourne fonction qui rend (\Gamma^i_{jk}(x,y))).
* `geodesic_solver(metric, p0, v0, tspan, method='rk4'|'symplectic'|'verlet') -> trajectory` (option: reparamétrisation par longueur). 
* `gauss_curvature(metric) -> callable(x,y)` ; `ricci_scalar(metric)` ; `jacobi_equation_solver(metric, geodesic, initial_variation)`
* `exponential_map(metric, p, v)` ; `distance(metric, p, q, method='shooting'|'optimize')`
* `laplace_beltrami_operator(metric) -> Symbol/psiop-compatible` (retourne symbole principal + corrections) ; `hodge_star(metric)` et `de_rham_laplacian()` (formes 0/1/2). 

**Numérique & tests**

* Intégration de géodésiques par solveurs robustes ; méthode de tir pour distance géodésique ; vérification numérique du théorème de Gauss–Bonnet sur domaines discrétisés. 

**Visualisation (pour `geometry_2d`)**

* Carte de courbure (K(x,y)), champ de vecteurs de transport parallèle, tracé de géodésiques, triangles géodésiques et calcul du défaut angulaire, cercles géodésiques et déformation. (voir idées visuelles dans les deux fichiers).

**Points d'intégration avec `psiop.py`**

* Construction automatique du symbole du Laplacien (symbole principal (g^{ij}\xi_i\xi_j)) pour créer opérateurs pseudo-différentiels et tester Egorov / transport. Fournir méthode `to_symbol()` compatible `psiop` pour quantification Weyl. 

---

# 3. Géométrie symplectique — 1D (espace de phase 2D)

**Résumé des fonctionnalités clés (mathématiques & dynamiques)**

* Forme symplectique standard (\omega = dx\wedge dp) ou forme localement définie (\omega_{ij}(z)). 
* Champ hamiltonien (X_H=\omega^{ij}\partial_j H) et équations de Hamilton (\dot z = X_H). 
* Crochet de Poisson standard ({f,g}=\partial_x f\partial_p g - \partial_p f\partial_x g). 
* Portrait de phase (courbes de niveau), points fixes et classification, action–angle, périodes (T(E)), séparatrices. 

**API recommandée (module `symplectic_1d`)**

* `class SymplecticForm1D` (supporte forme canonique et formes dépendantes de (z)).
* `hamiltonian_flow(H, z0, tspan, integrator='symplectic') -> trajectory` (intégrateurs symplectiques: Verlet, Stormer-Verlet, symplectic Euler). 
* `poisson_bracket(f, g)` ; `action_integral(H, E) -> I(E)` ; `frequency(H, I) -> dH/dI`
* `phase_portrait(H, x_range, p_range)` ; `find_fixed_points(H)` ; `linearize_at_fixed_point(H, z0) -> jacobian` 

**Visualisation**

* Portrait de phase 2D, courbes de niveau d'énergie, animation du flot. Export direct vers `geometry_2d` (la même UI peut servir pour visualiser plans de phase 2D). 

**Quantification / liens**

* Calcul d'actions (bohr–sommerfeld), construction de génératrices (S(x,I)) pour liaisons semi-classiques avec `psiop.py`. 

---

# 4. Géométrie symplectique — 2D (espace de phase 4D)

**Résumé des fonctionnalités clés (mathématiques & dynamiques)**

* Forme symplectique sur ((x_1,p_1,x_2,p_2)), crochets de Poisson 4D, flots hamiltoniens 4D. 
* Sections de Poincaré, application de premier retour (P:\Sigma\to\Sigma), matrice de monodromie, exposants de Lyapunov, tores invariants, résonances. 
* Linearisation du flot, stabilité des orbites périodiques. 

**API recommandée (module `symplectic_2d`)**

* `class SymplecticForm2D` (interface générique).
* `hamiltonian_flow_4d(H, z0, tspan, integrator='symplectic')` ; `poincare_section(H, Sigma_def, z0, tmax)` ; `first_return_map(points_on_section) -> P`
* `monodromy_matrix(H, periodic_orbit) -> ndarray` ; `lyapunov_exponents(trajectory)`. 
* Outils de projection/visualisation : `project(trajectory, plane)` pour afficher ((x_1,x_2)), ((x_1,p_1)), etc. 

**Visualisation**

* Nuage de points de Poincaré, cartes de bifurcation, tores invariants projetés, animations. 

**Quantification / liens**

* Transport de symboles (Egorov classique) : (a_t=a\circ\phi_t) ; vérification numérique de la préservation de la forme et de la conservation d'énergie. Fournir routine pour comparer (\frac{1}{i\hbar}[{\rm Op}(a),{\rm Op}(b)]) et (\mathrm{Op}({a,b})) (approx.).

---

# 5. Analyse microlocale — 1D

**Fonctions / concepts à inclure**

* Variété caractéristique (\operatorname{Char}(P)={(x,\xi): p(x,\xi)=0}). Propagation des singularités le long des bicaractéristiques. 
* Flot bicaractéristique (sur le fibré cotangent) et intégration numérique (liaison avec `symplectic_1d`). 
* Développement WKB : équation eikonale et équations de transport (ordres successifs). Implementation d'un solveur WKB symbolique / numérique en 1D (phase + amplitude). 
* Conditions de Bohr–Sommerfeld pour valeurs propres (1D) : calcul des actions (\frac{1}{2\pi}\oint \xi,dx). 

**API recommandée (module `microlocal_1d`)**

* `characteristic_variety(symbol)` ; `bicharacteristic_flow(symbol, z0, tspan)` (retourne trajectoire sur (T^*M)). 
* `wkb_ansatz(symbol, order=1) -> (S(x), a(x))` (eikonale + transport) ; `bohr_sommerfeld_quantization(H, contour) -> energies`. 

**Tests & visualisation**

* Comparer approximations WKB aux valeurs propres numériques (solver) ; visualiser phases, caustiques en 1D (amplitude très grande localisée). 

**Interactions `psiop.py`**

* Utiliser symboles extraits (par `psiop.symbol(...)`) pour construire `characteristic_variety` et pour tester quantification semi-classique (Moyal, commutateur vs. Poisson).

---

# 6. Analyse microlocale — 2D

**Fonctions / concepts à inclure**

* Flot bicaractéristique sur (T^*M) 4D, propagation des singularités, caustiques et points de retournement (fronts d'onde). 
* WKB multidimensionnel (éikonale implicite, transport le long de bicaractéristiques), maslov index/phase géométrique. 
* Sections de phase et génération de caustiques ; calcul numérique de surfaces de phase et front d'onde. 
* Conditions semi-classiques (Bohr–Sommerfeld généralisées) pour systèmes intégrables et non intégrables (approximation locale sur tores invariants le cas échéant). 

**API recommandée (module `microlocal_2d`)**

* `bichar_flow_2d(symbol, z0, tspan)` ; `propagate_singularity(initial_sing_support, tspan)`
* `wkb_multidim(symbol, initial_phase, order=1)` ; `compute_maslov_index(path)`
* `compute_caustics(symbol, region, resolution)` (retourne géométrie des caustiques pour visualisation). 

**Visualisation & Validation**

* Visualiser projections des bicaractéristiques, fronts d'onde, cartes d'intensité (amplitude WKB), et comparer aux simulations PDE (résultats de `solver.py`). Vérifier localisation des singularités le long des trajectoires. 

**Interactions `psiop.py`**

* Fournir outils pour construire symboles (principal & sous-principal), calculer produit de Moyal et tester la correspondance Poisson-commutateur numériquement ; exporter opérateurs pseudo-différentiels approchés pour simulations PDE.



