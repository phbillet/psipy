

## Package `riemannian.py`

### Noyau commun (1D et 2D)

1. Extraction de la métrique inverse g^{ij}(x) depuis le hamiltonien H(x, ξ)
2. Inversion pour obtenir g_{ij}(x)
3. Calcul du déterminant det(g) et de √|g|
4. Symboles de Christoffel Γ^k_{ij}
5. Équations des géodésiques (forme hamiltonienne et forme accélération)
6. Intégration numérique des géodésiques
7. Transport parallèle le long d'une courbe
8. Laplacien de Beltrami Δ_g
9. Élément de volume dV_g = √|g| dx

### Spécifique 1D

10. Longueur d'arc entre deux points
11. Reparamétrisation par longueur d'arc
12. Lien avec opérateurs de Sturm-Liouville
13. Potentiel effectif pour la réduction géodésique

### Spécifique 2D

14. Courbure de Gauss K
15. Tenseur de Riemann R^l_{ijk} (une seule composante indépendante)
16. Tenseur de Ricci R_{ij} = K g_{ij}
17. Courbure scalaire R = 2K
18. Équation de Jacobi (déviation géodésique)
19. Courbure géodésique κ_g d'une courbe
20. Théorème de Gauss-Bonnet (vérification numérique)
21. Application exponentielle exp_p(v)
22. Distance géodésique d(p, q)
23. Coordonnées normales au voisinage d'un point
24. Opérateur de Hodge ⋆ sur les 1-formes
25. Laplacien de Hodge-de Rham sur les formes

### Visualisation (geometry_1d / geometry_2d)

26. Tracé de géodésiques sur une surface
27. Champ de vecteurs parallèlement transportés
28. Carte de courbure K(x, y)
29. Cercles géodésiques et leur déformation
30. Triangles géodésiques et défaut angulaire

---

## Package `symplectic.py`

### Noyau commun (1D et 2D en position)

1. Définition d'une forme symplectique ω_{ij}(z)
2. Calcul de la matrice inverse ω^{ij} (structure de Poisson)
3. Crochet de Poisson {f, g}
4. Vérification de l'identité de Jacobi
5. Champ de vecteurs hamiltonien X_H
6. Équations de Hamilton
7. Intégration du flot hamiltonien (méthodes génériques)
8. Intégrateurs symplectiques (Störmer-Verlet, leapfrog)
9. Vérification de la préservation de ω par le flot
10. Conservation de H le long des trajectoires
11. Calcul des intégrales premières {H, F} = 0
12. Volume de Liouville et sa conservation

### Spécifique 1D en position (espace des phases 2D)

13. Portrait de phase (courbes de niveau de H)
14. Points fixes et leur classification (elliptique, hyperbolique)
15. Linéarisation au voisinage d'un point fixe
16. Période des orbites fermées T(E)
17. Variables action-angle (I, θ)
18. Action I(E) = (1/2π) ∮ ξ dx
19. Fréquence ω(E) = ∂H/∂I
20. Séparatrices et orbites homoclines
21. Fonction génératrice S(x, I)

### Spécifique 2D en position (espace des phases 4D)

22. Section de Poincaré (définition de la surface Σ)
23. Application de premier retour P : Σ → Σ
24. Calcul du temps de retour T(z)
25. Points fixes de l'application de Poincaré
26. Stabilité des orbites périodiques (valeurs propres)
27. Matrice de monodromie
28. Exposants de Lyapunov
29. Détection d'intégrabilité (existence d'une seconde intégrale)
30. Tores invariants (visualisation par sections)
31. Résonances et chaînes d'îlots
32. Application moment μ pour les symétries
33. Réduction symplectique (cas avec symétrie)

### Quantification et lien avec psiop.py

34. Produit de Moyal f ⋆ g (développement en ℏ)
35. Commutateur de Moyal [f, g]_⋆
36. Symbole de Weyl d'un opérateur
37. Correspondance Poisson-commutateur

### Visualisation

38. Portrait de phase 2D avec courbes de niveau
39. Champ de vecteurs hamiltonien
40. Animation du flot
41. Section de Poincaré (nuage de points)
42. Diagramme de bifurcation
43. Carte des exposants de Lyapunov

---

## Package `microlocal.py` (pont entre les deux)

1. Flot bicaractéristique du symbole principal
2. Variété caractéristique Char(P) = {p(x, ξ) = 0}
3. Propagation des singularités le long des bicaractéristiques
4. Développement WKB (équation eikonale + transport)
5. Conditions de Bohr-Sommerfeld pour les valeurs propres
6. Calcul de la phase géométrique
7. Caustiques et points de retournement
