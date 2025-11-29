from sympy import symbols, Matrix, simplify
from sympy.diffgeom import Manifold, Patch, CoordSystem

import numpy as np

def verify_symplectic_structure(dim=1):
    """Vérifier que ω est fermée et non-dégénérée"""
    
    if dim == 1:
        x, xi = symbols('x xi', real=True)
        
        # Forme symplectique ω = dξ ∧ dx
        # En coordonnées : ω = [[0, -1], [1, 0]]
        omega = Matrix([[0, -1], [1, 0]])
        
        print("Forme symplectique ω:")
        print(omega)
        
        # Non-dégénérescence : det(ω) ≠ 0
        det_omega = omega.det()
        print(f"\ndet(ω) = {det_omega}")
        assert det_omega != 0, "ω est dégénérée !"
        
        # Antisymétrie : ω = -ωᵀ
        print(f"\nω + ωᵀ = {simplify(omega + omega.T)}")
        assert (omega + omega.T).is_zero_matrix
        
    elif dim == 2:
        x, y, xi, eta = symbols('x y xi eta', real=True)
        
        # ω = dξ∧dx + dη∧dy
        # Matrice : [[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]]
        omega = Matrix([
            [0,  0, -1,  0],
            [0,  0,  0, -1],
            [1,  0,  0,  0],
            [0,  1,  0,  0]
        ])
        
        print("Forme symplectique 2D ω:")
        print(omega)
        print(f"\ndet(ω) = {omega.det()}")
        
    return omega

# Test
omega = verify_symplectic_structure(dim=1)

def hamiltonian_vector_field(H_expr, vars_x):
    """Calcule le champ hamiltonien X_H à partir de H quelconque"""
    from sympy import symbols, diff
    
    dim = len(vars_x)
    
    if dim == 1:
        x, = vars_x
        xi = symbols('xi', real=True)
        
        # Équations de Hamilton : ẋ = ∂H/∂ξ, ξ̇ = -∂H/∂x
        dx_dt = diff(H_expr, xi)
        dxi_dt = -diff(H_expr, x)
        
        print("Champ hamiltonien X_H:")
        print(f"  dx/dt = ∂H/∂ξ = {dx_dt}")
        print(f"  dξ/dt = -∂H/∂x = {dxi_dt}")
        
        return {'dx/dt': dx_dt, 'dxi/dt': dxi_dt}
        
    elif dim == 2:
        x, y = vars_x
        xi, eta = symbols('xi eta', real=True)
        
        # X_H = (∂H/∂ξ, ∂H/∂η, -∂H/∂x, -∂H/∂y)
        dx_dt = diff(H_expr, xi)
        dy_dt = diff(H_expr, eta)
        dxi_dt = -diff(H_expr, x)
        deta_dt = -diff(H_expr, y)
        
        print("Champ hamiltonien 2D X_H:")
        print(f"  dx/dt = {dx_dt}")
        print(f"  dy/dt = {dy_dt}")
        print(f"  dξ/dt = {dxi_dt}")
        print(f"  dη/dt = {deta_dt}")
        
        return {
            'dx/dt': dx_dt, 
            'dy/dt': dy_dt,
            'dxi/dt': dxi_dt, 
            'deta/dt': deta_dt
        }

# Exemple : H quelconque
from sympy import symbols, exp, sin, cos

x, xi = symbols('x xi', real=True)

# Hamiltoniens de complexité croissante
hamiltonians = {
    "Libre": xi**2 / 2,
    "Harmonique": xi**2/2 + x**2/2,
    "Pendule": xi**2/2 - cos(x),
    "Non-polynomial": xi**2/2 + x**4/4 + sin(xi*x),
    "Transcendant": exp(xi**2) * sin(x)
}

for name, H in hamiltonians.items():
    print(f"\n{'='*60}")
    print(f"Hamiltonien : {name}")
    print(f"H(x,ξ) = {H}")
    print('='*60)
    X_H = hamiltonian_vector_field(H, [x])

def verify_symplectic_preservation(H_expr, vars_x, t_max=1.0, n_times=10):
    """
    Version simplifiée et plus robuste
    """
    from scipy.integrate import solve_ivp
    from sympy import lambdify, symbols
    import numpy as np
    
    print("\n" + "="*70)
    print("VÉRIFICATION DE LA PRÉSERVATION SYMPLECTIQUE")
    print("="*70)
    
    dim = len(vars_x)
    
    if dim == 1:
        x, = vars_x
        xi = symbols('xi', real=True)
        
        # Système hamiltonien
        X_H = hamiltonian_vector_field(H_expr, vars_x)
        f_x = lambdify((x, xi), X_H['dx/dt'], 'numpy')
        f_xi = lambdify((x, xi), X_H['dxi/dt'], 'numpy')
        
        def system(t, y):
            return [f_x(y[0], y[1]), f_xi(y[0], y[1])]
        
        # Condition initiale
        y0 = np.array([1.0, 0.5])
        
        print(f"\nCondition initiale : (x₀, ξ₀) = ({y0[0]}, {y0[1]})")
        print(f"Hamiltonien : H = {H_expr}")
        
        # Intégration
        t_eval = np.linspace(0, t_max, 1000)
        sol = solve_ivp(system, [0, t_max], y0, t_eval=t_eval, 
                       method='DOP853', rtol=1e-10, atol=1e-12)
        
        # 1. Vérifier conservation de H
        H_func = lambdify((x, xi), H_expr, 'numpy')
        H_values = H_func(sol.y[0], sol.y[1])
        
        print(f"\n[1] Conservation de l'énergie H :")
        print(f"    H(t=0)     = {H_values[0]:.12f}")
        print(f"    H(t={t_max:.1f})   = {H_values[-1]:.12f}")
        print(f"    |ΔH|/H₀    = {abs(H_values[-1]-H_values[0])/abs(H_values[0]):.2e}")
        
        # 2. Vérifier det(Jacobien) = 1 via méthode des variations
        epsilon = 1e-7
        times_to_check = np.linspace(0, t_max, n_times)
        
        print(f"\n[2] Conservation de la structure symplectique (det J = 1) :")
        print(f"    Méthode : différences finies sur le jacobien")
        
        max_det_error = 0.0
        
        for t_check in times_to_check:
            # Intégrer 4 trajectoires perturbées
            sol_ref = solve_ivp(system, [0, t_check], y0, method='DOP853', rtol=1e-10)
            
            sol_dx = solve_ivp(system, [0, t_check], y0 + [epsilon, 0], 
                              method='DOP853', rtol=1e-10)
            sol_mx = solve_ivp(system, [0, t_check], y0 + [-epsilon, 0], 
                              method='DOP853', rtol=1e-10)
            
            sol_dxi = solve_ivp(system, [0, t_check], y0 + [0, epsilon], 
                               method='DOP853', rtol=1e-10)
            sol_mxi = solve_ivp(system, [0, t_check], y0 + [0, -epsilon], 
                               method='DOP853', rtol=1e-10)
            
            # États finaux
            final_ref = sol_ref.y[:, -1]
            final_dx = sol_dx.y[:, -1]
            final_mx = sol_mx.y[:, -1]
            final_dxi = sol_dxi.y[:, -1]
            final_mxi = sol_mxi.y[:, -1]
            
            # Jacobien
            J = np.array([
                [(final_dx[0] - final_mx[0])/(2*epsilon), 
                 (final_dxi[0] - final_mxi[0])/(2*epsilon)],
                [(final_dx[1] - final_mx[1])/(2*epsilon), 
                 (final_dxi[1] - final_mxi[1])/(2*epsilon)]
            ])
            
            det_J = np.linalg.det(J)
            det_error = abs(det_J - 1.0)
            max_det_error = max(max_det_error, det_error)
            
            if t_check == 0 or t_check >= t_max - 0.01 or det_error > 1e-6:
                print(f"    t={t_check:6.2f} : det(J) = {det_J:.12f}  (|error| = {det_error:.2e})")
        
        print(f"\n    Erreur maximale : {max_det_error:.2e}")
        
        # 3. Vérifier via la forme symplectique ω
        print(f"\n[3] Vérification directe : Jᵀ ω J = ω")
        
        omega = np.array([[0, -1], [1, 0]])
        omega_transformed = J.T @ omega @ J
        omega_error = np.linalg.norm(omega_transformed - omega)
        
        print(f"    ω =")
        print(f"    {omega}")
        print(f"\n    Jᵀ ω J =")
        print(f"    {omega_transformed}")
        print(f"\n    ||Jᵀ ω J - ω|| = {omega_error:.2e}")
        
        # Conclusion
        print("\n" + "="*70)
        print("CONCLUSION :")
        print("="*70)
        
        all_good = (abs(H_values[-1]-H_values[0])/abs(H_values[0]) < 1e-8 and 
                    max_det_error < 1e-6)
        
        if all_good:
            print("✓ Le flot hamiltonien PRÉSERVE la structure symplectique")
            print("✓ Conservation de H à la précision numérique")
            print("✓ det(Jacobien) = 1 vérifié")
            print("\n➜ Confirmation du Théorème de Liouville !")
        else:
            print("⚠ Erreurs numériques détectées (probablement dues à l'intégration)")
            print(f"  - Erreur sur H : {abs(H_values[-1]-H_values[0])/abs(H_values[0]):.2e}")
            print(f"  - Erreur sur det(J) : {max_det_error:.2e}")
        
        return sol, H_values
        
# Test avec différents hamiltoniens
print("\n" + "="*70)
print("TEST 1 : Oscillateur harmonique H = (ξ² + x²)/2")
print("="*70)
verify_symplectic_preservation(xi**2/2 + x**2/2, [x], t_max=10.0)

print("\n" + "="*70)
print("TEST 2 : Hamiltonien non-linéaire H = ξ²/2 + x⁴/4")
print("="*70)
verify_symplectic_preservation(xi**2/2 + x**4/4, [x], t_max=5.0)

def poisson_bracket(f_expr, g_expr, vars_x):
    """Crochet de Poisson {f, g} induit par ω"""
    from sympy import symbols, diff, simplify
    
    dim = len(vars_x)
    
    if dim == 1:
        x, = vars_x
        xi = symbols('xi', real=True)
        
        # {f, g} = ∂f/∂x · ∂g/∂ξ - ∂f/∂ξ · ∂g/∂x
        bracket = (
            diff(f_expr, x) * diff(g_expr, xi) - 
            diff(f_expr, xi) * diff(g_expr, x)
        )
        
    elif dim == 2:
        x, y = vars_x
        xi, eta = symbols('xi eta', real=True)
        
        bracket = (
            diff(f_expr, x) * diff(g_expr, xi) - diff(f_expr, xi) * diff(g_expr, x) +
            diff(f_expr, y) * diff(g_expr, eta) - diff(f_expr, eta) * diff(g_expr, y)
        )
    
    return simplify(bracket)

# Vérifier l'identité de Jacobi {f, {g, h}} + permutations circulaires = 0
from sympy import symbols

x, xi = symbols('x xi', real=True)

f = x**2
g = xi**2
h = x*xi

bracket_fg_h = poisson_bracket(poisson_bracket(f, g, [x]), h, [x])
bracket_gh_f = poisson_bracket(poisson_bracket(g, h, [x]), f, [x])
bracket_hf_g = poisson_bracket(poisson_bracket(h, f, [x]), g, [x])

jacobi_identity = simplify(bracket_fg_h + bracket_gh_f + bracket_hf_g)

print(f"f = {f}, g = {g}, h = {h}")
print(f"\n{{f, {{g, h}}}} + {{g, {{h, f}}}} + {{h, {{f, g}}}} = {jacobi_identity}")
print(f"\nIdentité de Jacobi vérifiée : {jacobi_identity == 0}")

def verify_canonical_relations(vars_x):
    """Vérifie les relations de commutation canoniques"""
    from sympy import symbols
    
    dim = len(vars_x)
    
    if dim == 1:
        x, = vars_x
        xi = symbols('xi', real=True)
        
        # Crochets fondamentaux
        bracket_x_xi = poisson_bracket(x, xi, [x])
        bracket_x_x = poisson_bracket(x, x, [x])
        bracket_xi_xi = poisson_bracket(xi, xi, [x])
        
        print("Relations canoniques 1D:")
        print(f"  {{x, ξ}} = {bracket_x_xi}  (doit être 1)")
        print(f"  {{x, x}} = {bracket_x_x}  (doit être 0)")
        print(f"  {{ξ, ξ}} = {bracket_xi_xi}  (doit être 0)")
        
        assert bracket_x_xi == 1
        assert bracket_x_x == 0
        assert bracket_xi_xi == 0
        
    elif dim == 2:
        x, y = vars_x
        xi, eta = symbols('xi eta', real=True)
        
        print("Relations canoniques 2D:")
        print(f"  {{x, ξ}} = {poisson_bracket(x, xi, [x, y])}")
        print(f"  {{y, η}} = {poisson_bracket(y, eta, [x, y])}")
        print(f"  {{x, η}} = {poisson_bracket(x, eta, [x, y])}")
        print(f"  {{y, ξ}} = {poisson_bracket(y, xi, [x, y])}")

verify_canonical_relations([x])

def visualize_symplectic_geometry(H_expr, vars_x, xlim=(-2, 2), klim=(-2, 2)):
    """Visualise la géométrie symplectique pour un H quelconque"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    
    x, = vars_x
    xi = symbols('xi', real=True)
    
    # Champ hamiltonien
    X_H = hamiltonian_vector_field(H_expr, vars_x)
    
    # Grille
    x_vals = np.linspace(*xlim, 30)
    xi_vals = np.linspace(*klim, 30)
    X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')
    
    # Évaluer le champ
    from sympy import lambdify
    f_x = lambdify((x, xi), X_H['dx/dt'], 'numpy')
    f_xi = lambdify((x, xi), X_H['dxi/dt'], 'numpy')
    
    U = f_x(X, XI)
    V = f_xi(X, XI)
    
    # Évaluer H
    H_func = lambdify((x, xi), H_expr, 'numpy')
    H_vals = H_func(X, XI)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1 : Champ hamiltonien + courbes de niveau de H
    ax1 = axes[0]
    
    # Courbes de niveau (surfaces d'énergie)
    contours = ax1.contour(X, XI, H_vals, levels=15, colors='blue', alpha=0.4)
    ax1.clabel(contours, inline=True, fontsize=8)
    
    # Champ de vecteurs (toujours tangent aux courbes de niveau!)
    ax1.quiver(X[::2, ::2], XI[::2, ::2], U[::2, ::2], V[::2, ::2], 
               color='red', alpha=0.7, scale=20, width=0.003)
    
    ax1.set_xlabel('x (position)', fontsize=12)
    ax1.set_ylabel('ξ (impulsion)', fontsize=12)
    ax1.set_title(f'Champ hamiltonien X_H\nH = {H_expr}', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2 : Forme symplectique (visualisation abstraite)
    ax2 = axes[1]
    
    # Visualiser ω comme une 2-forme via son action sur des vecteurs
    # ω(v, w) = v_x·w_ξ - v_ξ·w_x
    
    # Choisir quelques points
    test_points = [(0, 0), (1, 0.5), (-0.5, -1)]
    
    for (x0, xi0) in test_points:
        # Vecteur tangent au flot
        v = np.array([f_x(x0, xi0), f_xi(x0, xi0)])
        
        # Vecteur orthogonal symplectique : J·v où J = [[0, -1], [1, 0]]
        J = np.array([[0, -1], [1, 0]])
        w = J @ v
        
        # Vérifier : ω(v, w) = det([v, w])
        omega_vw = np.linalg.det(np.column_stack([v, w]))
        
        # Dessiner
        ax2.arrow(x0, xi0, v[0]*0.2, v[1]*0.2, 
                  head_width=0.1, head_length=0.05, fc='red', ec='red', alpha=0.7)
        ax2.arrow(x0, xi0, w[0]*0.2, w[1]*0.2, 
                  head_width=0.1, head_length=0.05, fc='blue', ec='blue', alpha=0.7)
        
        ax2.text(x0+0.3, xi0+0.3, f'ω(v,w)={omega_vw:.2f}', fontsize=9)
    
    ax2.contour(X, XI, H_vals, levels=15, colors='gray', alpha=0.2, linewidths=0.5)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('ξ', fontsize=12)
    ax2.set_title('Structure symplectique ω = dξ∧dx', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(xlim)
    ax2.set_ylim(klim)
    ax2.legend(['X_H (rouge)', 'J·X_H (bleu)'], loc='upper right')
    
    plt.tight_layout()
    plt.show()

# Exemples
import sympy as sp

visualize_symplectic_geometry(xi**2/2 + x**2/2, [x])  # Harmonique
visualize_symplectic_geometry(xi**2/2 - sp.cos(x), [x])  # Pendule

def quantize_hamiltonian(H_classical, vars_x, mode='weyl'):
    """
    Quantifie un hamiltonien classique en opérateur pseudo-différentiel
    
    H_classique(x, ξ) → Ĥ opérateur
    
    mode : 'weyl' (symétrique) ou 'kn' (Kohn-Nirenberg)
    """
    from psiop import PseudoDifferentialOperator
    
    # Créer l'opérateur
    H_op = PseudoDifferentialOperator(
        expr=H_classical,
        vars_x=vars_x,
        mode='symbol'
    )
    
    print(f"Hamiltonien classique : H = {H_classical}")
    print(f"Quantification ({mode}):")
    
    # Propriétés de l'opérateur quantifié
    print(f"  - Ordre : {H_op.symbol_order()}")
    print(f"  - Auto-adjoint : {H_op.is_self_adjoint()}")
    
    # Commutateur avec x et ξ (relations de Heisenberg)
    x_op = PseudoDifferentialOperator(vars_x[0], vars_x, mode='symbol')
    xi_op = PseudoDifferentialOperator(symbols('xi', real=True), vars_x, mode='symbol')
    
    comm_H_x = H_op.commutator_symbolic(x_op, order=1, mode=mode)
    comm_H_xi = H_op.commutator_symbolic(xi_op, order=1, mode=mode)
    
    print(f"\n  [Ĥ, x̂] = {simplify(comm_H_x)}")
    print(f"  [Ĥ, p̂] = {simplify(comm_H_xi)}")
    
    # Correspondance classique-quantique
    print(f"\n  Principe de correspondance:")
    print(f"  [Ĥ, x̂] ≈ iℏ {{H, x}} = iℏ (∂H/∂ξ)")
    print(f"  [Ĥ, p̂] ≈ iℏ {{H, ξ}} = -iℏ (∂H/∂x)")
    
    return H_op

# Test
from sympy import symbols

x = symbols('x', real=True)
xi = symbols('xi', real=True)

print("="*70)
print("QUANTIFICATION : Oscillateur harmonique")
print("="*70)
H_harm = xi**2/2 + x**2/2
H_op_harm = quantize_hamiltonian(H_harm, [x], mode='weyl')

print("\n" + "="*70)
print("QUANTIFICATION : Hamiltonien anharmonique")
print("="*70)
H_anharm = xi**2/2 + x**4/4
H_op_anharm = quantize_hamiltonian(H_anharm, [x], mode='weyl')

def canonical_symplectic_form_construction():
    """
    Construction pas-à-pas de la forme symplectique canonique
    sur le fibré cotangent T*M
    """
    from sympy import symbols, diff, simplify
    from sympy.diffgeom import Manifold, Patch, CoordSystem
    
    print("="*70)
    print("CONSTRUCTION CANONIQUE DE LA FORME SYMPLECTIQUE")
    print("="*70)
    
    # Étape 1 : Fibré cotangent T*M
    print("\n[Étape 1] Fibré cotangent T*M")
    print("-"*70)
    print("Pour une variété M de dimension n, T*M a dimension 2n")
    print("Coordonnées locales : (x¹, ..., xⁿ, ξ₁, ..., ξₙ)")
    print("où xⁱ ∈ M (base) et ξᵢ ∈ Tₓ*M (fibre cotangente)")
    
    # Exemple 1D
    x = symbols('x', real=True)
    xi = symbols('xi', real=True)
    
    print(f"\nExemple 1D : M = ℝ, T*M = ℝ × ℝ")
    print(f"Point générique : (x, ξ) ∈ T*ℝ")
    
    # Étape 2 : Forme de Liouville (forme tautologique)
    print("\n[Étape 2] Forme de Liouville θ (forme tautologique)")
    print("-"*70)
    print("Définition intrinsèque (indépendante des coordonnées) :")
    print("  θ : T(T*M) → ℝ")
    print("  θₚ(v) = p(π*(v))  où p ∈ T*M, v ∈ Tₚ(T*M)")
    print("  π : T*M → M est la projection canonique")
    
    print("\nEn coordonnées locales :")
    print("  θ = ξ₁dx¹ + ... + ξₙdxⁿ = Σᵢ ξᵢdxⁱ")
    
    print(f"\nEn 1D : θ = ξ dx")
    
    # Étape 3 : Forme symplectique ω = dθ
    print("\n[Étape 3] Forme symplectique ω = dθ")
    print("-"*70)
    print("ω est la différentielle extérieure de θ")
    print("\nCalcul en 1D :")
    print("  θ = ξ dx")
    print("  dθ = d(ξ dx)")
    print("     = dξ ∧ dx + ξ d(dx)")
    print("     = dξ ∧ dx       (car d² = 0)")
    print("  ω = dξ ∧ dx")
    
    # Étape 4 : Propriétés de ω
    print("\n[Étape 4] Propriétés fondamentales de ω")
    print("-"*70)
    
    # Propriété 1 : Fermée
    print("\n✓ Propriété 1 : ω est FERMÉE")
    print("  dω = d(dθ) = 0  (car d² = 0)")
    print("  En fait, ω est EXACTE : ω = dθ")
    
    # Propriété 2 : Non-dégénérée
    print("\n✓ Propriété 2 : ω est NON-DÉGÉNÉRÉE")
    print("  En coordonnées, ω s'écrit comme une matrice antisymétrique")
    print("  de déterminant non-nul")
    
    from sympy import Matrix
    omega_matrix_1d = Matrix([[0, -1], [1, 0]])
    print(f"\n  En 1D : ω = [[0, -1], [1, 0]]")
    print(f"  det(ω) = {omega_matrix_1d.det()} ≠ 0 ✓")
    
    # Propriété 3 : Canonique (invariante)
    print("\n✓ Propriété 3 : ω est CANONIQUE")
    print("  ω est invariante par changements de coordonnées canoniques")
    print("  (= transformations qui préservent la structure de T*M)")
    
    # Étape 5 : Universalité
    print("\n[Étape 5] UNIVERSALITÉ")
    print("-"*70)
    print("✨ Point crucial : ω existe AVANT qu'on choisisse un hamiltonien H")
    print("✨ C'est une structure GÉOMÉTRIQUE de T*M, pas une propriété de H")
    print("✨ Tout H utilise la MÊME ω pour définir sa dynamique")
    
    return omega_matrix_1d

omega = canonical_symplectic_form_construction()

def from_omega_to_flow(H_expr, vars_x):
    """
    Montre comment la structure symplectique ω permet de passer
    de H (fonction) à X_H (champ de vecteurs) à Φₜ (flot)
    """
    from sympy import symbols, diff, simplify, lambdify
    from scipy.integrate import solve_ivp
    import numpy as np
    import matplotlib.pyplot as plt
    
    dim = len(vars_x)
    
    if dim == 1:
        x, = vars_x
        xi = symbols('xi', real=True)
        
        print("="*70)
        print("DE ω AU FLOT HAMILTONIEN : MÉCANISME COMPLET")
        print("="*70)
        
        # Étape 1 : La structure symplectique ω
        print("\n[Étape 1] Structure symplectique ω = dξ ∧ dx")
        print("-"*70)
        print("En coordonnées matricielles :")
        print("  ω = [[0, -1], [1, 0]]")
        
        omega = np.array([[0, -1], [1, 0]])
        omega_inv = np.linalg.inv(omega)
        
        print(f"\nω⁻¹ = {omega_inv}")
        
        # Étape 2 : Le hamiltonien H
        print(f"\n[Étape 2] Hamiltonien H(x, ξ) = {H_expr}")
        print("-"*70)
        
        # Étape 3 : Gradient de H
        print("\n[Étape 3] Gradient dH = (∂H/∂x, ∂H/∂ξ)ᵀ")
        print("-"*70)
        
        dH_dx = diff(H_expr, x)
        dH_dxi = diff(H_expr, xi)
        
        print(f"  ∂H/∂x = {dH_dx}")
        print(f"  ∂H/∂ξ = {dH_dxi}")
        print(f"\n  dH = ({dH_dx}, {dH_dxi})ᵀ")
        
        # Étape 4 : Champ hamiltonien X_H = ω⁻¹ · dH
        print("\n[Étape 4] Champ hamiltonien X_H via ω⁻¹")
        print("-"*70)
        print("Définition : ι_{X_H} ω = dH")
        print("En coordonnées : X_H = ω⁻¹ · dH")
        
        print(f"\nX_H = ω⁻¹ · dH")
        print(f"    = [[0, 1], [-1, 0]] · ({dH_dx}, {dH_dxi})ᵀ")
        print(f"    = ({dH_dxi}, -{dH_dx})ᵀ")
        
        dx_dt = dH_dxi
        dxi_dt = -dH_dx
        
        print(f"\n  dx/dt = ∂H/∂ξ = {dx_dt}")
        print(f"  dξ/dt = -∂H/∂x = {dxi_dt}")
        
        # Étape 5 : Intégration du flot
        print("\n[Étape 5] Flot hamiltonien Φₜ par intégration de X_H")
        print("-"*70)
        
        f_x = lambdify((x, xi), dx_dt, 'numpy')
        f_xi = lambdify((x, xi), dxi_dt, 'numpy')
        
        def hamiltonian_system(t, y):
            return [f_x(y[0], y[1]), f_xi(y[0], y[1])]
        
        # Condition initiale
        y0 = [1.0, 0.5]
        t_span = [0, 10]
        t_eval = np.linspace(*t_span, 200)
        
        sol = solve_ivp(hamiltonian_system, t_span, y0, 
                       t_eval=t_eval, method='DOP853')
        
        print(f"  Condition initiale : (x₀, ξ₀) = {y0}")
        print(f"  Intégration de t=0 à t={t_span[1]}")
        
        # Étape 6 : Vérification de la conservation
        print("\n[Étape 6] Vérifications")
        print("-"*70)
        
        # Conservation de H
        H_func = lambdify((x, xi), H_expr, 'numpy')
        H_values = H_func(sol.y[0], sol.y[1])
        
        print(f"\n✓ Conservation de H :")
        print(f"  H(t=0) = {H_values[0]:.10f}")
        print(f"  H(t={t_span[1]}) = {H_values[-1]:.10f}")
        print(f"  |ΔH|/H₀ = {abs(H_values[-1] - H_values[0])/abs(H_values[0]):.2e}")
        
        # Conservation de l'aire (volume symplectique)
        # Calculer numériquement le jacobien du flot
        epsilon = 1e-6
        t_test = 5.0
        idx = np.argmin(np.abs(sol.t - t_test))
        x0, xi0 = sol.y[0, idx], sol.y[1, idx]
        
        # Perturbations
        sol_x_plus = solve_ivp(hamiltonian_system, [0, t_test], 
                               [y0[0] + epsilon, y0[1]], dense_output=True)
        sol_x_minus = solve_ivp(hamiltonian_system, [0, t_test], 
                                [y0[0] - epsilon, y0[1]], dense_output=True)
        sol_xi_plus = solve_ivp(hamiltonian_system, [0, t_test], 
                                [y0[0], y0[1] + epsilon], dense_output=True)
        sol_xi_minus = solve_ivp(hamiltonian_system, [0, t_test], 
                                 [y0[0], y0[1] - epsilon], dense_output=True)
        
        # Jacobien
        dx_dx0 = (sol_x_plus.sol(t_test)[0] - sol_x_minus.sol(t_test)[0]) / (2*epsilon)
        dx_dxi0 = (sol_xi_plus.sol(t_test)[0] - sol_xi_minus.sol(t_test)[0]) / (2*epsilon)
        dxi_dx0 = (sol_x_plus.sol(t_test)[1] - sol_x_minus.sol(t_test)[1]) / (2*epsilon)
        dxi_dxi0 = (sol_xi_plus.sol(t_test)[1] - sol_xi_minus.sol(t_test)[1]) / (2*epsilon)
        
        J = np.array([[dx_dx0, dx_dxi0], [dxi_dx0, dxi_dxi0]])
        det_J = np.linalg.det(J)
        
        print(f"\n✓ Conservation de la structure symplectique (Théorème de Liouville) :")
        print(f"  det(DΦₜ) à t={t_test} : {det_J:.10f}")
        print(f"  (doit être 1 pour préserver ω)")
        
        # Visualisation
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1 : Trajectoire dans l'espace des phases
        ax1 = axes[0, 0]
        
        # Courbes de niveau de H
        x_grid = np.linspace(-2, 2, 100)
        xi_grid = np.linspace(-2, 2, 100)
        X, XI = np.meshgrid(x_grid, xi_grid)
        H_grid = H_func(X, XI)
        
        contours = ax1.contour(X, XI, H_grid, levels=15, colors='lightblue', alpha=0.5)
        ax1.clabel(contours, inline=True, fontsize=8)
        
        # Trajectoire
        ax1.plot(sol.y[0], sol.y[1], 'r-', linewidth=2, label='Trajectoire')
        ax1.plot(y0[0], y0[1], 'go', markersize=10, label='Point initial')
        ax1.plot(sol.y[0, -1], sol.y[1, -1], 'rs', markersize=10, label='Point final')
        
        ax1.set_xlabel('x (position)', fontsize=11)
        ax1.set_ylabel('ξ (impulsion)', fontsize=11)
        ax1.set_title('Trajectoire dans l\'espace des phases', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot 2 : Conservation de H
        ax2 = axes[0, 1]
        ax2.plot(sol.t, H_values, 'b-', linewidth=2)
        ax2.axhline(H_values[0], color='r', linestyle='--', 
                   label=f'H₀ = {H_values[0]:.6f}')
        ax2.set_xlabel('Temps t', fontsize=11)
        ax2.set_ylabel('H(x(t), ξ(t))', fontsize=11)
        ax2.set_title('Conservation de l\'énergie', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3 : Évolution des coordonnées
        ax3 = axes[1, 0]
        ax3.plot(sol.t, sol.y[0], 'b-', linewidth=2, label='x(t)')
        ax3.plot(sol.t, sol.y[1], 'r-', linewidth=2, label='ξ(t)')
        ax3.set_xlabel('Temps t', fontsize=11)
        ax3.set_ylabel('Coordonnées', fontsize=11)
        ax3.set_title('Évolution temporelle', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4 : Champ de vecteurs hamiltonien
        ax4 = axes[1, 1]
        
        x_vals = np.linspace(-2, 2, 20)
        xi_vals = np.linspace(-2, 2, 20)
        X_field, XI_field = np.meshgrid(x_vals, xi_vals, indexing='ij')
        
        U = f_x(X_field, XI_field)
        V = f_xi(X_field, XI_field)
        
        ax4.quiver(X_field, XI_field, U, V, alpha=0.6, scale=20, width=0.003)
        ax4.contour(X, XI, H_grid, levels=15, colors='lightblue', alpha=0.3)
        ax4.plot(sol.y[0], sol.y[1], 'r-', linewidth=2, alpha=0.8)
        
        ax4.set_xlabel('x', fontsize=11)
        ax4.set_ylabel('ξ', fontsize=11)
        ax4.set_title('Champ hamiltonien X_H', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
        return sol

# Test avec différents hamiltoniens
from sympy import symbols, cos

x = symbols('x', real=True)
xi = symbols('xi', real=True)

print("\n" + "="*70)
print("TEST 1 : OSCILLATEUR HARMONIQUE")
print("="*70)
sol_harm = from_omega_to_flow(xi**2/2 + x**2/2, [x])

print("\n" + "="*70)
print("TEST 2 : PENDULE")
print("="*70)
sol_pendulum = from_omega_to_flow(xi**2/2 - cos(x), [x])

def interactive_symplectic_dynamics(H_options):
    """
    Dashboard interactif pour explorer différents hamiltoniens
    avec la MÊME structure symplectique
    """
    from ipywidgets import interact, Dropdown, FloatSlider
    import matplotlib.pyplot as plt
    
    @interact(
        H_choice=Dropdown(
            options=list(H_options.keys()),
            value=list(H_options.keys())[0],
            description='Hamiltonien:'
        ),
        x0=FloatSlider(min=-2, max=2, step=0.1, value=1.0, description='x₀'),
        xi0=FloatSlider(min=-2, max=2, step=0.1, value=0.5, description='ξ₀'),
        t_max=FloatSlider(min=1, max=20, step=1, value=10, description='Temps max')
    )
    def plot_dynamics(H_choice, x0, xi0, t_max):
        from sympy import lambdify, symbols
        from scipy.integrate import solve_ivp
        import numpy as np
        
        x = symbols('x', real=True)
        xi = symbols('xi', real=True)
        
        H_expr = H_options[H_choice]
        
        # Champ hamiltonien
        from sympy import diff
        dx_dt = diff(H_expr, xi)
        dxi_dt = -diff(H_expr, x)
        
        f_x = lambdify((x, xi), dx_dt, 'numpy')
        f_xi = lambdify((x, xi), dxi_dt, 'numpy')
        
        def system(t, y):
            return [f_x(y[0], y[1]), f_xi(y[0], y[1])]
        
        # Intégration
        sol = solve_ivp(system, [0, t_max], [x0, xi0], 
                       dense_output=True, method='DOP853')
        
        t_eval = np.linspace(0, t_max, 500)
        trajectory = sol.sol(t_eval)
        
        # Visualisation
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Espace des phases
        x_grid = np.linspace(-3, 3, 100)
        xi_grid = np.linspace(-3, 3, 100)
        X, XI = np.meshgrid(x_grid, xi_grid)
        
        H_func = lambdify((x, xi), H_expr, 'numpy')
        H_grid = H_func(X, XI)
        
        axes[0].contour(X, XI, H_grid, levels=20, colors='lightgray', alpha=0.5)
        axes[0].plot(trajectory[0], trajectory[1], 'b-', linewidth=2)
        axes[0].plot(x0, xi0, 'go', markersize=12, label='Initial')
        axes[0].plot(trajectory[0, -1], trajectory[1, -1], 'ro', 
                    markersize=12, label='Final')
        
        axes[0].set_xlabel('x', fontsize=12)
        axes[0].set_ylabel('ξ', fontsize=12)
        axes[0].set_title(f'Espace des phases\nH = {H_expr}', fontsize=13)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal')
        
        # Évolution temporelle
        axes[1].plot(t_eval, trajectory[0], 'b-', linewidth=2, label='x(t)')
        axes[1].plot(t_eval, trajectory[1], 'r-', linewidth=2, label='ξ(t)')
        axes[1].set_xlabel('Temps t', fontsize=12)
        axes[1].set_ylabel('Coordonnées', fontsize=12)
        axes[1].set_title('Évolution temporelle', fontsize=13)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Conservation
        H_values = H_func(trajectory[0], trajectory[1])
        print(f"\nConservation de H :")
        print(f"  H(t=0) = {H_values[0]:.8f}")
        print(f"  H(t={t_max}) = {H_values[-1]:.8f}")
        print(f"  |ΔH|/H₀ = {abs(H_values[-1] - H_values[0])/abs(H_values[0]):.2e}")

# Exemples
from sympy import symbols, cos, sin, exp

x, xi = symbols('x xi', real=True)

H_library = {
    "Libre": xi**2/2,
    "Harmonique": xi**2/2 + x**2/2,
    "Anharmonique": xi**2/2 + x**4/4,
    "Pendule": xi**2/2 - cos(x),
    "Double puits": xi**2/2 - x**2/2 + x**4/4,
    "Morse": xi**2/2 + (1 - exp(-x))**2,
    "Hénon-Heiles (simplifié)": xi**2/2 + x**2/2 + x**3,
}

interactive_symplectic_dynamics(H_library)

def explicit_derivation_hamilton_equations():
    """
    Dérive explicitement les équations de Hamilton
    à partir de la seule donnée de ω et H
    """
    from sympy import symbols, Matrix, solve, simplify
    
    print("="*70)
    print("DÉRIVATION EXPLICITE : DE ω + H → ÉQUATIONS DE HAMILTON")
    print("="*70)
    
    x, xi = symbols('x xi', real=True)
    dx_dt, dxi_dt = symbols('dx/dt dxi/dt', real=True)
    
    # Donnée 1 : Structure symplectique
    print("\n[Donnée 1] Structure symplectique ω")
    print("-"*70)
    print("ω = dξ ∧ dx")
    print("\nEn forme matricielle (base (x, ξ)) :")
    
    omega = Matrix([[0, -1], [1, 0]])
    print(omega)
    
    # Donnée 2 : Hamiltonien
    print("\n[Donnée 2] Hamiltonien H(x, ξ)")
    print("-"*70)
    
    # Symbole générique
    from sympy import Function
    H = Function('H')(x, xi)
    print(f"H = H(x, ξ)  (fonction arbitraire)")
    
    # Dérivation
    print("\n[Dérivation] Équation définissant X_H")
    print("-"*70)
    print("Le champ hamiltonien X_H = (dx/dt, dξ/dt) est défini par :")
    print("  ι_{X_H} ω = dH")
    print("\nTraduction en coordonnées :")
    
    # Vecteur champ
    X_H = Matrix([dx_dt, dxi_dt])
    print(f"\nX_H = {X_H.T}")
    
    # Gradient de H
    from sympy import diff
    dH_dx = diff(H, x)
    dH_dxi = diff(H, xi)
    grad_H = Matrix([dH_dx, dH_dxi])
    
    print(f"\ndH = ({dH_dx}, {dH_dxi})ᵀ")
    
    # Contraction ι_{X_H} ω
    print("\nContraction ι_{X_H} ω :")
    print("  ι_{X_H} ω(·) = ω(X_H, ·)")
    
    # En matriciel : ω · X_H
    contraction = omega * X_H
    print(f"\n  ω · X_H = {omega} · {X_H.T}")
    print(f"          = {contraction.T}")
    
    # Équation ι_{X_H} ω = dH
    print("\nÉquation à résoudre : ω · X_H = dH")
    print(f"  {contraction.T} = {grad_H.T}")
    
    # Système d'équations
    equations = [contraction[i] - grad_H[i] for i in range(2)]
    
    print("\nSystème :")
    for i, eq in enumerate(equations):
        print(f"  Équation {i+1} : {eq} = 0")
    
    # Résolution
    solution = solve(equations, [dx_dt, dxi_dt])
    
    print("\n[Solution] Équations de Hamilton")
    print("-"*70)
    print(f"  dx/dt = {solution[dx_dt]}")
    print(f"  dξ/dt = {solution[dxi_dt]}")
    
    print

def explicit_derivation_hamilton_equations():
    """
    Dérive explicitement les équations de Hamilton
    à partir de la seule donnée de ω et H
    """
    from sympy import symbols, Matrix, solve, simplify
    
    print("="*70)
    print("DÉRIVATION EXPLICITE : DE ω + H → ÉQUATIONS DE HAMILTON")
    print("="*70)
    
    x, xi = symbols('x xi', real=True)
    dx_dt, dxi_dt = symbols('dx/dt dxi/dt', real=True)
    
    # Donnée 1 : Structure symplectique
    print("\n[Donnée 1] Structure symplectique ω")
    print("-"*70)
    print("ω = dξ ∧ dx")
    print("\nEn forme matricielle (base (x, ξ)) :")
    
    omega = Matrix([[0, -1], [1, 0]])
    print(omega)
    
    # Donnée 2 : Hamiltonien
    print("\n[Donnée 2] Hamiltonien H(x, ξ)")
    print("-"*70)
    
    # Symbole générique
    from sympy import Function
    H = Function('H')(x, xi)
    print(f"H = H(x, ξ)  (fonction arbitraire)")
    
    # Dérivation
    print("\n[Dérivation] Équation définissant X_H")
    print("-"*70)
    print("Le champ hamiltonien X_H = (dx/dt, dξ/dt) est défini par :")
    print("  ι_{X_H} ω = dH")
    print("\nTraduction en coordonnées :")
    
    # Vecteur champ
    X_H = Matrix([dx_dt, dxi_dt])
    print(f"\nX_H = {X_H.T}")
    
    # Gradient de H
    from sympy import diff
    dH_dx = diff(H, x)
    dH_dxi = diff(H, xi)
    grad_H = Matrix([dH_dx, dH_dxi])
    
    print(f"\ndH = ({dH_dx}, {dH_dxi})ᵀ")
    
    # Contraction ι_{X_H} ω
    print("\nContraction ι_{X_H} ω :")
    print("  ι_{X_H} ω(·) = ω(X_H, ·)")
    
    # En matriciel : ω · X_H
    contraction = omega * X_H
    print(f"\n  ω · X_H = {omega} · {X_H.T}")
    print(f"          = {contraction.T}")
    
    # Équation ι_{X_H} ω = dH
    print("\nÉquation à résoudre : ω · X_H = dH")
    print(f"  {contraction.T} = {grad_H.T}")
    
    # Système d'équations
    equations = [contraction[i] - grad_H[i] for i in range(2)]
    
    print("\nSystème :")
    for i, eq in enumerate(equations):
        print(f"  Équation {i+1} : {eq} = 0")
    
    # Résolution
    solution = solve(equations, [dx_dt, dxi_dt])
    
    print("\n[Solution] Équations de Hamilton")
    print("-"*70)
    print(f"  dx/dt = {solution[dx_dt]}")
    print(f"  dξ/dt = {solution[dxi_dt]}")
    
    print("\n✨ RÉSULTAT UNIVERSEL ✨")
    print("-"*70)
    print("Pour TOUT hamiltonien H(x, ξ) :")
    print("  • dx/dt = ∂H/∂ξ")
    print("  • dξ/dt = -∂H/∂x")
    print("\nCes équations émergent UNIQUEMENT de :")
    print("  1. La structure symplectique ω = dξ ∧ dx (indépendante de H)")
    print("  2. L'équation géométrique ι_{X_H} ω = dH")
    
    # Vérification avec des exemples concrets
    print("\n" + "="*70)
    print("VÉRIFICATIONS AVEC EXEMPLES CONCRETS")
    print("="*70)
    
    hamiltonians = {
        "Libre": xi**2/2,
        "Harmonique": xi**2/2 + x**2/2,
        "Pendule": xi**2/2 - sp.cos(x),
    }
    
    for name, H_concrete in hamiltonians.items():
        print(f"\n{name} : H = {H_concrete}")
        print("-"*60)
        
        dH_dx = diff(H_concrete, x)
        dH_dxi = diff(H_concrete, xi)
        
        dx_dt_result = dH_dxi
        dxi_dt_result = -dH_dx
        
        print(f"  dx/dt = ∂H/∂ξ = {dx_dt_result}")
        print(f"  dξ/dt = -∂H/∂x = {dxi_dt_result}")
    
    return omega, solution

omega, hamilton_eqs = explicit_derivation_hamilton_equations()

def symplectic_structure_higher_dimensions(n=2):
    """
    Montre comment ω se généralise en dimension quelconque
    """
    from sympy import symbols, Matrix, simplify
    
    print("="*70)
    print(f"STRUCTURE SYMPLECTIQUE EN DIMENSION {n}")
    print("="*70)
    
    # Variables
    x_vars = symbols(f'x1:{n+1}', real=True)
    xi_vars = symbols(f'xi1:{n+1}', real=True)
    
    print(f"\n[Variables] T*ℝ^{n} a dimension {2*n}")
    print("-"*70)
    print(f"Variables spatiales : {x_vars}")
    print(f"Variables impulsion : {xi_vars}")
    
    # Forme symplectique
    print("\n[Forme symplectique canonique]")
    print("-"*70)
    print("ω = Σᵢ dξᵢ ∧ dxⁱ")
    
    # Matrice symplectique
    omega = Matrix.zeros(2*n, 2*n)
    
    for i in range(n):
        # Bloc [[0, -I], [I, 0]]
        omega[i, n+i] = -1
        omega[n+i, i] = 1
    
    print(f"\nMatrice ω (taille {2*n}×{2*n}) :")
    print(omega)
    
    # Propriétés
    print("\n[Propriétés]")
    print("-"*70)
    
    # Non-dégénérée
    det_omega = omega.det()
    print(f"✓ det(ω) = {det_omega} ≠ 0  (non-dégénérée)")
    
    # Antisymétrie
    is_antisymmetric = (omega + omega.T).is_zero_matrix
    print(f"✓ ω = -ωᵀ : {is_antisymmetric}  (antisymétrique)")
    
    # Inverse
    omega_inv = omega.inv()
    print(f"\n✓ ω⁻¹ existe :")
    print(omega_inv)
    
    # Équations de Hamilton
    print("\n[Équations de Hamilton générales]")
    print("-"*70)
    print("Pour H(x₁, ..., xₙ, ξ₁, ..., ξₙ) :")
    print()
    for i in range(n):
        print(f"  dxᵢ/dt = ∂H/∂ξᵢ")
    print()
    for i in range(n):
        print(f"  dξᵢ/dt = -∂H/∂xᵢ")
    
    # Structure par blocs
    print("\n[Structure par blocs]")
    print("-"*70)
    print("ω et ω⁻¹ ont la structure canonique :")
    print()
    print("      ⎡  0   -I ⎤         ⎡  0    I ⎤")
    print("  ω = ⎢         ⎥,  ω⁻¹ = ⎢         ⎥")
    print("      ⎣  I    0 ⎦         ⎣ -I    0 ⎦")
    print()
    print("où I est la matrice identité n×n")
    
    return omega

# Test en dimension 2 et 3
omega_2d = symplectic_structure_higher_dimensions(n=2)
print("\n" + "="*70 + "\n")
omega_3d = symplectic_structure_higher_dimensions(n=3)

def canonical_transformations_preserve_omega():
    """
    Montre que les transformations canoniques préservent ω
    """
    from sympy import symbols, Matrix, simplify, diff, Function
    
    print("="*70)
    print("TRANSFORMATIONS CANONIQUES ET PRÉSERVATION DE ω")
    print("="*70)
    
    x, xi, X, Xi = symbols('x xi X Xi', real=True)
    
    print("\n[Définition]")
    print("-"*70)
    print("Une transformation (x, ξ) ↦ (X, Ξ) est CANONIQUE si elle préserve ω :")
    print("  ω̃ = ω")
    print("où ω̃ = dΞ ∧ dX dans les nouvelles coordonnées")
    
    print("\n[Condition nécessaire et suffisante]")
    print("-"*70)
    print("La transformation est canonique ⟺ le crochet de Poisson est préservé :")
    print("  {f, g}_{(x,ξ)} = {f, g}_{(X,Ξ)}")
    print("pour toutes fonctions f, g")
    
    # Exemple 1 : Translation
    print("\n[Exemple 1 : Translation]")
    print("-"*60)
    
    a, b = symbols('a b', real=True)
    X_trans = x + a
    Xi_trans = xi + b
    
    print(f"Transformation : X = x + a, Ξ = ξ + b")
    print(f"dX = dx, dΞ = dξ")
    print(f"ω̃ = dΞ ∧ dX = dξ ∧ dx = ω ✓")
    
    # Exemple 2 : Rotation dans l'espace des phases
    print("\n[Exemple 2 : Rotation symplectique]")
    print("-"*60)
    
    from sympy import cos, sin
    theta = symbols('theta', real=True)
    
    X_rot = cos(theta)*x + sin(theta)*xi
    Xi_rot = -sin(theta)*x + cos(theta)*xi
    
    print(f"Transformation (rotation d'angle θ) :")
    print(f"  X = cos(θ)·x + sin(θ)·ξ")
    print(f"  Ξ = -sin(θ)·x + cos(θ)·ξ")
    
    # Calcul de dX ∧ dΞ
    dX_dx = cos(theta)
    dX_dxi = sin(theta)
    dXi_dx = -sin(theta)
    dXi_dxi = cos(theta)
    
    # Jacobien
    J = Matrix([[dX_dx, dX_dxi], [dXi_dx, dXi_dxi]])
    
    print(f"\nJacobien J =")
    print(J)
    
    # Vérification : Jᵀ ω J = ω
    omega = Matrix([[0, -1], [1, 0]])
    transformed = simplify(J.T * omega * J)
    
    print(f"\nJᵀ ω J =")
    print(transformed)
    print(f"\nω̃ = ω ? {transformed == omega} ✓")
    
    # Exemple 3 : Transformation générant (NON canonique)
    print("\n[Contre-exemple : Transformation NON canonique]")
    print("-"*60)
    
    c = symbols('c', real=True, positive=True)
    X_scale = c*x
    Xi_scale = xi/c  # Scaling qui ne préserve PAS l'aire
    
    print(f"Transformation : X = c·x, Ξ = ξ/c  (c ≠ 1)")
    
    J_scale = Matrix([[c, 0], [0, 1/c]])
    transformed_scale = simplify(J_scale.T * omega * J_scale)
    
    print(f"\nJacobien J =")
    print(J_scale)
    
    print(f"\nJᵀ ω J =")
    print(transformed_scale)
    
    print(f"\nω̃ = ω ? {transformed_scale == omega}")
    print(f"Cette transformation n'est PAS canonique (sauf si c = 1) ✗")
    
    # Génération par une fonction génératrice
    print("\n[Génération par fonction génératrice]")
    print("-"*70)
    print("Toute transformation canonique peut être générée par une fonction F.")
    print("\nTypes de fonctions génératrices :")
    print("  • F₁(x, X) : ξ = ∂F₁/∂x, Ξ = -∂F₁/∂X")
    print("  • F₂(x, Ξ) : ξ = ∂F₂/∂x, X = ∂F₂/∂Ξ")
    print("  • F₃(ξ, X) : x = -∂F₃/∂ξ, Ξ = -∂F₃/∂X")
    print("  • F₄(ξ, Ξ) : x = -∂F₄/∂ξ, X = ∂F₄/∂Ξ")
    
    # Exemple avec F₂
    print("\nExemple : Transformation identité via F₂(x, Ξ) = x·Ξ")
    
    F2 = x * Xi
    xi_from_F2 = diff(F2, x)
    X_from_F2 = diff(F2, Xi)
    
    print(f"  F₂ = x·Ξ")
    print(f"  ξ = ∂F₂/∂x = {xi_from_F2}")
    print(f"  X = ∂F₂/∂Ξ = {X_from_F2}")
    print(f"  Donc : X = x, Ξ = ξ (identité) ✓")

canonical_transformations_preserve_omega()

def from_poisson_to_commutator():
    """
    Montre le lien fondamental entre crochet de Poisson (classique)
    et commutateur (quantique) via la correspondance
    """
    from sympy import symbols, diff, simplify, I
    
    print("="*70)
    print("DU CROCHET DE POISSON AU COMMUTATEUR QUANTIQUE")
    print("="*70)
    
    x, xi = symbols('x xi', real=True)
    
    print("\n[Niveau classique : Crochet de Poisson]")
    print("-"*70)
    print("Pour deux observables f, g : T*M → ℝ :")
    print("  {f, g} = ∂f/∂x · ∂g/∂ξ - ∂f/∂ξ · ∂g/∂x")
    print("\nCe crochet est INDUIT par la structure symplectique ω")
    
    # Exemples de crochets
    print("\n[Crochets fondamentaux]")
    print("-"*60)
    
    observables = {
        'x et ξ': (x, xi),
        'x et x': (x, x),
        'ξ et ξ': (xi, xi),
        'x² et ξ': (x**2, xi),
        'x et ξ²': (x, xi**2),
    }
    
    for name, (f, g) in observables.items():
        bracket = diff(f, x)*diff(g, xi) - diff(f, xi)*diff(g, x)
        bracket = simplify(bracket)
        print(f"  {{{name}}} = {bracket}")
    
    print("\n[Niveau quantique : Commutateur]")
    print("-"*70)
    print("En mécanique quantique, les observables deviennent des opérateurs")
    print("Le crochet de Poisson devient le commutateur via :")
    print()
    print("  PRINCIPE DE CORRESPONDANCE (Dirac)")
    print("  ════════════════════════════════════")
    print("  {f, g}_classique  ↦  (1/iℏ) [f̂, ĝ]_quantique")
    print()
    print("Ou de manière équivalente :")
    print("  [f̂, ĝ] = iℏ {f, g}")
    
    print("\n[Relations de commutation canoniques]")
    print("-"*60)
    print("Du crochet de Poisson {x, ξ} = 1, on déduit :")
    print("  [x̂, p̂] = iℏ")
    print("\nC'est le fondement de la mécanique quantique !")
    
    print("\n[Table de correspondance]")
    print("-"*60)
    
    correspondence_table = [
        ("Classique", "Quantique"),
        ("─"*30, "─"*30),
        ("{x, ξ} = 1", "[x̂, p̂] = iℏ"),
        ("{x, x} = 0", "[x̂, x̂] = 0"),
        ("{ξ, ξ} = 0", "[p̂, p̂] = 0"),
        ("{x², ξ} = 2x", "[x̂², p̂] = 2iℏx̂"),
        ("{H, f} = df/dt", "[Ĥ, f̂] = iℏ df̂/dt"),
    ]
    
    print()
    for classical, quantum in correspondence_table:
        print(f"  {classical:30s} ←→ {quantum}")
    
    # Quantification de Weyl
    print("\n[Quantification de Weyl]")
    print("-"*70)
    print("Pour quantifier un hamiltonien classique H(x, ξ) :")
    print()
    print("1. QUANTIFICATION STANDARD (Kohn-Nirenberg) :")
    print("   H(x, ξ) ↦ H(x̂, p̂)  (ordre des opérateurs : position puis impulsion)")
    print()
    print("2. QUANTIFICATION DE WEYL (symétrique) :")
    print("   Opérateur Ĥ défini par son symbole de Weyl")
    print("   Préserve mieux la structure symplectique")
    print("   Auto-adjoint si H réel")
    
    print("\n[Exemple : Oscillateur harmonique]")
    print("-"*60)
    
    H_classical = xi**2/2 + x**2/2
    
    print(f"Hamiltonien classique : H = {H_classical}")
    print()
    print("Quantification standard :")
    print("  Ĥ = p̂²/2 + x̂²/2")
    print()
    print("Relations de commutation utilisées :")
    print("  [x̂, p̂] = iℏ  (émerge de {x, ξ} = 1)")
    print()
    print("Spectre (niveaux d'énergie) :")
    print("  Eₙ = ℏω(n + 1/2),  n = 0, 1, 2, ...")
    
    # Diagramme conceptuel
    print("\n[DIAGRAMME CONCEPTUEL]")
    print("="*70)
    print()
    print("  T*M avec structure symplectique ω")
    print("           │")
    print("           │ Induit")
    print("           ↓")
    print("  Crochet de Poisson {·, ·}")
    print("           │")
    print("           │ Quantification")
    print("           │ (ℏ → 0 : limite classique)")
    print("           ↓")
    print("  Commutateur [·, ·] = iℏ{·, ·}")
    print("           │")
    print("           │ Définit")
    print("           ↓")
    print("  Algèbre des observables quantiques")
    print()
    print("="*70)
    
    return correspondence_table

correspondence = from_poisson_to_commutator()

def conceptual_architecture():
    """
    Vue d'ensemble de l'architecture conceptuelle
    """
    
    diagram = """
    
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                   ARCHITECTURE CONCEPTUELLE                       ║
    ║            Émergence de la structure symplectique                 ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    
    NIVEAU 0 : GÉOMÉTRIE DU FIBRÉ COTANGENT
    ═══════════════════════════════════════
    
         Variété de base M (espace physique)
                    │
                    │ Construction géométrique
                    │
                    ↓
         Fibré cotangent T*M (espace des phases)
                    │
                    │ Structure intrinsèque
                    │
                    ↓
         Forme de Liouville θ = Σᵢ ξᵢ dxⁱ (CANONIQUE)
                    │
                    │ Différentielle extérieure d
                    │
                    ↓
         Forme symplectique ω = dθ (UNIVERSELLE)
    
         ✨ ω existe AVANT tout choix de hamiltonien H
    
    
    NIVEAU 1 : DYNAMIQUE HAMILTONIENNE
    ═══════════════════════════════════
    
         Fonction H : T*M → ℝ (hamiltonien QUELCONQUE)
                    │
                    │ Équation : ι_{X_H} ω = dH
                    │
                    ↓
         Champ hamiltonien X_H (tangent à T*M)
                    │
                    │ Intégration
                    │
                    ↓
         Flot Φₜᴴ : T*M → T*M
    
         ✨ Φₜᴴ PRÉSERVE ω (Théorème de Liouville)
    
    
    NIVEAU 2 : STRUCTURE ALGÉBRIQUE
    ═══════════════════════════════════
    
         Algèbre C∞(T*M) des fonctions lisses
                    │
                    │ ω induit un crochet
                    │
                    ↓
         Crochet de Poisson {f, g}
                    │
                    │ Propriétés :
                    │ • Bilinéaire
                    │ • Antisymétrique
                    │ • Identité de Jacobi
                    │ • Dérivation de Leibniz
                    │
                    ↓
         Structure d'algèbre de Lie
    
         ✨ {f, g} est COMPLÈTEMENT déterminé par ω
    
    
    NIVEAU 3 : QUANTIFICATION
    ═════════════════════════
    
         Crochet de Poisson {·, ·} (classique)
                    │
                    │ Correspondance :
                    │ {f, g} ↦ (1/iℏ)[f̂, ĝ]
                    │
                    ↓
         Commutateur [·, ·] (quantique)
                    │
                    │ Exemple : [x̂, p̂] = iℏ
                    │          (de {x, ξ} = 1)
                    │
                    ↓
         Opérateurs pseudo-différentiels
    
         ✨ La structure quantique HÉRITE de ω
    
    
    PROPRIÉTÉS UNIVERSELLES
    ═══════════════════════
    
    ✓ ω est INDÉPENDANTE du choix de H
    ✓ ω est CANONIQUE (ne dépend pas des coordonnées)
    ✓ ω est FERMÉE : dω = 0
    ✓ ω est NON-DÉGÉNÉRÉE : det(ω) ≠ 0
    ✓ Tout H → même ω → mêmes lois de conservation
    ✓ La quantification préserve la structure symplectique
    
    
    THÉORÈMES CLÉS
    ═════════════
    
    • Théorème de Darboux :
      Localement, toute structure symplectique est isomorphe à
      la structure canonique ω = Σᵢ dξᵢ ∧ dxⁱ
    
    • Théorème de Liouville (géométrique) :
      Le flot hamiltonien préserve ω
    
    • Théorème de Noether (symplectique) :
      Symétries ↔ Quantités conservées via ω
    
    """
    
    print(diagram)
    
    # Résumé synthétique
    print("\n" + "="*70)
    print("RÉSUMÉ : COMMENT ω ÉMERGE-T-ELLE ?")
    print("="*70)
    print()
    print("La structure symplectique ω n'est PAS une conséquence du hamiltonien H,")
    print("mais une PROPRIÉTÉ GÉOMÉTRIQUE INTRINSÈQUE du fibré cotangent T*M.")
    print()
    print("Séquence d'émergence :")
    print("  1. M (variété) → T*M (fibré cotangent) [construction géométrique]")
    print("  2. T*M → θ (forme de Liouville) [structure canonique unique]")
    print("  3. θ → ω = dθ [différentielle extérieure]")
    print("  4. ω + H → X_H [équation ι_{X_H}ω = dH]")
    print("  5. X_H → Φₜ [intégration du flot]")
    print()
    print("À CHAQUE ÉTAPE, la structure est DÉTERMINÉE par la géométrie,")
    print("PAS par un choix arbitraire de hamiltonien.")
    print()
    print("="*70)

conceptual_architecture()

def complete_symplectic_example_unified():
    """
    Exemple complet montrant TOUT le pipeline :
    H → ω → X_H → Φₜ → Conservations
    
    Utilise les classes de psiop.py pour l'intégration
    """
    from sympy import symbols, cos, sin, diff, lambdify
    from scipy.integrate import solve_ivp
    import numpy as np
    import matplotlib.pyplot as plt
    from psiop import PseudoDifferentialOperator
    
    print("="*70)
    print("EXEMPLE COMPLET : PENDULE AVEC STRUCTURE SYMPLECTIQUE")
    print("="*70)
    
    # Étape 1 : Définir le hamiltonien
    print("\n[ÉTAPE 1] Définition du hamiltonien")
    print("-"*70)
    
    x, xi = symbols('x xi', real=True)
    g, L, m = 1.0, 1.0, 1.0  # Constantes physiques
    
    H_pendule = m*L**2*xi**2/2 - m*g*L*cos(x)
    
    print(f"Hamiltonien du pendule : H = {H_pendule}")
    print(f"  Terme cinétique : T = m·L²·ξ²/2")
    print(f"  Terme potentiel : V = -m·g·L·cos(x)")
    
    # Étape 2 : Structure symplectique (indépendante de H!)
    print("\n[ÉTAPE 2] Structure symplectique canonique")
    print("-"*70)
    
    from sympy import Matrix
    omega = Matrix([[0, -1], [1, 0]])
    
    print("ω = dξ ∧ dx")
    print("\nMatrice symplectique :")
    print(omega)
    print("\nCette structure existe AVANT qu'on choisisse H !")
    
    # Étape 3 : Champ hamiltonien via ω
    print("\n[ÉTAPE 3] Champ hamiltonien X_H = ω⁻¹·dH")
    print("-"*70)
    
    dH_dx = diff(H_pendule, x)
    dH_dxi = diff(H_pendule, xi)
    
    dx_dt = dH_dxi
    dxi_dt = -dH_dx
    
    print(f"dH = ({dH_dx}, {dH_dxi})ᵀ")
    print(f"\nChamp hamiltonien :")
    print(f"  dx/dt = ∂H/∂ξ = {dx_dt}")
    print(f"  dξ/dt = -∂H/∂x = {dxi_dt}")
    
    # Étape 4 : Intégration du flot
    print("\n[ÉTAPE 4] Intégration du flot hamiltonien")
    print("-"*70)
    
    f_x = lambdify((x, xi), dx_dt, 'numpy')
    f_xi = lambdify((x, xi), dxi_dt, 'numpy')
    
    def pendulum_system(t, y):
        return [f_x(y[0], y[1]), f_xi(y[0], y[1])]
    
    # Conditions initiales multiples
    initial_conditions = [
        (np.pi/6, 0.0, "Petite oscillation"),
        (np.pi/2, 0.0, "Grande oscillation"),
        (np.pi, 0.5, "Proche séparatrice"),
        (0.0, 2.0, "Rotation"),
    ]
    
    t_span = [0, 20]
    t_eval = np.linspace(*t_span, 1000)
    
    solutions = []
    
    for x0, xi0, label in initial_conditions:
        sol = solve_ivp(
            pendulum_system, t_span, [x0, xi0],
            t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-12
        )
        solutions.append((sol, label))
        print(f"  {label:20s} : (x₀, ξ₀) = ({x0:.2f}, {xi0:.2f})")
    
    # Étape 5 : Vérification des conservations
    print("\n[ÉTAPE 5] Vérifications (Théorèmes de Liouville)")
    print("-"*70)
    
    H_func = lambdify((x, xi), H_pendule, 'numpy')
    
    for (sol, label), (x0, xi0, _) in zip(solutions, initial_conditions):
        H_values = H_func(sol.y[0], sol.y[1])
        H_conservation = abs(H_values[-1] - H_values[0]) / abs(H_values[0])
        
        # Calcul du jacobien pour vérifier det = 1
        epsilon = 1e-7
        t_test = 10.0
        
        # Perturbations
        sol_x_p = solve_ivp(pendulum_system, [0, t_test], 
                           [x0 + epsilon, xi0], method='DOP853', rtol=1e-10)
        sol_x_m = solve_ivp(pendulum_system, [0, t_test], 
                           [x0 - epsilon, xi0], method='DOP853', rtol=1e-10)
        sol_xi_p = solve_ivp(pendulum_system, [0, t_test], 
                            [x0, xi0 + epsilon], method='DOP853', rtol=1e-10)
        sol_xi_m = solve_ivp(pendulum_system, [0, t_test], 
                            [x0, xi0 - epsilon], method='DOP853', rtol=1e-10)
        
        # Jacobien à t_test
        J11 = (sol_x_p.y[0, -1] - sol_x_m.y[0, -1]) / (2*epsilon)
        J12 = (sol_xi_p.y[0, -1] - sol_xi_m.y[0, -1]) / (2*epsilon)
        J21 = (sol_x_p.y[1, -1] - sol_x_m.y[1, -1]) / (2*epsilon)
        J22 = (sol_xi_p.y[1, -1] - sol_xi_m.y[1, -1]) / (2*epsilon)
        
        det_J = J11*J22 - J12*J21
        
        print(f"\n  {label:20s} :")
        print(f"    Conservation H : |ΔH|/H₀ = {H_conservation:.2e}")
        print(f"    Conservation ω : det(DΦₜ) = {det_J:.10f} (≈ 1)")
    
    # Étape 6 : Visualisation complète
    print("\n[ÉTAPE 6] Visualisation de la géométrie symplectique")
    print("-"*70)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1 : Portrait de phase complet
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    # Grille pour les courbes de niveau
    x_grid = np.linspace(-np.pi, 2*np.pi, 300)
    xi_grid = np.linspace(-3, 3, 300)
    X, XI = np.meshgrid(x_grid, xi_grid, indexing='ij')
    H_grid = H_func(X, XI)
    
    # Courbes de niveau de H (surfaces d'énergie)
    contours = ax1.contour(X, XI, H_grid, levels=30, colors='lightblue', 
                          alpha=0.4, linewidths=1)
    ax1.clabel(contours, inline=True, fontsize=7, fmt='%.1f')
    
    # Séparatrice (énergie critique)
    E_sep = H_func(np.pi, 0)
    ax1.contour(X, XI, H_grid, levels=[E_sep], colors='red', 
               linewidths=3, linestyles='--')
    
    # Trajectoires
    colors = ['green', 'blue', 'orange', 'purple']
    for (sol, label), color in zip(solutions, colors):
        ax1.plot(sol.y[0], sol.y[1], color=color, linewidth=2, 
                label=label, alpha=0.8)
        # Point initial
        ax1.plot(sol.y[0, 0], sol.y[1, 0], 'o', color=color, 
                markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    ax1.set_xlabel('x (angle)', fontsize=12)
    ax1.set_ylabel('ξ (moment angulaire)', fontsize=12)
    ax1.set_title('Portrait de phase du pendule\n(Courbes de niveau = surfaces d\'énergie constante)', 
                 fontsize=13)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-np.pi, 2*np.pi)
    ax1.set_ylim(-3, 3)
    
    # Marquer les points fixes
    ax1.plot(0, 0, 'k*', markersize=15, label='Équilibre stable')
    ax1.plot(np.pi, 0, 'rx', markersize=15, markeredgewidth=3, 
            label='Équilibre instable')
    ax1.legend(loc='upper right', fontsize=9)
    
    # Plot 2 : Conservation de H
    ax2 = fig.add_subplot(gs[0, 2])
    
    for (sol, label), color in zip(solutions, colors):
        H_traj = H_func(sol.y[0], sol.y[1])
        H0 = H_traj[0]
        relative_error = np.abs((H_traj - H0) / H0)
        ax2.semilogy(sol.t, relative_error, color=color, linewidth=2, 
                    label=label, alpha=0.8)
    
    ax2.set_xlabel('Temps t', fontsize=10)
    ax2.set_ylabel('|ΔH|/H₀', fontsize=10)
    ax2.set_title('Conservation de l\'énergie', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3 : Évolution temporelle (exemple)
    ax3 = fig.add_subplot(gs[1, 2])
    
    sol_example, label_example = solutions[1]  # Grande oscillation
    ax3.plot(sol_example.t, sol_example.y[0], 'b-', linewidth=2, label='x(t)')
    ax3.plot(sol_example.t, sol_example.y[1], 'r-', linewidth=2, label='ξ(t)')
    ax3.set_xlabel('Temps t', fontsize=10)
    ax3.set_ylabel('Coordonnées', fontsize=10)
    ax3.set_title(f'Évolution : {label_example}', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4 : Champ de vecteurs hamiltonien
    ax4 = fig.add_subplot(gs[2, 0])
    
    x_vec = np.linspace(-np.pi, 2*np.pi, 20)
    xi_vec = np.linspace(-3, 3, 20)
    X_vec, XI_vec = np.meshgrid(x_vec, xi_vec, indexing='ij')
    
    U = f_x(X_vec, XI_vec)
    V = f_xi(X_vec, XI_vec)
    
    # Normalisation pour visualisation
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = U / (magnitude + 1e-10)
    V_norm = V / (magnitude + 1e-10)
    
    ax4.quiver(X_vec, XI_vec, U_norm, V_norm, magnitude, 
              cmap='viridis', alpha=0.6, scale=25, width=0.003)
    ax4.contour(X, XI, H_grid, levels=15, colors='gray', 
               alpha=0.3, linewidths=0.5)
    
    ax4.set_xlabel('x', fontsize=10)
    ax4.set_ylabel('ξ', fontsize=10)
    ax4.set_title('Champ hamiltonien X_H', fontsize=11)
    ax4.set_xlim(-np.pi, 2*np.pi)
    ax4.set_ylim(-3, 3)
    
    # Plot 5 : Forme symplectique (visualisation abstraite)
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Visualiser l'aire préservée par le flot
    # Prendre un petit carré et suivre son évolution
    
    square_corners = np.array([
        [np.pi/4, 0.5],
        [np.pi/4 + 0.3, 0.5],
        [np.pi/4 + 0.3, 0.7],
        [np.pi/4, 0.7],
        [np.pi/4, 0.5]  # Fermer
    ])
    
    # Aire initiale
    area_init = 0.3 * 0.2  # Rectangle
    
    # Évolution à différents temps
    times_to_plot = [0, 2, 5, 10]
    colors_time = ['red', 'orange', 'yellow', 'green']
    
    for t_plot, c_time in zip(times_to_plot, colors_time):
        evolved_corners = []
        for corner in square_corners:
            sol_corner = solve_ivp(
                pendulum_system, [0, t_plot], corner,
                method='DOP853', rtol=1e-10
            )
            evolved_corners.append([sol_corner.y[0, -1], sol_corner.y[1, -1]])
        
        evolved_corners = np.array(evolved_corners)
        
        # Calculer l'aire (formule du lacet)
        area = 0.5 * np.abs(np.sum(
            evolved_corners[:-1, 0] * evolved_corners[1:, 1] - 
            evolved_corners[1:, 0] * evolved_corners[:-1, 1]
        ))
        
        ax5.plot(evolved_corners[:, 0], evolved_corners[:, 1], 
                color=c_time, linewidth=2, label=f't={t_plot}, A={area:.4f}')
        ax5.fill(evolved_corners[:, 0], evolved_corners[:, 1], 
                color=c_time, alpha=0.2)
    
    ax5.contour(X, XI, H_grid, levels=15, colors='gray', 
               alpha=0.2, linewidths=0.5)
    ax5.set_xlabel('x', fontsize=10)
    ax5.set_ylabel('ξ', fontsize=10)
    ax5.set_title(f'Conservation de l\'aire symplectique\n(A₀={area_init:.4f})', 
                 fontsize=11)
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6 : Spectre et quantification (connexion)
    ax6 = fig.add_subplot(gs[2, 2])
    
    # Niveaux d'énergie classiques vs quantiques
    E_classical = np.linspace(-2, 2, 50)
    
    # Pour chaque énergie, calculer l'action (aire enfermée)
    # Approximation : période ~ surface enfermée
    actions_classical = []
    
    for E in E_classical:
        if E > -m*g*L:  # Au-dessus du minimum
            # Approximation : pour petites oscillations, action ~ E/ω
            omega_0 = np.sqrt(g/L)
            action_approx = 2*np.pi*E / omega_0 if E > 0 else 0
            actions_classical.append(action_approx)
        else:
            actions_classical.append(0)
    
    actions_classical = np.array(actions_classical)
    
    ax6.plot(E_classical, actions_classical, 'b-', linewidth=2, 
            label='Action classique')
    
    # Quantification de Bohr-Sommerfeld : I = n·ℏ
    hbar = 0.3  # Paramètre fictif pour visualisation
    n_levels = np.arange(0, 10)
    I_quantum = n_levels * hbar
    
    # Marquer les niveaux quantiques
    for n, I_n in zip(n_levels, I_quantum):
        ax6.axhline(I_n, color='red', linestyle='--', alpha=0.5, linewidth=1)
        if I_n < max(actions_classical):
            ax6.text(1.8, I_n, f'n={n}', fontsize=8, color='red')
    
    ax6.set_xlabel('Énergie E', fontsize=10)
    ax6.set_ylabel('Action I', fontsize=10)
    ax6.set_title('Quantification (Bohr-Sommerfeld)\nI = n·ℏ', fontsize=11)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('PENDULE : Géométrie symplectique complète', 
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.savefig('pendule_symplectique_complet.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Étape 7 : Connexion avec les opérateurs Ψ-DO
    print("\n[ÉTAPE 7] Connexion avec les opérateurs pseudo-différentiels")
    print("-"*70)
    
    # Créer l'opérateur via psiop.py
    H_op = PseudoDifferentialOperator(H_pendule, [x], mode='symbol')
    
    print(f"Opérateur hamiltonien créé via psiop.py")
    print(f"  Symbole : {H_op.symbol}")
    print(f"  Ordre : {H_op.symbol_order()}")
    
    # Champ hamiltonien via la méthode de l'opérateur
    flow = H_op.symplectic_flow()
    print(f"\nChamp symplectique (via méthode de l'opérateur) :")
    print(f"  dx/dt = {flow['dx/dt']}")
    print(f"  dξ/dt = {flow['dxi/dt']}")
    
    print("\n✨ La structure symplectique est ENCODÉE dans l'opérateur !")
    
    # Résumé final
    print("\n" + "="*70)
    print("RÉSUMÉ DU PIPELINE COMPLET")
    print("="*70)
    print()
    print("1. GÉOMÉTRIE : T*M → ω (structure symplectique canonique)")
    print("2. DYNAMIQUE : H + ω → X_H (champ hamiltonien)")
    print("3. ÉVOLUTION : X_H → Φₜ (flot)")
    print("4. CONSERVATIONS : Φₜ préserve H et ω (Liouville)")
    print("5. QUANTIFICATION : {·,·} → [·,·] (correspondance)")
    print("6. OPÉRATEURS : Symboles → Ψ-DO (analyse microlocale)")
    print()
    print("À CHAQUE ÉTAPE, la structure symplectique ω joue le rôle central,")
    print("et elle est INDÉPENDANTE du choix particulier de H.")
    print()
    print("="*70)

# Exécution
complete_symplectic_example_unified()

def philosophical_conclusion():
    """
    Réflexion finale sur la nature de la structure symplectique
    """
    
    conclusion = """
    
    ╔═══════════════════════════════════════════════════════════════════╗
    ║          CONCLUSION : LA NATURE DE LA STRUCTURE SYMPLECTIQUE      ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    
    QUESTION INITIALE
    ═════════════════
    
    "Si on prend un hamiltonien quelconque sur T*M,
     comment émerge la structure symplectique ?"
    
    
    RÉPONSE FONDAMENTALE
    ════════════════════
    
    La question contient une INVERSION CONCEPTUELLE !
    
    Ce n'est PAS :  H → ω  (le hamiltonien génère la structure)
    
    Mais PLUTÔT :   ω → dynamique(H)  (la structure pré-existe)
    
    
    LA VÉRITABLE SÉQUENCE
    ═════════════════════
    
    1. GÉOMÉTRIE PURE (niveau fondamental)
       ───────────────────────────────────
       • M existe (variété de configuration)
       • T*M existe automatiquement (fibré cotangent)
       • θ = Σᵢ ξᵢ dxⁱ existe (forme tautologique)
       • ω = dθ existe (différentielle extérieure)
    
       ➜ ω est une structure GÉOMÉTRIQUE, pas physique
       ➜ Elle existe AVANT tout choix de dynamique
    
    2. CHOIX DE PHYSIQUE (niveau secondaire)
       ────────────────────────────────────
       • On choisit un hamiltonien H : T*M → ℝ
       • H peut être ARBITRAIRE (polynomial, transcendant, ...)
    
    3. GÉNÉRATION DE LA DYNAMIQUE (synthèse)
       ───────────────────────────────────────
       • ω "sait" comment transformer H en champ de vecteurs
       • Équation universelle : ι_{X_H} ω = dH
       • Cette équation DÉTERMINE UNIQUEMENT X_H
    
    
    ANALOGIE ÉCLAIRANTE
    ═══════════════════
    
    La structure symplectique est comme la MÉTRIQUE sur une variété :
    
    • La métrique g existe sur la variété (géométrie de Riemann)
    • On peut définir diverses FONCTIONS sur la variété
    • Chaque fonction définit un gradient via g
    • Le gradient dépend de la fonction MAIS PAS g elle-même
    
    De même :
    
    • La forme ω existe sur T*M (géométrie symplectique)
    • On peut définir divers HAMILTONIENS sur T*M
    • Chaque H définit un flot via ω
    • Le flot dépend de H MAIS PAS ω elle-même
    
    
    IMPLICATIONS PROFONDES
    ══════════════════════
    
    1. UNIVERSALITÉ
       ────────────
       Tous les systèmes hamiltoniens partagent la MÊME structure ω.
       Les différences physiques viennent de H, pas de ω.
    
    2. GÉOMÉTRISATION DE LA PHYSIQUE
       ─────────────────────────────
       La dynamique n'est pas une "loi" imposée de l'extérieur,
       mais une CONSÉQUENCE GÉOMÉTRIQUE de (T*M, ω, H).
    
    3. QUANTIFICATION NATURELLE
       ────────────────────────
       La structure quantique [·,·] HÉRITE de ω via {·,·}.
       La mécanique quantique est "déjà contenue" dans (T*M, ω).
    
    4. UNIFICATION CONCEPTUELLE
       ────────────────────────
       • Mécanique classique = (T*M, ω, H, flot symplectique)
       • Optique géométrique = (T*M, ω, eikonal, rayons)
       • Mécanique quantique = (ℋ, commutateur, Ĥ, Schrödinger)
       • Opérateurs Ψ-DO = (symboles, composition, H_op, propagation)
    
       TOUS utilisent la MÊME structure sous-jacente !
    
    
    THÉORÈME DE DARBOUX (le "miracle")
    ═══════════════════════════════════
    
    Localement, toute structure symplectique (M, ω) est isomorphe à
    la structure canonique (ℝ²ⁿ, dξ ∧ dx).
    
    ➜ Il n'y a qu'UNE SEULE structure symplectique (à isomorphisme près)
    ➜ Contrairement à la géométrie riemannienne (infinité de métriques)
    ➜ C'est pourquoi la mécanique hamiltonienne est si UNIVERSELLE
    
    
    PERSPECTIVE MODERNE
    ═══════════════════
    
    La structure symplectique n'est pas une "découverte" de la physique,
    mais une NÉCESSITÉ MATHÉMATIQUE du formalisme lagrangien :
    
         Lagrangien L(q, q̇, t)
                │
                │ Transformation de Legendre
                │
                ↓
         Hamiltonien H(q, p, t) sur T*Q
                │
                │ Automatiquement
                │
                ↓
         Structure symplectique ω = dp ∧ dq
    
    ➜ Dès qu'on passe au formalisme hamiltonien, ω APPARAÎT
    ➜ Ce n'est pas un choix, c'est une CONSÉQUENCE
    
    
    RÉPONSE À LA QUESTION ORIGINALE
    ════════════════════════════════
    
    "Comment émerge la structure symplectique à partir d'un hamiltonien ?"
    
    RÉPONSE : Elle n'émerge PAS du hamiltonien !
    
    La structure symplectique ω :
    • Existe AVANT le choix de H
    • Est DÉTERMINÉE par la géométrie de T*M
    • Est UNIQUE (théorème de Darboux)
    • Est CANONIQUE (indépendante des coordonnées)
    
    Le hamiltonien H :
    • Est choisi APRÈS pour décrire la physique
    • UTILISE ω pour générer la dynamique
    • Différents H → différentes dynamiques
    • MAIS tous partagent la même ω
    
    
    L'INSIGHT FONDAMENTAL
    ═════════════════════
    
    La structure symplectique est la "GRAMMAIRE" de la dynamique.
    Le hamiltonien est le "CONTENU SÉMANTIQUE".
    
    Tout comme :
    • La grammaire d'une langue existe avant qu'on écrive une phrase
    • La métrique d'un espace existe avant qu'on trace une courbe
    • Les règles d'un jeu existent avant qu'on joue une partie
    
    De même :
    • ω existe avant qu'on choisisse H
    • ω définit les RÈGLES de la dynamique
    • H définit quel système particulier on étudie
    
    
    CITATION FINALE (Arnold)
    ════════════════════════
    
    "La mécanique hamiltonienne n'est pas une théorie physique,
     mais une structure mathématique dans laquelle vivent
     toutes les théories physiques conservatives."
    
    Et cette structure, c'est (T*M, ω).
    
    
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                    FIN DE LA DÉMONSTRATION                        ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    
    print(conclusion)

philosophical_conclusion()

def beyond_cotangent_bundle():
    """
    Exemples de structures symplectiques au-delà de T*M
    """
    
    print("="*70)
    print("AU-DELÀ DU FIBRÉ COTANGENT : AUTRES VARIÉTÉS SYMPLECTIQUES")
    print("="*70)
    
    examples = """
    
    1. SPHÈRE S² (géométrie classique)
    ═══════════════════════════════════
    
    • M = S² (sphère de rayon R)
    • Forme symplectique : ω = (1/R²) sin(θ) dθ ∧ dφ
    • Interprétation : aire sur la sphère
    • Application : moment angulaire en mécanique quantique
    
    Hamiltonien typique : H = J²/2I (corps rigide)
    ➜ Même principe : ω pré-existe, H définit la dynamique
    
    
    2. ESPACE PROJECTIF ℂℙⁿ (géométrie complexe)
    ═════════════════════════════════════════════
    
    • M = ℂℙⁿ (espace projectif complexe)
    • Forme de Fubini-Study : ω_FS
    • Interprétation : géométrie des espaces d'états quantiques
    • Application : théorie de l'information quantique
    
    ➜ La structure symplectique encode la géométrie de l'espace des états
    
    
    3. ORBITES COADJOINTES (théorie de Lie)
    ════════════════════════════════════════
    
    • G = groupe de Lie (ex: SO(3))
    • M = orbite coadjointe dans g* (dual de l'algèbre de Lie)
    • Forme de Kirillov-Kostant-Souriau : ω_KKS
    • Application : systèmes avec symétrie
    
    Exemple : SO(3) → orbites = sphères dans ℝ³
    ➜ Moment angulaire classique
    
    
    4. VARIÉTÉS DE POISSON (généralisation)
    ═══════════════════════════════════════
    
    • Structure plus générale que symplectique
    • Crochet de Poisson peut être dégénéré
    • Feuilletage en variétés symplectiques
    • Application : réduction symplectique, systèmes intégrables
    
    
    5. VARIÉTÉS DE CONTACT (dimension impaire)
    ══════════════════════════════════════════
    
    • Dimension 2n+1 (vs 2n pour symplectique)
    • Forme de contact α avec dα non-dégénérée
    • Application : optique géométrique, thermodynamique
    
    Relation : T*M × ℝ est naturellement une variété de contact
    
    
    PRINCIPE UNIFICATEUR
    ════════════════════
    
    Dans TOUS ces cas :
    
    1. La structure géométrique (ω, ou son analogue) existe AVANT H
    2. H utilise cette structure pour générer la dynamique
    3. Des propriétés de conservation émergent de la géométrie
    
    La structure symplectique sur T*M n'est qu'un CAS PARTICULIER
    (mais le plus important en physique) d'un principe plus général.
    """
    
    print(examples)
    
    # Exemple concret : sphère S²
    print("\n" + "="*70)
    print("EXEMPLE : DYNAMIQUE HAMILTONIENNE SUR LA SPHÈRE S²")
    print("="*70)
    
    from sympy import symbols, sin, cos, diff, simplify
    
    theta, phi = symbols('theta phi', real=True)
    p_theta, p_phi = symbols('p_theta p_phi', real=True)
    
    # Structure symplectique sur T*S² (approximation locale)
    print("\nCoordonnées : (θ, φ) sur S², (p_θ, p_φ) impulsions conjuguées")
    print("\nForme symplectique (coordonnées locales) :")
    print("  ω = dp_θ ∧ dθ + dp_φ ∧ dφ / sin²(θ)")
    print("      └─────────┘   └────────────────┘")
    print("       standard      facteur métrique")
    
    # Hamiltonien d'un corps rigide
    I1, I2, I3 = 1.0, 1.0, 1.5  # Moments d'inertie
    
    H_rigid = p_theta**2/(2*I1) + p_phi**2/(2*I3*sin(theta)**2)
    
    print(f"\nHamiltonien (corps rigide) : H = {H_rigid}")
    
    # Équations de Hamilton modifiées par la métrique
    print("\nÉquations de Hamilton (avec métrique) :")
    
    dH_dtheta = diff(H_rigid, theta)
    dH_dphi = diff(H_rigid, phi)
    dH_dptheta = diff(H_rigid, p_theta)
    dH_dpphi = diff(H_rigid, p_phi)
    
    print(f"  dθ/dt = ∂H/∂p_θ = {dH_dptheta}")
    print(f"  dφ/dt = ∂H/∂p_φ = {dH_dpphi}")
    print(f"  dp_θ/dt = -∂H/∂θ = {simplify(-dH_dtheta)}")
    print(f"  dp_φ/dt = -∂H/∂φ = {simplify(-dH_dphi)}")
    
    print("\n➜ Même structure : ω définit les équations, H les spécialise")

beyond_cotangent_bundle()

def print_bibliography():
    """
    Ressources pour approfondir
    """
    
    biblio = """
    
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                    BIBLIOGRAPHIE COMMENTÉE                        ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    
    LIVRES FONDAMENTAUX
    ═══════════════════
    
    [1] V.I. Arnold - "Mathematical Methods of Classical Mechanics"
        └─ LA référence absolue sur la géométrie symplectique
        └─ Exposition claire de ω comme structure pré-existante
        └─ Niveau : Master/Doctorat
    
    [2] R. Abraham & J.E. Marsden - "Foundations of Mechanics"
        └─ Approche géométrique moderne
        └─ Théorème de Darboux, réduction symplectique
        └─ Niveau : Doctorat
    
    [3] A. Weinstein - "Lectures on Symplectic Manifolds"
        └─ Court mais dense, focus sur la géométrie
        └─ Orbites coadjointes, applications physiques
        └─ Niveau : Master avancé
    
    
    GÉOMÉTRIE SYMPLECTIQUE
    ══════════════════════
    
    [4] D. McDuff & D. Salamon - "Introduction to Symplectic Topology"
        └─ Aspects topologiques (capacités, chirurgie)
        └─ Moins orienté physique, plus mathématique
        └─ Niveau : Doctorat
    
    [5] A. Cannas da Silva - "Lectures on Symplectic Geometry"
        └─ Introduction accessible et complète
        └─ Fibré cotangent, quantification, exemples
        └─ Niveau : Master
        └─ Gratuit en ligne : https://people.math.ethz.ch/~acannas/
    
    
    LIEN AVEC LES Ψ-DO
    ══════════════════
    
    [6] L. Hörmander - "The Analysis of Linear PDO" (Vol. III-IV)
        └─ Opérateurs pseudo-différentiels et symboles
        └─ Lien entre symboles classiques et quantiques
        └─ Niveau : Doctorat avancé
    
    [7] A. Martinez - "An Introduction to Semiclassical Analysis"
        └─ Analyse semi-classique (limite ℏ → 0)
        └─ Quantification de Weyl, corrections WKB
        └─ Niveau : Doctorat
    
    [8] M. Zworski - "Semiclassical Analysis"
        └─ Moderne, complet, bien écrit
        └─ Résonnances, diffusion, spectre asymptotique
        └─ Niveau : Doctorat
        └─ Gratuit : https://math.berkeley.edu/~zworski/
    
    
    APPLICATIONS PHYSIQUES
    ══════════════════════
    
    [9] L.D. Landau & E.M. Lifshitz - "Mechanics" (Vol. 1)
        └─ Approche physique classique
        └─ Principe variationnel, transformation de Legendre
        └─ Niveau : Licence/Master
    
    [10] J.V. José & E.J. Saletan - "Classical Dynamics"
         └─ Pont entre physique et mathématiques
         └─ Nombreux exemples concrets
         └─ Niveau : Master
    
    
    GÉOMÉTRIE ET QUANTIFICATION
    ════════════════════════════
    
    [11] N.M.J. Woodhouse - "Geometric Quantization"
         └─ Comment passer du classique au quantique géométriquement
         └─ Fibrés en droites, connexions, polarisations
         └─ Niveau : Doctorat
    
    [12] B. Kostant - "Quantization and Unitary Representations"
         └─ Approche via orbites coadjointes
         └─ Lien profond entre géométrie et représentations
         └─ Niveau : Doctorat avancé
    
    
    ARTICLES HISTORIQUES
    ════════════════════
    
    [13] P.A.M. Dirac - "The Principles of Quantum Mechanics" (1930)
         └─ Introduction du crochet de Poisson en mécanique quantique
         └─ Principe de correspondance classique-quantique
    
    [14] J.-M. Souriau - "Structure des Systèmes Dynamiques" (1970)
         └─ Formalisation moderne de la mécanique symplectique
         └─ Forme de Kirillov-Kostant-Souriau
    
    
    COURS EN LIGNE (GRATUITS)
    ═════════════════════════
    
    • MIT OCW 18.155 (Differential Analysis)
      https://ocw.mit.edu/
    
    • Stanford - Symplectic Geometry (A. Cannas da Silva)
      Vidéos + notes complètes
    
    • Berkeley - Microlocal Analysis (M. Zworski)
      Notes de cours gratuites
    
    • IHÉS - Séminaires de géométrie symplectique
      Vidéos en ligne
    
    
    LOGICIELS ET CODES
    ══════════════════
    
    • SymPy : calcul symbolique (Python)
      └─ Géométrie différentielle, calcul de Poisson
    
    • SageMath : mathématiques pures
      └─ Variétés, formes différentielles
    
    • Mathematica : notebooks interactifs
      └─ Visualisation de flots hamiltoniens
    
    
    EXERCICES ET PROBLÈMES
    ══════════════════════
    
    [15] F. Scheck - "Mechanics: From Newton's Laws to Deterministic Chaos"
         └─ Nombreux exercices corrigés
         └─ Progression pédagogique
    
    [16] J.R. Taylor - "Classical Mechanics"
         └─ Approche intuitive avec applications
         └─ Bon pour débuter
    
    
    PERSPECTIVES MODERNES
    ═════════════════════
    
    [17] V. Guillemin & S. Sternberg - "Symplectic Techniques in Physics"
         └─ Applications récentes (intégrabilité, chaos)
         └─ Systèmes complètement intégrables
    
    [18] M. Audin - "Torus Actions on Symplectic Manifolds"
         └─ Actions de groupe, réduction
         └─ Polytopes moment
    
    
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                      PARCOURS CONSEILLÉ                           ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    DÉBUTANT (Licence/M1) :
    • [9] Landau & Lifshitz (physique)
    • [16] Taylor (exercices)
    • [5] Cannas da Silva (géométrie)
    
    INTERMÉDIAIRE (M2) :
    • [1] Arnold (fondamental !)
    • [10] José & Saletan
    • [7] Martinez (semi-classique)
    
    AVANCÉ (Doctorat) :
    • [2] Abraham & Marsden
    • [6] Hörmander (Ψ-DO)
    • [8] Zworski (analyse)
    • [11] Woodhouse (quantification)
    
    
    CONSEIL FINAL
    ═════════════
    
    La meilleure façon de comprendre l'émergence de ω est de :
    
    1. CALCULER : implémenter des exemples (comme ce notebook !)
    2. VISUALISER : tracer les flots, les portraits de phase
    3. COMPARER : plusieurs hamiltoniens, même structure
    4. GÉNÉRALISER : passer à S², ℂℙⁿ, orbites coadjointes
    
    La compréhension profonde vient de la PRATIQUE, pas seulement
    de la lecture théorique.
    """
    
    print(biblio)

print_bibliography()

def ultimate_summary():
    """
    Résumé ultra-concis de toute la discussion
    """
    
    summary = """
    
    ╔═══════════════════════════════════════════════════════════════════╗
    ║        RÉSUMÉ ULTIME : ÉMERGENCE DE LA STRUCTURE SYMPLECTIQUE    ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    
    QUESTION
    ════════
    
    "Comment émerge la structure symplectique à partir d'un hamiltonien
     quelconque sur le fibré cotangent ?"
    
    
    RÉPONSE EN 3 POINTS
    ═══════════════════
    
    1️⃣  ELLE N'ÉMERGE PAS DU HAMILTONIEN
       
       • ω existe AVANT tout choix de H
       • ω est une structure GÉOMÉTRIQUE de T*M
       • ω = d(Σᵢ ξᵢ dxⁱ) est CANONIQUE
    
    
    2️⃣  LE HAMILTONIEN UTILISE LA STRUCTURE PRÉ-EXISTANTE
       
       • Équation universelle : ι_{X_H} ω = dH
       • ω transforme TOUT H en champ de vecteurs X_H
       • Différents H → différentes dynamiques, MÊME ω
    
    
    3️⃣  LA STRUCTURE EST UNIVERSELLE
       
       • Théorème de Darboux : une seule structure (localement)
       • Conservation automatique : Φₜᴴ préserve ω
       • Quantification : {·,·} → [·,·] = iℏ{·,·}
    
    
    SÉQUENCE LOGIQUE
    ════════════════
    
         Variété M
            │
            ↓ construction géométrique
         Fibré T*M
            │
            ↓ forme tautologique θ = ξ·dx
         Structure ω = dθ
            │
            ↓ choix physique
         Hamiltonien H
            │
            ↓ équation ι_{X_H}ω = dH
         Champ X_H = (∂H/∂ξ, -∂H/∂x)
            │
            ↓ intégration
         Flot Φₜᴴ (préserve ω et H)
    
    
    FORMULES CLÉS
    ═════════════
    
    • Forme symplectique :    ω = Σᵢ dξᵢ ∧ dxⁱ
    • Champ hamiltonien :     ẋⁱ = ∂H/∂ξᵢ, ξ̇ᵢ = -∂H/∂xⁱ
    • Crochet de Poisson :    {f,g} = Σᵢ(∂f/∂xⁱ·∂g/∂ξᵢ - ∂f/∂ξᵢ·∂g/∂xⁱ)
    • Conservation :          dH/dt = {H, H} = 0
    • Quantification :        [f̂,ĝ] = iℏ{f,g}
    
    
    INSIGHT PROFOND
    ═══════════════
    
    ω n'est pas une "loi physique" mais une NÉCESSITÉ GÉOMÉTRIQUE.
    
    Dès qu'on considère T*M, ω apparaît automatiquement via θ = ξ·dx.
    C'est la structure qui rend POSSIBLE la dynamique hamiltonienne.
    
    Sans ω → pas d'équations de Hamilton
    Sans ω → pas de crochet de Poisson
    Sans ω → pas de conservation de la mesure de Liouville
    Sans ω → pas de quantification cohérente
    
    ω est la "GRAMMAIRE" de toute dynamique conservative.
    
    
    CODE MINIMAL
    ════════════
    
    from sympy import symbols, diff
    
    # Structure symplectique (AVANT H)
    x, xi = symbols('x xi', real=True)
    omega = [[0, -1], [1, 0]]  # Matrice symplectique
    
    # Hamiltonien quelconque (APRÈS ω)
    H = xi**2/2 + V(x)  # Forme générique
    
    # Équations de Hamilton (DÉTERMINÉES par ω)
    dx_dt = diff(H, xi)   # = ∂H/∂ξ
    dxi_dt = -diff(H, x)  # = -∂H/∂x
    
    # Conservation (CONSÉQUENCE de ω)
    dH_dt = dx_dt * diff(H, x) + dxi_dt * diff(H, xi)  # = 0
    
    
    LE MOT DE LA FIN
    ════════════════
    
    La structure symplectique n'émerge pas du hamiltonien.
    
    Elle EST le théâtre où se joue la pièce hamiltonienne.
    
    H choisit l'intrigue, mais ω écrit les règles de la dramaturgie.
    
    
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                           FIN                                     ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    
    print(summary)

ultimate_summary()
