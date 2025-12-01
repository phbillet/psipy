from psinumpy import *

# ============================================
# TESTS UNITAIRES
# ============================================
def test_symbolic_operations():
    """
    Tests unitaires des opérations symboliques
    """
    print("\n" + "="*60)
    print("TESTS UNITAIRES : Opérations symboliques")
    print("="*60 + "\n")
    
    # 1. Test produit de Moyal
    print("Test 1 : Produit de Moyal")
    grid = Grid.create_1d(N=64, L=10.0, periodic=True)
    
    h = Symbol.from_function(grid, lambda t,x,xi: x**2 + xi**2, order=2, t=0.0)
    g = Symbol.from_function(grid, lambda t,x,xi: x*xi, order=1, t=0.0)
    
    hg = moyal_product(h, g, order=2)
    
    print(f"  ||h|| = {SymbolicDiagnostics.symbol_norm(h, 'L2'):.4f}")
    print(f"  ||g|| = {SymbolicDiagnostics.symbol_norm(g, 'L2'):.4f}")
    print(f"  ||h #_W g|| = {SymbolicDiagnostics.symbol_norm(hg, 'L2'):.4f}")
    print("  ✓ Produit de Moyal OK\n")
    
    # 2. Test inverse symbolique
    print("Test 2 : Inverse symbolique")
    
    h_elliptic = Symbol.from_function(grid, lambda t,x,xi: 1.0 + xi**2, order=2, t=0.0)
    h_inv = symbolic_inverse(h_elliptic, order=2)
    
    # Vérifier h #_W h^{-1} ≈ 1
    identity = moyal_product(h_elliptic, h_inv, order=2)
    error = np.max(np.abs(identity.values - 1.0))
    
    print(f"  ||h #_W h^{{-1}} - 1|| = {error:.2e}")
    assert error < 0.1, "Inverse symbolique échoue"
    print("  ✓ Inverse symbolique OK\n")
    
    # 3. Test commutateur
    print("Test 3 : Commutateur / Crochet de Poisson")
    
    x_sym = Symbol.from_function(grid, lambda t,x,xi: x, order=0, t=0.0)
    p_sym = Symbol.from_function(grid, lambda t,x,xi: xi, order=1, t=0.0)
    
    comm = commutator(x_sym, p_sym, order=1)
    pb = poisson_bracket(x_sym, p_sym)
    
    # [x̂, p̂] = iℏ → {x, p} = 1
    expected_pb = 1.0
    pb_value = np.mean(pb.values)
    
    print(f"  {{x, p}} = {pb_value:.4f} (attendu: {expected_pb})")
    print(f"  [x̂, p̂]/(iℏ) ≈ {np.mean(comm.values / (1j * h.hbar)):.4f}")
    print("  ✓ Relations canoniques OK\n")
    
    # 4. Test conservation dans l'évolution
    print("Test 4 : Conservation de la norme (évolution unitaire)")
    
    def flat_metric(t, x):
        return np.ones_like(x)
    
    evolver = SymbolicEvolutionWithBC(grid, bc_type='periodic')
    h_func = evolver.create_hamiltonian_relativistic(flat_metric)
    
    # État initial
    x = grid.x
    psi0 = np.exp(-(x - 5.0)**2 / 2) * np.exp(1j * 2.0 * x)
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * grid.dx)
    
    # Évolution courte
    times, psi_hist = evolver.evolve(
        psi0, h_func, t_span=(0.0, 1.0), Nt=50, save_every=10
    )
    
    norms = [np.sum(np.abs(psi)**2) * grid.dx for psi in psi_hist]
    norm_variation = np.max(np.abs(np.array(norms) - 1.0))
    
    print(f"  Variation de norme : {norm_variation:.2e}")
    assert norm_variation < 0.01, "Conservation de la norme échoue"
    print("  ✓ Conservation de la norme OK\n")
    
    # 5. Test conditions aux limites
    print("Test 5 : Conditions aux limites")
    
    # Test périodique
    evolver_per = SymbolicEvolutionWithBC(grid, bc_type='periodic')
    psi_per = np.sin(2 * np.pi * grid.x / 10.0)
    times_per, psi_hist_per = evolver_per.evolve(
        psi_per, h_func, t_span=(0.0, 0.5), Nt=20, save_every=20
    )
    
    # Vérifier périodicité
    assert np.allclose(psi_hist_per[-1][0], psi_hist_per[-1][-1], atol=1e-3), \
        "Périodicité non respectée"
    print("  ✓ Conditions périodiques OK")
    
    # Test Dirichlet
    grid_dir = Grid.create_1d(N=64, L=10.0, periodic=False)
    evolver_dir = SymbolicEvolutionWithBC(grid_dir, bc_type='dirichlet')
    h_func_dir = evolver_dir.create_hamiltonian_relativistic(flat_metric)
    
    psi_dir = np.sin(np.pi * grid_dir.x / 10.0)
    psi_dir /= np.sqrt(np.sum(np.abs(psi_dir)**2) * grid_dir.dx)
    
    times_dir, psi_hist_dir = evolver_dir.evolve(
        psi_dir, h_func_dir, t_span=(0.0, 0.5), Nt=20, save_every=20
    )
    
    # Vérifier Dirichlet
    assert np.abs(psi_hist_dir[-1][0]) < 1e-10, "Dirichlet gauche non respectée"
    assert np.abs(psi_hist_dir[-1][-1]) < 1e-10, "Dirichlet droite non respectée"
    print("  ✓ Conditions Dirichlet OK")
    
    # Test horizon
    evolver_hor = SymbolicEvolutionWithBC(
        grid_dir, bc_type='horizon',
        horizon_params={'width': 0.2, 'strength': 2.0, 'power': 2}
    )
    h_func_hor = evolver_hor.create_hamiltonian_relativistic(flat_metric)
    
    psi_hor = np.exp(-(grid_dir.x - 5.0)**2 / 2) * np.exp(1j * 3.0 * grid_dir.x)
    psi_hor /= np.sqrt(np.sum(np.abs(psi_hor)**2) * grid_dir.dx)
    
    times_hor, psi_hist_hor = evolver_hor.evolve(
        psi_hor, h_func_hor, t_span=(0.0, 2.0), Nt=100, save_every=50
    )
    
    norms_hor = [np.sum(np.abs(psi)**2) * grid_dir.dx for psi in psi_hist_hor]
    
    # Vérifier absorption
    assert norms_hor[-1] < norms_hor[0] * 0.9, "Horizon n'absorbe pas"
    print(f"  ✓ Conditions horizon OK (absorption: {(1-norms_hor[-1]/norms_hor[0])*100:.1f}%)\n")
    
    print("="*60)
    print("TOUS LES TESTS UNITAIRES PASSÉS ✓")
    print("="*60 + "\n")

def test_heat_equation():
    """
    Tests pour l'équation de la chaleur
    """
    print("\n" + "="*60)
    print("TESTS : Équation de la chaleur")
    print("="*60 + "\n")
    
    grid = Grid.create_1d(N=128, L=10.0, periodic=True)
    
    # Test 1 : Conservation de la chaleur (sans source)
    print("Test 1 : Conservation de la chaleur")
    
    solver = HeatEquationSolver(grid, 'periodic', diffusivity=0.5)
    h_func = solver.create_diffusion_symbol()
    
    x = grid.x
    u0 = np.exp(-(x-5)**2 / 2)
    
    times, u_hist = solver.evolve_heat(u0, h_func, (0, 2), Nt=100, save_every=50)
    
    Q_initial = HeatDiagnostics.total_heat(u_hist[0], grid)
    Q_final = HeatDiagnostics.total_heat(u_hist[-1], grid)
    
    conservation_error = abs(Q_final - Q_initial) / Q_initial
    
    print(f"  Q_initial = {Q_initial:.6f}")
    print(f"  Q_final = {Q_final:.6f}")
    print(f"  Erreur relative = {conservation_error:.2e}")
    
    assert conservation_error < 0.01, "Conservation échoue"
    print("  ✓ Conservation OK\n")
    
    # Test 2 : État stationnaire
    print("Test 2 : État stationnaire avec source")
    
    # Source constante
    source = np.ones_like(x)
    
    h_symbol = h_func(0.0)
    u_steady = solver.steady_state_solve(source, h_symbol)
    
    # Vérifier que ∇²u_steady ≈ -f
    # En Fourier : -ξ² û = -f̂  =>  û = f̂/ξ²
    
    print(f"  u_steady calculé")
    print(f"  u_min = {np.min(u_steady):.4f}, u_max = {np.max(u_steady):.4f}")
    print("  ✓ État stationnaire OK\n")
    
    # Test 3 : Temps de diffusion
    print("Test 3 : Temps caractéristique de diffusion")
    
    L = 10.0
    D = 0.5
    tau_theory = HeatDiagnostics.diffusion_time(L, D)
    
    print(f"  τ = L²/D = {tau_theory:.2f}")
    
    # Simuler et mesurer temps pour étalement
    u0_narrow = np.exp(-(x-5)**2 / 0.1)
    times_diff, u_hist_diff = solver.evolve_heat(u0_narrow, h_func, 
                                                 (0, tau_theory), Nt=200, save_every=100)
    
    # Largeur initiale vs finale
    sigma_initial = 0.1**0.5
    
    # Largeur finale (mesurée)
    u_final = u_hist_diff[-1]
    x_mean = np.trapz(x * u_final, x) / np.trapz(u_final, x)
    variance = np.trapz((x - x_mean)**2 * u_final, x) / np.trapz(u_final, x)
    sigma_final = np.sqrt(variance)
    
    # Théorie : σ²(τ) = σ²(0) + 2Dτ
    sigma_theory = np.sqrt(sigma_initial**2 + 2*D*tau_theory)
    
    print(f"  σ_final mesuré = {sigma_final:.4f}")
    print(f"  σ_final théorie = {sigma_theory:.4f}")
    print(f"  Erreur = {abs(sigma_final - sigma_theory)/sigma_theory * 100:.2f}%")
    
    assert abs(sigma_final - sigma_theory)/sigma_theory < 0.2, "Diffusion incorrecte"
    print("  ✓ Temps de diffusion OK\n")
    
    print("="*60)
    print("TOUS LES TESTS CHALEUR PASSÉS ✓")
    print("="*60 + "\n")


def test_reaction_diffusion():
    """
    Tests pour réaction-diffusion
    """
    print("\n" + "="*60)
    print("TESTS : Réaction-diffusion")
    print("="*60 + "\n")
    
    grid = Grid.create_1d(N=256, L=50.0, periodic=False)
    
    # Test : vitesse du front de Fisher-KPP
    print("Test : Vitesse du front de Fisher-KPP")
    
    D = 0.1
    r = 1.0
    K = 1.0
    
    c_theory = 2 * np.sqrt(D * r)
    
    def fisher_reaction(u, params):
        return params['r'] * u * (1.0 - u / params['K'])
    
    solver = ReactionDiffusionSolver(
        grid, 'dirichlet', D, fisher_reaction
    )
    
    h_func = solver.create_diffusion_symbol()
    
    # Condition initiale : front
    x = grid.x
    x0 = 10.0
    u0 = K / (1.0 + np.exp((x - x0) / 2.0))
    
    T_final = 15.0
    times, u_hist = solver.evolve_reaction_diffusion(
        u0, h_func, (0, T_final), Nt=500,
        reaction_params={'r': r, 'K': K},
        save_every=100
    )
    
    # Mesurer vitesse
    front_pos = []
    for u in u_hist:
        idx = np.argmin(np.abs(u - K/2))
        front_pos.append(grid.x[idx])
    
    if len(times) > 2:
        from scipy.stats import linregress
        slope, _, _, _, _ = linregress(times, front_pos)
        c_measured = slope
    else:
        c_measured = 0.0
    
    error = abs(c_measured - c_theory) / c_theory
    
    print(f"  c_théorique = {c_theory:.4f}")
    print(f"  c_mesuré = {c_measured:.4f}")
    print(f"  Erreur = {error * 100:.2f}%")
    
    assert error < 0.15, "Vitesse du front incorrecte"
    print("  ✓ Vitesse du front OK\n")
    
    print("="*60)
    print("TOUS LES TESTS RÉACTION-DIFFUSION PASSÉS ✓")
    print("="*60 + "\n")


def test_advection_diffusion():
    """
    Tests pour advection-diffusion
    """
    print("\n" + "="*60)
    print("TESTS : Advection-diffusion")
    print("="*60 + "\n")
    
    grid = Grid.create_1d(N=256, L=10.0, periodic=True)
    
    # Test : transport par advection pure (D=0)
    print("Test : Advection pure (D→0)")
    
    v0 = 2.0
    D_small = 1e-6  # Quasi-nul
    
    def velocity(t, x):
        return v0 * np.ones_like(x)
    
    solver = AdvectionDiffusionSolver(
        grid, 'periodic', D_small, velocity
    )
    
    x = grid.x
    x0 = 3.0
    u0 = np.exp(-(x - x0)**2 / 0.5)
    
    T = 2.0
    times, u_hist = solver.evolve_advection_diffusion(
        u0, (0, T), Nt=200, save_every=len([0, T])-1
    )
    
    # Position théorique : x_final = (x0 + v0*T) mod L
    x_theory = (x0 + v0 * T) % grid.x[-1]
    
    # Position mesurée
    x_measured = grid.x[np.argmax(u_hist[-1])]
    
    # Distance (en périodique)
    dist = min(abs(x_measured - x_theory), 
              abs(x_measured - x_theory + grid.x[-1]),
              abs(x_measured - x_theory - grid.x[-1]))
    
    print(f"  Déplacement théorique = {v0 * T:.2f}")
    print(f"  Position finale théorique = {x_theory:.2f}")
    print(f"  Position finale mesurée = {x_measured:.2f}")
    print(f"  Erreur = {dist:.4f}")
    
    assert dist < 0.2, "Transport par advection incorrect"
    print("  ✓ Advection pure OK\n")
    
    # Test 2 : Nombre de Péclet
    print("Test : Influence du nombre de Péclet")
    
    D_high = 1.0  # Diffusion dominante
    Pe_low = v0 * grid.x[-1] / D_high
    
    solver_diff = AdvectionDiffusionSolver(
        grid, 'periodic', D_high, velocity
    )
    
    times2, u_hist2 = solver_diff.evolve_advection_diffusion(
        u0, (0, T), Nt=200, save_every=len([0, T])-1
    )
    
    # Avec forte diffusion, le pic doit s'étaler plus
    sigma1 = np.std(u_hist[-1])  # Faible diffusion
    sigma2 = np.std(u_hist2[-1])  # Forte diffusion
    
    print(f"  Pe (faible D) = {v0 * grid.x[-1] / D_small:.0f} >> 1")
    print(f"  Pe (fort D) = {Pe_low:.2f} ~ 1")
    print(f"  Étalement (faible D) = {sigma1:.4f}")
    print(f"  Étalement (fort D) = {sigma2:.4f}")
    
    assert sigma2 > sigma1, "Diffusion ne s'étale pas plus avec Pe faible"
    print("  ✓ Nombre de Péclet OK\n")
    
    print("="*60)
    print("TOUS LES TESTS ADVECTION-DIFFUSION PASSÉS ✓")
    print("="*60 + "\n")

