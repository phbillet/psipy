import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, fftfreq
from scipy.ndimage import convolve
from dataclasses import dataclass
from typing import Callable, Tuple

@dataclass
class Grid:
    """Grille de calcul pour (x,ξ)"""
    x: np.ndarray  # Points spatiaux
    xi: np.ndarray  # Points de moment (espace de Fourier)
    dx: float
    dxi: float
    dim: int  # 1 ou 2
    
    @classmethod
    def create_1d(cls, N, L, periodic=True):
        """Crée une grille 1D"""
        x = np.linspace(0, L, N, endpoint=not periodic)
        dx = L / N
        
        if periodic:
            xi = 2 * np.pi * fftfreq(N, dx)
        else:
            # Pour Dirichlet, modes sinus
            xi = np.pi * np.arange(1, N+1) / L
        
        dxi = xi[1] - xi[0] if len(xi) > 1 else 1.0
        return cls(x, xi, dx, dxi, dim=1)
    
    @classmethod
    def create_2d(cls, Nx, Ny, Lx, Ly, periodic=True):
        """Crée une grille 2D"""
        x = np.linspace(0, Lx, Nx, endpoint=not periodic)
        y = np.linspace(0, Ly, Ny, endpoint=not periodic)
        dx = Lx / Nx
        dy = Ly / Ny
        
        if periodic:
            xi_x = 2 * np.pi * fftfreq(Nx, dx)
            xi_y = 2 * np.pi * fftfreq(Ny, dy)
        else:
            xi_x = np.pi * np.arange(1, Nx+1) / Lx
            xi_y = np.pi * np.arange(1, Ny+1) / Ly
        
        # Grille complète
        X, Y = np.meshgrid(x, y, indexing='ij')
        XI_X, XI_Y = np.meshgrid(xi_x, xi_y, indexing='ij')
        
        return cls(
            x=(X, Y), 
            xi=(XI_X, XI_Y),
            dx=dx,
            dxi=(xi_x[1]-xi_x[0], xi_y[1]-xi_y[0]),
            dim=2
        )


class Symbol:
    """
    Représente un symbole h(t, x, ξ) d'un opérateur pseudodifférentiel
    
    Stockage : valeurs sur grille (x, ξ) pour efficacité
    """
    
    def __init__(self, grid: Grid, order: int = 0):
        self.grid = grid
        self.order = order  # Ordre du symbole (m dans S^m)
        self.hbar = 1.0
        
        # Valeurs du symbole sur la grille phase-space
        if grid.dim == 1:
            self.values = np.zeros((len(grid.x), len(grid.xi)), dtype=complex)
        else:
            Nx, Ny = grid.x[0].shape
            N_xi_x, N_xi_y = grid.xi[0].shape
            self.values = np.zeros((Nx, Ny, N_xi_x, N_xi_y), dtype=complex)
    
    @classmethod
    def from_function(cls, grid: Grid, func: Callable, order: int = 0, t: float = 0.0):
        """
        Crée un symbole à partir d'une fonction h(t, x, ξ)
        """
        symbol = cls(grid, order)
        
        if grid.dim == 1:
            X, XI = np.meshgrid(grid.x, grid.xi, indexing='ij')
            symbol.values = func(t, X, XI)
        else:
            X, Y = grid.x
            XI_X, XI_Y = grid.xi
            # Créer une grille 4D : (x, y, ξ_x, ξ_y)
            shape = X.shape + XI_X.shape
            x_broadcast = X[:, :, np.newaxis, np.newaxis]
            y_broadcast = Y[:, :, np.newaxis, np.newaxis]
            xi_x_broadcast = XI_X[np.newaxis, np.newaxis, :, :]
            xi_y_broadcast = XI_Y[np.newaxis, np.newaxis, :, :]
            
            symbol.values = func(t, x_broadcast, y_broadcast, 
                               xi_x_broadcast, xi_y_broadcast)
        
        return symbol
    
    def update(self, func: Callable, t: float):
        """Met à jour les valeurs pour un nouveau temps t"""
        if self.grid.dim == 1:
            X, XI = np.meshgrid(self.grid.x, self.grid.xi, indexing='ij')
            self.values = func(t, X, XI)
        else:
            X, Y = self.grid.x
            XI_X, XI_Y = self.grid.xi
            shape = X.shape + XI_X.shape
            x_b = X[:, :, np.newaxis, np.newaxis]
            y_b = Y[:, :, np.newaxis, np.newaxis]
            xi_x_b = XI_X[np.newaxis, np.newaxis, :, :]
            xi_y_b = XI_Y[np.newaxis, np.newaxis, :, :]
            
            self.values = func(t, x_b, y_b, xi_x_b, xi_y_b)
    
    def gradient_x(self):
        """Calcule ∂h/∂x"""
        if self.grid.dim == 1:
            return np.gradient(self.values, self.grid.dx, axis=0)
        else:
            grad_x = np.gradient(self.values, self.grid.dx, axis=0)
            grad_y = np.gradient(self.values, self.grid.dx, axis=1)
            return grad_x, grad_y
    
    def gradient_xi(self):
        """Calcule ∂h/∂ξ"""
        if self.grid.dim == 1:
            return np.gradient(self.values, self.grid.dxi, axis=1)
        else:
            axis_xi_x = 2
            axis_xi_y = 3
            grad_xi_x = np.gradient(self.values, self.grid.dxi[0], axis=axis_xi_x)
            grad_xi_y = np.gradient(self.values, self.grid.dxi[1], axis=axis_xi_y)
            return grad_xi_x, grad_xi_y

def moyal_product(h: Symbol, g: Symbol, order: int = 2) -> Symbol:
    """
    Calcule le produit de Moyal h #_W g
    
    Développement asymptotique :
    (h #_W g)(x,ξ) = exp[(ℏ/2i)(∂_ξ^h ∂_x^g - ∂_x^h ∂_ξ^g)] [h(x,ξ)g(x,ξ)]
    
    ≈ h·g + (ℏ/2i){h,g} + O(ℏ²)
    
    Parameters:
    -----------
    h, g : Symbol
        Les deux symboles à composer
    order : int
        Ordre du développement (0: produit simple, 1: +Poisson, 2: corrections)
    
    Returns:
    --------
    result : Symbol
        Symbole de Ĥ ∘ Ĝ
    """
    assert h.grid == g.grid, "Les symboles doivent avoir la même grille"
    
    result = Symbol(h.grid, order=h.order + g.order)
    hbar = h.hbar
    
    # Ordre 0 : produit simple
    result.values = h.values * g.values
    
    if order >= 1:
        # Ordre 1 : crochet de Poisson
        if h.grid.dim == 1:
            dh_dx = h.gradient_x()
            dh_dxi = h.gradient_xi()
            dg_dx = g.gradient_x()
            dg_dxi = g.gradient_xi()
            
            poisson = dh_dxi * dg_dx - dh_dx * dg_dxi
            result.values = result.values.astype(np.complex128)
            result.values += (hbar / (2j)) * poisson
        
        else:  # dim == 2
            dh_dx, dh_dy = h.gradient_x()
            dg_dx, dg_dy = g.gradient_x()
            dh_dxi_x, dh_dxi_y = h.gradient_xi()
            dg_dxi_x, dg_dxi_y = g.gradient_xi()
            
            poisson = (dh_dxi_x * dg_dx + dh_dxi_y * dg_dy - 
                      dh_dx * dg_dxi_x - dh_dy * dg_dxi_y)
            result.values = result.values.astype(np.complex128)
            result.values += (hbar / (2j)) * poisson
    
    if order >= 2:
        # Ordre 2 : corrections quantiques
        if h.grid.dim == 1:
            # ∂²h/∂ξ² · ∂²g/∂x²
            d2h_dxi2 = np.gradient(dh_dxi, h.grid.dxi, axis=1)
            d2g_dx2 = np.gradient(dg_dx, g.grid.dx, axis=0)
            
            # ∂²h/∂x² · ∂²g/∂ξ²
            d2h_dx2 = np.gradient(dh_dx, h.grid.dx, axis=0)
            d2g_dxi2 = np.gradient(dg_dxi, g.grid.dxi, axis=1)
            
            # Termes croisés
            d2h_dxdxi = np.gradient(dh_dx, h.grid.dxi, axis=1)
            d2g_dxdxi = np.gradient(dg_dx, g.grid.dxi, axis=1)
            
            correction2 = -(hbar**2 / 8) * (
                d2h_dxi2 * d2g_dx2 + d2h_dx2 * d2g_dxi2 
                - 2 * d2h_dxdxi * d2g_dxdxi
            )
            result.values += correction2
        
        else:
            # 2D : plus complexe, formules similaires mais avec plus de termes
            # Pour simplicité, on peut s'arrêter à l'ordre 1 en 2D
            pass
    
    return result


def poisson_bracket(h: Symbol, g: Symbol) -> Symbol:
    """
    Calcule le crochet de Poisson {h, g} = ∂_ξh·∂_xg - ∂_xh·∂_ξg
    
    C'est la limite classique (ℏ→0) du commutateur [Ĥ,Ĝ]/(iℏ)
    """
    result = Symbol(h.grid, order=h.order + g.order - 1)
    
    if h.grid.dim == 1:
        dh_dx = h.gradient_x()
        dh_dxi = h.gradient_xi()
        dg_dx = g.gradient_x()
        dg_dxi = g.gradient_xi()
        
        result.values = dh_dxi * dg_dx - dh_dx * dg_dxi
    
    else:
        dh_dx, dh_dy = h.gradient_x()
        dg_dx, dg_dy = g.gradient_x()
        dh_dxi_x, dh_dxi_y = h.gradient_xi()
        dg_dxi_x, dg_dxi_y = g.gradient_xi()
        
        result.values = (dh_dxi_x * dg_dx + dh_dxi_y * dg_dy - 
                        dh_dx * dg_dxi_x - dh_dy * dg_dxi_y)
    
    return result


def commutator(h: Symbol, g: Symbol, order: int = 1) -> Symbol:
    """
    Calcule le symbole du commutateur [Ĥ, Ĝ]
    
    [Ĥ, Ĝ] = Ĥ∘Ĝ - Ĝ∘Ĥ
    
    Au premier ordre : σ([Ĥ,Ĝ]) = iℏ{h,g}
    """
    if order == 1:
        # Formule rapide au premier ordre
        pb = poisson_bracket(h, g)
        result = Symbol(h.grid, order=h.order + g.order - 1)
        result.values = 1j * h.hbar * pb.values
        return result
    else:
        # Calcul exact via compositions
        hg = moyal_product(h, g, order=order)
        gh = moyal_product(g, h, order=order)
        result = Symbol(h.grid, order=h.order + g.order)
        result.values = hg.values - gh.values
        return result

def symbolic_inverse(h: Symbol, order: int = 2, regularization: float = 1e-12) -> Symbol:
    """
    Calcule le symbole de Ĥ⁻¹ par développement asymptotique
    
    Pour un opérateur elliptique de symbole h(x,ξ) d'ordre m,
    le symbole de l'inverse est b(x,ξ) d'ordre -m avec :
    
    b₀ = 1/h
    bⱼ₊₁ = -b₀ #_W (h #_W bⱼ - δⱼ₀)
    
    Parameters:
    -----------
    h : Symbol
        Symbole à inverser (doit être elliptique)
    order : int
        Ordre du développement (nombre de termes de correction)
    regularization : float
        Régularisation pour éviter division par zéro
    
    Returns:
    --------
    b : Symbol
        Symbole de Ĥ⁻¹
    """
    # Vérifier l'ellipticité (au moins approximativement)
    min_abs = np.min(np.abs(h.values))
    if min_abs < regularization * 10:
        print(f"Avertissement : symbole proche de zéro (min = {min_abs}), "
              f"l'inverse peut être imprécis")
    
    # Symbole principal : b₀ = 1/h (avec régularisation)
    b0 = Symbol(h.grid, order=-h.order)
    h_reg = h.values + 1j * regularization  # Régularisation complexe
    b0.values = 1.0 / h_reg
    
    if order == 0:
        return b0
    
    # Termes de correction
    b_terms = [b0]
    
    for j in range(order):
        # Calculer h #_W bⱼ
        h_bj = moyal_product(h, b_terms[j], order=2)
        
        # Soustraire δⱼ₀ (= 1 si j=0, 0 sinon)
        if j == 0:
            h_bj.values -= 1.0
        
        # bⱼ₊₁ = -b₀ #_W (h #_W bⱼ - δⱼ₀)
        b_next = moyal_product(b0, h_bj, order=2)
        b_next.values *= -1.0
        
        b_terms.append(b_next)
    
    # Somme de tous les termes
    result = Symbol(h.grid, order=-h.order)
    result.values = sum(b.values for b in b_terms)
    
    return result


def symbolic_sqrt(h: Symbol, regularization: float = 1e-12) -> Symbol:
    """
    Calcule √h pour un symbole positif (utile pour hamiltoniens relativistes)
    
    Pour h = γⁱʲξᵢξⱼ, donne le symbole de √Ĥ
    """
    result = Symbol(h.grid, order=h.order/2)
    
    # Régularisation pour éviter √ de nombres négatifs ou très petits
    h_reg = h.values + regularization
    h_reg = np.where(np.real(h_reg) > 0, h_reg, regularization)
    
    result.values = np.sqrt(h_reg)
    
    return result

class WeylQuantization:
    """
    Transforme un symbole h(x,ξ) en opérateur Ĥ agissant sur ψ(x)
    
    Formule de quantification de Weyl :
    (Ĥψ)(x) = (2π)⁻ⁿ ∫∫ e^{i(x-y)·ξ} h((x+y)/2, ξ) ψ(y) dy dξ
    """
    def __init__(self, grid: Grid, boundary: str = 'periodic'):
        """
        Parameters:
        -----------
        grid : Grid
            Grille de calcul
        boundary : str
            Type de conditions aux limites : 'periodic', 'dirichlet', 'horizon'
        """
        self.grid = grid
        self.boundary = boundary
        
        # Pré-calculer les opérateurs FFT selon les conditions
        if boundary == 'periodic':
            self.fft_forward = fft if grid.dim == 1 else fft2
            self.fft_backward = ifft if grid.dim == 1 else ifft2

        elif boundary == 'horizon':
            # Pour 'horizon' on peut réutiliser la FFT périodique
            # L'absorption est gérée via le potentiel -i*Gamma dans le hamiltonien.
            # Si tu préfères un traitement particulier, remplacer ici.
            self.fft_forward = fft if grid.dim == 1 else fft2
            self.fft_backward = ifft if grid.dim == 1 else ifft2

        elif boundary == 'dirichlet':
            # Transformée en sinus discrète (DST)
            from scipy.fftpack import dst, idst
            if grid.dim == 1:
                self.fft_forward = lambda x: dst(x, type=1)
                self.fft_backward = lambda x: idst(x, type=1) / (2*(len(x)+1))
            else:
                # DST 2D
                def dst2d(x):
                    return dst(dst(x, type=1, axis=0), type=1, axis=1)
                def idst2d(x):
                    N, M = x.shape
                    return idst(idst(x, type=1, axis=0), type=1, axis=1) / (4*(N+1)*(M+1))
                self.fft_forward = dst2d
                self.fft_backward = idst2d
        else:
            # Fallback : utiliser la FFT (sécurisé)
            self.fft_forward = fft if grid.dim == 1 else fft2
            self.fft_backward = ifft if grid.dim == 1 else ifft2

    
    def apply_symbol_multiplicative(self, symbol: Symbol, psi: np.ndarray, 
                                   space: str = 'position') -> np.ndarray:
        """
        Applique un symbole multiplicatif (diagonal en x ou en ξ)
        
        Parameters:
        -----------
        symbol : Symbol
            Doit être de la forme h(x) (ordre 0 en ξ) ou h(ξ) (ordre 0 en x)
        psi : np.ndarray
            Fonction d'onde
        space : str
            'position' si h = h(x), 'momentum' si h = h(ξ)
        
        Returns:
        --------
        result : np.ndarray
            Résultat de l'application
        """
        if space == 'position':
            # Opérateur multiplicatif en espace réel : (Ĥψ)(x) = h(x)ψ(x)
            if self.grid.dim == 1:
                h_x = symbol.values[:, 0]  # Prendre la première colonne (indépendant de ξ)
            else:
                h_x = symbol.values[:, :, 0, 0]
            
            return h_x * psi
        
        elif space == 'momentum':
            # Opérateur multiplicatif en espace de Fourier
            # 1. FFT de ψ
            psi_k = self.fft_forward(psi)
            
            # 2. Multiplication par h(ξ)
            if self.grid.dim == 1:
                h_xi = symbol.values[0, :]  # Indépendant de x
            else:
                h_xi = symbol.values[0, 0, :, :]
            
            psi_k_result = h_xi * psi_k
            
            # 3. IFFT
            return self.fft_backward(psi_k_result)
        
        else:
            raise ValueError(f"space doit être 'position' ou 'momentum', pas '{space}'")
    
    def apply_symbol_general(self, symbol: Symbol, psi: np.ndarray, 
                            method: str = 'phase_space') -> np.ndarray:
        """
        Applique un symbole général h(x,ξ) via quantification de Weyl
        
        Plusieurs méthodes possibles :
        - 'phase_space' : transformée de Wigner explicite (coûteux mais exact)
        - 'splitting' : décomposition T+V si possible
        - 'pseudospectral' : approximation pseudo-spectrale
        
        Parameters:
        -----------
        symbol : Symbol
            Symbole général h(x,ξ)
        psi : np.ndarray
            Fonction d'onde
        method : str
            Méthode de quantification
        
        Returns:
        --------
        result : np.ndarray
            (Ĥψ)(x)
        """
        if method == 'splitting':
            # Applicable si h(x,ξ) = T(ξ) + V(x)
            # Non implémenté ici, voir section suivante
            raise NotImplementedError("Utiliser apply_hamiltonian_split")
        
        elif method == 'pseudospectral':
            # Approximation : alterner multiplications en x et en ξ
            # Convient pour opérateurs "presque" séparables
            
            result = psi.copy()
            
            # Appliquer terme par terme si décomposable
            # Sinon, approximation itérative
            
            # Pour l'instant, méthode simple : projection sur grille
            if self.grid.dim == 1:
                # Intégrer sur ξ avec poids
                h_avg_xi = np.mean(symbol.values, axis=1)
                result = h_avg_xi * result
                
                # Puis partie dépendant de ξ
                psi_k = self.fft_forward(result)
                h_avg_x = np.mean(symbol.values, axis=0)
                psi_k = h_avg_x * psi_k
                result = self.fft_backward(psi_k)
            
            return result
        
        elif method == 'phase_space':
            # Transformation de Wigner complète (très coûteux)
            return self._apply_via_wigner(symbol, psi)
        
        else:
            raise ValueError(f"Méthode '{method}' non reconnue")
    
    def _apply_via_wigner(self, symbol: Symbol, psi: np.ndarray) -> np.ndarray:
        """
        Application via transformée de Wigner (coûteux, O(N³) en 1D)
        
        W_ψ(x,ξ) = ∫ ψ*(x-y/2) ψ(x+y/2) e^{-iyξ} dy
        
        Ĥψ ↔ h(x,ξ) * W_ψ(x,ξ) (produit en phase space)
        """
        N = len(self.grid.x)
        
        # Transformée de Wigner de ψ
        W_psi = np.zeros((N, N), dtype=complex)
        
        for i, x in enumerate(self.grid.x):
            for k, xi in enumerate(self.grid.xi):
                # Intégrale en y
                integrand = np.zeros(N, dtype=complex)
                for j, y in enumerate(self.grid.x):
                    # Indices pour x±y/2 avec périodicité
                    idx_plus = int((i + j/2) % N)
                    idx_minus = int((i - j/2) % N)
                    
                    integrand[j] = (np.conj(psi[idx_minus]) * psi[idx_plus] * 
                                   np.exp(-1j * y * xi))
                
                W_psi[i, k] = np.trapz(integrand, self.grid.x)
        
        # Multiplication par le symbole
        result_W = symbol.values * W_psi
        
        # Transformée de Wigner inverse (projection sur ψ)
        result = np.zeros(N, dtype=complex)
        for i in range(N):
            result[i] = np.trapz(result_W[i, :], self.grid.xi)
        
        return result

class SplittingEvolution:
    """
    Évolution par splitting de Trotter pour H = T(ξ) + V(x)
    
    exp(-iĤΔt) ≈ exp(-iV̂Δt/2) exp(-iT̂Δt) exp(-iV̂Δt/2) + O(Δt³)
    """
    
    def __init__(self, grid: Grid, boundary: str = 'periodic'):
        self.grid = grid
        self.quantizer = WeylQuantization(grid, boundary)
        self.boundary = boundary
    
    def evolve_step(self, psi: np.ndarray, T_symbol: Symbol, V_symbol: Symbol, 
                    dt: float, method: str = 'strang') -> np.ndarray:
        """
        Un pas d'évolution par splitting
        
        Parameters:
        -----------
        psi : np.ndarray
            État à t
        T_symbol : Symbol
            Partie cinétique T(ξ) (diagonal en momentum)
        V_symbol : Symbol  
            Partie potentiel V(x) (diagonal en position)
        dt : float
            Pas de temps
        method : str
            'strang' (ordre 2) ou 'lie' (ordre 1)
        
        Returns:
        --------
        psi_new : np.ndarray
            État à t+dt
        """
        if method == 'strang':
            # Splitting de Strang (symétrique, ordre 2)
            # e^{-iHdt} ≈ e^{-iV dt/2} e^{-iT dt} e^{-iV dt/2}
            
            # 1. Demi-pas de potentiel
            psi = self._apply_potential(psi, V_symbol, dt/2)
            
            # 2. Pas complet cinétique
            psi = self._apply_kinetic(psi, T_symbol, dt)
            
            # 3. Demi-pas de potentiel
            psi = self._apply_potential(psi, V_symbol, dt/2)
        
        elif method == 'lie':
            # Splitting de Lie (ordre 1, plus simple)
            # e^{-iHdt} ≈ e^{-iV dt} e^{-iT dt}
            
            psi = self._apply_potential(psi, V_symbol, dt)
            psi = self._apply_kinetic(psi, T_symbol, dt)
        
        else:
            raise ValueError(f"Méthode '{method}' non reconnue")
        
        return psi
    
    def _apply_potential(self, psi: np.ndarray, V: Symbol, dt: float) -> np.ndarray:
        """
        Applique exp(-iV̂dt) : multiplication en espace réel
        """
        if self.grid.dim == 1:
            V_x = V.values[:, 0]  # V ne dépend que de x
        else:
            V_x = V.values[:, :, 0, 0]
        
        return psi * np.exp(-1j * V_x * dt)
    
    def _apply_kinetic(self, psi: np.ndarray, T: Symbol, dt: float) -> np.ndarray:
        """
        Applique exp(-iT̂dt) : multiplication en espace de Fourier
        """
        # 1. FFT
        psi_k = self.quantizer.fft_forward(psi)
        
        # 2. Multiplication par exp(-iT(ξ)dt)
        if self.grid.dim == 1:
            T_xi = T.values[0, :]  # T ne dépend que de ξ
        else:
            T_xi = T.values[0, 0, :, :]
        
        psi_k = psi_k * np.exp(-1j * T_xi * dt)
        
        # 3. IFFT
        return self.quantizer.fft_backward(psi_k)
    
    def evolve(self, psi0: np.ndarray, T_symbol_func: Callable, 
               V_symbol_func: Callable, t_span: Tuple[float, float], 
               Nt: int, save_every: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Évolution complète de t0 à tf
        
        Parameters:
        -----------
        psi0 : np.ndarray
            État initial
        T_symbol_func : Callable
            Fonction t -> Symbol donnant T(t,ξ)
        V_symbol_func : Callable
            Fonction t -> Symbol donnant V(t,x)
        t_span : Tuple[float, float]
            (t0, tf) intervalle de temps
        Nt : int
            Nombre de pas de temps
        save_every : int
            Sauvegarder un état tous les save_every pas
        
        Returns:
        --------
        times : np.ndarray
            Instants sauvegardés
        psi_history : np.ndarray
            États sauvegardés, shape = (Nt//save_every + 1, *psi0.shape)
        """
        t0, tf = t_span
        dt = (tf - t0) / Nt
        
        # Stockage
        n_saved = Nt // save_every + 1
        psi_history = np.zeros((n_saved,) + psi0.shape, dtype=complex)
        times = np.zeros(n_saved)
        
        psi = psi0.copy()
        psi_history[0] = psi
        times[0] = t0
        
        save_idx = 1
        
        for n in range(Nt):
            t = t0 + n * dt
            
            # Obtenir les symboles au temps t
            T = T_symbol_func(t)
            V = V_symbol_func(t)
            
            # Évolution
            psi = self.evolve_step(psi, T, V, dt, method='strang')
            
            # Sauvegarde
            if (n + 1) % save_every == 0 and save_idx < n_saved:
                psi_history[save_idx] = psi
                times[save_idx] = t + dt
                save_idx += 1
        
        return times, psi_history

class PseudoDifferentialOperator:
    """
    Opérateur pseudodifférentiel complet avec symbole + évolution
    """
    
    def __init__(self, grid: Grid, boundary: str = 'periodic'):
        self.grid = grid
        self.boundary = boundary
        self.quantizer = WeylQuantization(grid, boundary)
        self.evolver = SplittingEvolution(grid, boundary)
    
    def decompose_hamiltonian(self, h: Symbol) -> Tuple[Symbol, Symbol]:
        """
        Décompose h(x,ξ) = T(ξ) + V(x) si possible
        
        Returns:
        --------
        T, V : Symbol
            Parties cinétique et potentielle
        """
        # Approche simple : moyenniser
        if self.grid.dim == 1:
            # T(ξ) = moyenne sur x de h(x,ξ)
            T_values_1d = np.mean(h.values, axis=0)
            
            # V(x) = moyenne sur ξ de h(x,ξ)
            V_values_1d = np.mean(h.values, axis=1)
            
            # Créer les symboles
            T = Symbol(self.grid, order=h.order)
            V = Symbol(self.grid, order=0)
            
            # Broadcasting pour remplir la grille complète
            T.values = T_values_1d[np.newaxis, :]
            V.values = V_values_1d[:, np.newaxis]
        
        else:  # dim == 2
            # T(ξ_x, ξ_y) = moyenne sur (x,y)
            T_values_2d = np.mean(h.values, axis=(0, 1))
            
            # V(x,y) = moyenne sur (ξ_x, ξ_y)
            V_values_2d = np.mean(h.values, axis=(2, 3))
            
            T = Symbol(self.grid, order=h.order)
            V = Symbol(self.grid, order=0)
            
            T.values = T_values_2d[np.newaxis, np.newaxis, :, :]
            V.values = V_values_2d[:, :, np.newaxis, np.newaxis]
        
        return T, V
    
    def evolve_with_symbol(self, psi0: np.ndarray, 
                          hamiltonian_func: Callable[[float], Symbol],
                          t_span: Tuple[float, float], Nt: int,
                          save_every: int = 1,
                          method: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """
        Évolution avec un hamiltonien symbolique dépendant du temps
        
        Parameters:
        -----------
        psi0 : np.ndarray
            État initial
        hamiltonian_func : Callable
            Fonction t -> Symbol donnant h(t,x,ξ)
        t_span : Tuple[float, float]
            Intervalle de temps
        Nt : int
            Nombre de pas
        save_every : int
            Fréquence de sauvegarde
        method : str
            'auto', 'splitting', ou 'general'
        
        Returns:
        --------
        times, psi_history
        """
        if method == 'auto':
            # Tenter décomposition T+V
            h_test = hamiltonian_func(t_span[0])
            T_test, V_test = self.decompose_hamiltonian(h_test)
            
            # Vérifier si la décomposition est bonne
            h_recomposed = Symbol(self.grid, order=h_test.order)
            h_recomposed.values = T_test.values + V_test.values
            
            error = np.max(np.abs(h_test.values - h_recomposed.values))
            
            if error < 0.01 * np.max(np.abs(h_test.values)):
                print(f"Décomposition T+V réussie (erreur: {error:.2e}), "
                      f"utilisation du splitting")
                method = 'splitting'
            else:
                print(f"Décomposition T+V imprécise (erreur: {error:.2e}), "
                      f"utilisation de la méthode générale")
                method = 'general'
        
        if method == 'splitting':
            # Créer des fonctions qui retournent T et V à chaque instant
            def T_func(t):
                h = hamiltonian_func(t)
                T, _ = self.decompose_hamiltonian(h)
                return T
            
            def V_func(t):
                h = hamiltonian_func(t)
                _, V = self.decompose_hamiltonian(h)
                return V
            
            return self.evolver.evolve(psi0, T_func, V_func, t_span, Nt, save_every)
        
        elif method == 'general':
            # Méthode générale (plus lente)
            return self._evolve_general(psi0, hamiltonian_func, t_span, Nt, save_every)
        
        else:
            raise ValueError(f"Méthode '{method}' non reconnue")
    
    def _evolve_general(self, psi0: np.ndarray, 
                       hamiltonian_func: Callable[[float], Symbol],
                       t_span: Tuple[float, float], Nt: int,
                       save_every: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Évolution sans décomposition T+V (méthode générale, plus coûteuse)
        
        Utilise l'exponentielle de matrice ou approximations
        """
        t0, tf = t_span
        dt = (tf - t0) / Nt
        
        n_saved = Nt // save_every + 1
        psi_history = np.zeros((n_saved,) + psi0.shape, dtype=complex)
        times = np.zeros(n_saved)
        
        psi = psi0.copy()
        psi_history[0] = psi
        times[0] = t0
        
        save_idx = 1
        
        for n in range(Nt):
            t = t0 + n * dt
            h = hamiltonian_func(t)
            
            # Application directe : approximation d'ordre 1
            # ψ(t+dt) ≈ (1 - iĤdt)ψ(t)
            # ou mieux : ψ(t+dt) ≈ ψ(t) - iĤdt·ψ(t) + O(dt²)
            
            # Pour l'instant, méthode d'Euler implicite simplifiée
            # TODO: Implémenter Runge-Kutta 4 ou méthodes exponentielles
            
            H_psi = self.quantizer.apply_symbol_general(h, psi, method='pseudospectral')
            psi = psi - 1j * dt * H_psi
            
            # Normalisation (optionnelle)
            psi = psi / np.sqrt(np.sum(np.abs(psi)**2) * self.grid.dx)
            
            if (n + 1) % save_every == 0 and save_idx < n_saved:
                psi_history[save_idx] = psi
                times[save_idx] = t + dt
                save_idx += 1
        
        return times, psi_history

class BoundaryConditions:
    """
    Gestion des conditions aux limites au niveau symbolique
    """
    
    def __init__(self, grid: Grid, bc_type: str, 
                 horizon_params: dict = None):
        """
        Parameters:
        -----------
        grid : Grid
        bc_type : str
            'periodic', 'dirichlet', 'horizon'
        horizon_params : dict
            Pour 'horizon' : {'width': δ, 'strength': Γ₀, 'power': p}
        """
        self.grid = grid
        self.bc_type = bc_type
        self.horizon_params = horizon_params or {
            'width': 0.1,    # Fraction du domaine
            'strength': 1.0,  # Force d'absorption
            'power': 2        # Exposant du profil
        }
    
    def create_absorbing_potential(self) -> Symbol:
        """
        Crée un potentiel absorbant Γ(x) pour les conditions d'horizon
        
        Ajouter -iΓ(x) au hamiltonien près des bords
        """
        if self.bc_type != 'horizon':
            # Pas d'absorption
            Gamma = Symbol(self.grid, order=0)
            Gamma.values[:] = 0.0
            return Gamma
        
        Gamma = Symbol(self.grid, order=0)
        
        width = self.horizon_params['width']
        Gamma0 = self.horizon_params['strength']
        p = self.horizon_params['power']
        
        if self.grid.dim == 1:
            L = self.grid.x[-1] - self.grid.x[0]
            delta = width * L
            
            Gamma_values = np.zeros(len(self.grid.x))
            
            for i, x in enumerate(self.grid.x):
                # Bord gauche
                if x < delta:
                    Gamma_values[i] = Gamma0 * ((delta - x) / delta) ** p
                
                # Bord droit
                elif x > L - delta:
                    Gamma_values[i] = Gamma0 * ((x - (L - delta)) / delta) ** p
            
            # Broadcasting sur la grille (x, ξ)
            Gamma.values = Gamma_values[:, np.newaxis]
        
        else:  # dim == 2
            Lx = self.grid.x[0][-1, 0] - self.grid.x[0][0, 0]
            Ly = self.grid.x[1][0, -1] - self.grid.x[1][0, 0]
            delta_x = width * Lx
            delta_y = width * Ly
            
            X, Y = self.grid.x
            Gamma_values = np.zeros_like(X)
            
            # Contributions des 4 bords
            # Bord x = 0
            mask_left = X < delta_x
            Gamma_values[mask_left] += Gamma0 * ((delta_x - X[mask_left]) / delta_x) ** p
            
            # Bord x = Lx
            mask_right = X > Lx - delta_x
            Gamma_values[mask_right] += Gamma0 * ((X[mask_right] - (Lx - delta_x)) / delta_x) ** p
            
            # Bord y = 0
            mask_bottom = Y < delta_y
            Gamma_values[mask_bottom] += Gamma0 * ((delta_y - Y[mask_bottom]) / delta_y) ** p
            
            # Bord y = Ly
            mask_top = Y > Ly - delta_y
            Gamma_values[mask_top] += Gamma0 * ((Y[mask_top] - (Ly - delta_y)) / delta_y) ** p
            
            # Broadcasting sur (x, y, ξ_x, ξ_y)
            Gamma.values = Gamma_values[:, :, np.newaxis, np.newaxis]
        
        return Gamma
    
    def modify_hamiltonian(self, h: Symbol) -> Symbol:
        """
        Modifie le hamiltonien pour inclure les conditions aux limites
        
        Pour horizon : h → h - iΓ(x)
        """
        if self.bc_type == 'horizon':
            Gamma = self.create_absorbing_potential()
            h_modified = Symbol(self.grid, order=h.order)
            h_modified.values = h.values - 1j * Gamma.values
            return h_modified
        else:
            return h
    
    def apply_to_state(self, psi: np.ndarray) -> np.ndarray:
        """
        Applique les conditions aux limites directement sur l'état
        
        Pour Dirichlet : forcer ψ = 0 aux bords
        Pour periodic : déjà géré par FFT
        Pour horizon : absorption incluse dans hamiltonien
        """
        if self.bc_type == 'dirichlet':
            psi_bc = psi.copy()
            
            if self.grid.dim == 1:
                psi_bc[0] = 0.0
                psi_bc[-1] = 0.0
            else:
                psi_bc[0, :] = 0.0
                psi_bc[-1, :] = 0.0
                psi_bc[:, 0] = 0.0
                psi_bc[:, -1] = 0.0
            
            return psi_bc
        
        else:
            return psi

class SymbolicEvolutionWithBC:
    """
    Classe unifiée : symbole + évolution + conditions aux limites
    """
    
    def __init__(self, grid: Grid, bc_type: str, 
                 horizon_params: dict = None):
        self.grid = grid
        self.bc_type = bc_type
        
        self.bc = BoundaryConditions(grid, bc_type, horizon_params)
        self.operator = PseudoDifferentialOperator(grid, bc_type)
    
    def create_hamiltonian_relativistic(self, metric_func: Callable,
                                       potential_func: Callable = None) -> Callable:
        """
        Crée un hamiltonien relativiste h = √(γⁱʲ ξᵢξⱼ) + V
        
        Parameters:
        -----------
        metric_func : Callable
            Fonction (t, x) -> γⁱʲ(t,x) métrique spatiale
        potential_func : Callable, optional
            Fonction (t, x) -> V(t,x)
        
        Returns:
        --------
        h_func : Callable
            Fonction t -> Symbol
        """
        def h_func(t):
            if self.grid.dim == 1:
                # Grilles
                X, XI = np.meshgrid(self.grid.x, self.grid.xi, indexing='ij')
                
                # Métrique γ(t,x)
                gamma = metric_func(t, self.grid.x)
                
                # Partie cinétique : √(γ(x) ξ²)
                T_values = np.sqrt(gamma[:, np.newaxis] * XI**2)
                
                # Potentiel
                if potential_func is not None:
                    V_values = potential_func(t, self.grid.x)[:, np.newaxis]
                else:
                    V_values = 0.0
                
                # Symbole total
                h = Symbol(self.grid, order=1)
                h.values = T_values + V_values
            
            else:  # dim == 2
                X, Y = self.grid.x
                XI_X, XI_Y = self.grid.xi
                
                # Métrique γⁱʲ(t,x,y) : matrice 2x2 en chaque point
                gamma = metric_func(t, X, Y)  # Shape: (Nx, Ny, 2, 2)
                
                # Broadcasting pour phase space
                x_b = X[:, :, np.newaxis, np.newaxis]
                y_b = Y[:, :, np.newaxis, np.newaxis]
                xi_x_b = XI_X[np.newaxis, np.newaxis, :, :]
                xi_y_b = XI_Y[np.newaxis, np.newaxis, :, :]
                
                gamma_b = gamma[:, :, np.newaxis, np.newaxis, :, :]
                
                # T = √(γⁱʲ ξᵢ ξⱼ)
                T_values = np.sqrt(
                    gamma_b[:, :, :, :, 0, 0] * xi_x_b**2 +
                    2 * gamma_b[:, :, :, :, 0, 1] * xi_x_b * xi_y_b +
                    gamma_b[:, :, :, :, 1, 1] * xi_y_b**2
                )
                
                if potential_func is not None:
                    V_values = potential_func(t, x_b, y_b)
                else:
                    V_values = 0.0
                
                h = Symbol(self.grid, order=1)
                h.values = T_values + V_values
            
            # Modifier pour conditions aux limites
            h = self.bc.modify_hamiltonian(h)
            
            return h
        
        return h_func
    
    def evolve(self, psi0: np.ndarray, hamiltonian_func: Callable,
               t_span: Tuple[float, float], Nt: int,
               save_every: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Évolution complète avec conditions aux limites
        """
        # Appliquer conditions initiales
        psi0 = self.bc.apply_to_state(psi0)
        
        # Évolution
        times, psi_history = self.operator.evolve_with_symbol(
            psi0, hamiltonian_func, t_span, Nt, save_every
        )
        
        # Appliquer conditions à chaque état sauvegardé
        for i in range(len(psi_history)):
            psi_history[i] = self.bc.apply_to_state(psi_history[i])
        
        return times, psi_history

def example_relativistic_evolution():
    """
    Exemple : paquet d'onde dans espace-temps courbe avec horizon
    """
    # 1. Créer la grille
    grid = Grid.create_1d(N=128, L=10.0, periodic=False)
    
    # 2. Conditions aux limites : horizon (absorption aux bords)
    evolver = SymbolicEvolutionWithBC(
        grid, 
        bc_type='horizon',
        horizon_params={'width': 0.15, 'strength': 2.0, 'power': 2}
    )
    
    # 3. Définir métrique dépendant du temps
    # Exemple : métrique qui "s'effondre" (simulation trou noir)
    def metric_collapse(t, x):
        """γ(t,x) = 1 - tanh(α(t)(x - x₀)²)"""
        x0 = 5.0  # Centre de l'effondrement
        alpha = 0.1 * t  # Force croissante
        return 1.0 - 0.5 * np.tanh(alpha * (x - x0)**2)
    
    # 4. Potentiel (optionnel)
    def potential(t, x):
        return 0.1 * x**2  # Puits harmonique faible
    
    # 5. Créer le hamiltonien symbolique
    h_func = evolver.create_hamiltonian_relativistic(
        metric_collapse, potential
    )
    
    # 6. État initial : paquet gaussien
    x = grid.x
    x0, sigma, k0 = 3.0, 0.5, 5.0
    psi0 = np.exp(-(x - x0)**2 / (2*sigma**2)) * np.exp(1j * k0 * x)
    psi0 = psi0 / np.sqrt(np.sum(np.abs(psi0)**2) * grid.dx)
    
    # 7. Évolution
    times, psi_history = evolver.evolve(
        psi0,
        h_func,
        t_span=(0.0, 2.0),
        Nt=256,
        save_every=4
    )
    
    # 8. Analyse
    print(f"Évolution de t={times[0]:.2f} à t={times[-1]:.2f}")
    print(f"Nombre d'états sauvegardés : {len(psi_history)}")
    
    # Vérifier conservation de la norme (avec absorption)
    norms = [np.sum(np.abs(psi)**2) * grid.dx for psi in psi_history]
    print(f"Norme initiale : {norms[0]:.4f}")
    print(f"Norme finale : {norms[-1]:.4f} (absorption aux horizons)")
    
    return times, psi_history, grid


def example_2d_periodic():
    """
    Exemple 2D avec conditions périodiques
    """
    # 1. Grille 2D
    grid = Grid.create_2d(Nx=128, Ny=128, Lx=10.0, Ly=10.0, periodic=True)
    
    # 2. Conditions périodiques
    evolver = SymbolicEvolutionWithBC(grid, bc_type='periodic')
    
    # 3. Métrique : espace-temps ondulé
    def metric_wave(t, X, Y):
        """Onde gravitationnelle se propageant"""
        omega = 2.0
        k = 1.0
        
        # Métrique γⁱʲ comme matrice 2x2
        gamma = np.zeros(X.shape + (2, 2))
        
        perturbation = 0.1 * np.sin(k * X - omega * t)
        
        gamma[:, :, 0, 0] = 1.0 + perturbation
        gamma[:, :, 1, 1] = 1.0 - perturbation
        gamma[:, :, 0, 1] = 0.0
        gamma[:, :, 1, 0] = 0.0
        
        return gamma
    
    # 4. Hamiltonien
    h_func = evolver.create_hamiltonian_relativistic(metric_wave)
    
    # 5. État initial : paquet 2D
    X, Y = grid.x
    x0, y0, sigma = 5.0, 5.0, 0.7
    kx0, ky0 = 3.0, 3.0
    
    psi0 = (np.exp(-((X - x0)**2 + (Y - y0)**2) / (2*sigma**2)) *
            np.exp(1j * (kx0 * X + ky0 * Y)))
    psi0 = psi0 / np.sqrt(np.sum(np.abs(psi0)**2) * grid.dx**2)
    
    # 6. Évolution
    times, psi_history = evolver.evolve(
        psi0, h_func,
        t_span=(0.0, 5.0),
        Nt=256,
        save_every=8
    )
    
    return times, psi_history, grid

from functools import lru_cache
import hashlib

class CachedSymbol(Symbol):
    """
    Symbole avec mise en cache des calculs coûteux
    """
    
    def __init__(self, grid: Grid, order: int = 0, cache_size: int = 128):
        super().__init__(grid, order)
        self._gradient_x_cache = None
        self._gradient_xi_cache = None
        self._values_hash = None
        self.cache_size = cache_size
    
    def _compute_hash(self):
        """Hash des valeurs pour détecter les changements"""
        return hashlib.md5(self.values.tobytes()).hexdigest()
    
    def gradient_x(self, use_cache: bool = True):
        """Gradient avec cache"""
        current_hash = self._compute_hash()
        
        if use_cache and self._values_hash == current_hash and self._gradient_x_cache is not None:
            return self._gradient_x_cache
        
        # Recalculer
        self._gradient_x_cache = super().gradient_x()
        self._values_hash = current_hash
        
        return self._gradient_x_cache
    
    def gradient_xi(self, use_cache: bool = True):
        """Gradient avec cache"""
        current_hash = self._compute_hash()
        
        if use_cache and self._values_hash == current_hash and self._gradient_xi_cache is not None:
            return self._gradient_xi_cache
        
        self._gradient_xi_cache = super().gradient_xi()
        self._values_hash = current_hash
        
        return self._gradient_xi_cache


class AdaptiveSymbolicEvolution:
    """
    Évolution avec pas de temps adaptatif et réutilisation de symboles
    """
    
    def __init__(self, grid: Grid, bc_type: str, 
                 horizon_params: dict = None,
                 adaptive: bool = True):
        self.grid = grid
        self.bc_type = bc_type
        self.adaptive = adaptive
        
        self.bc = BoundaryConditions(grid, bc_type, horizon_params)
        self.operator = PseudoDifferentialOperator(grid, bc_type)
        
        # Cache pour symboles déjà calculés
        self._symbol_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cached_symbol(self, h_func: Callable, t: float, 
                         tolerance: float = 1e-10) -> Symbol:
        """
        Récupère un symbole du cache si disponible
        
        Parameters:
        -----------
        h_func : Callable
            Fonction générant le symbole
        t : float
            Temps
        tolerance : float
            Tolérance pour considérer deux temps identiques
        
        Returns:
        --------
        h : Symbol
        """
        # Chercher dans le cache un temps proche
        for t_cached, h_cached in self._symbol_cache.items():
            if abs(t - t_cached) < tolerance:
                self._cache_hits += 1
                return h_cached
        
        # Pas trouvé : calculer
        self._cache_misses += 1
        h = h_func(t)
        
        # Ajouter au cache (avec limite de taille)
        if len(self._symbol_cache) < 100:  # Limite arbitraire
            self._symbol_cache[t] = h
        
        return h
    
    def estimate_timestep(self, h: Symbol, psi: np.ndarray, 
                         safety: float = 0.5) -> float:
        """
        Estime un pas de temps adaptatif basé sur le symbole
        
        Critère CFL : dt ≤ C × dx / max(|v_groupe|)
        où v_groupe = ∂h/∂ξ
        
        Parameters:
        -----------
        h : Symbol
            Hamiltonien
        psi : np.ndarray
            État actuel
        safety : float
            Facteur de sécurité (< 1)
        
        Returns:
        --------
        dt : float
            Pas de temps suggéré
        """
        # Calculer la vitesse de groupe maximale
        grad_xi = h.gradient_xi()
        
        if self.grid.dim == 1:
            v_max = np.max(np.abs(grad_xi))
        else:
            grad_xi_x, grad_xi_y = grad_xi
            v_max = np.max(np.sqrt(grad_xi_x**2 + grad_xi_y**2))
        
        # Condition CFL
        if v_max > 1e-10:
            dt_cfl = safety * self.grid.dx / v_max
        else:
            dt_cfl = 1.0  # Pas de limite si vitesse nulle
        
        return dt_cfl
    
    def evolve_adaptive(self, psi0: np.ndarray, 
                       hamiltonian_func: Callable,
                       t_span: Tuple[float, float],
                       dt_initial: float = 0.01,
                       dt_min: float = 1e-6,
                       dt_max: float = 0.1,
                       save_every_dt: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Évolution avec pas de temps adaptatif
        
        Parameters:
        -----------
        psi0 : np.ndarray
            État initial
        hamiltonian_func : Callable
            Fonction t -> Symbol
        t_span : Tuple[float, float]
            Intervalle de temps
        dt_initial : float
            Pas de temps initial
        dt_min, dt_max : float
            Limites sur dt
        save_every_dt : float, optional
            Sauvegarder tous les save_every_dt (None = sauvegarder tout)
        
        Returns:
        --------
        times, psi_history
        """
        t0, tf = t_span
        t = t0
        dt = dt_initial
        
        psi = psi0.copy()
        psi = self.bc.apply_to_state(psi)
        
        # Stockage
        times = [t0]
        psi_history = [psi.copy()]
        last_save_time = t0
        
        # Décomposition T+V du premier symbole
        h_test = hamiltonian_func(t0)
        T_test, V_test = self.operator.decompose_hamiltonian(h_test)
        
        print(f"Évolution adaptative de t={t0} à t={tf}")
        print(f"dt initial: {dt}")
        
        step_count = 0
        
        while t < tf:
            # Obtenir symbole (avec cache)
            h = self.get_cached_symbol(hamiltonian_func, t)
            
            # Décomposer
            T, V = self.operator.decompose_hamiltonian(h)
            
            # Estimer dt optimal si adaptatif
            if self.adaptive:
                dt_suggested = self.estimate_timestep(h, psi, safety=0.5)
                dt = np.clip(dt_suggested, dt_min, dt_max)
            
            # Ajuster pour ne pas dépasser tf
            if t + dt > tf:
                dt = tf - t
            
            # Un pas d'évolution
            psi = self.operator.evolver.evolve_step(psi, T, V, dt, method='strang')
            psi = self.bc.apply_to_state(psi)
            
            t += dt
            step_count += 1
            
            # Sauvegarder ?
            if save_every_dt is None or (t - last_save_time) >= save_every_dt:
                times.append(t)
                psi_history.append(psi.copy())
                last_save_time = t
            
            # Affichage périodique
            if step_count % 100 == 0:
                norm = np.sum(np.abs(psi)**2) * self.grid.dx
                print(f"  t={t:.4f}, dt={dt:.6f}, ||ψ||²={norm:.6f}, "
                      f"cache: {self._cache_hits}/{self._cache_hits+self._cache_misses}")
        
        print(f"Évolution terminée : {step_count} pas, {len(times)} états sauvegardés")
        
        return np.array(times), np.array(psi_history)

    def create_hamiltonian_relativistic(self, metric_func, potential_func=None):
        # Réutiliser l’implémentation existante dans SymbolicEvolutionWithBC
        wrapper = SymbolicEvolutionWithBC(
            self.grid,
            self.bc_type,
            self.bc.horizon_params
        )
        return wrapper.create_hamiltonian_relativistic(metric_func, potential_func)

class HeatEquationSolver:
    """
    Résolution de l'équation de la chaleur via opérateurs pseudodifférentiels
    
    ∂_t u = D Ĥ_diff u
    
    où Ĥ_diff a le symbole h_diff(x,ξ) = -ξ²
    """
    
    def __init__(self, grid: Grid, bc_type: str = 'periodic', 
                 diffusivity: float = 1.0):
        """
        Parameters:
        -----------
        grid : Grid
            Grille spatiale
        bc_type : str
            Conditions aux limites
        diffusivity : float
            Coefficient de diffusion D
        """
        self.grid = grid
        self.bc_type = bc_type
        self.D = diffusivity
        
        self.bc = BoundaryConditions(grid, bc_type, horizon_params=None)
        self.quantizer = WeylQuantization(grid, bc_type)
    
    def create_diffusion_symbol(self, metric_func: Callable = None) -> Callable:
        """
        Crée le symbole de diffusion
        
        Cas isotrope : h_diff = -D ξ²
        Cas anisotrope : h_diff = -D γⁱʲ(x) ξᵢξⱼ  (diffusion dans métrique courbe)
        
        Parameters:
        -----------
        metric_func : Callable, optional
            (t,x) -> γ(t,x) pour diffusion anisotrope
            Si None, diffusion isotrope standard
        
        Returns:
        --------
        h_func : Callable
            Fonction t -> Symbol
        """
        if metric_func is None:
            # Diffusion isotrope standard
            if self.grid.dim == 1:
                def h_func(t):
                    h = Symbol(self.grid, order=2)
                    X, XI = np.meshgrid(self.grid.x, self.grid.xi, indexing='ij')
                    h.values = -self.D * XI**2
                    return h
            
            else:  # 2D
                def h_func(t):
                    h = Symbol(self.grid, order=2)
                    X, Y = self.grid.x
                    XI_X, XI_Y = self.grid.xi
                    
                    xi_x_b = XI_X[np.newaxis, np.newaxis, :, :]
                    xi_y_b = XI_Y[np.newaxis, np.newaxis, :, :]
                    
                    h.values = -self.D * (xi_x_b**2 + xi_y_b**2)
                    return h
        
        else:
            # Diffusion anisotrope avec métrique
            if self.grid.dim == 1:
                def h_func(t):
                    h = Symbol(self.grid, order=2)
                    X, XI = np.meshgrid(self.grid.x, self.grid.xi, indexing='ij')
                    
                    gamma = metric_func(t, self.grid.x)
                    h.values = -self.D * gamma[:, np.newaxis] * XI**2
                    return h
            
            else:  # 2D anisotrope
                def h_func(t):
                    h = Symbol(self.grid, order=2)
                    X, Y = self.grid.x
                    XI_X, XI_Y = self.grid.xi
                    
                    gamma = metric_func(t, X, Y)  # Shape: (Nx, Ny, 2, 2)
                    
                    xi_x_b = XI_X[np.newaxis, np.newaxis, :, :]
                    xi_y_b = XI_Y[np.newaxis, np.newaxis, :, :]
                    gamma_b = gamma[:, :, np.newaxis, np.newaxis, :, :]
                    
                    h.values = -self.D * (
                        gamma_b[:,:,:,:,0,0] * xi_x_b**2 +
                        2 * gamma_b[:,:,:,:,0,1] * xi_x_b * xi_y_b +
                        gamma_b[:,:,:,:,1,1] * xi_y_b**2
                    )
                    return h
        
        return h_func
    
    def evolve_heat(self, u0: np.ndarray, 
                    h_func: Callable,
                    t_span: Tuple[float, float],
                    Nt: int,
                    save_every: int = 1,
                    source_term: Callable = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Résout l'équation de la chaleur ∂_t u = Ĥ_diff u + f(t,x)
        
        Parameters:
        -----------
        u0 : np.ndarray
            Distribution initiale de température
        h_func : Callable
            Fonction t -> Symbol du laplacien
        t_span : Tuple[float, float]
            Intervalle de temps
        Nt : int
            Nombre de pas de temps
        save_every : int
            Fréquence de sauvegarde
        source_term : Callable, optional
            Fonction (t,x) -> f(t,x) terme source de chaleur
        
        Returns:
        --------
        times : np.ndarray
        u_history : np.ndarray
            Évolution de la température
        """
        t0, tf = t_span
        dt = (tf - t0) / Nt
        
        # Stockage
        n_saved = Nt // save_every + 1
        u_history = np.zeros((n_saved,) + u0.shape, dtype=float)
        times = np.zeros(n_saved)
        
        u = u0.copy()
        u_history[0] = u
        times[0] = t0
        
        save_idx = 1
        
        print(f"Résolution équation de la chaleur : t={t0} → {tf}")
        
        for n in range(Nt):
            t = t0 + n * dt
            
            # Symbole de diffusion au temps t
            h = h_func(t)
            
            # Décomposer en T(ξ) seulement (pas de partie V(x) pour diffusion pure)
            if self.grid.dim == 1:
                h_xi = h.values[0, :]  # Ne dépend que de ξ
            else:
                h_xi = h.values[0, 0, :, :]
            
            # Évolution : u(t+dt) = exp(dt Ĥ_diff) u(t)
            # Notez : pas de 'i' ici, c'est une équation réelle !
            
            # FFT
            u_k = self.quantizer.fft_forward(u)
            
            # Multiplication par exp(dt h(ξ))
            # Pour la chaleur : exp(dt × (-D ξ²)) = exp(-D ξ² dt)
            u_k = u_k * np.exp(dt * h_xi)
            
            # IFFT
            u = np.real(self.quantizer.fft_backward(u_k))
            
            # Ajouter terme source si présent
            if source_term is not None:
                if self.grid.dim == 1:
                    f = source_term(t, self.grid.x)
                else:
                    f = source_term(t, *self.grid.x)
                u += dt * f
            
            # Appliquer conditions aux limites
            u = self.bc.apply_to_state(u)
            
            # Sauvegarde
            if (n + 1) % save_every == 0 and save_idx < n_saved:
                u_history[save_idx] = u
                times[save_idx] = t + dt
                save_idx += 1
            
            if (n + 1) % (Nt // 10) == 0:
                print(f"  t={t+dt:.4f} ({n+1}/{Nt}), "
                      f"T_min={np.min(u):.4f}, T_max={np.max(u):.4f}, "
                      f"T_mean={np.mean(u):.4f}")
        
        print("✓ Résolution terminée")
        
        return times, u_history
    
    def steady_state_solve(self, source: np.ndarray, 
                          h_symbol: Symbol) -> np.ndarray:
        """
        Résout l'équation stationnaire : Ĥ_diff u = -f
        
        Pour trouver la distribution stationnaire de température
        
        Parameters:
        -----------
        source : np.ndarray
            Distribution de sources f(x)
        h_symbol : Symbol
            Symbole du laplacien
        
        Returns:
        --------
        u_steady : np.ndarray
            Distribution stationnaire
        """
        # Inverser symboliquement
        h_inv = symbolic_inverse(h_symbol, order=2, regularization=1e-10)
        
        # Appliquer : u = -Ĥ^{-1} f
        # En espace de Fourier : û(ξ) = -f̂(ξ) / h(ξ)
        
        f_k = self.quantizer.fft_forward(source)
        
        if self.grid.dim == 1:
            h_xi = h_symbol.values[0, :]
        else:
            h_xi = h_symbol.values[0, 0, :, :]
        
        # Division (attention au signe pour la chaleur)
        u_k = -f_k / h_xi
        
        u_steady = np.real(self.quantizer.fft_backward(u_k))
        
        return u_steady


class HeatDiagnostics:
    """
    Diagnostics spécifiques à l'équation de la chaleur
    """
    
    @staticmethod
    def total_heat(u: np.ndarray, grid: Grid) -> float:
        """Chaleur totale Q = ∫ u dx"""
        if grid.dim == 1:
            return np.trapz(u, grid.x)
        else:
            return np.sum(u) * grid.dx**2
    
    @staticmethod
    def entropy_production(u: np.ndarray, h_symbol: Symbol, 
                          grid: Grid) -> float:
        """
        Production d'entropie dS/dt = D ∫ |∇u|² dx
        
        Via le symbole : |∇u|² ~ ∫ |ξ|² |û(ξ)|² dξ
        """
        # Transformée de Fourier
        if grid.dim == 1:
            u_k = fft(u)
            xi = grid.xi
            integrand = grid.xi**2 * np.abs(u_k)**2
            return np.trapz(integrand, xi)
        else:
            u_k = fft2(u)
            XI_X, XI_Y = grid.xi
            integrand = (XI_X**2 + XI_Y**2) * np.abs(u_k)**2
            return np.sum(integrand) * grid.dxi[0] * grid.dxi[1]
    
    @staticmethod
    def diffusion_time(L: float, D: float) -> float:
        """
        Temps caractéristique de diffusion τ = L² / D
        """
        return L**2 / D

class ReactionDiffusionSolver(HeatEquationSolver):
    """
    Équation de réaction-diffusion : ∂_t u = D ∇²u + R(u)
    
    où R(u) est un terme de réaction non-linéaire
    
    Exemples :
    - Fisher-KPP : R(u) = r u(1 - u/K)
    - Bistable : R(u) = r u(1 - u)(u - a)
    - Gray-Scott : système couplé u-v
    """
    
    def __init__(self, grid: Grid, bc_type: str = 'periodic',
                 diffusivity: float = 1.0,
                 reaction_func: Callable = None):
        """
        Parameters:
        -----------
        reaction_func : Callable
            Fonction (u, params) -> R(u)
        """
        super().__init__(grid, bc_type, diffusivity)
        self.reaction_func = reaction_func
    
    def evolve_reaction_diffusion(self, u0: np.ndarray,
                                  h_func: Callable,
                                  t_span: Tuple[float, float],
                                  Nt: int,
                                  reaction_params: dict = None,
                                  save_every: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Résout ∂_t u = D ∇²u + R(u)
        
        Utilise splitting de Strang :
        exp(dt(D∇² + R)) ≈ exp(dt R/2) exp(dt D∇²) exp(dt R/2)
        """
        t0, tf = t_span
        dt = (tf - t0) / Nt
        
        reaction_params = reaction_params or {}
        
        # Stockage
        n_saved = Nt // save_every + 1
        u_history = np.zeros((n_saved,) + u0.shape, dtype=float)
        times = np.zeros(n_saved)
        
        u = u0.copy()
        u_history[0] = u
        times[0] = t0
        
        save_idx = 1
        
        print(f"Résolution réaction-diffusion : t={t0} → {tf}")
        
        for n in range(Nt):
            t = t0 + n * dt
            
            # Splitting de Strang
            
            # 1. Demi-pas de réaction : u → u + (dt/2) R(u)
            if self.reaction_func is not None:
                R_u = self.reaction_func(u, reaction_params)
                u = u + (dt / 2) * R_u
            
            # 2. Pas complet de diffusion
            h = h_func(t)
            
            if self.grid.dim == 1:
                h_xi = h.values[0, :]
            else:
                h_xi = h.values[0, 0, :, :]
            
            u_k = self.quantizer.fft_forward(u)
            u_k = u_k * np.exp(dt * h_xi)
            u = np.real(self.quantizer.fft_backward(u_k))
            
            # 3. Demi-pas de réaction
            if self.reaction_func is not None:
                R_u = self.reaction_func(u, reaction_params)
                u = u + (dt / 2) * R_u
            
            # Appliquer conditions aux limites
            u = self.bc.apply_to_state(u)
            
            # Sauvegarde
            if (n + 1) % save_every == 0 and save_idx < n_saved:
                u_history[save_idx] = u
                times[save_idx] = t + dt
                save_idx += 1
            
            if (n + 1) % (Nt // 10) == 0:
                print(f"  t={t+dt:.4f} ({n+1}/{Nt}), "
                      f"u_min={np.min(u):.4f}, u_max={np.max(u):.4f}")
        
        print("✓ Résolution terminée")
        
        return times, u_history


class GrayScottSolver:
    """
    Système de réaction-diffusion de Gray-Scott (2 espèces)
    
    ∂_t u = D_u ∇²u - uv² + F(1-u)
    ∂_t v = D_v ∇²v + uv² - (F+k)v
    
    Génère des motifs auto-organisés (spots, stripes, labyrinthes, etc.)
    """
    
    def __init__(self, grid: Grid, bc_type: str = 'periodic',
                 D_u: float = 2e-5, D_v: float = 1e-5,
                 F: float = 0.055, k: float = 0.062):
        """
        Parameters:
        -----------
        D_u, D_v : float
            Coefficients de diffusion pour u et v
        F : float
            Taux d'alimentation
        k : float
            Taux de destruction
        
        Régimes intéressants (Pearson, 1993) :
        - α : F=0.010, k=0.047 (spots)
        - β : F=0.014, k=0.054 (stripes)
        - γ : F=0.018, k=0.051 (holes)
        - δ : F=0.022, k=0.059 (moving spots)
        - ε : F=0.030, k=0.062 (waves)
        - ζ : F=0.026, k=0.055 (spiral waves)
        - η : F=0.034, k=0.063 (spiral chaos)
        """
        self.grid = grid
        self.bc_type = bc_type
        self.D_u = D_u
        self.D_v = D_v
        self.F = F
        self.k = k
        
        self.quantizer = WeylQuantization(grid, bc_type)
        
        # Pré-calculer les exponentielles de diffusion
        if grid.dim == 2:
            XI_X, XI_Y = grid.xi
            laplacian_u = -D_u * (XI_X**2 + XI_Y**2)
            laplacian_v = -D_v * (XI_X**2 + XI_Y**2)
            
            self.exp_laplacian_u = None  # Sera calculé pour chaque dt
            self.exp_laplacian_v = None
            self.laplacian_u = laplacian_u
            self.laplacian_v = laplacian_v
    
    def reaction(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Termes de réaction
        
        R_u = -uv² + F(1-u)
        R_v = +uv² - (F+k)v
        """
        uv2 = u * v**2
        
        R_u = -uv2 + self.F * (1.0 - u)
        R_v = +uv2 - (self.F + self.k) * v
        
        return R_u, R_v
    
    def evolve(self, u0: np.ndarray, v0: np.ndarray,
              t_span: Tuple[float, float], Nt: int,
              save_every: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Évolution du système Gray-Scott
        
        Returns:
        --------
        times, u_history, v_history
        """
        t0, tf = t_span
        dt = (tf - t0) / Nt
        
        # Pré-calculer exponentielles
        exp_lap_u_half = np.exp(dt * self.laplacian_u / 2)
        exp_lap_u_full = np.exp(dt * self.laplacian_u)
        exp_lap_v_half = np.exp(dt * self.laplacian_v / 2)
        exp_lap_v_full = np.exp(dt * self.laplacian_v)
        
        # Stockage
        n_saved = Nt // save_every + 1
        u_history = np.zeros((n_saved,) + u0.shape, dtype=float)
        v_history = np.zeros((n_saved,) + v0.shape, dtype=float)
        times = np.zeros(n_saved)
        
        u = u0.copy()
        v = v0.copy()
        
        u_history[0] = u
        v_history[0] = v
        times[0] = t0
        
        save_idx = 1
        
        print(f"Évolution Gray-Scott : F={self.F}, k={self.k}")
        print(f"D_u={self.D_u:.2e}, D_v={self.D_v:.2e}")
        print(f"t={t0} → {tf} ({Nt} pas)")
        
        for n in range(Nt):
            t = t0 + n * dt
            
            # Splitting de Strang : diffusion-réaction-diffusion
            
            # 1. Demi-pas diffusion u
            u_k = fft2(u)
            u_k *= exp_lap_u_half
            u = np.real(ifft2(u_k))
            
            # 2. Demi-pas diffusion v
            v_k = fft2(v)
            v_k *= exp_lap_v_half
            v = np.real(ifft2(v_k))
            
            # 3. Pas complet réaction
            R_u, R_v = self.reaction(u, v)
            u += dt * R_u
            v += dt * R_v
            
            # 4. Demi-pas diffusion u
            u_k = fft2(u)
            u_k *= exp_lap_u_half
            u = np.real(ifft2(u_k))
            
            # 5. Demi-pas diffusion v
            v_k = fft2(v)
            v_k *= exp_lap_v_half
            v = np.real(ifft2(v_k))
            
            # Clamp pour stabilité numérique
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
            
            # Sauvegarde
            if (n + 1) % save_every == 0 and save_idx < n_saved:
                u_history[save_idx] = u
                v_history[save_idx] = v
                times[save_idx] = t + dt
                save_idx += 1
            
            if (n + 1) % (Nt // 10) == 0:
                print(f"  t={t+dt:.1f} ({n+1}/{Nt})")
        
        print("✓ Terminé")
        
        return times, u_history, v_history

class AdvectionDiffusionSolver(HeatEquationSolver):
    """
    Équation d'advection-diffusion : ∂_t u = D ∇²u - v·∇u
    
    où v(x) est un champ de vélocité
    
    Applications :
    - Transport de polluants
    - Écoulement thermique
    - Convection forcée
    """
    
    def __init__(self, grid: Grid, bc_type: str = 'periodic',
                 diffusivity: float = 1.0,
                 velocity_field: Callable = None):
        """
        Parameters:
        -----------
        velocity_field : Callable
            Fonction (t, x) -> v(t,x) en 1D
            ou (t, X, Y) -> (v_x, v_y) en 2D
        """
        super().__init__(grid, bc_type, diffusivity)
        self.velocity_field = velocity_field
    
    def create_advection_diffusion_symbol(self) -> Callable:
        """
        Crée le symbole complet h = -D ξ² - i v·ξ
        
        Partie diffusion : -D ξ²
        Partie advection : -i v·ξ
        """
        if self.velocity_field is None:
            # Pas d'advection, juste diffusion
            return super().create_diffusion_symbol()
        
        if self.grid.dim == 1:
            def h_func(t):
                h = Symbol(self.grid, order=1)
                X, XI = np.meshgrid(self.grid.x, self.grid.xi, indexing='ij')
                
                # Champ de vélocité
                v = self.velocity_field(t, self.grid.x)
                
                # h = -D ξ² - i v ξ
                h.values = -self.D * XI**2 - 1j * v[:, np.newaxis] * XI
                
                return h
        
        else:  # 2D
            def h_func(t):
                h = Symbol(self.grid, order=1)
                X, Y = self.grid.x
                XI_X, XI_Y = self.grid.xi
                
                # Champ de vélocité (v_x, v_y)
                v_x, v_y = self.velocity_field(t, X, Y)
                
                xi_x_b = XI_X[np.newaxis, np.newaxis, :, :]
                xi_y_b = XI_Y[np.newaxis, np.newaxis, :, :]
                v_x_b = v_x[:, :, np.newaxis, np.newaxis]
                v_y_b = v_y[:, :, np.newaxis, np.newaxis]
                
                # h = -D (ξ_x² + ξ_y²) - i (v_x ξ_x + v_y ξ_y)
                h.values = (-self.D * (xi_x_b**2 + xi_y_b**2) - 
                           1j * (v_x_b * xi_x_b + v_y_b * xi_y_b))
                
                return h
        
        return h_func
    
    def evolve_advection_diffusion(self, u0: np.ndarray,
                                   t_span: Tuple[float, float],
                                   Nt: int,
                                   save_every: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Résout ∂_t u = D ∇²u - v·∇u
        """
        h_func = self.create_advection_diffusion_symbol()
        
        t0, tf = t_span
        dt = (tf - t0) / Nt
        
        # Stockage
        n_saved = Nt // save_every + 1
        u_history = np.zeros((n_saved,) + u0.shape, dtype=complex)  # Complex car advection
        times = np.zeros(n_saved)
        
        u = u0.copy().astype(complex)
        u_history[0] = np.real(u)
        times[0] = t0
        
        save_idx = 1
        
        print(f"Résolution advection-diffusion : t={t0} → {tf}")
        
        for n in range(Nt):
            t = t0 + n * dt
            
            # Symbole au temps t
            h = h_func(t)
            
            if self.grid.dim == 1:
                h_xi = h.values[0, :]
            else:
                h_xi = h.values[0, 0, :, :]
            
            # Évolution : u(t+dt) = exp(dt h) u(t)
            u_k = self.quantizer.fft_forward(u)
            u_k = u_k * np.exp(dt * h_xi)
            u = self.quantizer.fft_backward(u_k)
            
            # Prendre partie réelle (advection peut introduire petite partie imaginaire)
            u = np.real(u)
            
            # Conditions aux limites
            u = self.bc.apply_to_state(u)
            
            # Sauvegarde
            if (n + 1) % save_every == 0 and save_idx < n_saved:
                u_history[save_idx] = u
                times[save_idx] = t + dt
                save_idx += 1
            
            if (n + 1) % (Nt // 10) == 0:
                print(f"  t={t+dt:.4f} ({n+1}/{Nt})")
        
        print("✓ Résolution terminée")
        
        return times, np.real(u_history)

class WaveEquationSolver:
    """
    Résolution de l'équation d'onde : ∂²_t u = c² ∇²u
    
    Reformulation du premier ordre :
      ∂_t u = v
      ∂_t v = c² ∇²u
    
    Applications :
    - Ondes acoustiques
    - Ondes électromagnétiques (scalaire)
    - Vibrations de membranes
    - Ondes sismiques
    """
    
    def __init__(self, grid: Grid, bc_type: str = 'periodic',
                 wave_speed: float = 1.0):
        """
        Parameters:
        -----------
        grid : Grid
            Grille spatiale
        bc_type : str
            Conditions aux limites
        wave_speed : float
            Vitesse de propagation c
        """
        self.grid = grid
        self.bc_type = bc_type
        self.c = wave_speed
        
        self.bc = BoundaryConditions(grid, bc_type, horizon_params=None)
        self.quantizer = WeylQuantization(grid, bc_type)
    
    def create_wave_symbol(self, metric_func: Callable = None) -> Callable:
        """
        Crée le symbole de l'opérateur d'onde
        
        Cas isotrope : h = ±ic|ξ|
        Cas anisotrope : h = ±ic√(γⁱʲ ξᵢξⱼ)
        
        Parameters:
        -----------
        metric_func : Callable, optional
            (t,x) -> γ(t,x) pour propagation anisotrope
            Si None, propagation isotrope standard
        
        Returns:
        --------
        h_func : Callable
            Fonction t -> (h_plus, h_minus) pour les deux branches
        """
        if metric_func is None:
            # Propagation isotrope
            if self.grid.dim == 1:
                def h_func(t):
                    X, XI = np.meshgrid(self.grid.x, self.grid.xi, indexing='ij')
                    
                    # Symbole : ±ic|ξ|
                    h_values = self.c * np.abs(XI)
                    
                    return h_values  # On retournera ±i × h_values
            
            else:  # 2D
                def h_func(t):
                    X, Y = self.grid.x
                    XI_X, XI_Y = self.grid.xi
                    
                    xi_x_b = XI_X[np.newaxis, np.newaxis, :, :]
                    xi_y_b = XI_Y[np.newaxis, np.newaxis, :, :]
                    
                    # h = c|ξ| = c√(ξ_x² + ξ_y²)
                    h_values = self.c * np.sqrt(xi_x_b**2 + xi_y_b**2)
                    
                    return h_values
        
        else:
            # Propagation anisotrope
            if self.grid.dim == 1:
                def h_func(t):
                    X, XI = np.meshgrid(self.grid.x, self.grid.xi, indexing='ij')
                    
                    gamma = metric_func(t, self.grid.x)
                    
                    # h = c√(γ(x)ξ²)
                    h_values = self.c * np.sqrt(gamma[:, np.newaxis] * XI**2)
                    
                    return h_values
            
            else:  # 2D anisotrope
                def h_func(t):
                    X, Y = self.grid.x
                    XI_X, XI_Y = self.grid.xi
                    
                    gamma = metric_func(t, X, Y)  # (Nx, Ny, 2, 2)
                    
                    xi_x_b = XI_X[np.newaxis, np.newaxis, :, :]
                    xi_y_b = XI_Y[np.newaxis, np.newaxis, :, :]
                    gamma_b = gamma[:, :, np.newaxis, np.newaxis, :, :]
                    
                    # h = c√(γⁱʲ ξᵢξⱼ)
                    h_values = self.c * np.sqrt(
                        gamma_b[:,:,:,:,0,0] * xi_x_b**2 +
                        2 * gamma_b[:,:,:,:,0,1] * xi_x_b * xi_y_b +
                        gamma_b[:,:,:,:,1,1] * xi_y_b**2
                    )
                    
                    return h_values
        
        return h_func
    
    def evolve_wave(self, u0: np.ndarray, v0: np.ndarray,
                    h_func: Callable,
                    t_span: Tuple[float, float],
                    Nt: int,
                    save_every: int = 1,
                    damping: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Résout l'équation d'onde avec conditions initiales u(0) et ∂_t u(0)
        
        Parameters:
        -----------
        u0 : np.ndarray
            Déplacement initial u(t=0, x)
        v0 : np.ndarray
            Vitesse initiale v(t=0, x) = ∂_t u(t=0, x)
        h_func : Callable
            Fonction t -> h(t,ξ) symbole
        t_span : Tuple[float, float]
            Intervalle de temps
        Nt : int
            Nombre de pas de temps
        save_every : int
            Fréquence de sauvegarde
        damping : float
            Coefficient d'amortissement γ (pour ∂²_t u + γ∂_t u = c²∇²u)
        
        Returns:
        --------
        times : np.ndarray
        u_history : np.ndarray
            Déplacement au cours du temps
        v_history : np.ndarray
            Vitesse au cours du temps
        """
        t0, tf = t_span
        dt = (tf - t0) / Nt
        
        # Stockage
        n_saved = Nt // save_every + 1
        u_history = np.zeros((n_saved,) + u0.shape, dtype=float)
        v_history = np.zeros((n_saved,) + v0.shape, dtype=float)
        times = np.zeros(n_saved)
        
        u = u0.copy()
        v = v0.copy()
        
        u_history[0] = u
        v_history[0] = v
        times[0] = t0
        
        save_idx = 1
        
        print(f"Résolution équation d'onde : t={t0} → {tf}")
        print(f"  Vitesse c = {self.c}")
        if damping > 0:
            print(f"  Amortissement γ = {damping}")
        
        for n in range(Nt):
            t = t0 + n * dt
            
            # Symbole au temps t
            h_values = h_func(t)
            
            # Évolution par splitting symplectique (préserve l'énergie)
            # Méthode de Störmer-Verlet / leapfrog
            
            # 1. Demi-pas pour v : v → v + (dt/2) c²∇²u
            #    En Fourier : v̂ → v̂ - (dt/2) c²ξ² û
            
            u_k = self.quantizer.fft_forward(u)
            
            if self.grid.dim == 1:
                h_xi = h_values[0, :]
                laplacian = -h_xi**2 / self.c**2  # -ξ²
            else:
                h_xi = h_values[0, 0, :, :]
                laplacian = -h_xi**2 / self.c**2
            
            # Amortissement : v → v - γv dt/2
            if damping > 0:
                v = v * (1 - damping * dt / 2)
            
            # Contribution du laplacien
            v_k = self.quantizer.fft_forward(v)
            v_k = v_k + (dt / 2) * (self.c**2 * laplacian * u_k)
            v = np.real(self.quantizer.fft_backward(v_k))
            
            # 2. Pas complet pour u : u → u + dt v
            u = u + dt * v
            
            # 3. Demi-pas pour v (même chose)
            u_k = self.quantizer.fft_forward(u)
            v_k = self.quantizer.fft_forward(v)
            v_k = v_k + (dt / 2) * (self.c**2 * laplacian * u_k)
            v = np.real(self.quantizer.fft_backward(v_k))
            
            # Amortissement
            if damping > 0:
                v = v * (1 - damping * dt / 2)
            
            # Conditions aux limites
            u = self.bc.apply_to_state(u)
            v = self.bc.apply_to_state(v)
            
            # Sauvegarde
            if (n + 1) % save_every == 0 and save_idx < n_saved:
                u_history[save_idx] = u
                v_history[save_idx] = v
                times[save_idx] = t + dt
                save_idx += 1
            
            if (n + 1) % (Nt // 10) == 0:
                E_kin = np.sum(v**2) * self.grid.dx / 2
                E_pot = np.sum(u**2) * self.grid.dx / 2
                print(f"  t={t+dt:.4f} ({n+1}/{Nt}), "
                      f"E_cin={E_kin:.4e}, E_pot={E_pot:.4e}")
        
        print("✓ Résolution terminée")
        
        return times, u_history, v_history
    
    def evolve_wave_damped_alternative(self, u0: np.ndarray, v0: np.ndarray,
                                      h_func: Callable,
                                      t_span: Tuple[float, float],
                                      Nt: int,
                                      damping: float = 0.0,
                                      save_every: int = 1):
        """
        Variante avec matrice exponentielle directe
        
        Pour le système :
          ∂_t [u] = [  0     1  ] [u]   +  [     0      ] (amortissement)
              [v]   [c²∇²   -γ ] [v]      [-γv         ]
        """
        # À implémenter si besoin d'une méthode alternative
        pass


class WaveDiagnostics:
    """
    Diagnostics pour l'équation d'onde
    """
    
    @staticmethod
    def total_energy(u: np.ndarray, v: np.ndarray, 
                    c: float, grid: Grid) -> float:
        """
        Énergie totale E = E_cin + E_pot
        
        E_cin = (1/2) ∫ v² dx  (énergie cinétique)
        E_pot = (c²/2) ∫ |∇u|² dx  (énergie potentielle)
        
        En Fourier : E_pot = (c²/2) ∫ |ξ|² |û|² dξ
        """
        if grid.dim == 1:
            # Énergie cinétique
            E_kin = 0.5 * np.trapz(v**2, grid.x)
            
            # Énergie potentielle (via Fourier)
            u_k = fft(u)
            E_pot = 0.5 * c**2 * np.sum(grid.xi**2 * np.abs(u_k)**2) * grid.dxi
        
        else:  # 2D
            E_kin = 0.5 * np.sum(v**2) * grid.dx**2
            
            u_k = fft2(u)
            XI_X, XI_Y = grid.xi
            xi_sq = XI_X**2 + XI_Y**2
            E_pot = 0.5 * c**2 * np.sum(xi_sq * np.abs(u_k)**2) * grid.dxi[0] * grid.dxi[1]
        
        return E_kin + E_pot
    
    @staticmethod
    def momentum(v: np.ndarray, grid: Grid) -> float:
        """
        Quantité de mouvement P = ∫ v dx
        """
        if grid.dim == 1:
            return np.trapz(v, grid.x)
        else:
            return np.sum(v) * grid.dx**2
    
    @staticmethod
    def wave_period(c: float, L: float, mode: int = 1) -> float:
        """
        Période d'oscillation T = 2L/(n c) pour le mode n
        
        En domaine périodique de longueur L
        """
        return 2 * L / (mode * c)
    
    @staticmethod
    def dispersion_relation(c: float, xi: np.ndarray) -> np.ndarray:
        """
        Relation de dispersion ω(ξ) = c|ξ|
        
        Fréquence angulaire en fonction du nombre d'onde
        """
        return c * np.abs(xi)

class DispersiveWaveSolver(WaveEquationSolver):
    """
    Équation d'onde dispersive : ∂²_t u = c²∇²u - α∇⁴u
    
    Le terme ∇⁴u introduit de la dispersion :
    - Différentes fréquences se propagent à des vitesses différentes
    - Formation de paquets d'onde qui s'étalent
    
    Relation de dispersion : ω² = c²ξ² + αξ⁴
    """
    
    def __init__(self, grid: Grid, bc_type: str = 'periodic',
                 wave_speed: float = 1.0,
                 dispersion: float = 0.0):
        """
        Parameters:
        -----------
        dispersion : float
            Coefficient de dispersion α
        """
        super().__init__(grid, bc_type, wave_speed)
        self.alpha = dispersion
    
    def create_dispersive_symbol(self) -> Callable:
        """
        Symbole avec dispersion : h² = c²ξ² + αξ⁴
        
        d'où h = √(c²ξ² + αξ⁴)
        """
        if self.grid.dim == 1:
            def h_func(t):
                X, XI = np.meshgrid(self.grid.x, self.grid.xi, indexing='ij')
                
                # h = √(c²ξ² + αξ⁴)
                h_values = np.sqrt(self.c**2 * XI**2 + self.alpha * XI**4)
                
                return h_values
        
        else:  # 2D
            def h_func(t):
                X, Y = self.grid.x
                XI_X, XI_Y = self.grid.xi
                
                xi_x_b = XI_X[np.newaxis, np.newaxis, :, :]
                xi_y_b = XI_Y[np.newaxis, np.newaxis, :, :]
                
                xi_sq = xi_x_b**2 + xi_y_b**2
                
                # h = √(c²|ξ|² + α|ξ|⁴)
                h_values = np.sqrt(self.c**2 * xi_sq + self.alpha * xi_sq**2)
                
                return h_values
        
        return h_func
    
    def group_velocity(self, xi: np.ndarray) -> np.ndarray:
        """
        Vitesse de groupe v_g = dω/dξ
        
        Pour ω = √(c²ξ² + αξ⁴) :
        v_g = (c²ξ + 2αξ³) / √(c²ξ² + αξ⁴)
        """
        omega_sq = self.c**2 * xi**2 + self.alpha * xi**4
        omega = np.sqrt(omega_sq)
        
        v_g = (self.c**2 * xi + 2 * self.alpha * xi**3) / (omega + 1e-10)
        
        return v_g



        
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class ParallelSymbolicOperations:
    """
    Opérations symboliques parallélisées
    """
    
    @staticmethod
    def parallel_moyal_product(symbols_h: list, symbols_g: list, 
                               order: int = 2, n_workers: int = None) -> list:
        """
        Calcule h_i #_W g_i en parallèle pour plusieurs paires
        
        Utile pour évoluer plusieurs états ou calculer plusieurs observables
        """
        if n_workers is None:
            n_workers = mp.cpu_count()
        
        def compute_one(args):
            h, g = args
            return moyal_product(h, g, order=order)
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(compute_one, zip(symbols_h, symbols_g)))
        
        return results
    
    @staticmethod
    def parallel_evolution(psi0_list: list, grid: Grid, 
                          hamiltonian_func: Callable,
                          t_span: Tuple[float, float], Nt: int,
                          bc_type: str = 'periodic',
                          n_workers: int = None) -> list:
        """
        Évolue plusieurs états initiaux en parallèle
        
        Utile pour :
        - Moyenner sur plusieurs conditions initiales
        - Études de Monte Carlo
        - Exploration de paramètres
        """
        if n_workers is None:
            n_workers = min(mp.cpu_count(), len(psi0_list))
        
        def evolve_one(psi0):
            evolver = SymbolicEvolutionWithBC(grid, bc_type)
            times, psi_hist = evolver.evolve(
                psi0, hamiltonian_func, t_span, Nt, save_every=Nt
            )
            return times, psi_hist
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(evolve_one, psi0_list))
        
        return results


class VectorizedSymbolOperations:
    """
    Opérations vectorisées optimisées avec NumPy
    """
    
    @staticmethod
    def batch_gradient(symbols: list, axis: str = 'x') -> list:
        """
        Calcule les gradients de plusieurs symboles en une seule passe
        
        Plus efficace que de boucler sur chaque symbole
        """
        if not symbols:
            return []
        
        grid = symbols[0].grid
        
        # Empiler tous les symboles en un grand array
        if grid.dim == 1:
            stacked = np.stack([s.values for s in symbols], axis=0)
            # Shape: (n_symbols, n_x, n_xi)
            
            if axis == 'x':
                grads = np.gradient(stacked, grid.dx, axis=1)
            else:  # axis == 'xi'
                grads = np.gradient(stacked, grid.dxi, axis=2)
            
            # Décomposer en liste de symboles
            results = []
            for i in range(len(symbols)):
                s_grad = Symbol(grid, order=symbols[i].order)
                s_grad.values = grads[i]
                results.append(s_grad)
            
            return results
        
        else:
            # Similaire pour 2D
            raise NotImplementedError("2D batch gradient pas encore implémenté")
class QuantumObservables:
    """
    Calcul d'observables à partir des symboles et états
    """
    
    def __init__(self, grid: Grid):
        self.grid = grid
    
    def expectation_value(self, psi: np.ndarray, O_symbol: Symbol) -> complex:
        """
        Valeur moyenne ⟨ψ|Ô|ψ⟩
        
        Approximation : ⟨Ô⟩ ≈ ∫ |ψ(x)|² O(x, p_class(x)) dx
        où p_class(x) = -i ∂_x ln(ψ(x))
        
        Ou via quantification de Weyl complète
        """
        if self.grid.dim == 1:
            # Méthode simple : O(x) seulement (ordre 0 en ξ)
            if np.allclose(O_symbol.values[:, 1:], O_symbol.values[:, :-1]):
                # O ne dépend que de x
                O_x = O_symbol.values[:, 0]
                integrand = np.conj(psi) * O_x * psi
                return np.trapz(integrand, self.grid.x)
            
            else:
                # Méthode complète via Wigner
                return self._expectation_via_wigner_1d(psi, O_symbol)
        
        else:
            raise NotImplementedError("2D pas encore implémenté")
    
    def _expectation_via_wigner_1d(self, psi: np.ndarray, O_symbol: Symbol) -> complex:
        """
        ⟨Ô⟩ = ∫∫ W_ψ(x,ξ) O(x,ξ) dx dξ / (2π)
        """
        N = len(self.grid.x)
        result = 0.0
        
        for i, x in enumerate(self.grid.x):
            for k, xi in enumerate(self.grid.xi):
                # Wigner en (x, ξ)
                W_psi = 0.0
                for j, y in enumerate(self.grid.x):
                    idx_plus = (i + j//2) % N
                    idx_minus = (i - j//2) % N
                    W_psi += (np.conj(psi[idx_minus]) * psi[idx_plus] * 
                             np.exp(-1j * y * xi) * self.grid.dx)
                
                result += W_psi * O_symbol.values[i, k] * self.grid.dx * self.grid.dxi
        
        return result / (2 * np.pi)
    
    def position_expectation(self, psi: np.ndarray) -> float:
        """⟨x⟩"""
        if self.grid.dim == 1:
            density = np.abs(psi)**2
            return np.trapz(self.grid.x * density, self.grid.x)
        else:
            X, Y = self.grid.x
            density = np.abs(psi)**2
            x_mean = np.sum(X * density) * self.grid.dx**2
            y_mean = np.sum(Y * density) * self.grid.dx**2
            return x_mean, y_mean
    
    def momentum_expectation(self, psi: np.ndarray) -> float:
        """⟨p⟩ = -i⟨ψ|∂_x|ψ⟩"""
        if self.grid.dim == 1:
            dpsi_dx = np.gradient(psi, self.grid.dx)
            integrand = np.conj(psi) * (-1j * dpsi_dx)
            return np.real(np.trapz(integrand, self.grid.x))
        else:
            raise NotImplementedError("2D momentum pas encore implémenté")
    
    def energy_expectation(self, psi: np.ndarray, h_symbol: Symbol) -> complex:
        """⟨E⟩ = ⟨ψ|Ĥ|ψ⟩"""
        return self.expectation_value(psi, h_symbol)
    
    def uncertainty(self, psi: np.ndarray, O_symbol: Symbol) -> float:
        """
        Incertitude ΔO = √(⟨Ô²⟩ - ⟨Ô⟩²)
        """
        O_mean = self.expectation_value(psi, O_symbol)
        
        # Créer O²
        O2_symbol = Symbol(self.grid, order=2*O_symbol.order)
        O2_symbol.values = O_symbol.values**2
        
        O2_mean = self.expectation_value(psi, O2_symbol)
        
        return np.sqrt(np.abs(O2_mean - O_mean**2))
    
    def von_neumann_entropy(self, psi: np.ndarray, 
                           subsystem_size: int = None) -> float:
        """
        Entropie de von Neumann d'un sous-système
        
        S = -Tr(ρ_A ln ρ_A)
        
        Nécessite de tracer sur partie du système
        """
        if self.grid.dim == 1 and subsystem_size is not None:
            # Matrice densité réduite via SVD
            # Reshape ψ en matrice
            N = len(psi)
            if N % subsystem_size != 0:
                raise ValueError("subsystem_size doit diviser N")
            
            M = N // subsystem_size
            psi_matrix = psi.reshape(subsystem_size, M)
            
            # SVD
            U, s, Vh = np.linalg.svd(psi_matrix, full_matrices=False)
            
            # Valeurs propres de ρ_A
            eigenvalues = s**2
            eigenvalues = eigenvalues[eigenvalues > 1e-14]  # Filtrer numériquement zéro
            
            # S = -Σ λ ln λ
            entropy = -np.sum(eigenvalues * np.log(eigenvalues))
            
            return entropy
        
        else:
            raise NotImplementedError("Entropie 2D ou sans sous-système")


class SymbolicDiagnostics:
    """
    Diagnostics sur les symboles eux-mêmes
    """
    
    @staticmethod
    def check_ellipticity(h: Symbol, threshold: float = 1e-10) -> Tuple[bool, float]:
        """
        Vérifie si h est elliptique : |h(x,ξ)| ≥ c⟨ξ⟩^m pour |ξ| grand
        
        Returns:
        --------
        is_elliptic : bool
        min_value : float
            Valeur minimale de |h|
        """
        h_abs = np.abs(h.values)
        min_val = np.min(h_abs)
        
        is_elliptic = min_val > threshold
        
        return is_elliptic, min_val
    
    @staticmethod
    def compute_classical_flow(h: Symbol, 
                              x0: float, xi0: float,
                              t_span: float, Nt: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule une bicaractéristique (trajectoire classique)
        
        Équations de Hamilton :
        dx/dt = ∂h/∂ξ
        dξ/dt = -∂h/∂x
        
        Returns:
        --------
        x_traj, xi_traj : np.ndarray
            Trajectoires dans l'espace des phases
        """
        from scipy.integrate import solve_ivp
        
        # Interpoler le symbole
        if h.grid.dim == 1:
            from scipy.interpolate import RectBivariateSpline
            
            h_interp = RectBivariateSpline(
                h.grid.x, h.grid.xi, h.values.real, kx=3, ky=3
            )
            
            def hamilton_eqs(t, y):
                x, xi = y
                
                # ∂h/∂ξ
                dh_dxi = h_interp(x, xi, dy=1, grid=False)[0]
                
                # ∂h/∂x
                dh_dx = h_interp(x, xi, dx=1, grid=False)[0]
                
                return [dh_dxi, -dh_dx]
            
            sol = solve_ivp(
                hamilton_eqs,
                [0, t_span],
                [x0, xi0],
                t_eval=np.linspace(0, t_span, Nt),
                method='RK45'
            )
            
            return sol.y[0], sol.y[1]
        
        else:
            raise NotImplementedError("Bicaractéristiques 2D pas encore implémentées")
    
    @staticmethod
    def symbol_norm(h: Symbol, kind: str = 'L2') -> float:
        """
        Norme du symbole
        
        Parameters:
        -----------
        kind : str
            'L2' : ∫∫ |h(x,ξ)|² dx dξ
            'Linf' : max |h(x,ξ)|
            'L1' : ∫∫ |h(x,ξ)| dx dξ
        """
        if kind == 'L2':
            if h.grid.dim == 1:
                return np.sqrt(np.sum(np.abs(h.values)**2) * h.grid.dx * h.grid.dxi)
            else:
                return np.sqrt(np.sum(np.abs(h.values)**2) * 
                             h.grid.dx**2 * h.grid.dxi[0] * h.grid.dxi[1])
        
        elif kind == 'Linf':
            return np.max(np.abs(h.values))
        
        elif kind == 'L1':
            if h.grid.dim == 1:
                return np.sum(np.abs(h.values)) * h.grid.dx * h.grid.dxi
            else:
                return np.sum(np.abs(h.values)) * h.grid.dx**2 * h.grid.dxi[0] * h.grid.dxi[1]
        
        else:
            raise ValueError(f"Norme '{kind}' non reconnue")




# ============================================
# BENCHMARKS DE PERFORMANCE
# ============================================
def benchmark_performance():
    """
    Mesure les performances des différentes opérations
    """
    import time
    
    print("\n" + "="*60)
    print("BENCHMARKS DE PERFORMANCE")
    print("="*60 + "\n")
    
    sizes_1d = [32, 64, 128, 256, 512]
    sizes_2d = [16, 32, 64]
    
    results = {
        '1D': {},
        '2D': {}
    }
    
    # Benchmarks 1D
    print("Benchmarks 1D :")
    print("-" * 60)
    print(f"{'Taille':<10} {'Symbole':<12} {'Moyal':<12} {'Inverse':<12} {'Évolution (10 pas)':<20}")
    print("-" * 60)
    
    for N in sizes_1d:
        grid = Grid.create_1d(N=N, L=10.0, periodic=True)
        
        # Temps création symbole
        t0 = time.time()
        h = Symbol.from_function(grid, lambda t,x,xi: x**2 + xi**2, order=2, t=0.0)
        t_symbol = (time.time() - t0) * 1000
        
        # Temps produit de Moyal
        g = Symbol.from_function(grid, lambda t,x,xi: x*xi, order=1, t=0.0)
        t0 = time.time()
        hg = moyal_product(h, g, order=2)
        t_moyal = (time.time() - t0) * 1000
        
        # Temps inverse
        h_ell = Symbol.from_function(grid, lambda t,x,xi: 1.0 + xi**2, order=2, t=0.0)
        t0 = time.time()
        h_inv = symbolic_inverse(h_ell, order=2)
        t_inverse = (time.time() - t0) * 1000
        
        # Temps évolution
        def metric(t, x):
            return np.ones_like(x)
        
        evolver = SymbolicEvolutionWithBC(grid, bc_type='periodic')
        h_func = evolver.create_hamiltonian_relativistic(metric)
        
        x = grid.x
        psi0 = np.exp(-(x - 5.0)**2 / 2)
        psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * grid.dx)
        
        t0 = time.time()
        times, psi_hist = evolver.evolve(
            psi0, h_func, t_span=(0.0, 0.1), Nt=10, save_every=1
        )
        t_evolution = (time.time() - t0) * 1000
        
        results['1D'][N] = {
            'symbol': t_symbol,
            'moyal': t_moyal,
            'inverse': t_inverse,
            'evolution': t_evolution
        }
        
        print(f"{N:<10} {t_symbol:>8.2f} ms  {t_moyal:>8.2f} ms  "
              f"{t_inverse:>8.2f} ms  {t_evolution:>15.2f} ms")
    
    print()
    
    # Benchmarks 2D
    print("\nBenchmarks 2D :")
    print("-" * 60)
    print(f"{'Taille':<10} {'Symbole':<12} {'Moyal':<12} {'Évolution (10 pas)':<20}")
    print("-" * 60)
    
    for N in sizes_2d:
        grid = Grid.create_2d(Nx=N, Ny=N, Lx=10.0, Ly=10.0, periodic=True)
        
        # Temps création symbole
        t0 = time.time()
        def h_2d(t, x, y, xi_x, xi_y):
            return x**2 + y**2 + xi_x**2 + xi_y**2
        h = Symbol.from_function(grid, h_2d, order=2, t=0.0)
        t_symbol = (time.time() - t0) * 1000
        
        # Temps produit de Moyal
        def g_2d(t, x, y, xi_x, xi_y):
            return x * xi_x + y * xi_y
        g = Symbol.from_function(grid, g_2d, order=1, t=0.0)
        
        t0 = time.time()
        hg = moyal_product(h, g, order=1)  # Ordre 1 seulement pour 2D
        t_moyal = (time.time() - t0) * 1000
        
        # Temps évolution
        def metric_2d(t, X, Y):
            gamma = np.zeros(X.shape + (2, 2))
            gamma[:, :, 0, 0] = 1.0
            gamma[:, :, 1, 1] = 1.0
            return gamma
        
        evolver = SymbolicEvolutionWithBC(grid, bc_type='periodic')
        h_func = evolver.create_hamiltonian_relativistic(metric_2d)
        
        X, Y = grid.x
        psi0 = np.exp(-((X - 5.0)**2 + (Y - 5.0)**2) / 2)
        psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * grid.dx**2)
        
        t0 = time.time()
        times, psi_hist = evolver.evolve(
            psi0, h_func, t_span=(0.0, 0.1), Nt=10, save_every=1
        )
        t_evolution = (time.time() - t0) * 1000
        
        results['2D'][N] = {
            'symbol': t_symbol,
            'moyal': t_moyal,
            'evolution': t_evolution
        }
        
        print(f"{N}×{N:<6} {t_symbol:>8.2f} ms  {t_moyal:>8.2f} ms  "
              f"{t_evolution:>15.2f} ms")
    
    print("\n" + "="*60)
    print("BENCHMARKS TERMINÉS")
    print("="*60 + "\n")
    
    # Analyse du scaling
    print("Analyse du scaling :")
    print("-" * 40)
    
    # 1D
    N_ref = 128
    for N in [256, 512]:
        if N in results['1D']:
            ratio_expected = (N / N_ref)**2  # O(N²) pour FFT
            ratio_actual = results['1D'][N]['evolution'] / results['1D'][N_ref]['evolution']
            print(f"1D: N={N_ref}→{N}, ratio attendu: {ratio_expected:.1f}×, "
                  f"observé: {ratio_actual:.1f}×")
    
    # 2D
    N_ref = 64
    for N in [128, 256]:
        if N in results['2D']:
            ratio_expected = (N / N_ref)**4  # O(N⁴) pour FFT 2D
            ratio_actual = results['2D'][N]['evolution'] / results['2D'][N_ref]['evolution']
            print(f"2D: N={N_ref}→{N}, ratio attendu: {ratio_expected:.1f}×, "
                  f"observé: {ratio_actual:.1f}×")
    
    print()
    
    return results

