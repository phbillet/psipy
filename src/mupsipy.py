"""
Opérateurs pseudodifférentiels : cœur minimal et générique
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fft2, ifft2, fftfreq, rfft, rfftfreq
from scipy.ndimage import convolve
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import multiprocessing as mp
from functools import partial


# ============================================================================
# GRILLE
# ============================================================================

@dataclass
class Grid:
    """Grille spatiale (x) et duale (ξ)"""
    x: np.ndarray       # Points spatiaux
    xi: np.ndarray      # Points de Fourier
    dx: float           # Pas spatial
    dxi: float          # Pas Fourier
    dim: int            # 1 ou 2
    
    @classmethod
    def create_1d(cls, N: int, L: float, periodic: bool = True):
        """Grille 1D"""
        x = np.linspace(0, L, N, endpoint=not periodic)
        dx = L / N
        
        if periodic:
            xi = 2 * np.pi * fftfreq(N, dx)
        else:
            # Dirichlet : modes sinus
            xi = np.pi * np.arange(1, N+1) / L
        
        dxi = xi[1] - xi[0] if len(xi) > 1 else 1.0
        return cls(x, xi, dx, dxi, dim=1)
    
    @classmethod
    def create_2d(cls, Nx: int, Ny: int, Lx: float, Ly: float, 
                  periodic: bool = True):
        """Grille 2D"""
        x_1d = np.linspace(0, Lx, Nx, endpoint=not periodic)
        y_1d = np.linspace(0, Ly, Ny, endpoint=not periodic)
        dx = Lx / Nx
        dy = Ly / Ny
        
        if periodic:
            xi_x = 2 * np.pi * fftfreq(Nx, dx)
            xi_y = 2 * np.pi * fftfreq(Ny, dy)
        else:
            xi_x = np.pi * np.arange(1, Nx+1) / Lx
            xi_y = np.pi * np.arange(1, Ny+1) / Ly
        
        X, Y = np.meshgrid(x_1d, y_1d, indexing='ij')
        XI_X, XI_Y = np.meshgrid(xi_x, xi_y, indexing='ij')
        
        return cls(
            x=(X, Y),
            xi=(XI_X, XI_Y),
            dx=dx,
            dxi=(xi_x[1]-xi_x[0], xi_y[1]-xi_y[0]),
            dim=2
        )


# ============================================================================
# SYMBOLE
# ============================================================================

class Symbol:
    """
    Symbole h(t, x, ξ) d'un opérateur pseudodifférentiel
    
    Stockage : valeurs sur grille (x, ξ)
    """
    
    def __init__(self, grid: Grid, order: int = 0):
        self.grid = grid
        self.order = order
        self.hbar = 1.0
        
        # Valeurs sur grille phase-space
        if grid.dim == 1:
            self.values = np.zeros((len(grid.x), len(grid.xi)), dtype=complex)
        else:
            Nx, Ny = grid.x[0].shape
            N_xi_x, N_xi_y = grid.xi[0].shape
            self.values = np.zeros((Nx, Ny, N_xi_x, N_xi_y), dtype=complex)
    
    @classmethod
    def from_function(cls, grid: Grid, func: Callable, order: int = 0, 
                     t: float = 0.0):
        """Crée symbole depuis fonction h(t, x, ξ)"""
        symbol = cls(grid, order)
        
        if grid.dim == 1:
            X, XI = np.meshgrid(grid.x, grid.xi, indexing='ij')
            symbol.values = func(t, X, XI)
        else:
            X, Y = grid.x
            XI_X, XI_Y = grid.xi
            shape = X.shape + XI_X.shape
            x_b = X[:, :, np.newaxis, np.newaxis]
            y_b = Y[:, :, np.newaxis, np.newaxis]
            xi_x_b = XI_X[np.newaxis, np.newaxis, :, :]
            xi_y_b = XI_Y[np.newaxis, np.newaxis, :, :]
            symbol.values = func(t, x_b, y_b, xi_x_b, xi_y_b)
        
        return symbol
    
    def gradient_x(self):
        """∂h/∂x"""
        if self.grid.dim == 1:
            return np.gradient(self.values, self.grid.dx, axis=0)
        else:
            grad_x = np.gradient(self.values, self.grid.dx, axis=0)
            grad_y = np.gradient(self.values, self.grid.dx, axis=1)
            return grad_x, grad_y
    
    def gradient_xi(self):
        """∂h/∂ξ"""
        if self.grid.dim == 1:
            return np.gradient(self.values, self.grid.dxi, axis=1)
        else:
            grad_xi_x = np.gradient(self.values, self.grid.dxi[0], axis=2)
            grad_xi_y = np.gradient(self.values, self.grid.dxi[1], axis=3)
            return grad_xi_x, grad_xi_y


# ============================================================================
# OPÉRATIONS SYMBOLIQUES
# ============================================================================

def moyal_product(h: Symbol, g: Symbol, order: int = 1) -> Symbol:
    """
    Produit de Moyal : h #_W g
    
    order=0 : h·g
    order=1 : h·g + (ℏ/2i){h,g}
    order=2 : + corrections O(ℏ²)
    """
    assert h.grid == g.grid
    result = Symbol(h.grid, order=h.order + g.order)
    hbar = h.hbar
    
    # Ordre 0
    result.values = (h.values * g.values).astype(complex)
    
    if order >= 1:
        # Crochet de Poisson
        if h.grid.dim == 1:
            dh_dx = h.gradient_x()
            dh_dxi = h.gradient_xi()
            dg_dx = g.gradient_x()
            dg_dxi = g.gradient_xi()
            poisson = dh_dxi * dg_dx - dh_dx * dg_dxi
        else:
            dh_dx, dh_dy = h.gradient_x()
            dg_dx, dg_dy = g.gradient_x()
            dh_dxi_x, dh_dxi_y = h.gradient_xi()
            dg_dxi_x, dg_dxi_y = g.gradient_xi()
            poisson = (dh_dxi_x * dg_dx + dh_dxi_y * dg_dy - 
                      dh_dx * dg_dxi_x - dh_dy * dg_dxi_y)
        
        result.values += (hbar / (2j)) * poisson
    
    if order >= 2:
        # Corrections O(ℏ²) - simplifiées
        if h.grid.dim == 1:
            d2h_dxi2 = np.gradient(dh_dxi, h.grid.dxi, axis=1)
            d2g_dx2 = np.gradient(dg_dx, g.grid.dx, axis=0)
            correction2 = -(hbar**2 / 8) * d2h_dxi2 * d2g_dx2
            result.values += correction2
    
    return result


def poisson_bracket(h: Symbol, g: Symbol) -> Symbol:
    """Crochet de Poisson {h,g} = ∂_ξh·∂_xg - ∂_xh·∂_ξg"""
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
    """Symbole du commutateur [Ĥ,Ĝ] = iℏ{h,g} + O(ℏ²)"""
    if order == 1:
        pb = poisson_bracket(h, g)
        result = Symbol(h.grid, order=h.order + g.order - 1)
        result.values = 1j * h.hbar * pb.values
        return result
    else:
        hg = moyal_product(h, g, order=order)
        gh = moyal_product(g, h, order=order)
        result = Symbol(h.grid, order=h.order + g.order)
        result.values = hg.values - gh.values
        return result


def symbolic_inverse(h: Symbol, order: int = 2, 
                    regularization: float = 1e-12) -> Symbol:
    """
    Inverse symbolique : symbole de Ĥ^{-1}
    
    b₀ = 1/h
    b_{j+1} = -b₀ #_W (h #_W bⱼ - δⱼ₀)
    """
    b0 = Symbol(h.grid, order=-h.order)
    h_reg = h.values + 1j * regularization
    b0.values = 1.0 / h_reg
    
    if order == 0:
        return b0
    
    b_terms = [b0]
    for j in range(order):
        h_bj = moyal_product(h, b_terms[j], order=2)
        if j == 0:
            h_bj.values -= 1.0
        b_next = moyal_product(b0, h_bj, order=2)
        b_next.values *= -1.0
        b_terms.append(b_next)
    
    result = Symbol(h.grid, order=-h.order)
    result.values = sum(b.values for b in b_terms)
    return result


# ============================================================================
# CALCULS PARALLÉLISÉS
# ============================================================================

def _eval_symbol_wrapper(args):
    """Wrapper pour parallélisation"""
    func, t, grid, x_chunk, xi_chunk = args
    if grid.dim == 1:
        X, XI = np.meshgrid(x_chunk, xi_chunk, indexing='ij')
        return func(t, X, XI)
    else:
        # 2D : plus complexe, à adapter si nécessaire
        return None


def parallel_symbol_eval(func: Callable, t: float, grid: Grid, 
                        n_workers: int = None) -> np.ndarray:
    """
    Évalue un symbole en parallèle
    
    Divise la grille en chunks et calcule en parallèle
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    if grid.dim == 1:
        # Diviser en chunks
        N = len(grid.x)
        chunk_size = N // n_workers
        chunks = []
        
        for i in range(n_workers):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_workers - 1 else N
            x_chunk = grid.x[start:end]
            chunks.append((func, t, grid, x_chunk, grid.xi))
        
        with mp.Pool(n_workers) as pool:
            results = pool.map(_eval_symbol_wrapper, chunks)
        
        return np.concatenate(results, axis=0)
    else:
        # 2D : évaluation directe (ou à paralléliser différemment)
        X, Y = grid.x
        XI_X, XI_Y = grid.xi
        shape = X.shape + XI_X.shape
        x_b = X[:, :, np.newaxis, np.newaxis]
        y_b = Y[:, :, np.newaxis, np.newaxis]
        xi_x_b = XI_X[np.newaxis, np.newaxis, :, :]
        xi_y_b = XI_Y[np.newaxis, np.newaxis, :, :]
        return func(t, x_b, y_b, xi_x_b, xi_y_b)


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def check_ellipticity(h: Symbol, threshold: float = 1e-10) -> Tuple[bool, float]:
    """Vérifie si h est elliptique"""
    h_abs = np.abs(h.values)
    min_val = np.min(h_abs)
    is_elliptic = min_val > threshold
    return is_elliptic, min_val


def symbol_norm(h: Symbol, kind: str = 'L2') -> float:
    """Norme du symbole"""
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

"""
Évolution temporelle générique
"""

# ============================================================================
# ÉVOLUTION GÉNÉRIQUE
# ============================================================================

class Evolver:
    """
    Évolution générique pour toute équation de la forme :
    
    i∂_t ψ = Ĥψ  (Schrödinger)
    ∂_t u = Ĥu    (Chaleur, Réaction-Diffusion, etc.)
    
    avec Ĥ défini par son symbole h(t,x,ξ)
    """
    
    def __init__(self, grid: Grid, bc_type: str = 'periodic'):
        """
        Parameters:
        -----------
        grid : Grid
        bc_type : 'periodic' ou 'dirichlet'
        """
        self.grid = grid
        self.bc_type = bc_type
        
        # Pré-calculer transformées
        if grid.dim == 1:
            self.fft_forward = fft if bc_type == 'periodic' else self._dst
            self.fft_backward = ifft if bc_type == 'periodic' else self._idst
        else:
            self.fft_forward = fft2 if bc_type == 'periodic' else self._dst2d
            self.fft_backward = ifft2 if bc_type == 'periodic' else self._idst2d
    
    def _dst(self, x):
        """Discrete Sine Transform (Dirichlet)"""
        from scipy.fftpack import dst
        return dst(x, type=1)
    
    def _idst(self, x):
        """Inverse DST"""
        from scipy.fftpack import idst
        return idst(x, type=1) / (2 * (len(x) + 1))
    
    def _dst2d(self, x):
        """DST 2D"""
        from scipy.fftpack import dst
        return dst(dst(x, type=1, axis=0), type=1, axis=1)
    
    def _idst2d(self, x):
        """Inverse DST 2D"""
        from scipy.fftpack import idst
        N, M = x.shape
        return idst(idst(x, type=1, axis=0), type=1, axis=1) / (4*(N+1)*(M+1))
    
    def apply_bc(self, psi: np.ndarray) -> np.ndarray:
        """Applique conditions aux limites"""
        if self.bc_type == 'dirichlet':
            psi = psi.copy()
            if self.grid.dim == 1:
                psi[0] = 0.0
                psi[-1] = 0.0
            else:
                psi[0, :] = 0.0
                psi[-1, :] = 0.0
                psi[:, 0] = 0.0
                psi[:, -1] = 0.0
        return psi
    
    def evolve_splitting(self, 
                        psi0: np.ndarray,
                        T_symbol_func: Callable,  # t -> Symbol de T(ξ)
                        V_symbol_func: Callable,  # t -> Symbol de V(x)
                        t_span: Tuple[float, float],
                        Nt: int,
                        imaginary: bool = True,  # True=Schrödinger, False=Chaleur
                        save_every: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Évolution par splitting de Strang : exp(dt(T+V)) ≈ exp(dt V/2) exp(dt T) exp(dt V/2)
        
        Parameters:
        -----------
        imaginary : bool
            True  → exp(-i dt H) pour Schrödinger
            False → exp(dt H) pour chaleur/diffusion
        """
        t0, tf = t_span
        dt = (tf - t0) / Nt
        
        factor = -1j if imaginary else 1.0
        
        # Stockage
        n_saved = Nt // save_every + 1
        psi_history = np.zeros((n_saved,) + psi0.shape, dtype=complex)
        times = np.zeros(n_saved)
        
        psi = psi0.copy().astype(complex)
        psi_history[0] = psi
        times[0] = t0
        
        save_idx = 1
        
        for n in range(Nt):
            t = t0 + n * dt
            
            # Obtenir symboles
            T = T_symbol_func(t)
            V = V_symbol_func(t)
            
            # Extraire valeurs
            if self.grid.dim == 1:
                T_xi = T.values[0, :]
                V_x = V.values[:, 0]
            else:
                T_xi = T.values[0, 0, :, :]
                V_x = V.values[:, :, 0, 0]
            
            # Splitting de Strang
            # 1. Demi-pas V
            psi = psi * np.exp(factor * V_x * dt / 2)
            
            # 2. Pas complet T (en Fourier)
            psi_k = self.fft_forward(psi)
            psi_k = psi_k * np.exp(factor * T_xi * dt)
            psi = self.fft_backward(psi_k)
            
            # 3. Demi-pas V
            psi = psi * np.exp(factor * V_x * dt / 2)
            
            # Conditions aux limites
            psi = self.apply_bc(psi)
            
            # Sauvegarde
            if (n + 1) % save_every == 0 and save_idx < n_saved:
                psi_history[save_idx] = psi
                times[save_idx] = t + dt
                save_idx += 1
        
        return times, psi_history
    
    def evolve_wave(self,
                   u0: np.ndarray,
                   v0: np.ndarray,
                   h_symbol_func: Callable,  # t -> Symbol de c²|ξ|²
                   t_span: Tuple[float, float],
                   Nt: int,
                   damping: float = 0.0,
                   save_every: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Équation d'onde : ∂²_t u = c²∇²u (avec amortissement optionnel)
        
        Reformulation : ∂_t u = v, ∂_t v = c²∇²u - γv
        Méthode : Störmer-Verlet (symplectique)
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
        
        for n in range(Nt):
            t = t0 + n * dt
            
            # Symbole h = c²|ξ|²
            h = h_symbol_func(t)
            
            if self.grid.dim == 1:
                h_xi = h.values[0, :]
                laplacian = -h_xi / (h_xi[1]**0.5)**2  # Approximation : -ξ²
            else:
                h_xi = h.values[0, 0, :, :]
                # Pour onde: h = c²(ξ_x² + ξ_y²), donc laplacian ≈ -h/c²
                c_sq = np.mean(h_xi) / (self.grid.dxi[0]**2 + self.grid.dxi[1]**2)
                laplacian = -h_xi / c_sq
            
            # Störmer-Verlet
            # 1. v^{n+1/2} = v^n + (dt/2) L[u^n] - (γ dt/2) v^n
            if damping > 0:
                v = v * (1 - damping * dt / 2)
            
            u_k = self.fft_forward(u)
            v_k = self.fft_forward(v)
            v_k = v_k + (dt / 2) * laplacian * u_k
            v = np.real(self.fft_backward(v_k))
            
            # 2. u^{n+1} = u^n + dt v^{n+1/2}
            u = u + dt * v
            
            # 3. v^{n+1} = v^{n+1/2} + (dt/2) L[u^{n+1}] - (γ dt/2) v^{n+1/2}
            u_k = self.fft_forward(u)
            v_k = self.fft_forward(v)
            v_k = v_k + (dt / 2) * laplacian * u_k
            v = np.real(self.fft_backward(v_k))
            
            if damping > 0:
                v = v * (1 - damping * dt / 2)
            
            # BC
            u = self.apply_bc(u)
            v = self.apply_bc(v)
            
            # Sauvegarde
            if (n + 1) % save_every == 0 and save_idx < n_saved:
                u_history[save_idx] = u
                v_history[save_idx] = v
                times[save_idx] = t + dt
                save_idx += 1
        
        return times, u_history, v_history


# ============================================================================
# HELPERS : DÉCOMPOSITION T+V
# ============================================================================

def decompose_hamiltonian(h: Symbol) -> Tuple[Symbol, Symbol]:
    """
    Décompose h(x,ξ) ≈ T(ξ) + V(x)
    
    T(ξ) = moyenne sur x
    V(x) = moyenne sur ξ
    """
    grid = h.grid
    T = Symbol(grid, order=h.order)
    V = Symbol(grid, order=0)
    
    if grid.dim == 1:
        T_values = np.mean(h.values, axis=0)
        V_values = np.mean(h.values, axis=1)
        T.values = T_values[np.newaxis, :]
        V.values = V_values[:, np.newaxis]
    else:
        T_values = np.mean(h.values, axis=(0, 1))
        V_values = np.mean(h.values, axis=(2, 3))
        T.values = T_values[np.newaxis, np.newaxis, :, :]
        V.values = V_values[:, :, np.newaxis, np.newaxis]
    
    return T, V