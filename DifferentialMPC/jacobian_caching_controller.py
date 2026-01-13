import torch
from torch import Tensor
from typing import Callable, Optional, Tuple, Dict, List, Union
import time
import numpy as np
from collections import OrderedDict
from .controller import DifferentiableMPCController, GradMethod
from .cost import GeneralQuadCost


class JacobianCache:
    """
    Intelligent cache per Jacobian matrices with LRU eviction and validity checking.
    """
    
    def __init__(
        self, 
        max_entries: int = 100,
        validity_threshold: float = 1e-2,
        enable_temporal_coherence: bool = True,
        cache_compression: bool = False,
        device: str = "cuda:0"
    ):
        self.max_entries = max_entries
        self.validity_threshold = validity_threshold
        self.enable_temporal_coherence = enable_temporal_coherence
        self.cache_compression = cache_compression
        self.device = device
        
        # Cache storage: key -> (A, B, x_center, u_center, timestamp, hit_count)
        self.cache = OrderedDict()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'evictions': 0,
            'total_queries': 0
        }
        
        # Temporal coherence tracking
        self.last_states = None
        self.last_controls = None
        
    def _create_cache_key(self, x: Tensor, u: Tensor, t: int) -> str:
        """Create cache key from state, control, and timestep"""
        # Use rounded values for numerical stability in key generation
        x_rounded = torch.round(x * 1000) / 1000  # 3 decimal precision
        u_rounded = torch.round(u * 1000) / 1000
        
        # Efficient hash-like key generation
        x_hash = torch.sum(x_rounded * torch.arange(1, x.numel() + 1, device=x.device)).item()
        u_hash = torch.sum(u_rounded * torch.arange(1, u.numel() + 1, device=u.device)).item()
        
        return f"{t}_{x_hash:.0f}_{u_hash:.0f}"
    
    def _is_valid(self, x: Tensor, u: Tensor, x_center: Tensor, u_center: Tensor) -> bool:
        """Check if cached entry is still valid for given state/control"""
        
        # State deviation check
        state_dev = torch.norm(x - x_center)
        control_dev = torch.norm(u - u_center)
        
        # Validity based on deviations
        return (state_dev < self.validity_threshold and 
                control_dev < self.validity_threshold)
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if len(self.cache) >= self.max_entries:
            # Remove oldest entry (LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats['evictions'] += 1
    
    def get(self, x: Tensor, u: Tensor, t: int) -> Optional[Tuple[Tensor, Tensor]]:
        """
        Get cached Jacobian matrices if available and valid.
        
        Args:
            x: State [nx] or [B, nx]  
            u: Control [nu] or [B, nu]
            t: Timestep
            
        Returns:
            (A, B) matrices if cache hit, None if cache miss
        """
        self.stats['total_queries'] += 1
        
        # Handle batch dimension
        if x.ndim > 1:
            # For batch, check each element and return first valid match
            for b in range(x.shape[0]):
                result = self.get(x[b], u[b], t)
                if result is not None:
                    A, B = result
                    # Expand to batch dimension
                    return A.unsqueeze(0).expand(x.shape[0], -1, -1), B.unsqueeze(0).expand(x.shape[0], -1, -1)
            return None
        
        # Single element case
        key = self._create_cache_key(x, u, t)
        
        # Check if key exists
        if key not in self.cache:
            self.stats['misses'] += 1
            return None
        
        # Get cached entry
        A, B, x_center, u_center, timestamp, hit_count = self.cache[key]
        
        # Validity check
        if not self._is_valid(x, u, x_center, u_center):
            # Invalid entry - remove and report miss
            del self.cache[key]
            self.stats['invalidations'] += 1
            self.stats['misses'] += 1
            return None
        
        # Valid hit - update LRU and stats
        self.cache[key] = (A, B, x_center, u_center, time.time(), hit_count + 1)
        self.cache.move_to_end(key)  # Move to end (most recent)
        self.stats['hits'] += 1
        
        return A, B
    
    def put(self, x: Tensor, u: Tensor, t: int, A: Tensor, B: Tensor):
        """
        Store Jacobian matrices in cache.
        
        Args:
            x: State [nx] or [B, nx]
            u: Control [nu] or [B, nu] 
            t: Timestep
            A: State Jacobian [nx, nx] or [B, nx, nx]
            B: Control Jacobian [nx, nu] or [B, nx, nu]
        """
        # Handle batch dimension
        if x.ndim > 1:
            # Store each batch element separately
            for b in range(x.shape[0]):
                self.put(x[b], u[b], t, A[b], B[b])
            return
        
        # Single element case
        key = self._create_cache_key(x, u, t)
        
        # Evict if necessary
        self._evict_lru()
        
        # Store with compression if enabled
        if self.cache_compression:
            # Simple compression: store as float32 if input is float64
            A_store = A.float() if A.dtype == torch.float64 else A.clone()
            B_store = B.float() if B.dtype == torch.float64 else B.clone()
        else:
            A_store = A.clone().detach()
            B_store = B.clone().detach()
        
        # Store entry
        self.cache[key] = (
            A_store, B_store, 
            x.clone().detach(), u.clone().detach(),
            time.time(), 0  # timestamp, hit_count
        )
    
    def update_temporal_coherence(self, X: Tensor, U: Tensor):
        """Update temporal coherence tracking for adaptive caching"""
        if self.enable_temporal_coherence:
            self.last_states = X.detach().clone() if X is not None else None
            self.last_controls = U.detach().clone() if U is not None else None
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / max(1, total)
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_entries
        }
    
    def clear(self):
        """Clear cache and reset statistics"""
        self.cache.clear()
        self.stats = {
            'hits': 0,
            'misses': 0, 
            'invalidations': 0,
            'evictions': 0,
            'total_queries': 0
        }


class JacobianCachingController(DifferentiableMPCController):
    """
    Jacobian Caching Controller per performance optimization con linearizzazioni costose.
    
    Caching Philosophy:
    - Cache intelligente delle linearizzazioni più costose computazionalmente
    - Validity checking basato su deviazioni state/control
    - Adaptive cache sizing basato su problem complexity
    - Temporal coherence per trajectory optimization
    """
    
    def __init__(
            self,
            f_dyn: Callable,
            total_time: float,
            step_size: float,
            horizon: int,
            cost_module: torch.nn.Module,
            # Jacobian caching parameters
            enable_caching: bool = True,
            cache_max_entries: int = 200,
            cache_validity_threshold: float = 1e-2,
            enable_temporal_coherence: bool = True,
            cache_compression: bool = True,
            adaptive_cache_sizing: bool = True,
            # Standard controller parameters
            u_min: Optional[torch.Tensor] = None,
            u_max: Optional[torch.Tensor] = None,
            reg_eps: float = 1e-6,
            device: str = "cuda:0",
            grad_method: GradMethod | str = GradMethod.ANALYTIC,
            f_dyn_jac: Optional[Callable] = None,
            max_iter: int = 50,
            verbose: int = 0
    ):
        # Initialize base controller
        super().__init__(
            f_dyn=f_dyn,
            total_time=total_time,
            step_size=step_size,
            horizon=horizon,
            cost_module=cost_module,
            u_min=u_min,
            u_max=u_max,
            reg_eps=reg_eps,
            device=device,
            grad_method=grad_method,
            f_dyn_jac=f_dyn_jac,
            max_iter=max_iter,
            verbose=verbose
        )
        
        # Jacobian caching configuration
        self.enable_caching = enable_caching
        self.cache_validity_threshold = cache_validity_threshold
        self.adaptive_cache_sizing = adaptive_cache_sizing
        
        # Adaptive cache size based on problem complexity
        if adaptive_cache_sizing:
            complexity_factor = (horizon * (self.nx + self.nu)) / 100
            adaptive_max_entries = int(cache_max_entries * (1 + complexity_factor * 0.1))
            cache_max_entries = min(adaptive_max_entries, cache_max_entries * 3)
        
        # Initialize Jacobian cache
        if self.enable_caching:
            self.jacobian_cache = JacobianCache(
                max_entries=cache_max_entries,
                validity_threshold=cache_validity_threshold,
                enable_temporal_coherence=enable_temporal_coherence,
                cache_compression=cache_compression,
                device=device
            )
        else:
            self.jacobian_cache = None
        
        # Performance monitoring
        self.linearization_times = []
        self.cache_performance = {
            'total_linearizations': 0,
            'cached_linearizations': 0,
            'time_saved': 0.0
        }
        
        if self.verbose > 0:
            print(f"Jacobian Caching Setup:")
            print(f"  Caching enabled: {self.enable_caching}")
            if self.enable_caching:
                print(f"  Cache max entries: {cache_max_entries}")
                print(f"  Validity threshold: {cache_validity_threshold}")
                print(f"  Temporal coherence: {enable_temporal_coherence}")
                print(f"  Cache compression: {cache_compression}")
    
    def linearize_dynamics(self, X: Tensor, U: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Override linearize_dynamics per utilizzare intelligent Jacobian caching.
        
        Args:
            X: State trajectories [B, T+1, nx] 
            U: Control trajectories [B, T, nu]
            
        Returns:
            A: State Jacobians [B, T, nx, nx]
            B: Control Jacobians [B, T, nx, nu]
        """
        if not self.enable_caching:
            # No caching - use parent implementation
            return super().linearize_dynamics(X, U)
        
        batch_size, T_plus_1, nx = X.shape
        _, T, nu = U.shape
        
        # Initialize output tensors
        A_out = torch.zeros(batch_size, T, nx, nx, device=X.device, dtype=X.dtype)
        B_out = torch.zeros(batch_size, T, nx, nu, device=X.device, dtype=X.dtype)
        
        # Linearize each timestep with caching
        cache_hits = 0
        cache_misses = 0
        
        for t in range(T):
            for b in range(batch_size):
                x_t = X[b, t, :]
                u_t = U[b, t, :]
                
                # Try cache first
                cached_result = self.jacobian_cache.get(x_t, u_t, t)
                
                if cached_result is not None:
                    # Cache hit
                    A_cached, B_cached = cached_result
                    A_out[b, t, :, :] = A_cached.to(dtype=X.dtype)  # Handle compression
                    B_out[b, t, :, :] = B_cached.to(dtype=X.dtype)
                    cache_hits += 1
                    
                else:
                    # Cache miss - compute linearization
                    start_time = time.time()
                    
                    if self.f_dyn_jac is not None:
                        # Use provided Jacobian function
                        A_t, B_t = self.f_dyn_jac(x_t, u_t, self.dt)
                    else:
                        # Numerical differentiation fallback
                        A_t, B_t = self._numerical_jacobian(x_t, u_t)
                    
                    linearization_time = time.time() - start_time
                    self.linearization_times.append(linearization_time)
                    
                    # Store results
                    A_out[b, t, :, :] = A_t
                    B_out[b, t, :, :] = B_t
                    
                    # Cache the result
                    self.jacobian_cache.put(x_t, u_t, t, A_t, B_t)
                    cache_misses += 1
        
        # Update cache performance statistics
        self.cache_performance['total_linearizations'] += cache_hits + cache_misses
        self.cache_performance['cached_linearizations'] += cache_hits
        
        if cache_hits > 0 and len(self.linearization_times) > 0:
            avg_linearization_time = np.mean(self.linearization_times[-cache_misses:])
            time_saved = cache_hits * avg_linearization_time
            self.cache_performance['time_saved'] += time_saved
        
        # Update temporal coherence
        self.jacobian_cache.update_temporal_coherence(X, U)
        
        if self.verbose > 1:
            cache_stats = self.jacobian_cache.get_stats()
            print(f"  Linearization cache: {cache_hits} hits, {cache_misses} misses "
                  f"(hit rate: {cache_stats['hit_rate']:.1%})")
        
        return A_out, B_out
    
    def _numerical_jacobian(self, x: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
        """Numerical Jacobian computation fallback"""
        eps = 1e-6
        
        # State Jacobian A = df/dx
        A = torch.zeros(self.nx, self.nx, device=x.device, dtype=x.dtype)
        f_center = self.f_dyn(x, u, self.dt)
        
        for i in range(self.nx):
            x_plus = x.clone()
            x_plus[i] += eps
            f_plus = self.f_dyn(x_plus, u, self.dt)
            A[:, i] = (f_plus - f_center) / eps
        
        # Control Jacobian B = df/du
        B = torch.zeros(self.nx, self.nu, device=x.device, dtype=x.dtype)
        
        for j in range(self.nu):
            u_plus = u.clone()
            u_plus[j] += eps
            f_plus = self.f_dyn(x, u_plus, self.dt)
            B[:, j] = (f_plus - f_center) / eps
        
        return A, B
    
    def solve_step(self, x0: Tensor, U_init: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Override solve_step per cache-aware optimization.
        """
        if not self.enable_caching:
            return super().solve_step(x0, U_init)
        
        # Pre-warm cache if this is a new trajectory
        if U_init is not None and self.jacobian_cache.last_states is None:
            self._prewarm_cache(x0, U_init)
        
        # Use parent solve_step with cached linearizations
        return super().solve_step(x0, U_init)
    
    def _prewarm_cache(self, x0: Tensor, U_init: Tensor):
        """Pre-warm cache with initial trajectory linearizations"""
        if not self.enable_caching:
            return
        
        if self.verbose > 1:
            print("Pre-warming Jacobian cache...")
        
        # Rollout initial trajectory
        X_init = self.rollout_trajectory(x0, U_init)
        
        # Pre-compute and cache linearizations along initial trajectory
        _ = self.linearize_dynamics(X_init, U_init)
        
        if self.verbose > 1:
            cache_stats = self.jacobian_cache.get_stats()
            print(f"Cache pre-warmed with {cache_stats['cache_size']} entries")
    
    def get_caching_stats(self) -> Dict:
        """Get comprehensive caching performance statistics"""
        if not self.enable_caching:
            return {"caching_enabled": False}
        
        cache_stats = self.jacobian_cache.get_stats()
        
        # Performance metrics
        cache_efficiency = (self.cache_performance['cached_linearizations'] / 
                          max(1, self.cache_performance['total_linearizations']))
        
        avg_linearization_time = np.mean(self.linearization_times) if self.linearization_times else 0.0
        
        return {
            "caching_enabled": True,
            "cache_statistics": cache_stats,
            "performance": {
                "total_linearizations": self.cache_performance['total_linearizations'],
                "cached_linearizations": self.cache_performance['cached_linearizations'],
                "cache_efficiency": cache_efficiency,
                "time_saved_seconds": self.cache_performance['time_saved'],
                "avg_linearization_time_ms": avg_linearization_time * 1000
            },
            "cache_config": {
                "max_entries": self.jacobian_cache.max_entries,
                "validity_threshold": self.jacobian_cache.validity_threshold,
                "temporal_coherence": self.jacobian_cache.enable_temporal_coherence,
                "compression": self.jacobian_cache.cache_compression
            }
        }
    
    def clear_cache(self):
        """Clear Jacobian cache and reset statistics"""
        if self.enable_caching:
            self.jacobian_cache.clear()
        
        self.linearization_times = []
        self.cache_performance = {
            'total_linearizations': 0,
            'cached_linearizations': 0,
            'time_saved': 0.0
        }
    
    def set_cache_validity_threshold(self, threshold: float):
        """Dynamically adjust cache validity threshold"""
        if self.enable_caching:
            self.jacobian_cache.validity_threshold = threshold
            self.cache_validity_threshold = threshold
            
            if self.verbose > 0:
                print(f"Cache validity threshold updated to {threshold}")
    
    def optimize_cache_settings(self, X_trajectory: Tensor, U_trajectory: Tensor):
        """
        Optimize cache settings basato su trajectory characteristics.
        
        Args:
            X_trajectory: Representative state trajectory [B, T+1, nx]
            U_trajectory: Representative control trajectory [B, T, nu]
        """
        if not self.enable_caching:
            return
        
        # Analyze trajectory characteristics
        state_variations = torch.std(X_trajectory, dim=1)  # [B, nx]
        control_variations = torch.std(U_trajectory, dim=1)  # [B, nu]
        
        max_state_var = torch.max(state_variations)
        max_control_var = torch.max(control_variations)
        
        # Adaptive validity threshold based on trajectory smoothness
        if max_state_var < 0.1 and max_control_var < 0.1:
            # Smooth trajectory - can use larger validity threshold
            optimal_threshold = min(self.cache_validity_threshold * 2.0, 0.05)
        elif max_state_var > 1.0 or max_control_var > 1.0:
            # Highly dynamic trajectory - use smaller validity threshold
            optimal_threshold = max(self.cache_validity_threshold * 0.5, 1e-3)
        else:
            # Keep current threshold
            optimal_threshold = self.cache_validity_threshold
        
        # Update threshold if significantly different
        if abs(optimal_threshold - self.cache_validity_threshold) > 1e-3:
            self.set_cache_validity_threshold(optimal_threshold)
            
            if self.verbose > 0:
                print(f"Optimized cache threshold: {optimal_threshold:.4f} "
                      f"(state_var: {max_state_var.item():.3f}, "
                      f"control_var: {max_control_var.item():.3f})")
    
    def reset(self) -> None:
        """Enhanced reset with cache management"""
        super().reset()
        
        if self.enable_caching:
            # Optionally clear cache on reset (configurable)
            # For now, keep cache for potential reuse across episodes
            cache_stats = self.jacobian_cache.get_stats()
            if self.verbose > 0:
                print(f"Reset with cache: {cache_stats['cache_size']} entries, "
                      f"{cache_stats['hit_rate']:.1%} hit rate")
