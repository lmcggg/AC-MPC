
import torch
from torch import Tensor
import numpy as np
import time
from typing import Callable, Optional, Tuple, Dict, List, Union
from collections import OrderedDict
from .jacobian_caching_controller import JacobianCachingController, JacobianCache
from .controller import GradMethod


class SpatialTemporalCache(JacobianCache):
    """
    Enhanced cache with spatial-temporal locality optimization.
    """
    
    def __init__(
        self,
        max_entries: int = 200,
        validity_threshold: float = 1e-2,
        spatial_clusters: int = 8,
        temporal_window: int = 5,
        enable_interpolation: bool = True,
        device: str = "cuda:0"
    ):
        super().__init__(max_entries, validity_threshold, True, True, device)
        
        self.spatial_clusters = spatial_clusters
        self.temporal_window = temporal_window
        self.enable_interpolation = enable_interpolation
        
        # Spatial clustering for locality
        self.spatial_cache_clusters = [OrderedDict() for _ in range(spatial_clusters)]
        self.cluster_centers = None
        
        # Temporal coherence tracking
        self.temporal_history = []
        
        # Interpolation cache
        self.interpolation_cache = OrderedDict()
        self.interpolation_stats = {'hits': 0, 'attempts': 0}
        
    def _get_spatial_cluster(self, x: Tensor) -> int:
        """Determine spatial cluster for state"""
        if self.cluster_centers is None:
            # Initialize cluster centers randomly
            self.cluster_centers = torch.randn(
                self.spatial_clusters, x.shape[-1], 
                device=x.device, dtype=x.dtype
            )
        
        # Find nearest cluster center
        distances = torch.norm(self.cluster_centers - x.unsqueeze(0), dim=1)
        cluster_id = torch.argmin(distances).item()
        
        return cluster_id
    
    def _update_cluster_centers(self, x: Tensor, cluster_id: int):
        """Update cluster center with exponential moving average"""
        if self.cluster_centers is not None:
            alpha = 0.1  # Learning rate
            self.cluster_centers[cluster_id] = (
                (1 - alpha) * self.cluster_centers[cluster_id] + alpha * x
            )
    
    def get_with_interpolation(self, x: Tensor, u: Tensor, t: int) -> Optional[Tuple[Tensor, Tensor]]:
        """
        Enhanced get with interpolation fallback.
        """
        # Standard cache lookup first
        result = self.get(x, u, t)
        if result is not None:
            return result
        
        if not self.enable_interpolation:
            return None
        
        # Try interpolation from nearby cached entries
        self.interpolation_stats['attempts'] += 1
        interpolated = self._interpolate_from_neighbors(x, u, t)
        
        if interpolated is not None:
            self.interpolation_stats['hits'] += 1
            return interpolated
        
        return None
    
    def _interpolate_from_neighbors(
        self, x: Tensor, u: Tensor, t: int
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """
        Interpolate Jacobian from nearby cache entries.
        """
        if len(self.cache) < 2:
            return None
        
        # Find closest cache entries
        candidates = []
        target_state = torch.cat([x, u])  # Combined state-control
        
        for key, (A, B, x_cached, u_cached, timestamp, hit_count) in self.cache.items():
            cached_state = torch.cat([x_cached, u_cached])
            distance = torch.norm(target_state - cached_state)
            
            if distance < self.validity_threshold * 3.0:  # Wider search radius
                candidates.append((distance.item(), A, B, x_cached, u_cached))
        
        if len(candidates) < 2:
            return None
        
        # Sort by distance and take closest
        candidates.sort(key=lambda x: x[0])
        
        # Weighted interpolation of closest 2-3 entries
        weights = []
        As, Bs = [], []
        
        for i in range(min(3, len(candidates))):
            dist, A, B, x_c, u_c = candidates[i]
            weight = 1.0 / (dist + 1e-6)  # Inverse distance weighting
            weights.append(weight)
            As.append(A)
            Bs.append(B)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted interpolation
        A_interp = sum(w * A for w, A in zip(weights, As))
        B_interp = sum(w * B for w, B in zip(weights, Bs))
        
        return A_interp, B_interp
    
    def put_with_clustering(self, x: Tensor, u: Tensor, t: int, A: Tensor, B: Tensor):
        """Enhanced put with spatial clustering"""
        # Standard put
        self.put(x, u, t, A, B)
        
        # Spatial clustering optimization
        cluster_id = self._get_spatial_cluster(x)
        key = self._create_cache_key(x, u, t)
        
        # Store in spatial cluster
        self.spatial_cache_clusters[cluster_id][key] = (A, B, x.clone(), u.clone(), time.time())
        
        # Update cluster center
        self._update_cluster_centers(x, cluster_id)
        
        # Maintain cluster size
        if len(self.spatial_cache_clusters[cluster_id]) > self.max_entries // self.spatial_clusters:
            # Remove oldest from this cluster
            oldest_key = next(iter(self.spatial_cache_clusters[cluster_id]))
            del self.spatial_cache_clusters[cluster_id][oldest_key]
    
    def get_interpolation_stats(self) -> Dict:
        """Get interpolation performance statistics"""
        hit_rate = (self.interpolation_stats['hits'] / 
                   max(1, self.interpolation_stats['attempts']))
        
        return {
            'interpolation_attempts': self.interpolation_stats['attempts'],
            'interpolation_hits': self.interpolation_stats['hits'],
            'interpolation_hit_rate': hit_rate,
            'spatial_clusters': self.spatial_clusters,
            'cluster_usage': [len(cluster) for cluster in self.spatial_cache_clusters]
        }


class HierarchicalCache:
    """
    Multi-level hierarchical cache: coarse + fine grain caching.
    """
    
    def __init__(
        self,
        coarse_entries: int = 50,
        fine_entries: int = 300,
        coarse_threshold: float = 5e-2,
        fine_threshold: float = 1e-3,
        device: str = "cuda:0"
    ):
        self.coarse_cache = JacobianCache(coarse_entries, coarse_threshold, device=device)
        self.fine_cache = JacobianCache(fine_entries, fine_threshold, device=device)
        
        self.hierarchy_stats = {
            'coarse_hits': 0,
            'fine_hits': 0,
            'total_queries': 0
        }
    
    def get(self, x: Tensor, u: Tensor, t: int) -> Optional[Tuple[Tensor, Tensor]]:
        """Hierarchical cache lookup: fine first, then coarse"""
        self.hierarchy_stats['total_queries'] += 1
        
        # Try fine cache first (most accurate)
        result = self.fine_cache.get(x, u, t)
        if result is not None:
            self.hierarchy_stats['fine_hits'] += 1
            return result
        
        # Fallback to coarse cache
        result = self.coarse_cache.get(x, u, t)
        if result is not None:
            self.hierarchy_stats['coarse_hits'] += 1
            return result
        
        return None
    
    def put(self, x: Tensor, u: Tensor, t: int, A: Tensor, B: Tensor):
        """Store in both caches with appropriate thresholds"""
        # Always store in fine cache
        self.fine_cache.put(x, u, t, A, B)
        
        # Store in coarse cache with reduced precision for broader coverage
        self.coarse_cache.put(x, u, t, A, B)
    
    def get_stats(self) -> Dict:
        """Get hierarchical cache statistics"""
        total_hits = self.hierarchy_stats['coarse_hits'] + self.hierarchy_stats['fine_hits']
        hit_rate = total_hits / max(1, self.hierarchy_stats['total_queries'])
        
        coarse_stats = self.coarse_cache.get_stats()
        fine_stats = self.fine_cache.get_stats()
        
        return {
            'hierarchical_hit_rate': hit_rate,
            'coarse_contribution': self.hierarchy_stats['coarse_hits'] / max(1, total_hits),
            'fine_contribution': self.hierarchy_stats['fine_hits'] / max(1, total_hits),
            'coarse_cache': coarse_stats,
            'fine_cache': fine_stats
        }


class EnhancedJacobianCachingController(JacobianCachingController):
    """
    Enhanced Jacobian Caching Controller with advanced optimizations.
    """
    
    def __init__(
            self,
            f_dyn: Callable,
            total_time: float,
            step_size: float,
            horizon: int,
            cost_module: torch.nn.Module,
            # Enhanced caching parameters
            enable_enhanced_caching: bool = True,
            caching_strategy: str = "spatial_temporal",  # "spatial_temporal", "hierarchical", "hybrid"
            enable_interpolation: bool = True,
            enable_predictive_loading: bool = True,
            spatial_clusters: int = 8,
            # Standard parameters
            cache_max_entries: int = 300,
            cache_validity_threshold: float = 1e-2,
            u_min: Optional[torch.Tensor] = None,
            u_max: Optional[torch.Tensor] = None,
            reg_eps: float = 1e-6,
            device: str = "cuda:0",
            grad_method: GradMethod | str = GradMethod.ANALYTIC,
            f_dyn_jac: Optional[Callable] = None,
            max_iter: int = 50,
            verbose: int = 0
    ):
        # Initialize parent without caching first
        super().__init__(
            f_dyn=f_dyn,
            total_time=total_time,
            step_size=step_size,
            horizon=horizon,
            cost_module=cost_module,
            enable_caching=False,  # We'll handle caching ourselves
            u_min=u_min,
            u_max=u_max,
            reg_eps=reg_eps,
            device=device,
            grad_method=grad_method,
            f_dyn_jac=f_dyn_jac,
            max_iter=max_iter,
            verbose=verbose
        )
        
        # Enhanced caching setup
        self.enable_enhanced_caching = enable_enhanced_caching
        self.caching_strategy = caching_strategy
        self.enable_interpolation = enable_interpolation
        self.enable_predictive_loading = enable_predictive_loading
        
        if self.enable_enhanced_caching:
            if caching_strategy == "spatial_temporal":
                self.enhanced_cache = SpatialTemporalCache(
                    max_entries=cache_max_entries,
                    validity_threshold=cache_validity_threshold,
                    spatial_clusters=spatial_clusters,
                    enable_interpolation=enable_interpolation,
                    device=device
                )
            elif caching_strategy == "hierarchical":
                self.enhanced_cache = HierarchicalCache(
                    coarse_entries=cache_max_entries // 4,
                    fine_entries=cache_max_entries,
                    coarse_threshold=cache_validity_threshold * 5,
                    fine_threshold=cache_validity_threshold,
                    device=device
                )
            else:  # hybrid
                self.enhanced_cache = SpatialTemporalCache(
                    max_entries=cache_max_entries,
                    validity_threshold=cache_validity_threshold,
                    spatial_clusters=spatial_clusters,
                    enable_interpolation=enable_interpolation,
                    device=device
                )
        
        # Performance tracking for enhanced features
        self.enhanced_stats = {
            'interpolation_uses': 0,
            'predictive_hits': 0,
            'total_enhanced_queries': 0
        }
        
        if self.verbose > 0:
            print(f"Enhanced Jacobian Caching Setup:")
            print(f"  Strategy: {caching_strategy}")
            print(f"  Interpolation: {enable_interpolation}")
            print(f"  Predictive loading: {enable_predictive_loading}")
            if hasattr(self.enhanced_cache, 'spatial_clusters'):
                print(f"  Spatial clusters: {spatial_clusters}")
    
    def linearize_dynamics(self, X: Tensor, U: Tensor) -> Tuple[Tensor, Tensor]:
        """Enhanced linearization with advanced caching"""
        if not self.enable_enhanced_caching:
            return super().linearize_dynamics(X, U)
        
        batch_size, T_plus_1, nx = X.shape
        _, T, nu = U.shape
        
        A_out = torch.zeros(batch_size, T, nx, nx, device=X.device, dtype=X.dtype)
        B_out = torch.zeros(batch_size, T, nx, nu, device=X.device, dtype=X.dtype)
        
        cache_hits = 0
        cache_misses = 0
        interpolation_hits = 0
        predictive_hits = 0
        
        for t in range(T):
            for b in range(batch_size):
                x_t = X[b, t, :]
                u_t = U[b, t, :]
                
                self.enhanced_stats['total_enhanced_queries'] += 1
                
                # Enhanced cache lookup with interpolation
                if hasattr(self.enhanced_cache, 'get_with_interpolation'):
                    cached_result = self.enhanced_cache.get_with_interpolation(x_t, u_t, t)
                else:
                    cached_result = self.enhanced_cache.get(x_t, u_t, t)
                
                if cached_result is not None:
                    # Cache hit (including interpolation)
                    A_cached, B_cached = cached_result
                    A_out[b, t, :, :] = A_cached.to(dtype=X.dtype)
                    B_out[b, t, :, :] = B_cached.to(dtype=X.dtype)
                    cache_hits += 1
                    
                    # Check if this was an interpolation hit
                    if hasattr(self.enhanced_cache, 'interpolation_stats'):
                        current_interp_hits = self.enhanced_cache.interpolation_stats['hits']
                        if current_interp_hits > self.enhanced_stats.get('last_interp_hits', 0):
                            interpolation_hits += 1
                            self.enhanced_stats['last_interp_hits'] = current_interp_hits
                    
                else:
                    # Cache miss - compute linearization
                    start_time = time.time()
                    
                    if self.f_dyn_jac is not None:
                        A_t, B_t = self.f_dyn_jac(x_t, u_t, self.dt)
                    else:
                        A_t, B_t = self._numerical_jacobian(x_t, u_t)
                    
                    linearization_time = time.time() - start_time
                    self.linearization_times.append(linearization_time)
                    
                    A_out[b, t, :, :] = A_t
                    B_out[b, t, :, :] = B_t
                    
                    # Enhanced cache storage
                    if hasattr(self.enhanced_cache, 'put_with_clustering'):
                        self.enhanced_cache.put_with_clustering(x_t, u_t, t, A_t, B_t)
                    else:
                        self.enhanced_cache.put(x_t, u_t, t, A_t, B_t)
                    
                    cache_misses += 1
        
        # Predictive pre-loading for next iteration
        if self.enable_predictive_loading:
            self._predictive_preload(X, U)
        
        # Update performance stats
        self.cache_performance['total_linearizations'] += cache_hits + cache_misses
        self.cache_performance['cached_linearizations'] += cache_hits
        self.enhanced_stats['interpolation_uses'] += interpolation_hits
        
        if self.verbose > 1:
            print(f"  Enhanced cache: {cache_hits} hits, {cache_misses} misses, "
                  f"{interpolation_hits} interpolations")
        
        return A_out, B_out
    
    def _predictive_preload(self, X: Tensor, U: Tensor):
        """Predictive pre-loading basato su trajectory patterns"""
        if not self.enable_predictive_loading:
            return
        
        # Simple predictive strategy: preload around current states
        batch_size, T_plus_1, nx = X.shape
        
        # Predict next likely states based on current trajectory
        for b in range(min(2, batch_size)):  # Only first 2 batch elements to avoid overhead
            for t in range(min(3, T_plus_1 - 1)):  # Only first few timesteps
                x_current = X[b, t, :]
                
                # Predict small perturbations around current state
                perturbations = torch.randn(5, nx, device=x_current.device) * 0.01
                u_current = U[b, min(t, U.shape[1] - 1), :] if t < U.shape[1] else torch.zeros(2, device=x_current.device)
                
                for pert in perturbations:
                    x_pred = x_current + pert
                    # Check if we should preload (not already cached)
                    if self.enhanced_cache.get(x_pred, u_current, t) is None:
                        # This would be a cache miss - could preload if resources allow
                        # For now, just count as predictive opportunity
                        self.enhanced_stats['predictive_hits'] += 1
    
    def get_enhanced_stats(self) -> Dict:
        """Get comprehensive enhanced caching statistics"""
        base_stats = super().get_caching_stats() if self.enable_caching else {}
        
        enhanced_stats = {
            'enhanced_caching_enabled': self.enable_enhanced_caching,
            'caching_strategy': self.caching_strategy,
            'interpolation_enabled': self.enable_interpolation,
            'enhanced_performance': self.enhanced_stats
        }
        
        # Strategy-specific stats
        if hasattr(self.enhanced_cache, 'get_interpolation_stats'):
            enhanced_stats['interpolation_stats'] = self.enhanced_cache.get_interpolation_stats()
        
        if hasattr(self.enhanced_cache, 'get_stats') and callable(self.enhanced_cache.get_stats):
            enhanced_stats['cache_stats'] = self.enhanced_cache.get_stats()
        
        return {**base_stats, **enhanced_stats}
    
    def optimize_cache_parameters(self, performance_history: List[Dict]):
        """Optimize cache parameters basato su performance history"""
        if not self.enable_enhanced_caching or len(performance_history) < 3:
            return
        
        # Analyze recent performance
        recent_hit_rates = [p.get('hit_rate', 0) for p in performance_history[-5:]]
        recent_times = [p.get('solve_time', 1) for p in performance_history[-5:]]
        
        avg_hit_rate = np.mean(recent_hit_rates)
        avg_time = np.mean(recent_times)
        
        # Adaptive optimization
        if avg_hit_rate < 0.5:
            # Low hit rate - increase validity threshold
            if hasattr(self.enhanced_cache, 'validity_threshold'):
                self.enhanced_cache.validity_threshold *= 1.1
                if self.verbose > 0:
                    print(f"Increased cache validity threshold to {self.enhanced_cache.validity_threshold:.4f}")
        
        elif avg_hit_rate > 0.8:
            # Very high hit rate - can be more strict
            if hasattr(self.enhanced_cache, 'validity_threshold'):
                self.enhanced_cache.validity_threshold *= 0.95
                if self.verbose > 0:
                    print(f"Decreased cache validity threshold to {self.enhanced_cache.validity_threshold:.4f}")
        
        # Adjust cache size based on performance
        if avg_time > 0.5:  # If solve time is high
            if hasattr(self.enhanced_cache, 'max_entries'):
                self.enhanced_cache.max_entries = min(self.enhanced_cache.max_entries * 1.1, 500)
                if self.verbose > 0:
                    print(f"Increased cache size to {self.enhanced_cache.max_entries}")


# Factory function per create enhanced controller
def create_enhanced_jacobian_controller(
    f_dyn: Callable,
    f_dyn_jac: Callable,
    total_time: float,
    step_size: float,
    horizon: int,
    cost_module,
    strategy: str = "spatial_temporal",
    device: str = "cuda:0",
    **kwargs
) -> EnhancedJacobianCachingController:
    """
    Factory function per creare enhanced jacobian caching controller.
    
    Args:
        strategy: "spatial_temporal", "hierarchical", or "hybrid"
    """
    return EnhancedJacobianCachingController(
        f_dyn=f_dyn,
        f_dyn_jac=f_dyn_jac,
        total_time=total_time,
        step_size=step_size,
        horizon=horizon,
        cost_module=cost_module,
        caching_strategy=strategy,
        enable_interpolation=True,
        enable_predictive_loading=True,
        device=device,
        verbose=1,
        **kwargs
    )