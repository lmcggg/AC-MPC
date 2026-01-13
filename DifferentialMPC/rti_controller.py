import torch
from torch import Tensor
from typing import Callable, Optional, Tuple, Dict
import time
import numpy as np
from .controller import DifferentiableMPCController, GradMethod, ILQRSolve
from .cost import GeneralQuadCost


class RTIController(DifferentiableMPCController):
    """
    Real-Time Iteration Controller per MPC con garanzie di timing deterministico.
    
    RTI Philosophy:
    - Preparation phase: Pre-computa linearizzazioni e QP setup (background)
    - Feedback phase: Solo 1 SQP iteration con warm-start (real-time)
    - Prediction: Shift + warm-start per prossimo timestep
    """
    
    def __init__(
            self,
            f_dyn: Callable,
            total_time: float,
            step_size: float, 
            horizon: int,
            cost_module: torch.nn.Module,
            u_min: Optional[torch.Tensor] = None,
            u_max: Optional[torch.Tensor] = None,
            reg_eps: float = 1e-6,
            device: str = "cuda:0",
            grad_method: GradMethod | str = GradMethod.ANALYTIC,
            f_dyn_jac: Optional[Callable] = None,
            # RTI-specific parameters
            enable_preparation_phase: bool = True,
            max_feedback_time_ms: float = 10.0,  # Maximum feedback phase time
            fallback_iterations: int = 3,        # Fallback if RTI fails
            warm_start_strategy: str = "shift_fill",  # "shift_fill" or "prediction"
            prediction_horizon_ratio: float = 0.8,   # For prediction warm-start
            timing_monitoring: bool = True,
            verbose: int = 0
    ):
        # Initialize base controller with RTI-optimized settings
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
            max_iter=1,  # RTI: Only 1 iteration per timestep
            verbose=verbose
        )
        
        # RTI-specific configuration
        self.enable_preparation_phase = enable_preparation_phase
        self.max_feedback_time_ms = max_feedback_time_ms
        self.fallback_iterations = fallback_iterations
        self.warm_start_strategy = warm_start_strategy
        self.prediction_horizon_ratio = prediction_horizon_ratio
        self.timing_monitoring = timing_monitoring
        
        # RTI state management
        self.preparation_ready = False
        self.X_prediction = None  # Pre-computed trajectory prediction
        self.U_prediction = None  # Pre-computed control prediction
        self.linearization_cache = None  # Cached A, B matrices
        self.quadraticization_cache = None  # Cached cost derivatives
        
        # Performance monitoring
        self.timing_stats = {
            'preparation_times': [],
            'feedback_times': [],
            'total_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'fallback_activations': 0
        }
    
    def prepare_step(self, x_current: Tensor, U_prev: Optional[Tensor] = None) -> Dict:
        """
        RTI Preparation Phase: Pre-computa linearizzazioni e setup QP per il prossimo timestep.
        Questa fase può essere eseguita in background mentre il controllo precedente è applicato.
        
        Args:
            x_current: Stato corrente [B, nx] o [nx]
            U_prev: Sequenza di controlli precedente per warm-start [B, T, nu] o [T, nu]
            
        Returns:
            Dict con informazioni di preparation per debugging
        """
        if not self.enable_preparation_phase:
            return {"status": "disabled"}
        
        prep_start = time.time()
        
        # Handle batch dimensions
        was_unbatched = x_current.ndim == 1
        if was_unbatched:
            x_current = x_current.unsqueeze(0)
        B = x_current.shape[0]
        
        # Warm-start strategy
        if U_prev is None or self.warm_start_strategy == "shift_fill":
            # Shift-and-fill strategy: shift previous controls + zero-fill
            if self.U_prediction is not None:
                U_warm = self._shift_and_fill_controls(self.U_prediction, B)
            else:
                U_warm = torch.zeros(B, self.horizon, self.nu, device=self.device, dtype=x_current.dtype)
        else:
            # Use provided previous controls
            if U_prev.ndim == 2:
                U_prev = U_prev.unsqueeze(0)
            U_warm = U_prev.expand(B, -1, -1) if U_prev.shape[0] == 1 else U_prev
            
        # Prediction step: rollout with warm-start controls
        X_pred = self.rollout_trajectory(x_current, U_warm)
        
        # Pre-compute linearizations and quadraticizations for efficiency
        try:
            # Cache linearization
            A_cache, B_cache = self.linearize_dynamics(X_pred, U_warm)
            
            # Cache quadraticization  
            x_ref_batch = self.cost_module.x_ref
            u_ref_batch = self.cost_module.u_ref
            if x_ref_batch.shape[0] == 1 and B > 1:
                x_ref_batch = x_ref_batch.expand(B, -1, -1)
            if u_ref_batch.shape[0] == 1 and B > 1:
                u_ref_batch = u_ref_batch.expand(B, -1, -1)
            
            l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN = self.cost_module.quadraticize(
                X_pred, U_warm, x_ref_override=x_ref_batch, u_ref_override=u_ref_batch
            )
            
            # Store caches (conditionally detach for memory vs gradient preservation)
            if getattr(self, 'preserve_gradients', False):
                # During training: preserve gradients for differentiability
                self.X_prediction = X_pred
                self.U_prediction = U_warm
                self.linearization_cache = (A_cache, B_cache)
                self.quadraticization_cache = (l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN)
            else:
                # During inference: detach for memory efficiency
                self.X_prediction = X_pred.detach()
                self.U_prediction = U_warm.detach()
                self.linearization_cache = (A_cache.detach(), B_cache.detach())
                self.quadraticization_cache = (l_x.detach(), l_u.detach(), l_xx.detach(), 
                                             l_xu.detach(), l_uu.detach(), l_xN.detach(), l_xxN.detach())
            
            self.preparation_ready = True
            self.timing_stats['cache_hits'] += 1
            
        except Exception as e:
            if self.verbose > 0:
                print(f"RTI preparation failed: {e}, falling back to online computation")
            self.preparation_ready = False
            self.timing_stats['cache_misses'] += 1
        
        prep_time = (time.time() - prep_start) * 1000  # Convert to ms
        self.timing_stats['preparation_times'].append(prep_time)
        
        return {
            "status": "ready" if self.preparation_ready else "failed",
            "prep_time_ms": prep_time,
            "warm_start_strategy": self.warm_start_strategy,
            "cache_ready": self.preparation_ready
        }
    
    def feedback_step(self, x_current: Tensor) -> Tuple[Tensor, Tensor, Dict]:
        """
        RTI Feedback Phase: Executes 1 SQP iteration usando pre-computed data.
        Questa fase deve essere deterministica e completare entro max_feedback_time_ms.
        
        Args:
            x_current: Stato corrente misurato [B, nx] or [nx]
            
        Returns:
            X_opt: Traiettoria ottima [B, T+1, nx]
            U_opt: Controlli ottimi [B, T, nu] 
            info: Dict con timing e diagnostic info
        """
        feedback_start = time.time()
        
        # Handle batch dimensions
        was_unbatched = x_current.ndim == 1
        if was_unbatched:
            x_current = x_current.unsqueeze(0)
        B = x_current.shape[0]
        
        try:
            if self.preparation_ready and self._validate_cache(x_current):
                # Fast path: Use pre-computed linearizations
                X_opt, U_opt = self._feedback_with_cache(x_current)
                cache_used = True
            else:
                # Fallback path: Online computation
                X_opt, U_opt = self._feedback_without_cache(x_current)
                cache_used = False
                self.timing_stats['fallback_activations'] += 1
            
        except Exception as e:
            if self.verbose > 0:
                print(f"RTI feedback failed: {e}, using fallback controller")
            X_opt, U_opt = self._emergency_fallback(x_current)
            cache_used = False
            self.timing_stats['fallback_activations'] += 1
        
        feedback_time = (time.time() - feedback_start) * 1000  # ms
        self.timing_stats['feedback_times'].append(feedback_time)
        
        # Check real-time constraint violation
        rt_constraint_violated = feedback_time > self.max_feedback_time_ms
        
        # Prepare for next iteration (shift predictions)
        self._shift_predictions()
        
        info = {
            "feedback_time_ms": feedback_time,
            "cache_used": cache_used,
            "rt_constraint_ok": not rt_constraint_violated,
            "max_time_ms": self.max_feedback_time_ms
        }
        
        if was_unbatched:
            X_opt = X_opt.squeeze(0)
            U_opt = U_opt.squeeze(0)
        
        return X_opt, U_opt, info
    
    def solve_step(self, x_current: Tensor, U_prev: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Complete RTI step: Preparation + Feedback in sequence.
        Use questo per single-step operation o quando non hai background preparation.
        """
        total_start = time.time()
        
        # Preparation phase
        self.prepare_step(x_current, U_prev)
        
        # Feedback phase  
        X_opt, U_opt, _ = self.feedback_step(x_current)
        
        total_time = (time.time() - total_start) * 1000
        self.timing_stats['total_times'].append(total_time)
        
        # In the context of ILQRSolve, solve_step should return only X and U
        return X_opt, U_opt
    
    def _shift_and_fill_controls(self, U_prev: Tensor, B: int) -> Tensor:
        """Shift controls and zero-fill for warm-start"""
        if U_prev.shape[0] == 1 and B > 1:
            U_prev = U_prev.expand(B, -1, -1)
        elif U_prev.shape[0] != B:
            # If shapes don't match and it's not a singleton, create new tensor
            U_prev = torch.zeros(B, U_prev.shape[1], U_prev.shape[2], 
                               device=U_prev.device, dtype=U_prev.dtype)
            
        # Shift: U[1:] + zero-fill last timestep
        U_shifted = torch.cat([
            U_prev[:, 1:, :],  # Shift by removing first timestep
            torch.zeros(B, 1, self.nu, device=self.device, dtype=U_prev.dtype)  # Zero-fill
        ], dim=1)
        
        return U_shifted
    
    def _validate_cache(self, x_current: Tensor) -> bool:
        """Validate if cached predictions are still usable"""
        if not self.preparation_ready or self.X_prediction is None:
            return False
        
        # Check if current state is close to predicted first state
        x_pred_0 = self.X_prediction[:, 0, :]  # First predicted state
        state_error = torch.norm(x_current - x_pred_0, dim=-1)
        max_error = torch.max(state_error)
        
        # Cache is valid if prediction error is reasonable
        return max_error < 1.0  # Tunable threshold
    
    def _feedback_with_cache(self, x_current: Tensor) -> Tuple[Tensor, Tensor]:
        """Fast feedback using cached linearizations"""
        B = x_current.shape[0]
        
        # Use cached linearizations and quadraticizations
        A_cache, B_cache = self.linearization_cache
        l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN = self.quadraticization_cache
        
        # Update initial state in prediction
        X_warm = self.X_prediction.clone()
        X_warm[:, 0, :] = x_current  # Fix initial state
        
        # Re-rollout with corrected initial state
        U_warm = self.U_prediction.clone()
        X_corrected = self.rollout_trajectory(x_current, U_warm)
        
        # Single SQP iteration with cached data
        try:
            from torch.func import vmap as _vmap
            K, k = _vmap(self.backward_lqr)(A_cache, B_cache, l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN)
        except:
            # Fallback if vmap fails
            K_list, k_list = [], []
            for b in range(B):
                K_b, k_b = self.backward_lqr(A_cache[b], B_cache[b], l_x[b], l_u[b], 
                                           l_xx[b], l_xu[b], l_uu[b], l_xN[b], l_xxN[b])
                K_list.append(K_b)
                k_list.append(k_b)
            K = torch.stack(K_list, dim=0)
            k = torch.stack(k_list, dim=0)
        
        # Forward pass with step size 1.0 (aggressive for RT)
        X_opt, U_opt = self._forward_pass_batch(x_current, X_corrected, U_warm, K, k)
        
        return X_opt, U_opt
    
    def _feedback_without_cache(self, x_current: Tensor) -> Tuple[Tensor, Tensor]:
        """Fallback feedback without cache - single iteration"""
        B = x_current.shape[0]
        
        # Use shift-and-fill warm-start
        if self.U_prediction is not None:
            U_init = self._shift_and_fill_controls(self.U_prediction, B)
        else:
            U_init = torch.zeros(B, self.horizon, self.nu, device=self.device, dtype=x_current.dtype)
        
        # Single SQP iteration
        X_warm = self.rollout_trajectory(x_current, U_init)
        
        # Quick linearization and quadraticization
        A, B_dyn = self.linearize_dynamics(X_warm, U_init)
        
        x_ref_batch = self.cost_module.x_ref
        u_ref_batch = self.cost_module.u_ref
        if x_ref_batch.shape[0] == 1 and B > 1:
            x_ref_batch = x_ref_batch.expand(B, -1, -1)
        if u_ref_batch.shape[0] == 1 and B > 1:  
            u_ref_batch = u_ref_batch.expand(B, -1, -1)
            
        l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN = self.cost_module.quadraticize(
            X_warm, U_init, x_ref_override=x_ref_batch, u_ref_override=u_ref_batch
        )
        
        # Backward pass
        try:
            from torch.func import vmap as _vmap
            K, k = _vmap(self.backward_lqr)(A, B_dyn, l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN)
        except:
            K_list, k_list = [], []
            for b in range(B):
                K_b, k_b = self.backward_lqr(A[b], B_dyn[b], l_x[b], l_u[b],
                                           l_xx[b], l_xu[b], l_uu[b], l_xN[b], l_xxN[b])
                K_list.append(K_b)
                k_list.append(k_b)
            K = torch.stack(K_list, dim=0)
            k = torch.stack(k_list, dim=0)
        
        # Forward pass
        X_opt, U_opt = self._forward_pass_batch(x_current, X_warm, U_init, K, k)
        
        return X_opt, U_opt
    
    def _emergency_fallback(self, x_current: Tensor) -> Tuple[Tensor, Tensor]:
        """Emergency fallback - return safe/reasonable controls"""
        B = x_current.shape[0]
        
        if self.verbose > 0:
            print("RTI Emergency fallback activated")
        
        # Safe fallback: use previous controls if available, otherwise zero
        if self.U_prediction is not None:
            U_safe = self._shift_and_fill_controls(self.U_prediction, B)
        else:
            U_safe = torch.zeros(B, self.horizon, self.nu, device=self.device, dtype=x_current.dtype)
        
        # Rollout with safe controls
        X_safe = self.rollout_trajectory(x_current, U_safe)
        
        return X_safe, U_safe
    
    def _forward_pass_batch(self, x0: Tensor, X_ref: Tensor, U_ref: Tensor, 
                           K: Tensor, k: Tensor, alpha: float = 1.0) -> Tuple[Tensor, Tensor]:
        """Batched forward pass for RTI"""
        B = x0.shape[0]
        X_new = [x0]
        U_new = []
        
        x_current = x0
        for t in range(self.horizon):
            dx = x_current - X_ref[:, t, :]  # [B, nx]
            du = alpha * k[:, t, :] + alpha * torch.einsum('bij,bj->bi', K[:, t, :, :], dx)  # [B, nu]
            u_new = U_ref[:, t, :] + du
            
            # Apply control constraints
            if self.u_min is not None:
                u_new = torch.max(u_new, self.u_min.expand_as(u_new))
            if self.u_max is not None:
                u_new = torch.min(u_new, self.u_max.expand_as(u_new))
                
            U_new.append(u_new)
            x_current = self.f_dyn(x_current, u_new, self.dt)
            X_new.append(x_current)
        
        return torch.stack(X_new, dim=1), torch.stack(U_new, dim=1)
    
    def _shift_predictions(self):
        """Shift predictions for next timestep"""
        if self.X_prediction is not None:
            # Shift trajectories by one timestep
            self.X_prediction = self._shift_and_fill_states(self.X_prediction)
            self.U_prediction = self._shift_and_fill_controls(self.U_prediction, self.X_prediction.shape[0])
            
        # Invalidate cached linearizations (they're state-dependent)
        self.preparation_ready = False
        self.linearization_cache = None
        self.quadraticization_cache = None
    
    def _shift_and_fill_states(self, X_prev: Tensor) -> Tensor:
        """Shift states and duplicate last state for warm-start"""
        B = X_prev.shape[0]
        
        # Shift: X[1:] + duplicate last state
        X_shifted = torch.cat([
            X_prev[:, 1:, :],  # Shift by removing first timestep
            X_prev[:, -1:, :]  # Duplicate last state
        ], dim=1)
        
        return X_shifted
    
    def get_timing_stats(self) -> Dict:
        """Get comprehensive timing statistics for RT analysis"""
        if not self.timing_monitoring:
            return {"monitoring": "disabled"}
        
        prep_times = self.timing_stats['preparation_times']
        feedback_times = self.timing_stats['feedback_times']
        total_times = self.timing_stats['total_times']
        
        stats = {
            "preparation": {
                "mean_ms": np.mean(prep_times) if prep_times else 0.0,
                "max_ms": np.max(prep_times) if prep_times else 0.0,
                "std_ms": np.std(prep_times) if prep_times else 0.0,
                "count": len(prep_times)
            },
            "feedback": {
                "mean_ms": np.mean(feedback_times) if feedback_times else 0.0,
                "max_ms": np.max(feedback_times) if feedback_times else 0.0,
                "std_ms": np.std(feedback_times) if feedback_times else 0.0,
                "count": len(feedback_times),
                "rt_violations": sum(1 for t in feedback_times if t > self.max_feedback_time_ms)
            },
            "total": {
                "mean_ms": np.mean(total_times) if total_times else 0.0,
                "max_ms": np.max(total_times) if total_times else 0.0,
                "std_ms": np.std(total_times) if total_times else 0.0,
                "count": len(total_times)
            },
            "cache": {
                "hits": self.timing_stats['cache_hits'],
                "misses": self.timing_stats['cache_misses'],
                "hit_rate": self.timing_stats['cache_hits'] / max(1, self.timing_stats['cache_hits'] + self.timing_stats['cache_misses'])
            },
            "fallbacks": self.timing_stats['fallback_activations'],
            "max_feedback_constraint_ms": self.max_feedback_time_ms
        }
        
        return stats
    
    def reset_timing_stats(self):
        """Reset timing statistics for new experiment"""
        self.timing_stats = {
            'preparation_times': [],
            'feedback_times': [],
            'total_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'fallback_activations': 0
        }
    
    def forward(self, x0: torch.Tensor, U_init: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Override forward method to enable gradient-preserving RTI execution.
        
        For differentiability, this method bypasses ILQRSolve.apply() and directly
        calls rti_step() while preserving gradients.
        """
        # This controller is now differentiable
        if getattr(self, 'preserve_gradients', False):
            # If called from a context that wants to preserve gradients, run the step directly
            return self.solve_step(x0, U_init)
        else:
            # Standard path through ILQRSolve autograd function
            B = x0.shape[0] if x0.ndim > 1 else 1
            if U_init is None:
                U_init = torch.zeros(B, self.horizon, self.nu, device=x0.device, dtype=x0.dtype)
            
            C, c, C_final, c_final = self.cost_module.C, self.cost_module.c, self.cost_module.C_final, self.cost_module.c_final
            x_ref, u_ref = self.cost_module.x_ref, self.cost_module.u_ref
            return ILQRSolve.apply(x0, C, c, C_final, c_final, x_ref, u_ref, self, U_init)
    
    def reset(self) -> None:
        """Enhanced reset for RTI controller"""
        super().reset()
        
        # RTI-specific cleanup
        self.preparation_ready = False
        self.X_prediction = None
        self.U_prediction = None
        self.linearization_cache = None
        self.quadraticization_cache = None
        
        # Clear memory aggressively for RT systems
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()