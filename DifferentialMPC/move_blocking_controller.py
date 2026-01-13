import torch
from torch import Tensor
from typing import Callable, Optional, Tuple, Dict, List, Union
import numpy as np
from .controller import DifferentiableMPCController, GradMethod
from .cost import GeneralQuadCost


class MoveBlockingController(DifferentiableMPCController):
    """
    Move Blocking Controller per riduzione DOF e efficienza computazionale.
    
    Move Blocking Philosophy:
    - Parametrizza controlli con blocchi costanti su intervalli temporali
    - Riduce variabili di ottimizzazione da T*nu a n_blocks*nu
    - Mantiene expressivity controllando block sizes e patterns
    - Ideale per orizzonti lunghi con dinamiche lente
    """
    
    def __init__(
            self,
            f_dyn: Callable,
            total_time: float,
            step_size: float,
            horizon: int,
            cost_module: torch.nn.Module,
            # Move blocking parameters
            blocking_pattern: Union[List[int], str] = "uniform",
            n_blocks: Optional[int] = None,
            block_sizes: Optional[List[int]] = None,
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
        # Initialize base controller first
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
        
        # Ensure controller attributes are properly initialized
        self.H_last = None
        
        # Move blocking configuration
        self.blocking_pattern = blocking_pattern
        self.n_blocks = n_blocks
        self.block_sizes = block_sizes
        
        # Setup blocking structure
        self._setup_move_blocking()
        
        if self.verbose > 0:
            print(f"Move Blocking Setup:")
            print(f"  Original DOF: {self.horizon * self.nu}")
            print(f"  Reduced DOF: {self.n_blocks * self.nu}")
            print(f"  Reduction ratio: {(self.n_blocks * self.nu) / (self.horizon * self.nu):.2%}")
            print(f"  Block pattern: {self.block_sizes}")
    
    def _setup_move_blocking(self):
        """Setup move blocking structure basata su pattern specificato"""
        
        if isinstance(self.blocking_pattern, str):
            if self.blocking_pattern == "uniform":
                self._setup_uniform_blocking()
            elif self.blocking_pattern == "exponential":
                self._setup_exponential_blocking()
            elif self.blocking_pattern == "custom":
                if self.block_sizes is None:
                    raise ValueError("block_sizes deve essere specificato per custom pattern")
                self._setup_custom_blocking()
            else:
                raise ValueError(f"Unknown blocking pattern: {self.blocking_pattern}")
        elif isinstance(self.blocking_pattern, list):
            # Direct block sizes specification
            self.block_sizes = self.blocking_pattern
            self._setup_custom_blocking()
        else:
            raise ValueError("blocking_pattern deve essere str o List[int]")
        
        # Create mapping tensors per efficiency
        self._create_blocking_tensors()
    
    def _setup_uniform_blocking(self):
        """Setup uniform block sizes"""
        if self.n_blocks is None:
            # Default: roughly sqrt(horizon) blocks for good trade-off
            self.n_blocks = max(1, int(np.sqrt(self.horizon)))
        
        # Distribute horizon uniformly across blocks
        base_size = self.horizon // self.n_blocks
        remainder = self.horizon % self.n_blocks
        
        self.block_sizes = [base_size] * self.n_blocks
        # Distribute remainder across first blocks
        for i in range(remainder):
            self.block_sizes[i] += 1
        
        if self.verbose > 1:
            print(f"Uniform blocking: {self.n_blocks} blocks with sizes {self.block_sizes}")
    
    def _setup_exponential_blocking(self):
        """Setup exponentially increasing block sizes (fine -> coarse)"""
        if self.n_blocks is None:
            self.n_blocks = max(3, int(np.log2(self.horizon)))
        
        # Exponential growth with base factor
        base_factor = 1.5
        base_size = 1
        
        self.block_sizes = []
        remaining_horizon = self.horizon
        
        for i in range(self.n_blocks - 1):
            block_size = max(1, int(base_size * (base_factor ** i)))
            block_size = min(block_size, remaining_horizon - (self.n_blocks - i - 1))
            self.block_sizes.append(block_size)
            remaining_horizon -= block_size
        
        # Last block gets remaining timesteps
        self.block_sizes.append(max(1, remaining_horizon))
        
        # Adjust if we overshot
        total_size = sum(self.block_sizes)
        if total_size != self.horizon:
            self.block_sizes[-1] += self.horizon - total_size
        
        if self.verbose > 1:
            print(f"Exponential blocking: {self.n_blocks} blocks with sizes {self.block_sizes}")
    
    def _setup_custom_blocking(self):
        """Setup custom block sizes"""
        if sum(self.block_sizes) != self.horizon:
            raise ValueError(f"Block sizes sum {sum(self.block_sizes)} != horizon {self.horizon}")
        
        self.n_blocks = len(self.block_sizes)
        
        if self.verbose > 1:
            print(f"Custom blocking: {self.n_blocks} blocks with sizes {self.block_sizes}")
    
    def _create_blocking_tensors(self):
        """Create efficient tensors per mapping blocked -> full controls"""
        
        # Create index mapping: block_idx -> timestep indices
        self.block_to_timestep_indices = []
        start_idx = 0
        
        for block_size in self.block_sizes:
            indices = list(range(start_idx, start_idx + block_size))
            self.block_to_timestep_indices.append(indices)
            start_idx += block_size
        
        # Create expansion matrix: [n_blocks, horizon] che replica blocked controls
        self.blocking_matrix = torch.zeros(
            self.n_blocks, self.horizon, 
            device=self.device, dtype=torch.get_default_dtype()
        )
        
        for block_idx, timestep_indices in enumerate(self.block_to_timestep_indices):
            for t_idx in timestep_indices:
                self.blocking_matrix[block_idx, t_idx] = 1.0
        
        if self.verbose > 2:
            print(f"Blocking matrix shape: {self.blocking_matrix.shape}")
            print(f"Block-to-timestep mapping: {self.block_to_timestep_indices}")
    
    def expand_blocked_controls(self, U_blocked: Tensor) -> Tensor:
        """
        Expand blocked controls to full temporal resolution.
        
        Args:
            U_blocked: Blocked controls [B, n_blocks, nu] o [n_blocks, nu]
            
        Returns:
            U_full: Full controls [B, horizon, nu] o [horizon, nu]
        """
        was_unbatched = U_blocked.ndim == 2
        if was_unbatched:
            U_blocked = U_blocked.unsqueeze(0)
        
        B, n_blocks, nu = U_blocked.shape
        assert n_blocks == self.n_blocks, f"Expected {self.n_blocks} blocks, got {n_blocks}"
        
        # Efficient expansion using broadcasting
        # U_blocked: [B, n_blocks, nu] -> [B, n_blocks, nu, 1]
        # blocking_matrix: [n_blocks, horizon] -> [1, n_blocks, 1, horizon]
        U_blocked_expanded = U_blocked.unsqueeze(-1)  # [B, n_blocks, nu, 1]
        blocking_matrix_expanded = self.blocking_matrix.unsqueeze(0).unsqueeze(2)  # [1, n_blocks, 1, horizon]
        
        # Broadcast multiplication and sum over blocks
        U_full = torch.sum(U_blocked_expanded * blocking_matrix_expanded, dim=1)  # [B, nu, horizon]
        U_full = U_full.transpose(-1, -2)  # [B, horizon, nu]
        
        if was_unbatched:
            U_full = U_full.squeeze(0)
        
        return U_full
    
    def compress_full_controls(self, U_full: Tensor) -> Tensor:
        """
        Compress full controls to blocked representation (averaging within blocks).
        
        Args:
            U_full: Full controls [B, horizon, nu] o [horizon, nu]
            
        Returns:
            U_blocked: Blocked controls [B, n_blocks, nu] o [n_blocks, nu]
        """
        was_unbatched = U_full.ndim == 2
        if was_unbatched:
            U_full = U_full.unsqueeze(0)
        
        B, horizon, nu = U_full.shape
        assert horizon == self.horizon, f"Expected horizon {self.horizon}, got {horizon}"
        
        U_blocked = torch.zeros(B, self.n_blocks, nu, device=U_full.device, dtype=U_full.dtype)
        
        for block_idx, timestep_indices in enumerate(self.block_to_timestep_indices):
            # Average controls within block
            block_controls = U_full[:, timestep_indices, :]  # [B, block_size, nu]
            U_blocked[:, block_idx, :] = torch.mean(block_controls, dim=1)  # [B, nu]
        
        if was_unbatched:
            U_blocked = U_blocked.squeeze(0)
        
        return U_blocked
    
    def solve_step(self, x0: Tensor, U_init: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Override solve_step per utilizzare move blocking optimization.
        
        Args:
            x0: Initial states [B, nx] o [nx]
            U_init: Initial full controls [B, horizon, nu] o [horizon, nu] (optional)
            
        Returns:
            X_opt: Optimal states [B, horizon+1, nx]
            U_opt: Optimal controls [B, horizon, nu]
        """
        was_unbatched = x0.ndim == 1
        if was_unbatched:
            x0 = x0.unsqueeze(0)
        B = x0.shape[0]
        
        # Initialize blocked controls
        if U_init is not None:
            if U_init.ndim == 2:
                U_init = U_init.unsqueeze(0)
            U_init_blocked = self.compress_full_controls(U_init)
        else:
            U_init_blocked = torch.zeros(B, self.n_blocks, self.nu, device=self.device, dtype=x0.dtype)
        
        # Extract batch-consistent references
        x_ref_batch = self.cost_module.x_ref
        u_ref_batch = self.cost_module.u_ref
        if x_ref_batch.shape[0] == 1 and B > 1:
            x_ref_batch = x_ref_batch.expand(B, -1, -1)
        if u_ref_batch.shape[0] == 1 and B > 1:
            u_ref_batch = u_ref_batch.expand(B, -1, -1)
        
        # Optimization loop on blocked variables
        U_blocked = U_init_blocked.clone()
        
        for iteration in range(self.max_iter):
            # Expand to full controls per trajectory rollout e cost computation
            U_full = self.expand_blocked_controls(U_blocked)
            
            # Standard trajectory rollout
            X = self.rollout_trajectory(x0, U_full)
            
            # Cost and gradients computation
            current_cost = self.cost_module.objective(X, U_full, x_ref_override=x_ref_batch, u_ref_override=u_ref_batch)
            
            if iteration == 0:
                best_cost = current_cost
                best_X, best_U_blocked = X.clone(), U_blocked.clone()
            
            # Linearization and quadraticization
            A, B_dyn = self.linearize_dynamics(X, U_full)
            l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN = self.cost_module.quadraticize(
                X, U_full, x_ref_override=x_ref_batch, u_ref_override=u_ref_batch
            )
            
            # Backward pass con blocked structure
            K_blocked, k_blocked = self._backward_lqr_blocked(
                A, B_dyn, l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN
            )
            
            # Forward pass con line search
            alpha = self._line_search_blocked(
                x0, X, U_blocked, K_blocked, k_blocked, 
                x_ref_batch, u_ref_batch, current_cost
            )
            
            # Update blocked controls with line search
            if alpha > 1e-8:  # Meaningful step size
                # Simple feedforward update for blocked structure
                U_blocked_new = U_blocked + alpha * k_blocked
                
                # Re-evaluate with new blocked controls
                U_full_new = self.expand_blocked_controls(U_blocked_new)
                X_new = self.rollout_trajectory(x0, U_full_new)
                new_cost = self.cost_module.objective(X_new, U_full_new, x_ref_override=x_ref_batch, u_ref_override=u_ref_batch)
                
                # Accept if improvement
                improvement = torch.mean(best_cost - new_cost)
                if improvement > 1e-6:
                    best_cost = new_cost
                    best_X, best_U_blocked = X_new.clone(), U_blocked_new.clone()
                    U_blocked = U_blocked_new  # Update for next iteration
                    self.converged = True
                    
                    if self.verbose > 1:
                        print(f"  Move blocking iter {iteration}: cost improvement = {improvement.item():.6f}")
                else:
                    if self.verbose > 1:
                        print(f"  Move blocking iter {iteration}: no improvement, breaking")
                    break
            else:
                if self.verbose > 1:
                    print(f"  Move blocking iter {iteration}: step size too small, breaking")
                break
        
        # Final expansion
        U_opt_full = self.expand_blocked_controls(best_U_blocked)
        
        if was_unbatched:
            best_X = best_X.squeeze(0)
            U_opt_full = U_opt_full.squeeze(0)
        
        return best_X, U_opt_full
    
    def _backward_lqr_blocked(
        self, A: Tensor, B_dyn: Tensor, l_x: Tensor, l_u: Tensor,
        l_xx: Tensor, l_xu: Tensor, l_uu: Tensor, l_xN: Tensor, l_xxN: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Backward pass LQR adaptato per move blocking structure.
        
        Returns:
            K_blocked: Feedback gains per blocks [B, n_blocks, nu, nx]  
            k_blocked: Feedforward per blocks [B, n_blocks, nu]
        """
        B_batch = A.shape[0]
        
        # Use vmap for efficient batch processing if possible, otherwise loop
        try:
            from torch.func import vmap as _vmap
            K_full, k_full = _vmap(self.backward_lqr)(A, B_dyn, l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN)
        except:
            # Fallback: manual loop over batch
            K_list, k_list = [], []
            for b in range(B_batch):
                K_b, k_b = self.backward_lqr(A[b], B_dyn[b], l_x[b], l_u[b], 
                                           l_xx[b], l_xu[b], l_uu[b], l_xN[b], l_xxN[b])
                K_list.append(K_b)
                k_list.append(k_b)
            K_full = torch.stack(K_list, dim=0)
            k_full = torch.stack(k_list, dim=0)
        
        # Compress to blocked representation
        K_blocked = torch.zeros(B_batch, self.n_blocks, self.nu, self.nx, 
                               device=A.device, dtype=A.dtype)
        k_blocked = torch.zeros(B_batch, self.n_blocks, self.nu, 
                               device=A.device, dtype=A.dtype)
        
        for block_idx, timestep_indices in enumerate(self.block_to_timestep_indices):
            # Average gains within each block
            K_block_avg = torch.mean(K_full[:, timestep_indices, :, :], dim=1)  # [B, nu, nx]
            k_block_avg = torch.mean(k_full[:, timestep_indices, :], dim=1)    # [B, nu]
            
            K_blocked[:, block_idx, :, :] = K_block_avg
            k_blocked[:, block_idx, :] = k_block_avg
        
        return K_blocked, k_blocked
    
    def _compute_blocked_feedback_update(self, X: Tensor, U_blocked: Tensor, K_blocked: Tensor) -> Tensor:
        """Compute feedback update per blocked controls"""
        B_batch = X.shape[0]
        
        # Expand current blocked controls
        U_full = self.expand_blocked_controls(U_blocked)
        
        # Compute state deviations from trajectory
        du_update_blocked = torch.zeros_like(U_blocked)
        
        for block_idx, timestep_indices in enumerate(self.block_to_timestep_indices):
            # Average state deviations within block  
            dx_block = torch.mean(X[:, timestep_indices, :], dim=1)  # [B, nx]
            
            # Apply averaged feedback gain
            du_block = torch.einsum('bji,bj->bi', K_blocked[:, block_idx, :, :], dx_block)
            du_update_blocked[:, block_idx, :] = du_block
        
        return du_update_blocked
    
    def _line_search_blocked(
        self, x0: Tensor, X_ref: Tensor, U_blocked_ref: Tensor,
        K_blocked: Tensor, k_blocked: Tensor,
        x_ref_batch: Tensor, u_ref_batch: Tensor, 
        current_cost: Tensor
    ) -> float:
        """Line search adaptato per blocked controls"""
        
        if hasattr(self, 'use_armijo_line_search') and self.use_armijo_line_search:
            return self._armijo_line_search_blocked(
                x0, X_ref, U_blocked_ref, K_blocked, k_blocked,
                x_ref_batch, u_ref_batch, current_cost
            )
        else:
            return self._standard_line_search_blocked(
                x0, X_ref, U_blocked_ref, K_blocked, k_blocked,
                x_ref_batch, u_ref_batch, current_cost
            )
    
    def _standard_line_search_blocked(
        self, x0: Tensor, X_ref: Tensor, U_blocked_ref: Tensor,
        K_blocked: Tensor, k_blocked: Tensor,
        x_ref_batch: Tensor, u_ref_batch: Tensor,
        current_cost: Tensor
    ) -> float:
        """Standard grid search per blocked controls"""
        
        alphas = [1.0, 0.5, 0.25, 0.125, 0.0625]
        best_alpha = 0.0
        best_improvement = -float('inf')
        
        for alpha in alphas:
            try:
                # Simple feedforward update test
                U_blocked_test = U_blocked_ref + alpha * k_blocked
                
                # Expand and evaluate
                U_full_test = self.expand_blocked_controls(U_blocked_test)
                X_test = self.rollout_trajectory(x0, U_full_test)
                
                test_cost = self.cost_module.objective(X_test, U_full_test, x_ref_override=x_ref_batch, u_ref_override=u_ref_batch)
                improvement = torch.mean(current_cost - test_cost)
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_alpha = alpha
                    
            except Exception as e:
                if self.verbose > 2:
                    print(f"Line search alpha {alpha} failed: {e}")
                continue
        
        return best_alpha
    
    def _armijo_line_search_blocked(
        self, x0: Tensor, X_ref: Tensor, U_blocked_ref: Tensor,
        K_blocked: Tensor, k_blocked: Tensor,
        x_ref_batch: Tensor, u_ref_batch: Tensor,
        current_cost: Tensor
    ) -> float:
        """Armijo line search per blocked controls"""
        
        # Simplified Armijo for blocked case - use standard grid as fallback
        return self._standard_line_search_blocked(
            x0, X_ref, U_blocked_ref, K_blocked, k_blocked,
            x_ref_batch, u_ref_batch, current_cost
        )
    
    def get_blocking_info(self) -> Dict:
        """Get informazioni su move blocking configuration"""
        return {
            "n_blocks": self.n_blocks,
            "block_sizes": self.block_sizes,
            "original_dof": self.horizon * self.nu,
            "reduced_dof": self.n_blocks * self.nu,
            "reduction_ratio": (self.n_blocks * self.nu) / (self.horizon * self.nu),
            "blocking_pattern": self.blocking_pattern,
            "compression_factor": self.horizon / self.n_blocks
        }
    
    def forward(self, x0: Tensor, U_init: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Override forward to bypass autograd ILQRSolve and use direct solve_step.
        Move blocking has its own optimization structure that doesn't need the autograd wrapper.
        """
        B = x0.shape[0] if x0.ndim > 1 else 1
        if U_init is None:
            U_init = torch.zeros(B, self.horizon, self.nu, device=x0.device, dtype=x0.dtype)
        
        # Use the move blocking solve_step directly
        return self.solve_step(x0, U_init)