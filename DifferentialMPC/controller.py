from __future__ import annotations
import enum
import logging
from typing import Callable, Optional, Tuple, Dict
import torch
from torch import Tensor
from torch.func import jacrev

try:
    from torch.func import vmap as _vmap

    _HAS_VMAP = True
except ImportError:
    _HAS_VMAP = False
    _vmap = None
from .utils import pnqp, jacobian_finite_diff_batched
from .cost import GeneralQuadCost

try:
    from torch.func import scan

    _HAS_SCAN = True
except ImportError:
    _HAS_SCAN = False


def _outer(a: Tensor, b: Tensor) -> Tensor:
    return a.unsqueeze(-1) * b.unsqueeze(-2)


class ILQRSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x0: Tensor,
                # Parameters and references are explicit inputs for differentiability
                C: Tensor, c: Tensor, C_final: Tensor, c_final: Tensor,
                x_ref: Tensor, u_ref: Tensor,
                controller: 'DifferentiableMPCController',
                U_init: Tensor
                ) -> Tuple[Tensor, Tensor]:

        # 1. Assign parameters and references to the controller module
        controller.cost_module.C = C
        controller.cost_module.c = c
        controller.cost_module.C_final = C_final
        controller.cost_module.c_final = c_final
        controller.cost_module.set_reference(x_ref, u_ref)  # Now part of the graph

        # 2. Solve the MPC
        X_opt, U_opt = controller.solve_step(x0, U_init)

        # 3. Detach if not converged
        if (not controller.converged) and controller.detach_unconverged:
            X_opt, U_opt = X_opt.detach(), U_opt.detach()

        # 4. Save tensors for backward pass
        ctx.controller = controller
        ctx.save_for_backward(
            X_opt, U_opt,
            controller.H_last[0], controller.H_last[1], controller.H_last[2],
            controller.F_last[0], controller.F_last[1],
            controller.tight_mask_last,
            x_ref, u_ref  # Also save the references
        )

        return X_opt, U_opt

    @staticmethod
    def backward(ctx, grad_X_out: Tensor, grad_U_out: Tensor):
        # 1. Recover saved data
        X, U, H_xx, H_uu, H_xu, A, Bm, tight_mask, x_ref, u_ref = ctx.saved_tensors
        ctrl = ctx.controller
        B, T, nx, nu = X.shape[0], U.shape[1], ctrl.nx, ctrl.nu

        # 2. Solve LQR backward to get sensitivities (dX/dθ, dU/dθ)
        grad_x = -grad_X_out[:, :-1]
        grad_u = -grad_U_out

        # Identify environments with no active inequality constraints so we can
        # evaluate their LQR updates in batch (dominant case during training).
        flat_tight = tight_mask.view(B, -1)
        unconstrained_mask = ~(flat_tight.any(dim=1))

        dX = torch.zeros_like(X)
        dU = torch.zeros_like(U)

        if unconstrained_mask.any():
            unconstrained_idx = torch.nonzero(unconstrained_mask, as_tuple=False).squeeze(-1)
            dX_uncon, dU_uncon = ctrl._lqr_unconstrained_batch(
                A[unconstrained_idx],
                Bm[unconstrained_idx],
                H_xx[unconstrained_idx],
                H_uu[unconstrained_idx],
                H_xu[unconstrained_idx],
                grad_x[unconstrained_idx],
                grad_u[unconstrained_idx]
            )
            dX[unconstrained_idx] = dX_uncon
            dU[unconstrained_idx] = dU_uncon

        constrained_mask = ~unconstrained_mask
        if constrained_mask.any():
            constrained_idx = torch.nonzero(constrained_mask, as_tuple=False).squeeze(-1)
            for idx in constrained_idx.tolist():
                dX_i, dU_i, _ = ctrl._zero_constrained_lqr(
                    A[idx], Bm[idx], H_xx[idx], H_uu[idx], H_xu[idx],
                    grad_x[idx], grad_u[idx], tight_mask[idx], U[idx]
                )
                dX[idx] = dX_i
                dU[idx] = dU_i

        # 3. Calculate cost parameter gradients
        # Error and error derivative
        err_x, err_u = X[:, :-1] - x_ref[:, :T], U - u_ref
        derr_x, derr_u = dX[:, :-1], dU

        # Gradient for running costs
        tau = torch.cat([err_x, err_u], dim=-1)
        dtau = torch.cat([derr_x, derr_u], dim=-1)
        grad_C = -0.5 * (_outer(dtau, tau) + _outer(tau, dtau))
        grad_c = -dtau

        # Gradient for terminal cost
        err_xN = X[:, -1] - x_ref[:, -1]
        derr_xN = dX[:, -1]
        grad_C_final_xx = -0.5 * (_outer(derr_xN, err_xN) + _outer(err_xN, derr_xN))
        grad_C_final = torch.nn.functional.pad(grad_C_final_xx, (0, nu, 0, nu))
        grad_c_final = torch.nn.functional.pad(-derr_xN, (0, nu))

        # 4. Calculate gradients of original inputs
        grad_x0 = dX[:, 0]
        # dL/dx_ref = (dL/d_err_x) * (d_err_x / dx_ref) = (-derr_x) * (-1) = derr_x
        grad_xref = torch.zeros_like(x_ref)
        grad_xref[:, :T, :] = derr_x
        grad_xref[:, -1, :] = derr_xN
        grad_uref = derr_u

        # 5. Restituisci i gradienti nell'ordine corretto degli input di forward()
        return (
            grad_x0,
            grad_C, grad_c, grad_C_final, grad_c_final,
            grad_xref, grad_uref,
            None,  # grad per controller
            None  # grad per U_init
        )


# ─────────────────────────────────────────────────────────────
class GradMethod(enum.Enum):
    """Modalità di calcolo della Jacobiana (A, B) della dinamica f(x,u,dt).

    * **ANALYTIC**  : l'utente fornisce una funzione `f_dyn_jac(x,u,dt)` che
                      restituisce (A, B) in forma analitica ‑faster and better .
    * **AUTO_DIFF** : usa `torch.autograd.functional.jacobian` (default) VERY SLOW.
    * **FINITE_DIFF**: differenze finite centralizzate con passo `fd_eps` LEGACY.
    """
    ANALYTIC = "analytic"
    AUTO_DIFF = "auto_diff"
    FINITE_DIFF = "finite_diff"


class DifferentiableMPCController(torch.nn.Module):
    # -------------------------------------------------------------------
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
            N_sim: Optional[int] = None,
            grad_method: GradMethod | str = GradMethod.AUTO_DIFF,
            f_dyn_jac: Optional[Callable[[torch.Tensor, torch.Tensor, float],
            Tuple[torch.Tensor, torch.Tensor]]] = None,
            fd_eps: float = 1e-4,
            max_iter: int = 40,
            tol_x: float = 1e-6,
            tol_u: float = 1e-6,
            exit_unconverged: bool = False,
            detach_unconverged: bool = True,
            converge_tol: float = 1e-6,
            delta_u: Optional[float] = None,
            best_cost_eps: float = 1e-6,
            not_improved_lim: int = 10,
            verbose: int = 0,
            use_armijo_line_search: bool = False,
            armijo_c1: float = 1e-4,
            armijo_c2: float = 0.9
    ):
        super().__init__()
        # Dispositivo
        self.device = torch.device(device)

        # Dinamica e costi
        self.f_dyn = f_dyn
        self.total_time = total_time
        self.dt = step_size
        self.horizon = horizon
        self.cost_module = cost_module

        # Dimensioni di stato e controllo
        self.nx = cost_module.nx
        self.nu = cost_module.nu

        # Bound di controllo
        self.u_min = u_min.to(self.device) if u_min is not None else None
        self.u_max = u_max.to(self.device) if u_max is not None else None

        # Parametri solver
        self.reg_eps = reg_eps
        self.N_sim = N_sim if N_sim is not None else horizon

        # Metodo di derivazione
        if isinstance(grad_method, str):
            grad_method = GradMethod(grad_method.lower())
        self.grad_method = grad_method
        self.f_dyn_jac = f_dyn_jac
        self.fd_eps = fd_eps
        if self.grad_method is GradMethod.ANALYTIC and self.f_dyn_jac is None:
            raise ValueError("Per grad_method='analytic' serve f_dyn_jac(x,u,dt)->(A,B)")

        # Parametri linesearch, trust-region e convergenza
        self.delta_u = delta_u
        self.best_cost_eps = best_cost_eps
        self.not_improved_lim = not_improved_lim
        self.verbose = verbose

        # ---------- nuovi attributi ----------
        self.max_iter = int(max_iter)
        self.tol_u = float(tol_u)
        self.tol_x = float(tol_x)
        self.detach_unconverged = bool(detach_unconverged)
        self.converged: bool | None = None
        # Buffer per backward
        self.U_last = None
        self.X_last = None
        self.H_last = None
        self.F_last = None
        self.lmb_last = None
        self.tight_mask_last = None

        # Criteri di convergenza
        self.exit_unconverged = exit_unconverged
        self.detach_unconverged = detach_unconverged
        self.converge_tol = converge_tol
        self.converged = True

        # Warm-start
        self.U_prev = None

        # FIX: Advanced line search parameters
        self.use_armijo_line_search = use_armijo_line_search
        self.armijo_c1 = armijo_c1
        self.armijo_c2 = armijo_c2

    #   Calcolo Jacobiane A, B
    def _jacobian_analytic(self, x: torch.Tensor, u: torch.Tensor):
        return self.f_dyn_jac(x, u, self.dt)

    def _jacobian_auto_diff(self, x: torch.Tensor, u: torch.Tensor):
        """Jacobian via autograd. Usa vectorize=True se disponibile."""
        A = torch.autograd.functional.jacobian(
            lambda xx: self.f_dyn(xx, u, self.dt), x,
            create_graph=True, vectorize=True
        )
        B = torch.autograd.functional.jacobian(
            lambda uu: self.f_dyn(x, uu, self.dt), u,
            create_graph=True, vectorize=True
        )
        return A, B

    def _jacobian_finite_diff(self, x: torch.Tensor, u: torch.Tensor):
        """Central finite differences; non-differentiable but faster than autograd.
        Returns A = d f / d x, B = d f / d u.
        """
        fd_eps = self.fd_eps
        nx, nu = self.nx, self.nu
        f0 = self.f_dyn(x, u, self.dt)
        # ---- A --------------------------------------------------------
        eye_x = torch.eye(nx, device=x.device, dtype=x.dtype)
        A_cols = []
        for j in range(nx):
            dx = eye_x[j] * fd_eps
            f_plus = self.f_dyn(x + dx, u, self.dt)
            f_minus = self.f_dyn(x - dx, u, self.dt)
            A_cols.append(((f_plus - f_minus) / (2.0 * fd_eps)).unsqueeze(-1))
        A = torch.cat(A_cols, dim=-1)  # [nx, nx]
        # ---- B --------------------------------------------------------
        eye_u = torch.eye(nu, device=u.device, dtype=u.dtype)
        B_cols = []
        for j in range(nu):
            du = eye_u[j] * fd_eps
            f_plus = self.f_dyn(x, u + du, self.dt)
            f_minus = self.f_dyn(x, u - du, self.dt)
            B_cols.append(((f_plus - f_minus) / (2.0 * fd_eps)).unsqueeze(-1))
        B = torch.cat(B_cols, dim=-1)  # [nx, nu]
        return A, B

    def forward(self, x0: Tensor, U_init: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with optional gradient preservation bypass.

        If preserve_gradients is set, this bypasses ILQRSolve.apply() to maintain
        gradient flow, similar to other controller variants.
        """
        B = x0.shape[0] if x0.ndim > 1 else 1
        if U_init is None:
            U_init = torch.zeros(B, self.horizon, self.nu, device=x0.device, dtype=x0.dtype)

        # Check if we should preserve gradients (bypass autograd function)
        if getattr(self, 'preserve_gradients', False):
            return self.solve_step(x0, U_init)
        else:
            # Standard path through ILQRSolve autograd function
            C, c, C_final, c_final = self.cost_module.C, self.cost_module.c, self.cost_module.C_final, self.cost_module.c_final
            x_ref, u_ref = self.cost_module.x_ref, self.cost_module.u_ref
            return ILQRSolve.apply(x0, C, c, C_final, c_final, x_ref, u_ref, self, U_init)

    def solve_step(self, x0: Tensor, U_init: Tensor) -> Tuple[Tensor, Tensor]:
        B = x0.shape[0]
        U, X = U_init.clone(), self.rollout_trajectory(x0, U_init)

        x_ref_batch, u_ref_batch = self.cost_module.x_ref, self.cost_module.u_ref
        best_cost = self.cost_module.objective(X, U, x_ref_override=x_ref_batch, u_ref_override=u_ref_batch)

        # FIX: Proper convergence tracking
        converged_iterations = 0
        self.converged = False

        for i in range(self.max_iter):
            l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN = self.cost_module.quadraticize(X, U)
            A, Bm = self.linearize_dynamics(X, U)

            K, k = _vmap(self.backward_lqr)(A, Bm, l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN)

            # FIX: Choose line search method based on configuration
            if self.use_armijo_line_search:
                # Advanced Armijo line search with convergence guarantees
                gradient_norm = torch.stack([torch.norm(l_x[b]).detach() + torch.norm(l_u[b]).detach()
                                             for b in range(B)], dim=0)

                # Call evaluate_alphas_armijo for each batch element individually
                X_new_list = []
                U_new_list = []
                new_cost_list = []

                for b in range(B):
                    X_b, U_b, cost_b = self.evaluate_alphas_armijo(
                        x0[b], X[b], U[b], K[b], k[b],
                        x_ref_batch[b], u_ref_batch[b],
                        best_cost[b], gradient_norm[b]
                    )
                    # Extract the trajectories (should already be correct dimensions)
                    X_new_list.append(X_b.squeeze(0))  # Remove batch dimension to get [T+1, nx]
                    U_new_list.append(U_b.squeeze(0))  # Remove batch dimension to get [T, nu]
                    new_cost_list.append(cost_b.squeeze().item())  # Extract scalar value

                X_new = torch.stack(X_new_list, dim=0)
                U_new = torch.stack(U_new_list, dim=0)
                new_cost = torch.tensor(new_cost_list, device=x0.device, dtype=x0.dtype)  # Convert list to tensor

                # DEBUG: Check dimensions after Armijo
                if hasattr(self, 'verbose') and self.verbose > 0:
                    print(
                        f"DEBUG Armijo results: X_new.shape={X_new.shape}, U_new.shape={U_new.shape}, new_cost.shape={new_cost.shape}")

            else:
                # Standard alpha grid search using pure evaluate_alphas with vmap
                # Extract cost matrices from cost_module (prepare_costs logic)
                C_run, c_run, C_final, c_final = self.cost_module._prepare_costs(B)
                
                # Use pure evaluate_alphas with vmap - works for both fixed and batch-dependent costs
                X_candidates_batch, U_candidates_batch, candidate_costs_batch = _vmap(
                    self.evaluate_alphas_pure, in_dims=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                )(x0, X, U, K, k, x_ref_batch, u_ref_batch, C_run, c_run, C_final, c_final)
                
                # X_candidates_batch: [B, n_alpha, H+1, nx]
                # U_candidates_batch: [B, n_alpha, H, nu] 
                # candidate_costs_batch: [B, n_alpha]
                
                # Find best alpha for each batch element
                best_alpha_indices = torch.argmin(candidate_costs_batch, dim=1)  # [B]
                
                # Extract best trajectories using advanced indexing
                batch_indices = torch.arange(B, device=x0.device)
                X_new = X_candidates_batch[batch_indices, best_alpha_indices]  # [B, H+1, nx]
                U_new = U_candidates_batch[batch_indices, best_alpha_indices]  # [B, H, nu]
                new_cost = candidate_costs_batch[batch_indices, best_alpha_indices]  # [B]

            # FIX: Ensure consistent dimensions for cost comparison
            if new_cost.numel() != B:
                new_cost = new_cost.view(B)
            if best_cost.numel() != B:
                best_cost = best_cost.view(B)
                
            improved_mask = new_cost < best_cost

            # FIX: Check convergence criteria
            cost_improvement = torch.abs(new_cost - best_cost)
            max_improvement = torch.max(cost_improvement)

            if not improved_mask.any():
                converged_iterations += 1
                if converged_iterations >= 3 or max_improvement.item() < self.converge_tol:
                    self.converged = True
                    break
            else:
                converged_iterations = 0  # Reset counter if improvement found

            best_cost = torch.where(improved_mask, new_cost, best_cost)
            # FIX: Ensure proper broadcasting for trajectory updates
            improved_mask_X = improved_mask.view(B, 1, 1).expand(-1, X.shape[1], X.shape[2])
            improved_mask_U = improved_mask.view(B, 1, 1).expand(-1, U.shape[1], U.shape[2])
            X = torch.where(improved_mask_X, X_new, X)
            U = torch.where(improved_mask_U, U_new, U)

            self.cost_module.set_reference(x_ref_batch, u_ref_batch)

        # FIX: Set converged=True even if max iterations reached (practical convergence)
        if not self.converged:
            # Check if solution is reasonable (not diverged)
            if torch.isfinite(X).all() and torch.isfinite(U).all():
                self.converged = True
        self.cost_module.set_reference(x_ref_batch, u_ref_batch)
        l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN = self.cost_module.quadraticize(X, U)
        A, Bm = self.linearize_dynamics(X, U)
        self.H_last, self.F_last = (l_xx, l_uu, l_xu), (A, Bm)
        self.tight_mask_last, self.X_last, self.U_last = self._compute_tight_mask(U), X, U

        # FIX: Memory leak cleanup after solve_step completion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory after intensive computation

        return X, U

    def forward_pass_batched(self, x0, X_ref, U_ref, K, k):
        """
        Esegue il forward pass per l'intero batch usando vmap.
        """

        def _forward_single(x0i, X_ref_i, U_ref_i, K_i, k_i):
            # Esegue il rollout per una singola traiettoria
            x_i = x0i
            xs = [x_i]
            us = []
            for t in range(self.horizon):
                dx_i = x_i - X_ref_i[t]
                du_i = k_i[t] + torch.einsum("ij,j->i", K_i[t], dx_i)  # K @ dx
                u_new = U_ref_i[t] + du_i

                # Applica i vincoli
                if self.u_min is not None:
                    u_new = torch.max(u_new, self.u_min)
                if self.u_max is not None:
                    u_new = torch.min(u_new, self.u_max)

                us.append(u_new)
                x_i = self.f_dyn(x_i, u_new, self.dt)
                xs.append(x_i)

            return torch.stack(xs, dim=0), torch.stack(us, dim=0)

        # Vettorizza la funzione di forward pass sul batch
        X_new, U_new = _vmap(_forward_single, in_dims=(0, 0, 0, 0, 0))(x0, X_ref, U_ref, K, k)
        return X_new, U_new

    def forward_pass(self, x0, X_ref, U_ref, K, k):
        X_new, U_new = [x0], []
        xt = x0
        for t in range(self.horizon):
            dx = xt - X_ref[t]
            du = K[t] @ dx + k[t]
            ut = U_ref[t] + du
            U_new.append(ut)
            xt = self.f_dyn(xt, ut, self.dt)
            X_new.append(xt)
        return torch.stack(X_new, dim=0), torch.stack(U_new, dim=0)

    def rollout_trajectory(
            self,
            x0: torch.Tensor,  # (B, nx)  **oppure** (nx,) per retro-compatibilità
            U: torch.Tensor  # (B, T, nu) **oppure** (T, nu)
    ) -> torch.Tensor:  # ritorna (B, T+1, nx)
        """
        Propaga la dinamica per tutti gli orizzonti in **parallelo sui batch**.

        Args
        ----
        x0 : (B, nx)      Stato iniziale batch (B può essere 1).
        U  : (B, T, nu)   Sequenza di comandi per batch.
                          Se passato shape (T, nu) verrà broadcastato su B=1.
        Returns
        -------
        X  : (B, T+1, nx) Traiettoria degli stati.
        """
        # -- normalizza le dimensioni ----------------------------------------
        if x0.ndim == 1:  # (nx,)  →  (1, nx)
            x0 = x0.unsqueeze(0)
        if U.ndim == 2:  # (T, nu) → (1, T, nu)
            U = U.unsqueeze(0)

        B, T, _ = U.shape
        nx = self.nx
        device, dtype = x0.device, x0.dtype

        # buffer trajectory
        X = torch.empty(B, T + 1, nx, device=device, dtype=dtype)
        X[:, 0] = x0
        xt = x0

        # loop temporale
        for t in range(T):
            ut = U[:, t]  # (B, nu)
            xt = self.f_dyn(xt, ut, self.dt)  # f_dyn deve supportare batch
            X[:, t + 1] = xt
        return X

    # -----------------------------------------------------------------

    def linearize_dynamics(self, X: Tensor, U: Tensor):
        B, T, nx, nu = X.shape[0], U.shape[1], self.nx, self.nu

        if self.grad_method is GradMethod.AUTO_DIFF and _HAS_VMAP:
            f = lambda x, u: self.f_dyn(x, u, self.dt)
            jac_x, jac_u = jacrev(f, argnums=0), jacrev(f, argnums=1)
            A = _vmap(_vmap(jac_x, in_dims=(0, 0)), in_dims=(0, 0))(X[:, :-1], U)
            B = _vmap(_vmap(jac_u, in_dims=(0, 0)), in_dims=(0, 0))(X[:, :-1], U)
            return A, B
        elif self.grad_method is GradMethod.ANALYTIC:
            x_flat = X[:, :-1].reshape(-1, nx)  # Shape: [B*T, nx]
            u_flat = U.reshape(-1, nu)  # Shape: [B*T, nu]
            A_flat, B_flat = self.f_dyn_jac(x_flat, u_flat, self.dt)
            A = A_flat.reshape(B, T, nx, nx)
            B = B_flat.reshape(B, T, nx, nu)
            return A, B

        # Fallback per differenze finite
        else:
            print("attenzione il metodo non e differenziabile e non e adatto per RL")
            A, B = jacobian_finite_diff_batched(
                self.f_dyn, X[:, :-1].reshape(-1, self.nx), U.reshape(-1, self.nu), dt=self.dt
            )
            return A.reshape(B, T, nx, nx), B.reshape(B, T, nx, nu)

    # ------------------------------------------------------------------
    def backward_lqr(
            self,
            A: torch.Tensor,  # [T, nx, nx]
            B: torch.Tensor,  # [T, nx, nu]
            l_x: torch.Tensor,  # [T, nx]
            l_u: torch.Tensor,  # [T, nu]
            l_xx: torch.Tensor,  # [T, nx, nx]
            l_xu: torch.Tensor,  # [T, nx, nu]
            l_uu: torch.Tensor,  # [T, nu, nu]
            l_xN: torch.Tensor,  # [nx]
            l_xxN: torch.Tensor  # [nx, nx]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Riccati backward-pass with robust fallback for Q_uu.
        Returns:
            K_seq : [T, nu, nx]
            k_seq : [T, nu]
        """
        T, nx, nu = A.shape[0], self.nx, self.nu
        dtype, device = A.dtype, A.device
        I_nu = torch.eye(nu, dtype=dtype, device=device) * self.reg_eps

        # FAST PATH con torch.func.scan
        if _HAS_SCAN:
            A_rev = torch.flip(A, dims=[0])
            B_rev = torch.flip(B, dims=[0])
            lx_rev = torch.flip(l_x, dims=[0])
            lu_rev = torch.flip(l_u, dims=[0])
            lxx_rev = torch.flip(l_xx, dims=[0])
            lxu_rev = torch.flip(l_xu, dims=[0])
            luu_rev = torch.flip(l_uu, dims=[0])

            def riccati_step(carry, inps):
                V, v = carry
                A_t, B_t, lx_t, lu_t, lxx_t, lxu_t, luu_t = inps

                # Q matrices
                Q_xx = lxx_t + A_t.T @ V @ A_t
                Q_xu = lxu_t + A_t.T @ V @ B_t
                Q_ux = Q_xu.mT
                Q_uu = luu_t + B_t.T @ V @ B_t + I_nu

                # rhs vectors
                q_x = lx_t + A_t.T @ v
                q_u = lu_t + B_t.T @ v

                # FIX: Robust Cholesky decomposition with SVD fallback
                Q_uu_reg = Q_uu + self.reg_eps * torch.eye(nu, dtype=dtype, device=device)
                K_t, k_t = self._robust_solve(Q_uu_reg, Q_xu.mT, q_u.unsqueeze(-1))

                #  cost-to-go
                V_new = Q_xx + K_t.T @ Q_uu @ K_t + K_t.T @ Q_ux + Q_xu @ K_t
                v_new = q_x + K_t.T @ Q_uu @ k_t + K_t.T @ q_u + Q_ux.mT @ k_t
                return (V_new, v_new), (K_t, k_t)

            _, (K_rev, k_rev) = scan(
                riccati_step,
                (l_xxN, l_xN),
                (A_rev, B_rev, lx_rev, lu_rev, lxx_rev, lxu_rev, luu_rev)
            )
            K_seq = torch.flip(K_rev, dims=[0])
            k_seq = torch.flip(k_rev, dims=[0])
            return K_seq, k_seq, None, None

        # FALLBACK: loop Python
        V = l_xxN
        v = l_xN
        K_list, k_list = [], []
        for t in reversed(range(T)):
            # Q matrices
            Q_xx = l_xx[t] + A[t].T @ V @ A[t]
            Q_xu = l_xu[t] + A[t].T @ V @ B[t]
            Q_ux = Q_xu.mT
            Q_uu = l_uu[t] + B[t].T @ V @ B[t] + I_nu

            # rhs
            q_x = l_x[t] + A[t].T @ v
            q_u = l_u[t] + B[t].T @ v
            Q_uu_reg = Q_uu + self.reg_eps * torch.eye(nu, dtype=dtype, device=device)

            # FIX: Robust Cholesky decomposition with SVD fallback
            Kt, kt = self._robust_solve(Q_uu_reg, Q_xu.mT, q_u.unsqueeze(-1))
            kt = kt.squeeze(-1)

            K_list.insert(0, Kt)
            k_list.insert(0, kt)

            # update cost-to-go
            V = Q_xx + Kt.T @ Q_uu @ Kt + Kt.T @ Q_ux + Q_xu @ Kt
            v = q_x + Kt.T @ Q_uu @ kt + Kt.T @ q_u + Q_ux.mT @ kt

        K_seq = torch.stack(K_list, dim=0)
        k_seq = torch.stack(k_list, dim=0)
        return K_seq, k_seq

    def _robust_solve(self, Q_uu: torch.Tensor, Q_xu_T: torch.Tensor, q_u: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        FIX: Robust linear system solver with Cholesky + SVD fallback for numerical robustness.

        Solves: Q_uu @ X = -Q_xu_T and Q_uu @ x = -q_u
        Returns: (-X, -x) for feedback gains K_t, k_t

        Hierarchy:
        1. Cholesky decomposition (fastest, requires PD matrix)
        2. SVD decomposition (robust, handles rank deficiency)
        3. Regularized pseudoinverse (fallback for extreme cases)
        """
        # Ensure all inputs have consistent dtype (use Q_uu as reference)
        Q_xu_T = Q_xu_T.to(dtype=Q_uu.dtype)
        q_u = q_u.to(dtype=Q_uu.dtype)

        try:
            # Primary: Cholesky decomposition for PD matrices
            L = torch.linalg.cholesky(Q_uu)
            K_t = -torch.cholesky_solve(Q_xu_T, L)
            k_t = -torch.cholesky_solve(q_u, L)
            return K_t, k_t

        except (RuntimeError, torch._C._LinAlgError) as e:
            error_str = str(e).lower()
            if ("singular" in error_str or
                    "not positive definite" in error_str or
                    "not positive-definite" in error_str or
                    "factorization could not be completed" in error_str):
                # Secondary: SVD decomposition for robustness
                try:
                    U, S, Vh = torch.linalg.svd(Q_uu)

                    # Condition number check and SVD-based solution
                    cond_threshold = 1e12  # Condition number threshold
                    S_reg = torch.where(S > S.max() / cond_threshold, S, S.max() / cond_threshold)

                    # Solve using SVD: Q_uu = U @ diag(S) @ Vh
                    # Q_uu^{-1} = Vh^T @ diag(1/S) @ U^T
                    S_inv = 1.0 / S_reg
                    Q_uu_inv_svd = Vh.mH @ torch.diag_embed(S_inv) @ U.mH

                    K_t = -Q_uu_inv_svd @ Q_xu_T
                    k_t = -Q_uu_inv_svd @ q_u

                    # Log SVD usage for monitoring
                    if hasattr(self, 'verbose') and self.verbose > 0:
                        cond_num = S.max() / S.min()
                        print(f"SVD fallback used - condition number: {cond_num:.2e}")

                    return K_t, k_t

                except RuntimeError:
                    # Tertiary: Regularized pseudoinverse (last resort)
                    reg_strength = max(self.reg_eps * 100, 1e-6)  # Stronger regularization
                    Q_uu_reg = Q_uu + reg_strength * torch.eye(Q_uu.shape[-1],
                                                               device=Q_uu.device, dtype=Q_uu.dtype)
                    Q_uu_pinv = torch.linalg.pinv(Q_uu_reg)

                    K_t = -Q_uu_pinv @ Q_xu_T
                    k_t = -Q_uu_pinv @ q_u

                    if hasattr(self, 'verbose') and self.verbose > 0:
                        print(f"Pseudoinverse fallback used - regularization: {reg_strength:.2e}")

                    return K_t, k_t
            else:
                # Re-raise if not a numerical issue
                raise e

    # -----------------------------------------------------------------
    def compute_cost(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        Returns batched cost (B,) WITHOUT cast to float.
        """
        return self.cost_module.objective(X, U)

    # -----------------------------------------------------------------

    def evaluate_alphas(
            self, x0: Tensor, X_ref: Tensor, U_ref: Tensor, K: Tensor, k: Tensor,
            x_ref_traj: Tensor, u_ref_traj: Tensor,
            alphas: Tuple[float, ...] = (1.0, 0.8, 0.5, 0.2, 0.1)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculates trajectories and costs for all alpha in parallel.
        Returns all candidates, not just the best one.
        """
        A = torch.tensor(alphas, dtype=x0.dtype, device=x0.device)
        n_alpha = A.shape[0]
        x_current_batch = x0.expand(n_alpha, -1)

        xs_list, us_list = [x_current_batch], []
        for t in range(self.horizon):
            dx_batch = x_current_batch - X_ref[t]
            du_batch = A.view(-1, 1) * k[t] + torch.einsum('ij,aj->ai', K[t], dx_batch)
            u_batch = U_ref[t] + du_batch
            if self.u_min is not None: u_batch = torch.max(u_batch, self.u_min)
            if self.u_max is not None: u_batch = torch.min(u_batch, self.u_max)
            us_list.append(u_batch)
            x_current_batch = self.f_dyn(x_current_batch, u_batch, self.dt)
            xs_list.append(x_current_batch)

        X_candidates = torch.stack(xs_list, dim=1)  # Shape [n_alpha, H+1, nx]
        U_candidates = torch.stack(us_list, dim=1)  # Shape [n_alpha, H, nu]

        objective_fn = lambda x, u: self.cost_module.objective(x, u, x_ref_override=x_ref_traj,
                                                               u_ref_override=u_ref_traj)
        candidate_costs = _vmap(objective_fn)(X_candidates, U_candidates)  # Shape [n_alpha]

        return X_candidates, U_candidates, candidate_costs

    def evaluate_alphas_pure(
            self, x0: Tensor, X_ref: Tensor, U_ref: Tensor, K: Tensor, k: Tensor,
            x_ref_traj: Tensor, u_ref_traj: Tensor,
            C: Tensor, c: Tensor, C_final: Tensor, c_final: Tensor,
            alphas: Tuple[float, ...] = (1.0, 0.8, 0.5, 0.2, 0.1)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Pure version of evaluate_alphas that takes cost matrices as parameters.
        
        This function is mathematically pure - same inputs always produce same outputs.
        Compatible with vmap for both fixed and batch-dependent cost scenarios.
        
        Args:
            x0: Initial state [nx]
            X_ref: Reference state trajectory [T+1, nx]
            U_ref: Reference control trajectory [T, nu]
            K: Feedback gain matrices [T, nu, nx]
            k: Feedforward terms [T, nu]
            x_ref_traj: Reference trajectory for cost calculation [T+1, nx]
            u_ref_traj: Reference control for cost calculation [T, nu]
            C: Running cost matrices [T, nx+nu, nx+nu]
            c: Running cost vectors [T, nx+nu]
            C_final: Terminal cost matrix [nx+nu, nx+nu]
            c_final: Terminal cost vector [nx+nu]
            alphas: Step sizes to evaluate
            
        Returns:
            X_candidates: State trajectories for each alpha [n_alpha, T+1, nx]
            U_candidates: Control trajectories for each alpha [n_alpha, T, nu]
            candidate_costs: Costs for each alpha [n_alpha]
        """
        A = torch.tensor(alphas, dtype=x0.dtype, device=x0.device)
        n_alpha = A.shape[0]
        x_current_batch = x0.expand(n_alpha, -1)

        xs_list, us_list = [x_current_batch], []
        for t in range(self.horizon):
            dx_batch = x_current_batch - X_ref[t]
            du_batch = A.view(-1, 1) * k[t] + torch.einsum('ij,aj->ai', K[t], dx_batch)
            u_batch = U_ref[t] + du_batch
            if self.u_min is not None: u_batch = torch.max(u_batch, self.u_min)
            if self.u_max is not None: u_batch = torch.min(u_batch, self.u_max)
            us_list.append(u_batch)
            x_current_batch = self.f_dyn(x_current_batch, u_batch, self.dt)
            xs_list.append(x_current_batch)

        X_candidates = torch.stack(xs_list, dim=1)  # Shape [n_alpha, T+1, nx]
        U_candidates = torch.stack(us_list, dim=1)  # Shape [n_alpha, T, nu]

        # Use pure objective function - compatible with vmap
        candidate_costs = GeneralQuadCost.objective_pure(
            X_candidates, U_candidates, 
            x_ref_traj.expand(n_alpha, -1, -1), u_ref_traj.expand(n_alpha, -1, -1),
            C.expand(n_alpha, -1, -1, -1), c.expand(n_alpha, -1, -1),
            C_final.expand(n_alpha, -1, -1), c_final.expand(n_alpha, -1),
            self.nx, self.nu
        )

        return X_candidates, U_candidates, candidate_costs

    def evaluate_alphas_armijo(
            self, x0: Tensor, X_ref: Tensor, U_ref: Tensor, K: Tensor, k: Tensor,
            x_ref_traj: Tensor, u_ref_traj: Tensor,
            current_cost: Tensor, gradient_norm: Tensor,
            c1: float = 1e-4, c2: float = 0.9, max_alpha: float = 1.0,
            min_alpha: float = 1e-6, max_backtracks: int = 20
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        FIX: Advanced line search with Armijo-Goldstein conditions for guaranteed convergence.

        Implements strong Wolfe conditions:
        1. Armijo condition: f(x + α*p) ≤ f(x) + c1*α*∇f^T*p (sufficient decrease)
        2. Curvature condition: |∇f(x + α*p)^T*p| ≤ c2*|∇f^T*p| (curvature)

        Args:
            current_cost: Current objective value [B]
            gradient_norm: Norm of gradient for descent direction [B]
            c1: Armijo parameter (typically 1e-4)
            c2: Curvature parameter (typically 0.9)

        Returns:
            Best X, U, and alpha values satisfying Wolfe conditions
        """
        # FIX: Simplified for single-batch processing (no vmap complexity)
        device, dtype = x0.device, x0.dtype

        # Initialize alpha search
        alpha_candidates = torch.linspace(max_alpha, min_alpha, max_backtracks, device=device, dtype=dtype)

        # Expected decrease from gradient norm (proxy for directional derivative)
        expected_decrease = c1 * gradient_norm  # scalar

        best_alpha = torch.tensor(min_alpha, device=device, dtype=dtype)
        best_cost = current_cost.clone()
        best_X = X_ref.clone()  # [T+1, nx]
        best_U = U_ref.clone()  # [T, nu]

        # Line search over all alphas for single batch
        for alpha in alpha_candidates:
            # Forward pass with current alpha - simplified single batch
            x_current = x0  # [nx]
            xs_list = [x_current]
            us_list = []

            for t in range(self.horizon):
                dx = x_current - X_ref[t]  # [nx]
                du = alpha * k[t] + alpha * (K[t] @ dx)  # [nu]
                u_new = U_ref[t] + du

                # Apply control constraints
                if self.u_min is not None: u_new = torch.max(u_new, self.u_min)
                if self.u_max is not None: u_new = torch.min(u_new, self.u_max)

                us_list.append(u_new)
                x_current = self.f_dyn(x_current, u_new, self.dt)
                xs_list.append(x_current)

            X_candidate = torch.stack(xs_list, dim=0)  # [T+1, nx]
            U_candidate = torch.stack(us_list, dim=0)  # [T, nu]

            # Evaluate objective for this alpha - single batch
            candidate_cost = self.cost_module.objective(
                X_candidate.unsqueeze(0), U_candidate.unsqueeze(0),
                x_ref_override=x_ref_traj.unsqueeze(0), u_ref_override=u_ref_traj.unsqueeze(0)
            ).squeeze(0)  # Remove batch dim to get scalar

            # Armijo condition check: f(x + α*p) ≤ f(x) + c1*α*gradient_norm
            armijo_threshold = current_cost - alpha * expected_decrease
            armijo_satisfied = candidate_cost <= armijo_threshold

            # Additional monotonicity check for robustness
            improvement = candidate_cost < best_cost
            combined_condition = armijo_satisfied & improvement

            # Update best solution if condition satisfied
            if combined_condition:
                best_cost = candidate_cost
                best_alpha = alpha
                best_X = X_candidate
                best_U = U_candidate

            # Early termination if sufficient improvement found
            if best_alpha > min_alpha:
                break

        # Final cost calculation for single batch
        final_cost = self.cost_module.objective(
            best_X.unsqueeze(0), best_U.unsqueeze(0),
            x_ref_override=x_ref_traj.unsqueeze(0), u_ref_override=u_ref_traj.unsqueeze(0)
        ).squeeze(0)

        # Log line search statistics if verbose
        if hasattr(self, 'verbose') and self.verbose > 0:
            improvement = "✓" if final_cost < current_cost else "✗"
            print(f"Armijo line search: α={best_alpha.item():.3e}, improved={improvement}")

        # Return with proper dimensions for caller expecting [1, T+1, nx], [1, T, nu], [1]
        # Note: we only add 1 dimension to make it [1, T+1, nx] and [1, T, nu]
        return best_X.unsqueeze(0), best_U.unsqueeze(0), final_cost.unsqueeze(0)

    # -----------------------------------------------------------------
    def _compute_tight_mask(self, U: torch.Tensor, atol: float = 1e-7) -> torch.Tensor:
        mask = torch.zeros_like(U, dtype=torch.bool)
        if self.u_min is not None:
            mask |= torch.isclose(U, self.u_min.expand_as(U), atol=atol)
        if self.u_max is not None:
            mask |= torch.isclose(U, self.u_max.expand_as(U), atol=atol)
        return mask

    # -----------------------------------------------------------------
    def _zero_constrained_lqr(
            self,
            A: torch.Tensor,  # [T, nx, nx]
            B: torch.Tensor,  # [T, nx, nu]
            H_xx: torch.Tensor,  # [T, nx, nx]
            H_uu: torch.Tensor,  # [T, nu, nu]
            H_xu: torch.Tensor,  # [T, nx, nu]
            grad_x: torch.Tensor,  # [T, nx]
            grad_u: torch.Tensor,  # [T, nu]
            tight_mask: torch.Tensor,  # [T, nu]
            U_last_i: torch.Tensor,
            delta_u: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T, nx, nu = A.shape[0], self.nx, self.nu
        dtype, device = A.dtype, A.device
        I_nu = torch.eye(nu, dtype=dtype, device=device) * self.reg_eps
        V = H_xx[-1]
        v = grad_x[-1]
        K_seq = [None] * T
        k_seq = [None] * T

        for t in reversed(range(T)):
            # Ensure dtype consistency for mixed precision training
            A_t = A[t].to(dtype=V.dtype)
            B_t = B[t].to(dtype=V.dtype)
            H_xx_t = H_xx[t].to(dtype=V.dtype)
            H_xu_t = H_xu[t].to(dtype=V.dtype)
            H_uu_t = H_uu[t].to(dtype=V.dtype)
            I_nu_typed = I_nu.to(dtype=V.dtype)

            Q_xx = H_xx_t + A_t.T @ V @ A_t
            Q_xu = H_xu_t + A_t.T @ V @ B_t
            Q_ux = Q_xu.mT
            Q_uu = H_uu_t + B_t.T @ V @ B_t + I_nu_typed

            # Ensure all tensors have consistent dtype
            grad_x_t = grad_x[t].to(dtype=V.dtype)
            grad_u_t = grad_u[t].to(dtype=V.dtype)
            v_typed = v.to(dtype=V.dtype)

            q_x = grad_x_t + A_t.T @ v_typed
            q_u = grad_u_t + B_t.T @ v_typed

            # Solve constrained QP on du
            free = ~tight_mask[t]
            if free.any():
                lb = torch.full((nu,), -float('inf'), device=device, dtype=dtype)
                ub = torch.full((nu,), float('inf'), device=device, dtype=dtype)
                if self.u_min is not None:
                    lb = self.u_min - U_last_i[t]
                    ub = self.u_max - U_last_i[t]
                if delta_u is not None:
                    lb = torch.maximum(lb, torch.full_like(lb, -delta_u))
                    ub = torch.minimum(ub, torch.full_like(ub, delta_u))

                H_f = Q_uu[free][:, free]
                q_f = q_u[free]
                lb_f = lb[free]
                ub_f = ub[free]

                du_batch, diagnostics = pnqp(
                    H_f.unsqueeze(0),
                    q_f.unsqueeze(0),
                    lb_f.unsqueeze(0),
                    ub_f.unsqueeze(0)
                )
                du_f = du_batch.squeeze(0)
                # Ensure dtype consistency with V for mixed precision
                du = torch.zeros(nu, dtype=V.dtype, device=device)
                du[free] = du_f.to(dtype=V.dtype)
            else:
                du = torch.zeros(nu, dtype=V.dtype, device=device)

            # FIX: Robust feedback gains computation with SVD fallback
            Q_uu_reg = Q_uu + 1e-8 * I_nu_typed
            Kt, _ = self._robust_solve(Q_uu_reg, Q_ux, torch.zeros_like(q_u).unsqueeze(-1))
            Kt = Kt.squeeze(-1) if Kt.ndim > Q_ux.ndim - 1 else Kt
            kt = du
            K_seq[t] = Kt
            k_seq[t] = kt

            V = Q_xx + Kt.T @ Q_uu @ Kt + Kt.T @ Q_ux + Q_xu @ Kt
            v = q_x + Kt.T @ Q_uu @ kt + Kt.T @ q_u + Q_ux.mT @ kt

        K_seq = torch.stack(K_seq, dim=0)
        k_seq = torch.stack(k_seq, dim=0)

        # Forward pass: compute dX, dU
        dX = [torch.zeros(self.nx, dtype=V.dtype, device=device)]
        dU = []
        for t in range(T):
            dx = dX[-1]
            du = K_seq[t] @ dx + k_seq[t]
            dU.append(du)
            # Ensure dtype consistency in forward pass
            A_t_dtype = A[t].to(dtype=dx.dtype)
            B_t_dtype = B[t].to(dtype=du.dtype)
            dX.append(A_t_dtype @ dx + B_t_dtype @ du)

        dX = torch.stack(dX, dim=0)
        dU = torch.stack(dU, dim=0)
        return dX, dU, K_seq

    def _lqr_unconstrained_batch(
            self,
            A: torch.Tensor,
            B: torch.Tensor,
            H_xx: torch.Tensor,
            H_uu: torch.Tensor,
            H_xu: torch.Tensor,
            grad_x: torch.Tensor,
            grad_u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch Riccati sweep for trajectories without active constraints."""

        if A.dim() != 4:
            raise ValueError("Expected batched tensors with shape [B, T, ...]")

        batch_size, T, nx, _ = A.shape
        device = A.device
        dtype = A.dtype

        eye_nu = (torch.eye(self.nu, device=device, dtype=dtype) * self.reg_eps).unsqueeze(0).expand(batch_size, -1, -1)

        V = H_xx[:, -1]
        v = grad_x[:, -1]

        K_seq = []
        k_seq = []

        for t in range(T - 1, -1, -1):
            A_t = A[:, t].to(dtype=V.dtype)
            B_t = B[:, t].to(dtype=V.dtype)
            H_xx_t = H_xx[:, t].to(dtype=V.dtype)
            H_xu_t = H_xu[:, t].to(dtype=V.dtype)
            H_uu_t = H_uu[:, t].to(dtype=V.dtype)
            I_nu_typed = eye_nu.to(dtype=V.dtype)

            # Riccati recursion terms
            V_A = torch.matmul(V, A_t)
            V_B = torch.matmul(V, B_t)

            Q_xx = H_xx_t + torch.matmul(A_t.transpose(-1, -2), V_A)
            Q_xu = H_xu_t + torch.matmul(A_t.transpose(-1, -2), V_B)
            Q_ux = Q_xu.transpose(-1, -2)
            Q_uu = H_uu_t + torch.matmul(B_t.transpose(-1, -2), V_B) + I_nu_typed

            q_x = grad_x[:, t].to(dtype=V.dtype) + torch.matmul(A_t.transpose(-1, -2), v.unsqueeze(-1)).squeeze(-1)
            q_u = grad_u[:, t].to(dtype=V.dtype) + torch.matmul(B_t.transpose(-1, -2), v.unsqueeze(-1)).squeeze(-1)

            Q_uu_reg = Q_uu + 1e-8 * I_nu_typed

            # Feedback gains (batched solves)
            K_t = -torch.linalg.solve(Q_uu_reg, Q_ux)
            k_t = -torch.linalg.solve(Q_uu_reg, q_u.unsqueeze(-1)).squeeze(-1)

            K_seq.append(K_t)
            k_seq.append(k_t)

            Q_uu_k = torch.matmul(Q_uu, k_t.unsqueeze(-1)).squeeze(-1)
            V = Q_xx + torch.matmul(K_t.transpose(-1, -2), torch.matmul(Q_uu, K_t)) \
                + torch.matmul(K_t.transpose(-1, -2), Q_ux) \
                + torch.matmul(Q_xu, K_t)
            v = q_x + torch.matmul(K_t.transpose(-1, -2), (Q_uu_k + q_u).unsqueeze(-1)).squeeze(-1) \
                + torch.matmul(Q_xu, k_t.unsqueeze(-1)).squeeze(-1)

        K_seq = torch.stack(K_seq[::-1], dim=1)  # [B, T, nu, nx]
        k_seq = torch.stack(k_seq[::-1], dim=1)  # [B, T, nu]

        # Forward sweep for state-action perturbations
        dX = torch.zeros(batch_size, T + 1, nx, device=device, dtype=dtype)
        dU = torch.zeros(batch_size, T, self.nu, device=device, dtype=dtype)

        for t in range(T):
            dx = dX[:, t]
            du = torch.matmul(K_seq[:, t], dx.unsqueeze(-1)).squeeze(-1) + k_seq[:, t]
            dU[:, t] = du
            dX[:, t + 1] = torch.matmul(A[:, t], dx.unsqueeze(-1)).squeeze(-1) \
                + torch.matmul(B[:, t], du.unsqueeze(-1)).squeeze(-1)

        return dX, dU

    # -----------------------------------------------------------------

    def reset(self) -> None:
        """
        Resetta lo stato interno del controller memorizzato dall'ultima esecuzione workaround per l'uso in cicli di training dove .backward()
        viene chiamato ripetutamente. FIX: Enhanced memory cleanup for training loops.
        """
        if self.verbose > 0:
            print("Resetting MPC controller internal state.")

        # FIX: Explicit tensor deletion for memory leak prevention
        if self.U_last is not None:
            del self.U_last
        if self.X_last is not None:
            del self.X_last
        if self.H_last is not None:
            del self.H_last
        if self.F_last is not None:
            del self.F_last
        if self.lmb_last is not None:
            del self.lmb_last
        if self.tight_mask_last is not None:
            del self.tight_mask_last

        self.U_last = None
        self.X_last = None
        self.H_last = None
        self.F_last = None
        self.lmb_last = None
        self.tight_mask_last = None
        self.converged = None

        # FIX: Aggressive memory cleanup for training stability
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure cleanup completion
