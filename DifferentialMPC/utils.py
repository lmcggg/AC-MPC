# diff_mpc/utils.py

import torch
from torch.func import jacrev, vmap
from typing import Sequence, Union

TensorLike = Union[torch.Tensor, Sequence[float], float]

def bmv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Batch matrix-vector product."""
    return torch.einsum("...ij,...j->...i", A, x)

def bdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batch dot product."""
    return (a * b).sum(-1)

def bquad(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Batch quadratic form: x^T H x."""
    return torch.einsum("...i,...ij,...j->...", x, H, x)

def bger(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batch outer product."""
    return a.unsqueeze(-1) * b.unsqueeze(-2)

def eclamp(x: torch.Tensor, lo: TensorLike, hi: TensorLike) -> torch.Tensor:
    """Element-wise clamp."""
    lo_t = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
    hi_t = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
    return torch.max(torch.min(x, hi_t), lo_t)

def get_state_constraints(state: torch.Tensor, x_max: float, x_min: float) -> torch.Tensor:
    """Helper for simple state constraints."""
    constraint1 = state - x_max
    constraint2 = -state + x_min
    return torch.cat([constraint1, constraint2], dim=1)

def batched_jacobian(f, x, u):
    # External vmap for batch - jacrev for variables
    def f_wrapped(xu):
        x_, u_ = xu.split([x.shape[-1], u.shape[-1]], dim=-1)
        return f(x_, u_)

    xu = torch.cat([x, u], dim=-1)
    J = vmap(jacrev(f_wrapped))(xu)
    A, B = J.split([x.shape[-1], u.shape[-1]], dim=-1)
    return A, B
#fallback
def jacobian_finite_diff_batched(f_dyn, x, u, dt, eps: float = 1e-4):
    B, nx = x.shape
    nu = u.shape[-1]
    xu = torch.cat([x, u], dim=-1)                             # (B, nx+nu)
    eye = torch.eye(nx + nu, device=x.device, dtype=x.dtype)   # (nx+nu, nx+nu)
    perturb = eps * eye                                        # (nx+nu, nx+nu)
    def f_xu(z):
        x_, u_ = z[..., :nx], z[..., nx:]
        return f_dyn(x_, u_, dt)                               # (nx,)
    J = vmap(
        lambda p: (f_xu(xu + p) - f_xu(xu - p)) / (2 * eps),
        in_dims=0,
        out_dims=0,
    )(perturb)
    J = J.permute(1, 2, 0)
    A, Bm = J.split([nx, nu], dim=-1)
    return A, Bm

def pnqp(
    H: torch.Tensor,
    q: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    x_init: torch.Tensor | None = None,
    n_iter: int = 20,
    reg: float = 1e-10,
    tol: float = 1e-5,
    gamma: float = 0.1
) -> tuple[torch.Tensor, dict]:
    """
    Projected Newton for for box-constrained Quadratic Programs.
    Solves min 0.5 * x' H x + q' x s.t. lower <= x <= upper.
    """
    B, n = q.shape[:-1], q.shape[-1]
    I_reg = reg * torch.eye(n, dtype=H.dtype, device=H.device)

    if x_init is None:
        if n == 1:
            x = -(q / H.squeeze(-1))
        else:
            lu, piv = torch.linalg.lu_factor(H)
            x = -torch.linalg.lu_solve(lu, piv, q.unsqueeze(-1)).squeeze(-1)
    else:
        x = x_init.clone()
    x = eclamp(x, lower, upper)

    def obj(z):
        return 0.5 * bquad(z, H) + bdot(q, z)

    for it in range(n_iter):
        g = bmv(H, x) + q
        at_lo = (x <= lower) & (g > 0)
        at_hi = (x >= upper) & (g < 0)
        free = ~(at_lo | at_hi)
        mask = bger(free.float(), free.float())
        H_r = H * mask + I_reg
        g_r = g * free

        if n == 1:
            dx = -g_r / H_r.squeeze(-1)
        else:
            lu, piv = torch.linalg.lu_factor(H_r)
            dx = -torch.linalg.lu_solve(lu, piv, g_r.unsqueeze(-1)).squeeze(-1)

        if dx.abs().max() < tol:
            break

        alpha = torch.ones_like(dx[..., :1])
        f_x = obj(x)
        for _ in range(10):
            x_new = eclamp(x + alpha * dx, lower, upper)
            if (obj(x_new) <= f_x + gamma * bdot(g, x_new - x)).all():
                break
            alpha *= 0.5

        x = x_new
    return x, {"iters": it + 1}