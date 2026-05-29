from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine
from ._utils import rollout, score_sequences, _peak_bytes_shooting, _auto_batch_size, run_in_batches, _warmup_cuda


def icem(
    dynamics: ControlAffine,
    x0: Tensor,
    N_s: int,
    K: int,
    dt: float,
    constr_fn: Callable[[Tensor], Tensor],
    n_iter: int = 5,
    elite_frac: float = 0.1,
    noise_beta: float = 0.9,
    return_trajs: bool = False,
    batch_size: int | None = None,
) -> dict:
    """Improved Cross-Entropy Method (iCEM) MPC.

    Extends vanilla CEM (see ``cem``) with two key improvements from
    Pinneri et al. (2021) "Sample-efficient Cross-Entropy Method for
    Real-time Planning":

    1. **Colored noise** — AR(1) temporally-correlated perturbations instead
       of i.i.d. Gaussian.  This biases samples towards smooth control
       sequences and improves sample efficiency for dynamical systems.

    2. **Elite reuse** — the N_s_elite sequences from the previous iteration
       are carried into the next round's sample pool instead of being
       discarded.  This prevents good solutions from being lost and warms
       the distribution faster.

    Algorithm (per iteration i):
        1. Draw N_s_new = N_s - (N_s_elite if i > 0 else 0) new samples using
           AR(1) colored noise around the current (mean, std).
        2. Concatenate with the carried-over elite pool (if any).
        3. Score all N_s candidates; select top N_s_elite as the new elite pool.
        4. Refit mean and std from the elite pool.

    After n_iter iterations, return the best sample from the final round.

    Args:
        dynamics:     ControlAffine instance.
        x0:           (N, nx) initial states.
        N_s:          Total number of candidates per iteration (new + carried elite).
        K:            Prediction horizon (number of Euler steps).
        dt:           Integration time step.
        constr_fn:    (N, nx) -> (N,) or (N, 1). Higher values are safer.
        n_iter:       Number of iCEM refinement iterations.
        elite_frac:   Fraction of N_s retained as elite (0 < elite_frac ≤ 1).
        noise_beta:   AR(1) temporal correlation coefficient in [0, 1).
                      0 → i.i.d. Gaussian (same as CEM), 1 → constant noise.
                      Values around 0.9 produce smooth, temporally-correlated
                      control perturbations.
        return_trajs: If True, also return the best control sequence and
                      state trajectory per initial state.
        batch_size:   Maximum number of initial states processed in one GPU
                      call.  ``None`` (default) auto-selects from free GPU
                      memory; pass an explicit integer to override.

    Returns:
        dict with keys:
            "values"       — (N,)         best min-constraint per initial state.
            "control_traj" — (N, K, nu)   best control sequence
                                           (only if return_trajs=True).
            "state_traj"   — (N, K+1, nx) corresponding state trajectory
                                           (only if return_trajs=True).
    """
    N, nx = x0.shape
    nu = dynamics.nu
    u_min = dynamics.u_min  # (nu,)
    u_max = dynamics.u_max  # (nu,)
    N_s_elite = max(1, int(N_s * elite_frac))

    # Auto-batch: split x0 along dim=0 to stay within GPU memory budget.
    if batch_size is None:
        if x0.is_cuda:
            _warmup_cuda(dynamics, x0, dt, constr_fn)
            batch_size = _auto_batch_size(
                _peak_bytes_shooting(N_s, K, nx, nu, x0.element_size()), x0.device
            )
        else:
            batch_size = max(1, N)

    def _body(x_chunk: Tensor) -> dict:
        n = x_chunk.shape[0]
        mean = (
            ((u_min + u_max) / 2).reshape(1, 1, nu).expand(n, K, nu).clone()
        )
        std = ((u_max - u_min) / 4).reshape(1, 1, nu).expand(n, K, nu).clone()

        elite_pool: Tensor | None = None
        best_controls: Tensor | None = None
        best_vals: Tensor | None = None

        for _ in range(n_iter):
            N_s_new = N_s - (N_s_elite if elite_pool is not None else 0)

            noise = _colored_noise(n, N_s_new, K, nu, noise_beta, x_chunk.device, x_chunk.dtype)
            new_controls = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(
                min=u_min.reshape(1, 1, 1, -1),
                max=u_max.reshape(1, 1, 1, -1),
            )

            if elite_pool is not None:
                controls = torch.cat([elite_pool, new_controls], dim=1)
            else:
                controls = new_controls

            scores = score_sequences(dynamics, x_chunk, controls, dt, constr_fn)

            elite_idx = scores.topk(N_s_elite, dim=1).indices
            idx_exp = elite_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, nu)
            elite_pool = controls.gather(1, idx_exp)

            mean = elite_pool.mean(dim=1)
            std = elite_pool.std(dim=1).clamp(min=1e-3)

            best_s = scores.argmax(dim=1)
            n_idx = torch.arange(n, device=x_chunk.device)
            best_controls = controls[n_idx, best_s]
            best_vals = scores[n_idx, best_s]

        result = {"values": best_vals}
        if return_trajs:
            result["control_traj"] = best_controls
            result["state_traj"] = rollout(dynamics, x_chunk, best_controls, dt)
        return result

    # Always route through run_in_batches: the retry-on-OOM logic protects
    # against first-call workspace mismatches even when N <= batch_size.
    return run_in_batches(_body, x0, batch_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _colored_noise(
    N: int,
    N_s: int,
    K: int,
    nu: int,
    beta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Generate AR(1) temporally-correlated noise of shape (N, N_s, K, nu).

    The recurrence is:
        noise[:, :, 0, :] ~ N(0, 1)
        noise[:, :, t, :] = beta * noise[:, :, t-1, :] + sqrt(1 - beta^2) * N(0, 1)

    This keeps the marginal variance at 1 for all t while introducing temporal
    correlation controlled by beta (0 → i.i.d., near-1 → highly correlated).
    """
    noise = torch.empty(N, N_s, K, nu, device=device, dtype=dtype)
    noise[:, :, 0, :] = torch.randn(N, N_s, nu, device=device, dtype=dtype)
    scale = (1.0 - beta**2) ** 0.5
    for t in range(1, K):
        noise[:, :, t, :] = beta * noise[:, :, t - 1, :] + scale * torch.randn(
            N, N_s, nu, device=device, dtype=dtype
        )
    return noise
