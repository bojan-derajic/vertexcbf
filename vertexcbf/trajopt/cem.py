from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine
from ._utils import rollout, score_sequences, _peak_bytes_shooting, _auto_batch_size, run_in_batches, _warmup_cuda


def cem(
    dynamics: ControlAffine,
    x0: Tensor,
    N_s: int,
    K: int,
    dt: float,
    constr_fn: Callable[[Tensor], Tensor],
    n_iter: int = 5,
    elite_frac: float = 0.1,
    return_trajs: bool = False,
    batch_size: int | None = None,
) -> dict:
    """Cross-Entropy Method (CEM) MPC maximising min_{k=0..K} constr_fn(x_k).

    Iteratively refines a per-state Gaussian distribution over control sequences
    by fitting it to the highest-scoring (elite) samples.  This is well-suited
    to constraint satisfaction problems where the objective has a clear threshold.

    Algorithm (per iteration):
        1. Sample N_s sequences from N(mean, std^2): (N, N_s, K, nu), clamped to box.
        2. Roll out and score each by running minimum of constr_fn.
        3. Keep the top N_s_elite = ceil(elite_frac * N_s) sequences per state.
        4. Refit mean and std from the elite set (std floored at 1e-3 to avoid
           collapse).

    After n_iter iterations, the best sample from the final round is returned.

    Args:
        dynamics:     ControlAffine instance.
        x0:           (N, nx) initial states.
        N_s:          Number of sequences sampled per iteration.
        K:            Prediction horizon (number of Euler steps).
        dt:           Integration time step.
        constr_fn:    (N, nx) -> (N,) or (N, 1). Higher values are safer.
        n_iter:       Number of CEM refinement iterations.
        elite_frac:   Fraction of N_s samples retained as elite (0 < elite_frac ≤ 1).
                      E.g. 0.1 keeps the best 10 % of samples.
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

        best_controls: Tensor | None = None
        best_vals: Tensor | None = None

        for _ in range(n_iter):
            eps = torch.randn(n, N_s, K, nu, device=x_chunk.device, dtype=x_chunk.dtype)
            controls = (mean.unsqueeze(1) + std.unsqueeze(1) * eps).clamp(
                min=u_min.reshape(1, 1, 1, -1),
                max=u_max.reshape(1, 1, 1, -1),
            )

            scores = score_sequences(dynamics, x_chunk, controls, dt, constr_fn)

            elite_idx = scores.topk(N_s_elite, dim=1).indices
            idx_exp = elite_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, nu)
            elite_controls = controls.gather(1, idx_exp)

            mean = elite_controls.mean(dim=1)
            std = elite_controls.std(dim=1).clamp(min=1e-3)

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
