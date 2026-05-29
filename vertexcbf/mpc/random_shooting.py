from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine
from ._utils import rollout, score_sequences, _peak_bytes_shooting, _auto_batch_size, run_in_batches, _warmup_cuda


def random_shooting(
    dynamics: ControlAffine,
    x0: Tensor,
    N_s: int,
    K: int,
    dt: float,
    constr_fn: Callable[[Tensor], Tensor],
    return_trajs: bool = False,
    batch_size: int | None = None,
) -> dict:
    """Random shooting MPC maximising min_{k=0..K} constr_fn(x_k).

    Samples N_s control sequences uniformly from the control box [u_min, u_max],
    rolls out all trajectories in parallel via dynamics.euler_step, and returns
    the best-scoring sequence per initial state.

    Args:
        dynamics:     ControlAffine instance.
        x0:           (N, nx) initial states.
        N_s:          Number of random control sequences to sample per state.
        K:            Prediction horizon (number of Euler steps).
        dt:           Integration time step.
        constr_fn:    (N, nx) -> (N,) or (N, 1). Higher values are safer.
        return_trajs: If True, also return the best control sequence and
                      state trajectory per initial state.
        batch_size:   Maximum number of initial states processed in one GPU
                      call.  ``None`` (default) auto-selects a safe value from
                      free GPU memory; pass an explicit integer to override.

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
        u_rand = torch.rand(n, N_s, K, nu, device=x_chunk.device, dtype=x_chunk.dtype)
        controls = u_min + (u_max - u_min) * u_rand

        scores = score_sequences(dynamics, x_chunk, controls, dt, constr_fn)
        best_vals, best_s = scores.max(dim=1)

        result = {"values": best_vals}
        if return_trajs:
            best_controls = controls[
                torch.arange(n, device=x_chunk.device), best_s
            ]
            result["control_traj"] = best_controls
            result["state_traj"] = rollout(dynamics, x_chunk, best_controls, dt)
        return result

    # Always route through run_in_batches: the retry-on-OOM logic protects
    # against first-call workspace mismatches even when N <= batch_size.
    return run_in_batches(_body, x0, batch_size)
