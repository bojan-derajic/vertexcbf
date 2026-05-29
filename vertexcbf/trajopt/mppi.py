from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine
from ._utils import rollout, score_sequences, _peak_bytes_shooting, _auto_batch_size, run_in_batches, _warmup_cuda


def mppi(
    dynamics: ControlAffine,
    x0: Tensor,
    N_s: int,
    K: int,
    dt: float,
    constr_fn: Callable[[Tensor], Tensor],
    sigma: float | Tensor = 0.5,
    lam: float = 1.0,
    n_iter: int = 1,
    U_init: Tensor | None = None,
    return_trajs: bool = False,
    batch_size: int | None = None,
) -> dict:
    """Model Predictive Path Integral (MPPI) control.

    Maximises min_{k=0..K} constr_fn(x_k) via importance-weighted averaging of
    N_s sampled control perturbations around a nominal sequence.

    Algorithm (per iteration):
        1. Sample N_s Gaussian perturbations eps ~ N(0, sigma^2): (N_s, K, nu).
        2. Form perturbed sequences: U_perturbed[n,s] = clamp(U[n] + eps[s]).
        3. Roll out all N*N_s trajectories and score by running minimum of constr_fn.
        4. Compute per-state importance weights: w[n,s] = softmax(score[n,s] / lam).
        5. Update nominal: U[n] += sum_s w[n,s] * eps[s], then clamp to box.

    The same N_s perturbations are shared across all N initial states; the weight
    update is state-specific, so U becomes (N, K, nu) after the first iteration.

    Args:
        dynamics:     ControlAffine instance.
        x0:           (N, nx) initial states.
        N_s:          Number of trajectory samples per iteration.
        K:            Prediction horizon (number of Euler steps).
        dt:           Integration time step.
        constr_fn:    (M, nx) -> (M,) or (M, 1). Higher values are safer.
        sigma:        Gaussian noise standard deviation — scalar or (nu,) tensor.
                      Larger values explore more of the control space.
        lam:          Temperature (>0). Low lam → near-greedy (best sample wins);
                      high lam → uniform average over all samples.
        n_iter:       Number of MPPI refinement iterations.  One iteration is
                      standard for receding-horizon MPC; more iterations refine
                      the nominal sequence at higher compute cost.
        U_init:       Initial nominal control sequence — (K, nu) shared across all
                      states, or (N, K, nu) per-state.  Defaults to zeros.
        return_trajs: If True, also return the final nominal control sequence and
                      the corresponding state trajectory per initial state.
        batch_size:   Maximum number of initial states processed in one GPU
                      call.  ``None`` (default) auto-selects from free GPU
                      memory; pass an explicit integer to override.

    Returns:
        dict with keys:
            "values"       — (N,)         min-constraint of the final nominal traj.
            "control_traj" — (N, K, nu)   final nominal control sequence
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

    # Ensure sigma is a (nu,) tensor.
    sigma_t = torch.as_tensor(sigma, device=x0.device, dtype=x0.dtype)
    if sigma_t.dim() == 0:
        sigma_t = sigma_t.expand(nu)

    def _body(x_chunk: Tensor) -> dict:
        n = x_chunk.shape[0]

        if U_init is None:
            U = x_chunk.new_zeros(n, K, nu)
        elif U_init.dim() == 2:
            U = U_init.unsqueeze(0).expand(n, -1, -1).clone()
        else:
            # (N, K, nu): chunk must come from the outer x0 in the same order;
            # the chunk's offset relative to the original N is not known here,
            # so this branch is only meaningful when N <= batch_size (single chunk).
            U = U_init.clone()
            if U.shape[0] != n:
                # Fallback: take the leading n rows.  Callers passing per-state
                # U_init with batched calls should set batch_size >= N.
                U = U[:n].clone()

        for _ in range(n_iter):
            eps = torch.randn(N_s, K, nu, device=x_chunk.device, dtype=x_chunk.dtype) * sigma_t

            U_pert = (U.unsqueeze(1) + eps.unsqueeze(0)).clamp(
                min=u_min.reshape(1, 1, 1, -1),
                max=u_max.reshape(1, 1, 1, -1),
            )

            scores = score_sequences(dynamics, x_chunk, U_pert, dt, constr_fn)

            scores_shifted = (scores - scores.amax(dim=1, keepdim=True)) / max(lam, 1e-8)
            weights = torch.softmax(scores_shifted, dim=1)

            weighted_eps = torch.einsum("ns,sku->nku", weights, eps)
            U = (U + weighted_eps).clamp(
                min=u_min.reshape(1, 1, -1),
                max=u_max.reshape(1, 1, -1),
            )

        best_vals = score_sequences(
            dynamics, x_chunk, U.unsqueeze(1), dt, constr_fn
        ).squeeze(1)

        result = {"values": best_vals}
        if return_trajs:
            result["control_traj"] = U
            result["state_traj"] = rollout(dynamics, x_chunk, U, dt)
        return result

    # Always route through run_in_batches: the retry-on-OOM logic protects
    # against first-call workspace mismatches even when N <= batch_size.
    return run_in_batches(_body, x0, batch_size)
