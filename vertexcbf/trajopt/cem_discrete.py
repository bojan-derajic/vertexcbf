from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine
from ._utils import rollout, score_sequences, _peak_bytes_shooting, _auto_batch_size, run_in_batches, _warmup_cuda


def cem_discrete(
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
    """Cross-Entropy Method (CEM) MPC over the discrete vertex control set.

    Operates identically to :func:`beam_search` in that candidate controls are
    restricted to the 2**nu corners of the control box, but uses the CEM
    iterative-refinement loop instead of deterministic pruning.

    A per-state categorical distribution over the M = 2**nu vertices is
    maintained for each of the K timesteps.  Each iteration:

        1. Sample N_s vertex-index sequences from the current distribution:
           shape (N, N_s, K).
        2. Map indices to actual control vectors: (N, N_s, K, nu).
        3. Roll out and score each sequence by running minimum of constr_fn.
        4. Keep the top N_s_elite = ceil(elite_frac * N_s) sequences per state.
        5. Refit the per-step vertex distribution from elite empirical counts
           (Laplace-smoothed to avoid zero probabilities).

    After n_iter iterations the best sample from the final round is returned.

    Args:
        dynamics:     ControlAffine instance.
        x0:           (N, nx) initial states.
        N_s:          Number of sequences sampled per iteration.
        K:            Prediction horizon (number of Euler steps).
        dt:           Integration time step.
        constr_fn:    (N, nx) -> (N,) or (N, 1). Higher values are safer.
        n_iter:       Number of CEM refinement iterations.
        elite_frac:   Fraction of N_s samples retained as elite (0 < elite_frac ≤ 1).
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
    N_s_elite = max(1, int(N_s * elite_frac))

    # Discrete candidate controls: vertices of the control box.
    # get_control_vertices() returns (nu, 2**nu); .T gives (M, nu).
    vertices = dynamics.get_control_vertices().T  # (M, nu)
    M = vertices.shape[0]  # 2**nu

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
        probs = x_chunk.new_ones(n, K, M) / M

        best_controls: Tensor | None = None
        best_vals: Tensor | None = None

        for _ in range(n_iter):
            flat_probs = probs.reshape(n * K, M)
            sampled_idx = torch.multinomial(flat_probs, N_s, replacement=True)
            sampled_idx = sampled_idx.reshape(n, K, N_s).permute(0, 2, 1)

            controls = vertices[sampled_idx]

            scores = score_sequences(dynamics, x_chunk, controls, dt, constr_fn)

            elite_idx = scores.topk(N_s_elite, dim=1).indices

            elite_vertex_idx = sampled_idx.gather(
                1, elite_idx.unsqueeze(-1).expand(-1, -1, K)
            )

            counts = F.one_hot(elite_vertex_idx, num_classes=M).float().sum(dim=1)
            counts = counts + 1.0
            probs = counts / counts.sum(dim=-1, keepdim=True)

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
