from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine
from ._utils import (
    _peak_bytes_tree,
    _auto_batch_size,
    run_in_batches,
    _warmup_cuda,
    rollout,
)


def beam_search(
    dynamics: ControlAffine,
    x0: Tensor,
    B: int,
    K: int,
    dt: float,
    constr_fn: Callable[[Tensor], Tensor],
    return_trajs: bool = False,
    batch_size: int | None = None,
) -> dict:
    """Batched beam search maximising min_{k=0..K} constr_fn(x_k) over the vertices
    of the control box [u_min, u_max].

    The discrete candidate controls are the 2**nu corners of the control box,
    obtained via dynamics.get_control_vertices().  State propagation is done with
    dynamics.euler_step.

    Args:
        dynamics:     ControlAffine instance.
        x0:           (N, nx) initial states.
        B:            Beam width — max candidates retained after each step.
        K:            Prediction horizon (number of control steps).
        dt:           Euler integration time step.
        constr_fn:    Callable (N, nx) -> (N,) or (N, 1).  Constraint function.
                      Called on flattened batches, so it must handle arbitrary N.
        return_trajs: If True, also return the optimal control sequence and
                      state trajectory for each initial state.
        batch_size:   Maximum number of initial states processed in one GPU
                      call.  ``None`` (default) auto-selects from free GPU
                      memory; pass an explicit integer to override.

    Returns:
        dict with keys:
            "values":       (N,)          best min-constraint per initial state.
            "control_traj": (N, K, nu)    optimal control sequences
                                          (only if return_trajs=True).
            "state_traj":   (N, K+1, nx)  corresponding state trajectories
                                          (only if return_trajs=True).
    """
    N, nx = x0.shape
    nu = dynamics.nu

    # Discrete control candidates: vertices of the control box, shape (M, nu).
    # get_control_vertices() returns (nu, 2**nu); transpose gives (M, nu).
    controls = dynamics.get_control_vertices().T  # (M, nu)
    M = controls.shape[0]  # 2**nu

    # Auto-batch: split x0 along dim=0 to stay within GPU memory budget.
    if batch_size is None:
        if x0.is_cuda:
            _warmup_cuda(dynamics, x0, dt, constr_fn, warmup_n=B * M)
            batch_size = _auto_batch_size(
                _peak_bytes_tree(B, M, K, nx, nu, return_trajs, x0.element_size()),
                x0.device,
            )
        else:
            batch_size = max(1, N)  # CPU: one chunk, no GPU constraint.

    def _body(x_chunk: Tensor) -> dict:
        # ---------------------------------------------------------------------
        # Initialise beam for one chunk of initial states.
        # states:      (n, W, nx) — W = current beam width (starts at 1)
        # running_min: (n, W)     — min constr_fn value along each beam path so far
        # ---------------------------------------------------------------------
        n = x_chunk.shape[0]
        states = x_chunk.unsqueeze(1)  # (n, 1, nx)
        running_min = constr_fn(x_chunk).reshape(n).unsqueeze(1)  # (n, 1)

        if return_trajs:
            # Only the control history is tracked during search; the state
            # trajectory is reconstructed via a single rollout at the end.  This
            # avoids carrying an (n, W*M, k, nx) buffer through every expansion,
            # which is the dominant memory cost.
            control_hist = x_chunk.new_empty(n, 1, 0, nu)  # (n, 1, 0, nu)

        for _ in range(K):
            W = states.shape[1]
            states_exp = states.repeat_interleave(M, dim=1)  # (n, W*M, nx)
            rmin_exp = running_min.repeat_interleave(M, dim=1)  # (n, W*M)

            u_tiled = (
                controls.unsqueeze(0).expand(n * W, M, nu).reshape(n, W * M, nu)
            )

            if return_trajs:
                hist_exp = control_hist.repeat_interleave(M, dim=1)

            nWM = n * W * M
            states_next = dynamics.euler_step(
                states_exp.reshape(nWM, nx),
                u_tiled.reshape(nWM, nu),
                dt,
            ).reshape(n, W * M, nx)

            dist_next = constr_fn(states_next.reshape(nWM, nx)).reshape(n, W * M)
            rmin_next = torch.minimum(rmin_exp, dist_next)

            W_next = min(B, W * M)
            topk_vals, topk_idx = torch.topk(rmin_next, W_next, dim=1)

            states = states_next.gather(
                1, topk_idx.unsqueeze(-1).expand(-1, -1, nx)
            )
            running_min = topk_vals

            if return_trajs:
                h_so_far = hist_exp.shape[2]
                idx_hist = (
                    topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_so_far, nu)
                )
                hist_selected = torch.gather(hist_exp, 1, idx_hist)
                ctrl_selected = u_tiled.gather(
                    1, topk_idx.unsqueeze(-1).expand(-1, -1, nu)
                ).unsqueeze(2)
                control_hist = torch.cat([hist_selected, ctrl_selected], dim=2)

        best_vals, best_idx = running_min.max(dim=1)
        result = {"values": best_vals}

        if return_trajs:
            bi = best_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            best_ctrl = control_hist.gather(
                1, bi.expand(-1, -1, K, nu)
            ).squeeze(1)
            result["control_traj"] = best_ctrl
            result["state_traj"] = rollout(dynamics, x_chunk, best_ctrl, dt)

        return result

    # Always route through run_in_batches: even when N <= batch_size, the
    # retry-on-OOM logic protects against first-call workspace mismatches.
    return run_in_batches(_body, x0, batch_size)
