from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine
from ._utils import (
    rollout,
    _peak_bytes_tree,
    _auto_batch_size,
    run_in_batches,
    _warmup_cuda,
)


def branch_and_bound(
    dynamics: ControlAffine,
    x0: Tensor,
    B: int,
    K: int,
    dt: float,
    constr_fn: Callable[[Tensor], Tensor],
    n_restarts: int = 3,
    tie_noise: float = 1e-6,
    return_trajs: bool = False,
    batch_size: int | None = None,
) -> dict:
    """Batched Branch-and-Bound MPC maximising min_{k=0..K} constr_fn(x_k).

    Performs a level-by-level BnB tree search over the vertices of the control
    box [u_min, u_max].  The upper bound (UB) for any partial trajectory
    committed through step k is its running minimum of the constraint, since
    future steps can only decrease or maintain the minimum.

    Pruning rule: a child node is discarded whenever

        UB(child) = running_min <= global_lb

    where global_lb is the best *complete*-trajectory objective found so far.
    Pruned nodes cannot yield a strict improvement and are therefore safe to
    skip.

    Multiple restarts tighten global_lb progressively:
      - Restart 1 behaves like beam search (global_lb = -inf → no pruning),
        establishing an initial lower bound from the B best leaf nodes.
      - Restarts 2+ apply this tighter bound to prune unpromising branches
        earlier, freeing frontier capacity for other branches.  Small
        tie-breaking noise (tie_noise) diversifies which B candidates survive
        across restarts, allowing different subtrees to be explored.

    All N initial states are processed simultaneously in batched GPU-friendly
    tensor operations.  The inner Python loop runs only over the horizon K and
    n_restarts, both typically small.

    Args:
        dynamics:     ControlAffine instance.
        x0:           (N, nx) initial states.
        B:            Maximum frontier size per level (budget parameter).
                      Equivalent to beam width in beam search.
        K:            Prediction horizon (number of Euler steps).
        dt:           Integration time step.
        constr_fn:    (N, nx) -> (N,) or (N, 1). Higher values are safer.
        n_restarts:   Number of BnB passes. Pass 1 primes the global lower
                      bound; passes 2+ benefit from tighter BnB pruning and
                      explore different branches via tie-breaking noise.
        tie_noise:    Std of Gaussian noise added to selection scores in
                      passes 2+ to diversify branch exploration.  Set to 0
                      for fully deterministic behaviour.
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

    # Discrete control candidates: vertices of the control box, shape (M, nu).
    # get_control_vertices() returns (nu, 2**nu); .T gives (M, nu).
    ctrl_verts = dynamics.get_control_vertices().T  # (M, nu)
    M = ctrl_verts.shape[0]  # 2^nu

    # Auto-batch: split x0 along dim=0 to stay within GPU memory budget.
    if batch_size is None:
        if x0.is_cuda:
            _warmup_cuda(dynamics, x0, dt, constr_fn)
            batch_size = _auto_batch_size(
                _peak_bytes_tree(B, M, K, nx, nu, return_trajs, x0.element_size()),
                x0.device,
            )
        else:
            batch_size = max(1, N)  # CPU: one chunk, no GPU constraint.

    def _body(x_chunk: Tensor) -> dict:
        n = x_chunk.shape[0]

        # Global lower bound per state: best complete-trajectory min-constraint found.
        global_lb = x_chunk.new_full((n,), float("-inf"))
        best_ctrl = x_chunk.new_zeros(n, K, nu)

        for restart in range(n_restarts):
            states = x_chunk.unsqueeze(1)  # (n, 1, nx)
            rmin = constr_fn(x_chunk).reshape(n, 1).clone()  # (n, 1)

            if return_trajs:
                ctrl_hist = x_chunk.new_empty(n, 1, 0, nu)

            for k in range(K):
                W = states.shape[1]
                nWM = n * W * M

                states_exp = states.repeat_interleave(M, dim=1)
                rmin_exp = rmin.repeat_interleave(M, dim=1)

                u_tiled = (
                    ctrl_verts.unsqueeze(0)
                    .expand(n * W, M, nu)
                    .reshape(n, W * M, nu)
                )

                if return_trajs:
                    hist_exp = ctrl_hist.repeat_interleave(M, dim=1)

                states_next = dynamics.euler_step(
                    states_exp.reshape(nWM, nx),
                    u_tiled.reshape(nWM, nu),
                    dt,
                ).reshape(n, W * M, nx)

                c_next = constr_fn(states_next.reshape(nWM, nx)).reshape(n, W * M)
                rmin_next = torch.minimum(rmin_exp, c_next)

                if k == K - 1:
                    leaf_best_vals, leaf_best_idx = rmin_next.max(dim=1)
                    improved = leaf_best_vals > global_lb
                    global_lb = torch.where(improved, leaf_best_vals, global_lb)

                    if return_trajs:
                        n_idx = torch.arange(n, device=x_chunk.device)
                        leaf_hist = hist_exp[n_idx, leaf_best_idx]
                        last_u = u_tiled[n_idx, leaf_best_idx]
                        leaf_ctrl_seq = torch.cat(
                            [leaf_hist, last_u.unsqueeze(1)], dim=1
                        )
                        improved_exp = improved[:, None, None].expand_as(leaf_ctrl_seq)
                        best_ctrl = torch.where(improved_exp, leaf_ctrl_seq, best_ctrl)

                pruned = rmin_next <= global_lb.unsqueeze(1)
                sel_scores = rmin_next.masked_fill(pruned, float("-inf"))

                if tie_noise > 0.0 and restart > 0:
                    noise = tie_noise * torch.randn_like(sel_scores)
                    sel_scores = sel_scores + noise
                    sel_scores = sel_scores.masked_fill(pruned, float("-inf"))

                W_next = min(B, W * M)
                _, topk_idx = sel_scores.topk(W_next, dim=1)

                rmin = rmin_next.gather(1, topk_idx)
                states = states_next.gather(
                    1, topk_idx.unsqueeze(-1).expand(-1, -1, nx)
                )

                if return_trajs:
                    k_committed = hist_exp.shape[2]
                    idx_hist = (
                        topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k_committed, nu)
                    )
                    hist_selected = hist_exp.gather(1, idx_hist)
                    ctrl_now = u_tiled.gather(
                        1, topk_idx.unsqueeze(-1).expand(-1, -1, nu)
                    ).unsqueeze(2)
                    ctrl_hist = torch.cat([hist_selected, ctrl_now], dim=2)

        result = {"values": global_lb}
        if return_trajs:
            result["control_traj"] = best_ctrl
            result["state_traj"] = rollout(dynamics, x_chunk, best_ctrl, dt)
        return result

    # Always route through run_in_batches: even when N <= batch_size, the
    # retry-on-OOM logic protects against first-call workspace mismatches.
    return run_in_batches(_body, x0, batch_size)
