from __future__ import annotations

from typing import Callable, Literal

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


SamplingStrategy = Literal["softmax", "gumbel_topk", "rank", "epsilon_greedy"]
"""
Available sampling strategies (all sample B beams proportional to fitness):

- ``"softmax"``       — Multinomial sampling from a softmax distribution over scores.
                        ``temperature`` controls sharpness: T→0 recovers greedy beam search,
                        T→∞ gives uniform random exploration.

- ``"gumbel_topk"``   — Exact weighted sampling *without* replacement via the Gumbel-max
                        trick: Gumbel(0,1) noise is added to log-weights before taking
                        top-K.  Unlike multinomial, this is differentiable in principle and
                        avoids potential bias from sequential multinomial draws.  Same
                        ``temperature`` knob as softmax.

- ``"rank"``          — Probability proportional to *rank* (best score → rank 1 → highest
                        weight).  This is robust to score outliers that would collapse a
                        softmax distribution, and requires no temperature tuning.

- ``"epsilon_greedy"``— With probability ``epsilon`` (per initial state) keep a uniformly
                        random set of B beams; otherwise keep the deterministic top-K.
                        Simple exploration baseline; ``epsilon`` decays to 0 recovers
                        standard beam search.
"""


# ---------------------------------------------------------------------------
# Core sampling primitive
# ---------------------------------------------------------------------------


def _sample_indices(
    scores: Tensor,  # (N, C) — higher is better
    B: int,
    strategy: SamplingStrategy,
    temperature: float = 1.0,
    epsilon: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """
    Draw B (≤ C) indices *without replacement* per row, proportional to fitness.

    Returns:
        sampled_scores: (N, B_actual)  — scores at selected indices
        sampled_idx:    (N, B_actual)  — column indices into the C dimension
    """
    N, C = scores.shape
    B_actual = min(B, C)

    if strategy == "softmax":
        # Shift for numerical stability, then form a proper probability vector.
        shifted = (scores - scores.amax(dim=1, keepdim=True)) / max(temperature, 1e-8)
        probs = torch.softmax(shifted, dim=1)  # (N, C)
        idx = torch.multinomial(probs, B_actual, replacement=False)  # (N, B_actual)

    elif strategy == "gumbel_topk":
        # Gumbel-Top-K: add i.i.d. Gumbel(0,1) to log-weights, take top-K.
        # Mathematically equivalent to independent exponential race:
        #   key_i = Exp(1) / w_i  →  smallest keys selected  (Vitter / Efraimidis & Spirakis).
        # Here we use the log-space formulation for numerical stability.
        log_w = (scores - scores.amax(dim=1, keepdim=True)) / max(temperature, 1e-8)
        u = torch.rand_like(log_w).clamp(min=1e-20)
        gumbel = -torch.log(-torch.log(u))  # (N, C) i.i.d. Gumbel(0,1)
        perturbed = log_w + gumbel
        _, idx = torch.topk(perturbed, B_actual, dim=1)

    elif strategy == "rank":
        # Assign weight (C − rank + 1) to each candidate so rank-1 (best score)
        # gets the highest weight; rank-C gets weight 1.  No temperature needed.
        order = scores.argsort(
            dim=1, descending=True
        )  # (N, C)  positions sorted by score
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1,
            order,
            torch.arange(C, device=scores.device).unsqueeze(0).expand(N, C),
        )  # ranks[n, c] = rank of candidate c
        weights = (C - ranks).float() + 1.0  # (N, C), values in [1, C]
        probs = weights / weights.sum(dim=1, keepdim=True)
        idx = torch.multinomial(probs, B_actual, replacement=False)

    elif strategy == "epsilon_greedy":
        # Per-initial-state: with prob epsilon use a uniformly random beam set,
        # otherwise use deterministic top-K.
        _, top_idx = torch.topk(scores, B_actual, dim=1)  # (N, B_actual)
        rand_idx = torch.rand(N, C, device=scores.device).argsort(dim=1)[
            :, :B_actual
        ]  # (N, B_actual)
        explore = torch.rand(N, device=scores.device) < epsilon  # (N,) bool
        idx = torch.where(explore.unsqueeze(1), rand_idx, top_idx)

    else:
        raise ValueError(
            f"Unknown sampling strategy {strategy!r}. "
            f"Choose from: 'softmax', 'gumbel_topk', 'rank', 'epsilon_greedy'."
        )

    sampled_scores = scores.gather(1, idx)
    return sampled_scores, idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def stochastic_beam_search(
    dynamics: ControlAffine,
    x0: Tensor,
    B: int,
    K: int,
    dt: float,
    constr_fn: Callable[[Tensor], Tensor],
    strategy: SamplingStrategy = "gumbel_topk",
    temperature: float = 1.0,
    epsilon: float = 0.1,
    return_trajs: bool = False,
    batch_size: int | None = None,
) -> dict:
    """Stochastic beam search maximising ``min_{k=0..K} constr_fn(x_k)``.

    At each horizon step, instead of keeping the *deterministic* top-K beam
    entries, B entries are *sampled* with probability proportional to their
    running-minimum fitness.  This trades optimality for diversity, helping
    escape local optima in the constraint landscape.

    The control candidates are the 2**nu corners of the control box
    (``dynamics.get_control_vertices()``).  State propagation uses
    ``dynamics.euler_step``.

    Args:
        dynamics:     ``ControlAffine`` instance.
        x0:           ``(N, nx)`` initial states.
        B:            Beam width — number of candidates retained each step.
        K:            Prediction horizon (number of Euler steps).
        dt:           Integration time step.
        constr_fn:    ``(N, nx) → (N,)`` or ``(N, 1)``.  Higher is safer.
        strategy:     One of ``"softmax"``, ``"gumbel_topk"``, ``"rank"``,
                      ``"epsilon_greedy"``.  See module docstring for details.
        temperature:  Softness of sampling for ``"softmax"`` and
                      ``"gumbel_topk"``.  Lower → more greedy.  Ignored by
                      ``"rank"`` and ``"epsilon_greedy"``.
        epsilon:      Exploration probability for ``"epsilon_greedy"``.
                      Ignored by other strategies.
        return_trajs: If ``True``, also return the best control sequence and
                      state trajectory per initial state.
        batch_size:   Maximum number of initial states processed in one GPU
                      call.  ``None`` (default) auto-selects from free GPU
                      memory; pass an explicit integer to override.

    Returns:
        dict with keys:

        * ``"values"``        — ``(N,)``         best min-constraint per state.
        * ``"control_traj"``  — ``(N, K, nu)``   optimal control sequence
                                                  *(only if* ``return_trajs`` *is True)*.
        * ``"state_traj"``    — ``(N, K+1, nx)`` state trajectory
                                                  *(only if* ``return_trajs`` *is True)*.
    """
    N, nx = x0.shape
    nu = dynamics.nu

    # Discrete control candidates: corners of [u_min, u_max]^nu, shape (M, nu).
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
            batch_size = max(1, N)

    def _body(x_chunk: Tensor) -> dict:
        n = x_chunk.shape[0]
        states = x_chunk.unsqueeze(1)  # (n, 1, nx)
        running_min = constr_fn(x_chunk).reshape(n).unsqueeze(1)  # (n, 1)

        if return_trajs:
            # Only the control history is tracked during search; the state
            # trajectory is reconstructed via a single rollout at the end.
            control_hist = x_chunk.new_empty(n, 1, 0, nu)

        for _ in range(K):
            W = states.shape[1]
            states_exp = states.repeat_interleave(M, dim=1)
            rmin_exp = running_min.repeat_interleave(M, dim=1)

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

            sampled_vals, sampled_idx = _sample_indices(
                rmin_next, B, strategy,
                temperature=temperature, epsilon=epsilon,
            )

            states = states_next.gather(
                1, sampled_idx.unsqueeze(-1).expand(-1, -1, nx)
            )
            running_min = sampled_vals

            if return_trajs:
                k_so_far = hist_exp.shape[2]
                idx_hist = (
                    sampled_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k_so_far, nu)
                )
                hist_selected = torch.gather(hist_exp, 1, idx_hist)
                ctrl_selected = u_tiled.gather(
                    1, sampled_idx.unsqueeze(-1).expand(-1, -1, nu)
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
