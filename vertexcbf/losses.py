"""Loss functions for VertexCBF training.

The Neural CBF is parameterised as:

    h_Θ(x) = c(x) - r_Θ(x)

where ``c(x)`` is the constraint function defining the safe set and
``r_Θ(x)`` is the learned neural residual.

Two complementary loss terms:

* :func:`pde_loss` — enforces the Stationary Hamilton-Jacobi-Bellman
  Variational Inequality (HJB-VI) across a fine grid of states
  (physics-informed term).
* :func:`data_loss` — supervised regression against precomputed Neural CBF
  targets, e.g. produced by beam search (data-driven term).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


def pde_loss(
    residual_fn: nn.Module,
    constr_fn: Callable[[Tensor], Tensor],
    states: Tensor,
    xdot_vertices: Tensor,
) -> Tensor:
    """Stationary Hamilton-Jacobi-Bellman Variational Inequality (HJB-VI) loss.

    Given the Neural CBF ``h_Θ(x) = c(x) - r_Θ(x)``, the stationary HJB-VI
    requires:

        min( H(x, ∇h_Θ),  r_Θ(x) ) = 0

    where ``H(x, ∇h_Θ) = max_u { ∇h_Θ · (f(x) + g(x)u) }`` is the
    Hamiltonian.  Intuitively: either ``H ≤ 0`` (HJB condition satisfied in
    the interior) or ``r_Θ = 0`` (on the safe-set boundary ``h_Θ = c``).

    This is enforced by penalising violations of the VI:

        loss = mean( min(H(x, ∇h_Θ),  r_Θ(x))² )

    Args:
        residual_fn: Network outputting ``r_Θ(x)`` of shape ``(N, 1)``.
        constr_fn: Constraint function ``c(x)``, callable ``(N, nx) -> (N, 1)``.
        states: PDE grid states of shape ``(N, nx)``.  Must have
            ``requires_grad=True`` so that ``∇h_Θ`` can be computed.
        xdot_vertices: Precomputed ``f(x) + g(x) u_k`` at all ``2**nu``
            control vertices, shape ``(N, nx, 2**nu)``.  Detach this once
            before the training loop so it is not recomputed each epoch.

    Returns:
        Scalar PDE loss tensor.
    """
    constr = constr_fn(states)  # c(x),   (N, 1)
    residual = residual_fn(states)  # r_Θ(x), (N, 1)
    cbf = constr - residual  # h_Θ(x), (N, 1)

    grad_cbf = torch.autograd.grad(
        outputs=cbf,
        inputs=states,
        grad_outputs=torch.ones_like(cbf),
        create_graph=True,
    )[0].unsqueeze(
        1
    )  # ∇h_Θ, (N, 1, nx)

    # Hamiltonian: max over control vertices of  ∇h_Θ · xdot
    hamiltonian, _ = torch.max(torch.bmm(grad_cbf, xdot_vertices), dim=-1)  # (N, 1)
    hamiltonian = hamiltonian.squeeze(1)  # (N,)
    residual_flat = residual.squeeze(1)  # (N,)

    return torch.mean(torch.min(hamiltonian, residual_flat) ** 2)


def data_loss(
    residual_fn: nn.Module,
    constr_fn: Callable[[Tensor], Tensor],
    states: Tensor,
    target_values: Tensor,
) -> Tensor:
    """Supervised loss against precomputed Neural CBF targets.

    Computes MSE between the predicted Neural CBF
    ``h_Θ(x) = c(x) - r_Θ(x)`` and externally supplied targets (e.g. from
    beam search or any other numerical solver).

    Args:
        residual_fn: Network outputting ``r_Θ(x)`` of shape ``(N, 1)``.
        constr_fn: Constraint function ``c(x)``, callable ``(N, nx) -> (N, 1)``.
        states: States at which targets were computed, shape ``(N, nx)``.
        target_values: Ground-truth Neural CBF targets, shape ``(N,)``.

    Returns:
        Scalar data loss tensor.
    """
    constr = constr_fn(states)  # c(x),   (N, 1)
    residual = residual_fn(states)  # r_Θ(x), (N, 1)
    cbf_pred = (constr - residual).squeeze(1)  # h_Θ(x), (N,)
    return torch.mean((cbf_pred - target_values) ** 2)
