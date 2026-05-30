from typing import Callable, Sequence

import torch
from torch import Tensor


def composed_sdf(
    states: Tensor,
    sdfs: Sequence[Callable[[Tensor], Tensor]],
    alpha: float = 10.0,
) -> Tensor:
    """
    Compose multiple SDFs into a single SDF via softmin.

    The smooth min preserves the sign convention of the inputs:
      - For obstacle SDFs (negative inside, positive outside), the result is the
        SDF of the *union* of obstacles (closest obstacle wins).
      - For safe-set SDFs (positive inside, negative outside), the result is the
        SDF of the *intersection* of safe sets (most-violated constraint wins).

    All input SDFs must share the same convention; mixing them is not meaningful.

        softmin(d; alpha) = -(1/alpha) * log( sum_i exp(-alpha * d_i) )

    Args:
        states: (..., nx) tensor of states
        sdfs: sequence of callables, each mapping ``states`` to a (..., 1) SDF tensor
              (e.g. ``functools.partial(circle_sdf, center=[0,0], radius=1.0)``)
        alpha: softmin temperature (> 0). Higher values approach hard min. Default: 10.0.

    Returns:
        (..., 1) tensor of composed SDF values
    """
    if len(sdfs) == 0:
        raise ValueError("sdfs must contain at least one SDF")

    values = torch.cat([sdf(states) for sdf in sdfs], dim=-1)  # (..., n)
    return -(1.0 / alpha) * torch.logsumexp(-alpha * values, dim=-1, keepdim=True)
