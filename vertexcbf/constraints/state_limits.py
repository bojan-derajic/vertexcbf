import torch
from torch import Tensor


def state_limits_sdf(states: Tensor, limits: dict, alpha: float = 10.0) -> Tensor:
    """
    Signed distance function for box constraints on selected state dimensions.

    Each entry in ``limits`` constrains one state dimension to an interval.  Dimensions
    not listed are left unbounded.  Positive inside the safe region (all constrained
    states within bounds), negative outside (any state violates its bound).

    Individual per-dimension distances are aggregated via softmin (smooth approximation
    of the minimum), which is differentiable everywhere:

        softmin(d; alpha) = -(1/alpha) * log( sum_i exp(-alpha * d_i) )

    Larger ``alpha`` gives a tighter approximation of the hard min.

    Args:
        states: (..., nx) tensor of states
        limits: dict mapping state index → (lo, hi). Either bound may be ``None`` for a
                one-sided constraint, e.g.::

                    {0: (-1.0, 1.0), 2: (None, 0.5), 3: (0.0, None)}
        alpha: softmin temperature (> 0). Higher values approach hard min. Default: 10.0.

    Returns:
        (..., 1) tensor of SDF values (positive inside safe region, negative outside)
    """
    if not limits:
        raise ValueError("limits must contain at least one entry")

    distances = []
    for idx, (lo, hi) in limits.items():
        if idx < 0 or idx >= states.shape[-1]:
            raise ValueError(
                f"state index {idx} is out of bounds for states with {states.shape[-1]} dims"
            )
        if lo is None and hi is None:
            raise ValueError(
                f"at least one of lo or hi must be set for state index {idx}"
            )

        x = states[..., idx]

        if lo is not None and hi is not None:
            lo_t = torch.tensor(lo, dtype=states.dtype, device=states.device)
            hi_t = torch.tensor(hi, dtype=states.dtype, device=states.device)
            if lo_t >= hi_t:
                raise ValueError(
                    f"lo must be strictly less than hi for state index {idx}"
                )
            d = torch.minimum(x - lo_t, hi_t - x)
        elif lo is not None:
            d = x - torch.tensor(lo, dtype=states.dtype, device=states.device)
        else:
            d = torch.tensor(hi, dtype=states.dtype, device=states.device) - x

        distances.append(d)

    stacked = torch.stack(distances, dim=-1)  # (..., n)
    return -(1.0 / alpha) * torch.logsumexp(-alpha * stacked, dim=-1, keepdim=True)
