import torch
from torch import Tensor


def interval_sdf(states: Tensor, center: float, d: float) -> Tensor:
    """
    Signed distance function for a 1D symmetric interval [center - d, center + d].
    Intended for systems where only the first state is a position axis (e.g. 1D double integrator).

    Convention (safe-set): positive inside the interval, negative outside.
        c(x) = d - |x - center|

    Args:
        states: (..., nx) tensor of states; first state dim is position x
        center: scalar center of the interval
        d: half-width of the interval (so the interval is [center - d, center + d])

    Returns:
        (..., 1) tensor of SDF values (positive inside, negative outside)
    """
    x = states[..., :1]
    center_ = torch.tensor(center, dtype=states.dtype, device=states.device)
    d_ = torch.tensor(d, dtype=states.dtype, device=states.device)
    return d_ - (x - center_).abs()
