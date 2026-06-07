import torch
from torch import Tensor


def rectangle_sdf(states: Tensor, center: Tensor, a: float, b: float) -> Tensor:
    """
    Signed distance function for an axis-aligned rectangle.

    Args:
        states: (..., nx) tensor of states; first two dims are (x, y)
        center: (2,) tensor [cx, cy]
        a: half-length along x axis
        b: half-length along y axis

    Returns:
        (..., 1) tensor of SDF values (negative inside, positive outside)
    """
    xy = states[..., :2]
    center_ = torch.as_tensor(center, dtype=states.dtype, device=states.device)
    d = (xy - center_).abs() - torch.tensor(
        [a, b], dtype=states.dtype, device=states.device
    )
    sdf = torch.norm(d.clamp(min=0.0), dim=-1, keepdim=True) + d.max(
        dim=-1, keepdim=True
    ).values.clamp(max=0.0)
    return sdf
