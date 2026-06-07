import torch
from torch import Tensor


def circle_sdf(states: Tensor, center: Tensor, radius: float) -> Tensor:
    """
    Signed distance function for a circle.

    Args:
        states: (..., nx) tensor of states; first two dims are (x, y)
        center: (2,) tensor [cx, cy]
        radius: scalar radius

    Returns:
        (..., 1) tensor of SDF values (negative inside, positive outside)
    """
    xy = states[..., :2]
    center_ = torch.as_tensor(center, dtype=states.dtype, device=states.device)
    radius_ = torch.tensor(radius, dtype=states.dtype, device=states.device)
    dist = torch.norm(xy - center_, dim=-1, keepdim=True)
    return dist - radius_
