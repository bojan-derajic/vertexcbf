import torch
from torch import Tensor


def ball_3d_sdf(states: Tensor, center: Tensor, radius: float) -> Tensor:
    """
    Signed distance function for a 3D ball.

    Args:
        states: (..., nx) tensor of states; first three dims are (x, y, z)
        center: (3,) tensor [cx, cy, cz]
        radius: scalar radius

    Returns:
        (..., 1) tensor of SDF values (negative inside, positive outside)
    """
    xyz = states[..., :3]
    center_ = torch.as_tensor(center, dtype=states.dtype, device=states.device)
    radius_ = torch.tensor(radius, dtype=states.dtype, device=states.device)
    dist = torch.norm(xyz - center_, dim=-1, keepdim=True)
    return dist - radius_
