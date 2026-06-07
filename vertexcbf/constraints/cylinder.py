import torch
from torch import Tensor


def cylinder_sdf(
    states: Tensor, center: Tensor, direction: Tensor, radius: float
) -> Tensor:
    """
    Signed distance function for an infinite cylinder in 3D.

    The cylinder is defined by an axis (a line through ``center`` along ``direction``)
    and a ``radius``.  Points are projected onto the axis and the SDF is the distance
    from the point to the axis minus the radius.

    Convention (obstacle): negative inside the cylinder, positive outside.
        sdf(p) = ||( p - center ) - (( p - center ) · d̂ ) d̂|| - radius

    Args:
        states: (..., nx) tensor of states; first three dims are (x, y, z)
        center: (3,) tensor — a point on the cylinder axis
        direction: (3,) tensor — axis direction (need not be unit length)
        radius: scalar radius of the cylinder

    Returns:
        (..., 1) tensor of SDF values (negative inside, positive outside)
    """
    xyz = states[..., :3]
    center_ = torch.as_tensor(center, dtype=states.dtype, device=states.device)
    direction_ = torch.as_tensor(direction, dtype=states.dtype, device=states.device)

    d_hat = direction_ / direction_.norm()  # unit axis vector

    v = xyz - center_  # (..., 3) vector from center to point
    proj = (v * d_hat).sum(dim=-1, keepdim=True) * d_hat  # (..., 3) axial component
    perp = v - proj  # (..., 3) radial component

    dist = perp.norm(dim=-1, keepdim=True)  # (..., 1) distance to axis
    return dist - radius
