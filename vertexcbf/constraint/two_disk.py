import torch
from torch import Tensor


def two_disk_sdf(
    states: Tensor,
    robot_radius: float,
    obstacle_radius: float,
) -> Tensor:
    """
    Signed distance function for two disks colliding, in the *robot body frame*.

    The first two states are the obstacle's position relative to the robot,
    so the robot disk sits at the origin and the obstacle disk is centred at
    (px, py).  Collision happens when the disks overlap, i.e. when
    ``||(px, py)|| < robot_radius + obstacle_radius``.

    Convention (safe-set): positive when disks are separated, negative when
    they overlap.

        c(x) = ||(px, py)|| - (robot_radius + obstacle_radius)

    Args:
        states: (..., nx) tensor of states; first two dims are (px, py) in the
            robot body frame.
        robot_radius: radius of the robot disk (>= 0).
        obstacle_radius: radius of the obstacle disk (>= 0).

    Returns:
        (..., 1) tensor of SDF values (positive outside the collision set,
        negative inside).
    """
    xy = states[..., :2]
    r = torch.tensor(
        robot_radius + obstacle_radius, dtype=states.dtype, device=states.device
    )
    dist = torch.norm(xy, dim=-1, keepdim=True)
    return dist - r
