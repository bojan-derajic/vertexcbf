import torch
from torch import Tensor


def ee_sphere_sdf(
    states: Tensor,
    l1: float,
    l2: float,
    center: Tensor,
    radius: float,
    q1_idx: int = 0,
    q2_idx: int = 1,
    q3_idx: int = 2,
) -> Tensor:
    """
    Signed distance function for the end-effector of a 3-DOF RRR spatial
    manipulator with respect to a spherical obstacle.

    Forward kinematics (zero pose: arm fully extended along +x;
    q2 > 0 lifts; q3 measures the forearm relative to the upper arm):

        r(q2, q3) = l1*cos(q2) + l2*cos(q2 + q3)
        p_ee     = ( r * cos(q1),
                     r * sin(q1),
                     l1*sin(q2) + l2*sin(q2 + q3) )

    Convention (safe-set): positive outside the sphere, negative inside.
        c(x) = ||p_ee(x) - center|| - radius

    Args:
        states: (..., nx) tensor of states; q1, q2, q3 are read from the
            indices given below.
        l1: upper-arm length (> 0).
        l2: forearm length (> 0).
        center: (3,) tensor [cx, cy, cz] — obstacle centre in world coords.
        radius: obstacle radius (> 0).
        q1_idx, q2_idx, q3_idx: state indices of the three joint angles.
            Defaults match :class:`~vertexcbf.dynamics.Manipulator3DOF`.

    Returns:
        (..., 1) tensor of SDF values (positive outside the sphere, negative inside).
    """
    q1 = states[..., q1_idx]
    q2 = states[..., q2_idx]
    q3 = states[..., q3_idx]
    center_ = torch.as_tensor(center, dtype=states.dtype, device=states.device)
    radius_ = torch.tensor(radius, dtype=states.dtype, device=states.device)
    r_arm = l1 * torch.cos(q2) + l2 * torch.cos(q2 + q3)
    p_ee = torch.stack(
        [
            r_arm * torch.cos(q1),
            r_arm * torch.sin(q1),
            l1 * torch.sin(q2) + l2 * torch.sin(q2 + q3),
        ],
        dim=-1,
    )
    dist = torch.norm(p_ee - center_, dim=-1, keepdim=True)
    return dist - radius_
