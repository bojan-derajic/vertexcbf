import torch
from torch import Tensor


def manipulator_sphere_sdf(
    states: Tensor,
    l1: float,
    l2: float,
    center: Tensor,
    radius: float,
    link_radius: float = 0.0,
    alpha: float = 20.0,
    q1_idx: int = 0,
    q2_idx: int = 1,
    q3_idx: int = 2,
) -> Tensor:
    """
    Whole-arm signed distance function for a 3-DOF RRR spatial manipulator
    against a spherical obstacle.

    Each link is modelled as a capsule (line segment of length ``l1`` or ``l2``
    inflated by ``link_radius``).  The overall safe-set SDF is the softmin of
    the per-link capsule-vs-sphere distances, which corresponds to the
    intersection of the two link-safety constraints:

        c(x) = softmin_i ( dist(p_obstacle, link_i) - radius - link_radius )

    Convention (safe-set): positive when both links clear the inflated
    obstacle; negative when either link penetrates it.

    Args:
        states: (..., nx) tensor of states.  Joint angles are read from
            ``q1_idx``, ``q2_idx``, ``q3_idx``.
        l1: upper-arm length (> 0).
        l2: forearm length (> 0).
        center: (3,) tensor [cx, cy, cz] — obstacle centre in world coords.
        radius: obstacle radius (> 0).
        link_radius: capsule thickness applied to both links (>= 0).
            Default 0 reduces to "link centerlines must miss the sphere."
        alpha: softmin temperature (> 0).  Larger values approach the hard
            min over the two links.  Default 20.0.
        q1_idx, q2_idx, q3_idx: state indices of the three joint angles.

    Returns:
        (..., 1) tensor of SDF values (positive = both links safely outside
        the inflated obstacle, negative = at least one link in collision).
    """
    q1 = states[..., q1_idx]
    q2 = states[..., q2_idx]
    q3 = states[..., q3_idx]

    center_ = torch.as_tensor(center, dtype=states.dtype, device=states.device)
    radius_ = torch.tensor(radius, dtype=states.dtype, device=states.device)
    link_r = torch.tensor(link_radius, dtype=states.dtype, device=states.device)

    cos_q1 = torch.cos(q1)
    sin_q1 = torch.sin(q1)
    cos_q2 = torch.cos(q2)
    sin_q2 = torch.sin(q2)
    cos_q23 = torch.cos(q2 + q3)
    sin_q23 = torch.sin(q2 + q3)

    # Joint positions in world coordinates.
    shoulder = torch.zeros_like(torch.stack([q1, q1, q1], dim=-1))  # (..., 3)

    r_elbow = l1 * cos_q2
    elbow = torch.stack(
        [r_elbow * cos_q1, r_elbow * sin_q1, l1 * sin_q2],
        dim=-1,
    )  # (..., 3)

    r_ee = l1 * cos_q2 + l2 * cos_q23
    ee = torch.stack(
        [r_ee * cos_q1, r_ee * sin_q1, l1 * sin_q2 + l2 * sin_q23],
        dim=-1,
    )  # (..., 3)

    def _seg_distance(A: Tensor, B: Tensor, P: Tensor) -> Tensor:
        """Distance from point P to segment AB; returns shape (..., 1)."""
        d = B - A
        v = P - A
        dd = (d * d).sum(dim=-1, keepdim=True)
        vd = (v * d).sum(dim=-1, keepdim=True)
        t = torch.clamp(vd / dd, 0.0, 1.0)
        closest = A + t * d
        return torch.norm(P - closest, dim=-1, keepdim=True)

    d1 = _seg_distance(shoulder, elbow, center_) - radius_ - link_r  # (..., 1)
    d2 = _seg_distance(elbow, ee, center_) - radius_ - link_r  # (..., 1)

    # Softmin over the two links — intersection of safe sets ("every link
    # must clear the obstacle").  Hard min recovered as alpha -> infinity.
    stacked = torch.cat([d1, d2], dim=-1)  # (..., 2)
    return -(1.0 / alpha) * torch.logsumexp(-alpha * stacked, dim=-1, keepdim=True)
