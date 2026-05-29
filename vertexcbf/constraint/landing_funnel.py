import torch
from torch import Tensor


def landing_funnel_sdf(
    states: Tensor,
    px_pad: float = 0.5,
    slope: float = 0.5,
    vel_weight: float = 0.03,
    theta_weight: float = 0.15,
    eps: float = 1e-4,
    px_idx: int = 0,
    pz_idx: int = 1,
    vx_idx: int = 2,
    vz_idx: int = 3,
    theta_idx: int = 4,
) -> Tensor:
    """
    Smooth landing-funnel signed distance function (SDF).

    Convention:
        Positive inside safe set.
        Negative outside safe set.

    Safe set intuition:
        - Near the ground, horizontal error tolerance shrinks.
        - High velocity near the ground becomes unsafe.
        - Large tilt becomes unsafe.

    Constraint:

        h(x) =
            (px_pad + slope * pz)
            - sqrt(px^2 + eps)
            - vel_weight * (vx^2 + vz^2)
            - theta_weight * theta^2

    Args:
        states:
            (..., nx) tensor of states.

        px_pad:
            Half-width of landing pad at pz = 0.

        slope:
            Funnel widening rate with altitude.

        vel_weight:
            Penalty on translational velocity magnitude.

        theta_weight:
            Penalty on tilt angle magnitude.

        eps:
            Small smoothing constant for differentiability.

    Returns:
        (..., 1) tensor of signed-distance-like safety values.
    """

    px = states[..., px_idx : px_idx + 1]
    pz = states[..., pz_idx : pz_idx + 1]
    vx = states[..., vx_idx : vx_idx + 1]
    vz = states[..., vz_idx : vz_idx + 1]
    theta = states[..., theta_idx : theta_idx + 1]

    px_term = torch.sqrt(px.square() + eps)
    vel_term = vx.square() + vz.square()
    theta_term = theta.square()

    h = (
        (px_pad + slope * pz)
        - px_term
        - vel_weight * vel_term
        - theta_weight * theta_term
    )
    return h
