from typing import Optional, Union

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine


class LandingRocket(ControlAffine):
    """Planar landing-rocket system dynamics.

    State: x = [px, pz, vx, vz, theta, omega, m]
        px      : horizontal position
        pz      : vertical position (ground at pz = 0)
        vx, vz  : linear velocities
        theta   : pitch angle (periodic)
        omega   : angular rate
        m       : mass

    Input: u = [T, tau]
        T   : thrust magnitude (>= 0)
        tau : body torque

    Dynamics:
        d(px)/dt    = vx
        d(pz)/dt    = vz
        d(vx)/dt    = -(T / m) * sin(theta)
        d(vz)/dt    =  (T / m) * cos(theta) - g
        d(theta)/dt = omega
        d(omega)/dt = tau / I
        d(m)/dt     = -alpha * T
    """

    name = "landing_rocket"
    nx = 7
    nu = 2
    periodic_states = []
    clamp_states = [1, 3, 4, 6]  # pz, vz, theta and mass

    def __init__(
        self,
        x_min: Union[list, Tensor],
        x_max: Union[list, Tensor],
        u_min: Union[list, Tensor],
        u_max: Union[list, Tensor],
        gravity: float = 9.81,
        inertia: float = 1.0,
        alpha: float = 0.01,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(x_min, x_max, u_min, u_max, device=device, dtype=dtype)
        if self.u_min[0] < 0:
            raise ValueError(
                f"u_min[0] (minimum thrust) must be >= 0; got {self.u_min[0].item():.4g}. "
                "A rocket cannot produce negative thrust."
            )
        if self.x_min[6] <= 0:
            raise ValueError(
                f"x_min[6] (minimum mass) must be > 0; got {self.x_min[6].item():.4g}. "
                "Mass must remain strictly positive to avoid singular dynamics."
            )
        if inertia <= 0:
            raise ValueError(f"inertia must be > 0; got {inertia:.4g}.")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0; got {alpha:.4g}.")
        self.gravity = gravity
        self.inertia = inertia
        self.alpha = alpha

    def f(self, x: Tensor) -> Tensor:
        """Return the open-loop drift dynamics f(x).

        Args:
            x: State tensor of shape (N, 7).

        Returns:
            Drift term of shape (N, 7, 1).
        """
        vx = x[:, 2]
        vz = x[:, 3]
        omega = x[:, 5]

        drift = torch.stack(
            [
                vx,
                vz,
                torch.zeros_like(vx),
                torch.full_like(vz, -self.gravity),
                omega,
                torch.zeros_like(omega),
                torch.zeros_like(vx),
            ],
            dim=1,
        ).unsqueeze(-1)

        return drift

    def g(self, x: Tensor) -> Tensor:
        """Return the control input matrix g(x).

        Args:
            x: State tensor of shape (N, 7).

        Returns:
            Input matrix of shape (N, 7, 2); columns correspond to [T, tau].
        """
        theta = x[:, 4]
        m = x[:, 6]

        zeros = torch.zeros_like(theta)
        inv_I = torch.full_like(theta, 1.0 / self.inertia)
        neg_alpha = torch.full_like(theta, -self.alpha)

        g_matrix = torch.stack(
            [
                # px:    [0, 0]
                torch.stack([zeros, zeros], dim=1),
                # pz:    [0, 0]
                torch.stack([zeros, zeros], dim=1),
                # vx:    [-sin(theta)/m, 0]
                torch.stack([-torch.sin(theta) / m, zeros], dim=1),
                # vz:    [ cos(theta)/m, 0]
                torch.stack([torch.cos(theta) / m, zeros], dim=1),
                # theta: [0, 0]
                torch.stack([zeros, zeros], dim=1),
                # omega: [0, 1/I]
                torch.stack([zeros, inv_I], dim=1),
                # m:     [-alpha, 0]
                torch.stack([neg_alpha, zeros], dim=1),
            ],
            dim=1,
        )  # (N, 7, 2)

        return g_matrix
