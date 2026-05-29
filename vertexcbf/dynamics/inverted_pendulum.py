from typing import Optional, Union

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine


class InvertedPendulum(ControlAffine):
    """Inverted pendulum system dynamics.

    State: x = [theta, omega]
        theta : angle from upright (periodic)
        omega : angular velocity

    Input: u = [tau]
        tau : torque applied at the pivot

    Dynamics:
        d(theta)/dt = omega
        d(omega)/dt = (g / l) * sin(theta) + tau / (m * l^2)
    """

    name = "inverted_pendulum"
    nx = 2
    nu = 1
    periodic_states = []
    clamp_states = [0]

    def __init__(
        self,
        x_min: Union[list, Tensor],
        x_max: Union[list, Tensor],
        u_min: Union[list, Tensor],
        u_max: Union[list, Tensor],
        l: float,
        m: float,
        gravity: float = 9.81,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(x_min, x_max, u_min, u_max, device=device, dtype=dtype)
        self.l = l
        self.m = m
        self.gravity = gravity

    def f(self, x: Tensor) -> Tensor:
        """Return the open-loop drift dynamics."""
        return torch.stack(
            [
                x[:, 1],
                self.gravity * torch.sin(x[:, 0]) / self.l,
            ],
            dim=1,
        ).unsqueeze(-1)

    def g(self, x: Tensor) -> Tensor:
        """Return the control input matrix."""
        return torch.tensor(
            [
                [0.0],
                [1.0 / (self.m * self.l**2)],
            ],
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
