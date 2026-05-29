from typing import Optional, Union

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine


class VerticalDrone2D(ControlAffine):
    """2D vertical drone system dynamics.

    State: x = [z, vz]
        z : vertical position
        vz : vertical velocity

    Input: u = [az]
        az : vertical acceleration

    Dynamics:
        d(z)/dt = vz
        d(vz)/dt = K * az - g
    """

    name = "vertical_drone_2d"
    nx = 2
    nu = 1
    periodic_states = []
    clamp_states = []

    def __init__(
        self,
        x_min: Union[list, Tensor],
        x_max: Union[list, Tensor],
        u_min: Union[list, Tensor],
        u_max: Union[list, Tensor],
        K: float = 1.0,
        gravity: float = 9.81,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(x_min, x_max, u_min, u_max, device=device, dtype=dtype)
        self.K = K
        self.gravity = gravity

    def f(self, x: Tensor) -> Tensor:
        """Return the open-loop drift dynamics."""
        return torch.stack(
            [
                x[:, 1],
                torch.full_like(x[:, 1], -self.gravity),
            ],
            dim=1,
        ).unsqueeze(-1)

    def g(self, x: Tensor) -> Tensor:
        """Return the control input matrix."""
        return torch.tensor(
            [
                [0.0],
                [self.K],
            ],
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
