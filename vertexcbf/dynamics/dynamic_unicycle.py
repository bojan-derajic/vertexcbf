from typing import Optional, Union

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine


class DynamicUnicycle(ControlAffine):
    """Dynamic unicycle system dynamics.

    State: x = [px, py, theta, v, omega]
        px, py  : position
        theta   : heading angle (periodic)
        v       : linear speed
        omega   : angular velocity

    Input: u = [a, alpha]
        a     : linear acceleration
        alpha : angular acceleration

    Dynamics:
        d(px)/dt    = v * cos(theta)
        d(py)/dt    = v * sin(theta)
        d(theta)/dt = omega
        d(v)/dt     = a
        d(omega)/dt = alpha
    """

    name = "dynamic_unicycle"
    nx = 5
    nu = 2
    periodic_states = [2]
    clamp_states = []

    def __init__(
        self,
        x_min: Union[list, Tensor],
        x_max: Union[list, Tensor],
        u_min: Union[list, Tensor],
        u_max: Union[list, Tensor],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(x_min, x_max, u_min, u_max, device=device, dtype=dtype)

    def f(self, x: Tensor) -> Tensor:
        """Return the open-loop drift dynamics."""
        theta = x[:, 2]
        v = x[:, 3]
        omega = x[:, 4]
        return torch.stack(
            [
                v * torch.cos(theta),
                v * torch.sin(theta),
                omega,
                torch.zeros_like(v),
                torch.zeros_like(omega),
            ],
            dim=1,
        ).unsqueeze(-1)

    def g(self, x: Tensor) -> Tensor:
        """Return the control input matrix."""
        return torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
