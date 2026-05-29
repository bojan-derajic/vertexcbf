from typing import Optional, Union

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine


class DoubleIntegrator2D(ControlAffine):
    """2D double integrator system dynamics.

    State: x = [px, py, vx, vy]
        px, py : position
        vx, vy : velocity

    Input: u = [ax, ay]
        ax, ay : acceleration

    Dynamics:
        d(px)/dt = vx
        d(py)/dt = vy
        d(vx)/dt = ax
        d(vy)/dt = ay
    """

    name = "double_integrator_2d"
    nx = 4
    nu = 2
    periodic_states = []
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
        return torch.stack(
            [
                x[:, 2],
                x[:, 3],
                torch.zeros_like(x[:, 2]),
                torch.zeros_like(x[:, 3]),
            ],
            dim=1,
        ).unsqueeze(-1)

    def g(self, x: Tensor) -> Tensor:
        """Return the control input matrix."""
        return torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
