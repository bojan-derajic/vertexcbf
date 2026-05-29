from typing import Optional, Union

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine


class DoubleIntegrator3D(ControlAffine):
    """3D double integrator system dynamics.

    State: x = [px, py, pz, vx, vy, vz]
        px, py, pz : position
        vx, vy, vz : velocity

    Input: u = [ax, ay, az]
        ax, ay, az : acceleration

    Dynamics:
        d(px)/dt = vx
        d(py)/dt = vy
        d(pz)/dt = vz
        d(vx)/dt = ax
        d(vy)/dt = ay
        d(vz)/dt = az
    """

    name = "double_integrator_3d"
    nx = 6
    nu = 3
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
                x[:, 3],
                x[:, 4],
                x[:, 5],
                torch.zeros_like(x[:, 3]),
                torch.zeros_like(x[:, 4]),
                torch.zeros_like(x[:, 5]),
            ],
            dim=1,
        ).unsqueeze(-1)

    def g(self, x: Tensor) -> Tensor:
        """Return the control input matrix."""
        return torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
