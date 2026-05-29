from typing import Optional, Union

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine


class KinematicBicycle(ControlAffine):
    """Kinematic bicycle model system dynamics.

    State: x = [px, py, psi, v]
        px, py : position
        psi    : heading angle (periodic)
        v      : longitudinal speed

    Input: u = [tan_delta, a]
        tan_delta : tangent of the front steering angle (= tan(delta))
        a         : longitudinal acceleration

    Dynamics (centre-of-rear-axle formulation):
        d(px)/dt  = v * cos(psi)
        d(py)/dt  = v * sin(psi)
        d(psi)/dt = v * tan_delta / L
        d(v)/dt   = a
    """

    name = "kinematic_bicycle"
    nx = 4
    nu = 2
    periodic_states = [2]
    clamp_states = []

    def __init__(
        self,
        x_min: Union[list, Tensor],
        x_max: Union[list, Tensor],
        u_min: Union[list, Tensor],
        u_max: Union[list, Tensor],
        L: float,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(x_min, x_max, u_min, u_max, device=device, dtype=dtype)
        self.L = L

    def f(self, x: Tensor) -> Tensor:
        """Return the open-loop drift dynamics."""
        psi = x[:, 2]
        v = x[:, 3]
        return torch.stack(
            [
                v * torch.cos(psi),
                v * torch.sin(psi),
                torch.zeros_like(psi),
                torch.zeros_like(v),
            ],
            dim=1,
        ).unsqueeze(-1)

    def g(self, x: Tensor) -> Tensor:
        """Return the control input matrix."""
        v = x[:, 3]
        B = torch.zeros(
            (x.shape[0], self.nx, self.nu), device=self.device, dtype=self.dtype
        )
        B[:, 2, 0] = v / self.L
        B[:, 3, 1] = 1.0
        return B
