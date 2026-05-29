from typing import Optional, Union

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine


class DubinsCar(ControlAffine):
    """Dubins car system dynamics.

    State: x = [px, py, theta]
        px, py  : position
        theta   : heading angle (periodic)

    Input: u = [omega]
        omega : turning rate

    Dynamics:
        d(px)/dt    = v * cos(theta)
        d(py)/dt    = v * sin(theta)
        d(theta)/dt = omega

    where v is a constant forward speed.
    """

    name = "dubins_car"
    nx = 3
    nu = 1
    periodic_states = [2]
    clamp_states = []

    def __init__(
        self,
        x_min: Union[list, Tensor],
        x_max: Union[list, Tensor],
        u_min: Union[list, Tensor],
        u_max: Union[list, Tensor],
        v: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(x_min, x_max, u_min, u_max, device=device, dtype=dtype)
        self.v = v

    def f(self, x: Tensor) -> Tensor:
        """Return the open-loop drift dynamics."""
        return torch.stack(
            [
                self.v * torch.cos(x[:, 2]),
                self.v * torch.sin(x[:, 2]),
                torch.zeros_like(x[:, 2]),
            ],
            dim=1,
        ).unsqueeze(-1)

    def g(self, x: Tensor) -> Tensor:
        """Return the control input matrix."""
        return torch.tensor(
            [
                [0.0],
                [0.0],
                [1.0],
            ],
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
