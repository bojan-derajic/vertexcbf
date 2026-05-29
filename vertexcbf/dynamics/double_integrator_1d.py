from typing import Optional, Union

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine


class DoubleIntegrator1D(ControlAffine):
    """1D double integrator system dynamics.

    State: x = [p, v]
        p : position
        v : velocity

    Input: u = [a]
        a : acceleration

    Dynamics:
        d(p)/dt = v
        d(v)/dt = a
    """

    name = "double_integrator_1d"
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
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(x_min, x_max, u_min, u_max, device=device, dtype=dtype)

    def f(self, x: Tensor) -> Tensor:
        """Return the open-loop drift dynamics."""
        return torch.stack(
            [
                x[:, 1],
                torch.zeros_like(x[:, 1]),
            ],
            dim=1,
        ).unsqueeze(-1)

    def g(self, x: Tensor) -> Tensor:
        """Return the control input matrix."""
        return torch.tensor(
            [
                [0.0],
                [1.0],
            ],
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
