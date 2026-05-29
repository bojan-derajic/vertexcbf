from typing import Optional, Union

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine


class RelativeUnicycle(ControlAffine):
    """Relative dynamics between a dynamic-unicycle robot and a constant-velocity
    pedestrian/obstacle, expressed in the robot's body frame.

    State: x = [px, py, psi, vr, vp]
        px, py : pedestrian position in robot body frame
        psi    : pedestrian heading relative to robot heading (periodic)
        vr     : robot linear speed
        vp     : pedestrian speed (treated as an unknown constant parameter
                 lifted into the state)

    Input: u = [a, omega]
        a     : robot linear acceleration
        omega : robot angular velocity

    Dynamics:
        d(px)/dt  =  vp * cos(psi) - vr + omega * py
        d(py)/dt  =  vp * sin(psi)      - omega * px
        d(psi)/dt = -omega
        d(vr)/dt  =  a
        d(vp)/dt  =  0

    Notes:
        The omega-coupled terms in dpx/dpy are the Coriolis-like effect of
        expressing positions in a rotating (body) frame.  Together with the
        -omega term in dpsi/dt they make the system control-affine but with a
        state-dependent input matrix g(x).
    """

    name = "relative_unicycle"
    nx = 5
    nu = 2
    periodic_states = [2]
    clamp_states = [3, 4]

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
        psi = x[:, 2]
        vr = x[:, 3]
        vp = x[:, 4]
        zero = torch.zeros_like(vr)
        return torch.stack(
            [
                vp * torch.cos(psi) - vr,
                vp * torch.sin(psi),
                zero,
                zero,
                zero,
            ],
            dim=1,
        ).unsqueeze(-1)

    def g(self, x: Tensor) -> Tensor:
        """Return the (state-dependent) control input matrix g(x).

        Columns correspond to controls [a, omega]:

            g(x) = [[ 0,   py ],
                    [ 0,  -px ],
                    [ 0,  -1  ],
                    [ 1,   0  ],
                    [ 0,   0  ]]
        """
        N = x.shape[0]
        px = x[:, 0]
        py = x[:, 1]
        zero = torch.zeros_like(px)
        one = torch.ones_like(px)

        g = torch.zeros(N, self.nx, self.nu, device=self.device, dtype=self.dtype)
        g[:, 3, 0] = one          # a   -> dvr/dt
        g[:, 0, 1] = py           # omega -> dpx/dt
        g[:, 1, 1] = -px          # omega -> dpy/dt
        g[:, 2, 1] = -one         # omega -> dpsi/dt
        return g
