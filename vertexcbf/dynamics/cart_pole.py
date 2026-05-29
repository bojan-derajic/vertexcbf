from typing import Optional, Union

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine


class CartPole(ControlAffine):
    """Cart-pole (cart with inverted pendulum) system dynamics.

    State: x = [x, theta, x_dot, theta_dot]
        x         - horizontal cart position
        theta     - pendulum angle (0 = upright, pi = hanging down)
        x_dot     - cart velocity
        theta_dot - pendulum angular velocity

    Input: u = [f_x]
        f_x : horizontal force applied to the cart

    Dynamics (theta=0 upright convention, derived via Lagrangian mechanics):
        x_ddot     = [f_x + m_p*sin(theta)*(l*theta_dot^2 - g*cos(theta))] / D
        theta_ddot = [(m_c + m_p)*g*sin(theta) - m_p*l*theta_dot^2*cos(theta)*sin(theta)
                      - f_x*cos(theta)] / (l * D)
    where D = m_c + m_p * sin^2(theta)
    """

    name = "cart_pole"
    nx = 4
    nu = 1
    periodic_states = []
    clamp_states = [1, 2, 3]

    def __init__(
        self,
        x_min: Union[list, Tensor],
        x_max: Union[list, Tensor],
        u_min: Union[list, Tensor],
        u_max: Union[list, Tensor],
        m_c: float = 1.0,
        m_p: float = 0.1,
        l: float = 0.5,
        gravity: float = 9.81,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(x_min, x_max, u_min, u_max, device=device, dtype=dtype)
        self.m_c = m_c
        self.m_p = m_p
        self.l = l
        self.gravity = gravity

    def f(self, x: Tensor) -> Tensor:
        """Return the open-loop drift dynamics f(x)."""
        theta = x[:, 1]
        x_dot = x[:, 2]
        theta_dot = x[:, 3]

        D = self.m_c + self.m_p * torch.sin(theta) ** 2

        x_ddot = (
            self.m_p
            * torch.sin(theta)
            * (self.l * theta_dot**2 - self.gravity * torch.cos(theta))
        ) / D

        theta_ddot = (
            (self.m_c + self.m_p) * self.gravity * torch.sin(theta)
            - self.m_p * self.l * theta_dot**2 * torch.cos(theta) * torch.sin(theta)
        ) / (self.l * D)

        return torch.stack(
            [x_dot, theta_dot, x_ddot, theta_ddot],
            dim=1,
        ).unsqueeze(-1)

    def g(self, x: Tensor) -> Tensor:
        """Return the control input matrix g(x) such that xdot = f(x) + g(x)*u."""
        theta = x[:, 1]
        D = self.m_c + self.m_p * torch.sin(theta) ** 2

        zeros = torch.zeros_like(theta)
        g_matrix = torch.stack(
            [
                zeros,
                zeros,
                1.0 / D,
                -torch.cos(theta) / (self.l * D),
            ],
            dim=1,
        ).unsqueeze(
            -1
        )  # (N, 4, 1)

        return g_matrix
