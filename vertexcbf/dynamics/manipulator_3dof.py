from typing import Optional, Union

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine


class Manipulator3DOF(ControlAffine):
    """Three-link spatial revolute (RRR) manipulator dynamics.

    State: x = [q1, q2, q3, q1_dot, q2_dot, q3_dot]
        q1 - shoulder yaw   (rotation about world +z)
        q2 - shoulder pitch (about the local +y after the yaw rotation; q2 > 0 lifts the arm)
        q3 - elbow pitch    (forearm angle relative to the upper arm; q3 > 0 folds the tip up)

    At q1 = q2 = q3 = 0 the arm lies fully extended along the world +x axis.

    Input: u = [tau1, tau2, tau3]
        joint torques at the shoulder yaw, shoulder pitch, and elbow joints.

    Forward kinematics (l1 = upper-arm length, l2 = forearm length):
        r(q2, q3) = l1 * cos(q2) + l2 * cos(q2 + q3)
        p_ee = ( r * cos(q1),
                 r * sin(q1),
                 l1 * sin(q2) + l2 * sin(q2 + q3) )

    Dynamics (point-mass links at lc1, lc2 + diagonal joint inertias I1, I2, I3;
    gravity along -z):
        M(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = tau

    With this parameterisation the shoulder yaw is dynamically decoupled from the two
    pitch joints, so M is block-diagonal {1} (+) {2,3}, and M^{-1} is available in
    closed form (no linear solve needed).
    """

    name = "manipulator_3dof"
    nx = 6
    nu = 3
    periodic_states = []  # shoulder yaw wraps in [-pi, pi]
    clamp_states = [0, 1, 2, 3, 4, 5]

    def __init__(
        self,
        x_min: Union[list, Tensor],
        x_max: Union[list, Tensor],
        u_min: Union[list, Tensor],
        u_max: Union[list, Tensor],
        m1: float = 0.5,
        m2: float = 0.5,
        l1: float = 0.5,
        l2: float = 0.5,
        lc1: float = 0.25,
        lc2: float = 0.25,
        I1: float = 0.01,
        I2: float = 0.01,
        I3: float = 0.01,
        gravity: float = 9.81,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(x_min, x_max, u_min, u_max, device=device, dtype=dtype)
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.lc1 = lc1
        self.lc2 = lc2
        self.I1 = I1
        self.I2 = I2
        self.I3 = I3
        self.gravity = gravity

    def forward_kinematics(self, x: Tensor) -> Tensor:
        """Return the end-effector position p_ee in world coordinates.

        Args:
            x: State tensor of shape (..., nx). Only the first three components
                (q1, q2, q3) are used.

        Returns:
            Tensor of shape (..., 3) holding (px, py, pz).
        """
        q1 = x[..., 0]
        q2 = x[..., 1]
        q3 = x[..., 2]
        r_arm = self.l1 * torch.cos(q2) + self.l2 * torch.cos(q2 + q3)
        return torch.stack(
            [
                r_arm * torch.cos(q1),
                r_arm * torch.sin(q1),
                self.l1 * torch.sin(q2) + self.l2 * torch.sin(q2 + q3),
            ],
            dim=-1,
        )

    def _mass_components(self, q2: Tensor, q3: Tensor):
        """Compute the non-zero mass matrix entries and the inverse-block determinant.

        Returns the scalars M11, M22, M33, M23 (each shape (N,)) along with
        det_pitch = M22*M33 - M23**2, used for the closed-form 2x2 inverse of
        the lower (pitch-pitch) block.  Also returns the geometric helpers
        r_c (COM radius) and h_c (COM height) which the Coriolis terms reuse.
        """
        cos_q2 = torch.cos(q2)
        sin_q2 = torch.sin(q2)
        cos_q23 = torch.cos(q2 + q3)
        sin_q23 = torch.sin(q2 + q3)
        cos_q3 = torch.cos(q3)

        r_c = self.l1 * cos_q2 + self.lc2 * cos_q23
        h_c = self.l1 * sin_q2 + self.lc2 * sin_q23

        a = self.m1 * self.lc1**2
        b = self.m2
        l1lc2 = self.l1 * self.lc2

        M11 = a * cos_q2**2 + b * r_c**2 + self.I1
        M22 = a + b * (self.l1**2 + self.lc2**2 + 2.0 * l1lc2 * cos_q3) + self.I2
        M33 = torch.full_like(q3, b * self.lc2**2 + self.I3)
        M23 = b * (l1lc2 * cos_q3 + self.lc2**2)
        det_pitch = M22 * M33 - M23**2

        return M11, M22, M33, M23, det_pitch, r_c, h_c, sin_q23, cos_q3

    def f(self, x: Tensor) -> Tensor:
        q1 = x[:, 0]
        q2 = x[:, 1]
        q3 = x[:, 2]
        qd1 = x[:, 3]
        qd2 = x[:, 4]
        qd3 = x[:, 5]

        M11, M22, M33, M23, det_pitch, r_c, h_c, sin_q23, cos_q3 = (
            self._mass_components(q2, q3)
        )

        a = self.m1 * self.lc1**2
        b = self.m2
        l1lc2 = self.l1 * self.lc2
        sin_2q2 = torch.sin(2.0 * q2)
        sin_q3 = torch.sin(q3)

        # Coriolis * qdot
        Cqd1 = (
            -(a * sin_2q2 + 2.0 * b * r_c * h_c) * qd1 * qd2
            - 2.0 * b * r_c * self.lc2 * sin_q23 * qd1 * qd3
        )
        Cqd2 = (
            0.5 * a * sin_2q2 * qd1**2
            + b * r_c * h_c * qd1**2
            - 2.0 * b * l1lc2 * sin_q3 * qd2 * qd3
            - b * l1lc2 * sin_q3 * qd3**2
        )
        Cqd3 = b * r_c * self.lc2 * sin_q23 * qd1**2 + b * l1lc2 * sin_q3 * qd2**2

        # Gravity (along -z)
        g = self.gravity
        G2 = g * (self.m1 * self.lc1 + self.m2 * self.l1) * torch.cos(
            q2
        ) + self.m2 * g * self.lc2 * torch.cos(q2 + q3)
        G3 = self.m2 * g * self.lc2 * torch.cos(q2 + q3)

        # rhs = -C*qdot - G  (the un-actuated part of M*qddot = tau - C*qdot - G)
        rhs1 = -Cqd1
        rhs2 = -Cqd2 - G2
        rhs3 = -Cqd3 - G3

        # Closed-form inversion of the block-diagonal M.
        q1_ddot = rhs1 / M11
        q2_ddot = (M33 * rhs2 - M23 * rhs3) / det_pitch
        q3_ddot = (-M23 * rhs2 + M22 * rhs3) / det_pitch

        return torch.stack(
            [qd1, qd2, qd3, q1_ddot, q2_ddot, q3_ddot],
            dim=1,
        ).unsqueeze(-1)

    def g(self, x: Tensor) -> Tensor:
        q2 = x[:, 1]
        q3 = x[:, 2]

        M11, M22, M33, M23, det_pitch, *_ = self._mass_components(q2, q3)
        zeros = torch.zeros_like(q2)

        # M^{-1} (block-diagonal closed form)
        Minv_11 = 1.0 / M11
        Minv_22 = M33 / det_pitch
        Minv_23 = -M23 / det_pitch
        Minv_33 = M22 / det_pitch

        # g(x) = [[0; M^{-1}]] of shape (N, 6, 3): top 3 rows are zero (positions),
        # bottom 3 rows give qddot = M^{-1} tau.
        row1 = torch.stack([zeros, zeros, zeros], dim=-1)
        row2 = torch.stack([zeros, zeros, zeros], dim=-1)
        row3 = torch.stack([zeros, zeros, zeros], dim=-1)
        row4 = torch.stack([Minv_11, zeros, zeros], dim=-1)
        row5 = torch.stack([zeros, Minv_22, Minv_23], dim=-1)
        row6 = torch.stack([zeros, Minv_23, Minv_33], dim=-1)
        return torch.stack([row1, row2, row3, row4, row5, row6], dim=-2)
