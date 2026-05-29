from typing import Optional, Union

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine


class QuadrupedTrunk(ControlAffine):
    """Centroidal trunk model of a quadruped in quasi-static all-four-feet stance.

    Conventions:
        World frame   : ENU (x-east, y-north, z-up); gravity = [0, 0, -g].
        Body frame    : x-forward, y-left, z-up. Yaw is fixed at psi = 0
            (uncoupled from height / roll / pitch dynamics under this contact
            mode, and irrelevant to the tip-over / clearance safety story).
        Contact set   : all four feet in contact, with nominal body-frame foot
            positions r_i = [+/-a, +/-b, *].  The leg z-component drops out of
            the cross product because the foot forces are along body z.
        Contact forces: body-frame normal (vertical body-z) forces F_i >= 0
            (unilateral; legs push, do not pull).
        (px, py, psi) are dropped from the state because they are uncontrolled
            here and play no role in the safety constraint.

    State: x = [pz, phi, theta, vx, vy, vz, p, q, r]
        pz        : trunk height (m)
        phi, theta: roll, pitch (rad)
        vx, vy, vz: world-frame linear velocity (m/s)
        p, q, r   : body-frame angular velocity (rad/s)

    Input: u = [F1, F2, F3, F4]
        Body-frame normal contact forces at the four feet (N), in the order
            (FL, FR, RL, RR) = ([+a, +b, *], [+a, -b, *], [-a, +b, *], [-a, -b, *]).

    Kinematics:
        pz_dot    = vz
        phi_dot   = p + sin(phi)*tan(theta)*q + cos(phi)*tan(theta)*r
        theta_dot =              cos(phi)*q -            sin(phi)*r

    Translational (with psi = 0, the third column of R is [s_t*c_p, -s_p, c_t*c_p]):
        m*vx_dot = (sin(theta)*cos(phi))   * (F1 + F2 + F3 + F4)
        m*vy_dot = (-sin(phi))             * (F1 + F2 + F3 + F4)
        m*vz_dot = (cos(theta)*cos(phi))   * (F1 + F2 + F3 + F4) - m*g

    Rotational (cross product r_i x [0,0,F_i] = [b_i*F_i, -a_i*F_i, 0]):
        I_x*p_dot = b*(F1 - F2 + F3 - F4) - (I_z - I_y)*q*r
        I_y*q_dot = a*(-F1 - F2 + F3 + F4) - (I_x - I_z)*p*r
        I_z*r_dot = 0                       - (I_y - I_x)*p*q

    Yaw rate r is uncontrolled but kept in the state so its gyroscopic
    coupling into roll/pitch is honest.
    """

    name = "quadruped_trunk"
    nx = 9
    nu = 4
    periodic_states = []
    # Clamp the kinematic state and angles so Euler integration stays inside
    # the configured box (avoids the tan(theta) singularity at +/- pi/2).
    clamp_states = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    def __init__(
        self,
        x_min: Union[list, Tensor],
        x_max: Union[list, Tensor],
        u_min: Union[list, Tensor],
        u_max: Union[list, Tensor],
        mass: float = 12.0,
        Ix: float = 0.10,
        Iy: float = 0.20,
        Iz: float = 0.25,
        a: float = 0.20,
        b: float = 0.15,
        gravity: float = 9.81,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(x_min, x_max, u_min, u_max, device=device, dtype=dtype)
        if mass <= 0:
            raise ValueError(f"mass must be > 0; got {mass:.4g}.")
        for label, val in (("Ix", Ix), ("Iy", Iy), ("Iz", Iz)):
            if val <= 0:
                raise ValueError(f"{label} must be > 0; got {val:.4g}.")
        for label, val in (("a", a), ("b", b)):
            if val <= 0:
                raise ValueError(f"{label} must be > 0; got {val:.4g}.")
        if torch.any(self.u_min < 0):
            raise ValueError(
                "All u_min entries must be >= 0; legs apply unilateral contact forces. "
                f"Got u_min = {self.u_min.tolist()}."
            )

        self.mass = mass
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.a = a
        self.b = b
        self.gravity = gravity

    # ------------------------------------------------------------------
    # Drift and input matrix
    # ------------------------------------------------------------------

    def f(self, x: Tensor) -> Tensor:
        """Open-loop drift dynamics.

        Args:
            x: State tensor of shape (N, 9).

        Returns:
            Drift term of shape (N, 9, 1).
        """
        phi = x[:, 1]
        theta = x[:, 2]
        vz = x[:, 5]
        p = x[:, 6]
        q = x[:, 7]
        r = x[:, 8]

        cphi, sphi = torch.cos(phi), torch.sin(phi)
        ttheta = torch.tan(theta)

        # Position kinematics
        pz_dot = vz
        phi_dot = p + sphi * ttheta * q + cphi * ttheta * r
        theta_dot = cphi * q - sphi * r

        # Translational drift: only gravity (forces enter via g(x))
        zeros = torch.zeros_like(vz)
        vx_dot = zeros
        vy_dot = zeros
        vz_dot = torch.full_like(vz, -self.gravity)

        # Rotational drift: gyroscopic coupling only (torques enter via g(x))
        p_dot = -(self.Iz - self.Iy) / self.Ix * q * r
        q_dot = -(self.Ix - self.Iz) / self.Iy * p * r
        r_dot = -(self.Iy - self.Ix) / self.Iz * p * q

        return torch.stack(
            [pz_dot, phi_dot, theta_dot, vx_dot, vy_dot, vz_dot, p_dot, q_dot, r_dot],
            dim=1,
        ).unsqueeze(-1)

    def g(self, x: Tensor) -> Tensor:
        """Control-input matrix.

        Args:
            x: State tensor of shape (N, 9).

        Returns:
            Input matrix of shape (N, 9, 4); columns correspond to [F1, F2, F3, F4]
            (FL, FR, RL, RR).
        """
        phi = x[:, 1]
        theta = x[:, 2]
        N = x.shape[0]

        cphi, sphi = torch.cos(phi), torch.sin(phi)
        ctheta, stheta = torch.cos(theta), torch.sin(theta)
        zeros = torch.zeros(N, device=x.device, dtype=x.dtype)
        ones = torch.ones(N, device=x.device, dtype=x.dtype)

        # Translational: each foot contributes (1/m) * (third column of R with psi=0)
        # to (vx_dot, vy_dot, vz_dot).  All four columns are identical for the
        # translational rows (forces add scalarly along body z).
        tx = stheta * cphi / self.mass  # vx coefficient
        ty = -sphi / self.mass  # vy coefficient
        tz = ctheta * cphi / self.mass  # vz coefficient

        # Rotational moment arms per foot (body frame):
        #   p_dot: (b / I_x) * [+1, -1, +1, -1]
        #   q_dot: (a / I_y) * [-1, -1, +1, +1]
        #   r_dot: zeros
        b_Ix = self.b / self.Ix
        a_Iy = self.a / self.Iy

        # Assemble row-by-row so the shape is unambiguous.
        # Each row is one state dimension; each column is one input.
        def row(c1, c2, c3, c4):
            return torch.stack([c1, c2, c3, c4], dim=1)

        g_matrix = torch.stack(
            [
                # pz, phi, theta: no direct input
                row(zeros, zeros, zeros, zeros),
                row(zeros, zeros, zeros, zeros),
                row(zeros, zeros, zeros, zeros),
                # vx_dot
                row(tx, tx, tx, tx),
                # vy_dot
                row(ty, ty, ty, ty),
                # vz_dot
                row(tz, tz, tz, tz),
                # p_dot
                row(b_Ix * ones, -b_Ix * ones, b_Ix * ones, -b_Ix * ones),
                # q_dot
                row(-a_Iy * ones, -a_Iy * ones, a_Iy * ones, a_Iy * ones),
                # r_dot (uncontrolled)
                row(zeros, zeros, zeros, zeros),
            ],
            dim=1,
        )  # (N, 9, 4)

        return g_matrix
