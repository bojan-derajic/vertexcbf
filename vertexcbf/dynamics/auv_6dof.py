from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine


class AUV6DoF(ControlAffine):
    """6-DoF underwater vehicle (Fossen rigid-body model).

    Conventions:
        World frame   : ENU (x-east, y-north, z-up); gravity = [0, 0, -g].
        Body frame    : x-forward, y-left, z-up.
        Euler angles  : ZYX intrinsic (roll-pitch-yaw = phi-theta-psi).
        Translational mass is assumed isotropic (single scalar `m_t`,
            with added mass absorbed); inertia tensor is diagonal principal-axis.
        Neutrally buoyant (W = B = m_t * g); centre of buoyancy is offset
            `h_cb` above the centre of gravity along body +z, giving a
            self-righting restoring moment.

    State: x = [px, py, pz, phi, theta, psi, u, v, w, p, q, r]
        px, py, pz : world position (m)
        phi        : roll about body x (rad)
        theta      : pitch about body y (rad)
        psi        : yaw about body z (rad)
        u, v, w    : body-frame linear velocity (m/s)
        p, q, r    : body-frame angular velocity (rad/s)

    Input: tau = [Fx, Fy, Fz, taux, tauy, tauz]
        Body-frame force/moment (post thruster allocation).

    Kinematics (eta_dot = J(eta) nu):
        [px_dot, py_dot, pz_dot]^T = R(phi, theta, psi) [u, v, w]^T
        phi_dot   = p + sin(phi)*tan(theta)*q + cos(phi)*tan(theta)*r
        theta_dot =                cos(phi)*q -            sin(phi)*r
        psi_dot   =     sin(phi)/cos(theta)*q + cos(phi)/cos(theta)*r

    Dynamics:
        m_t*u_dot = Fx - m_t*(q*w - r*v) - (dl_x + dq_x*|u|)*u
        m_t*v_dot = Fy - m_t*(r*u - p*w) - (dl_y + dq_y*|v|)*v
        m_t*w_dot = Fz - m_t*(p*v - q*u) - (dl_z + dq_z*|w|)*w
        I_x*p_dot = taux - (I_z - I_y)*q*r - h_cb*B*cos(theta)*sin(phi) - (dl_p + dq_p*|p|)*p
        I_y*q_dot = tauy - (I_x - I_z)*p*r - h_cb*B*sin(theta)          - (dl_q + dq_q*|q|)*q
        I_z*r_dot = tauz - (I_y - I_x)*p*q                              - (dl_r + dq_r*|r|)*r

    The Euler-rate transform is singular at theta = +/- pi/2, so state bounds
    for theta should stay well inside that interval (default configs use ~+/-1.0 rad).
    """

    name = "auv_6dof"
    nx = 12
    nu = 6
    # phi and psi are 2*pi-periodic; theta is not (singularity at +/- pi/2).
    periodic_states = [3, 5]
    clamp_states = [
        4,
        6,
        7,
        8,
        9,
        10,
        11,
    ]  # keep theta inside (-pi/2, pi/2) to avoid the J(eta) singularity

    def __init__(
        self,
        x_min: Union[list, Tensor],
        x_max: Union[list, Tensor],
        u_min: Union[list, Tensor],
        u_max: Union[list, Tensor],
        m_t: float = 30.0,
        Ix: float = 0.5,
        Iy: float = 2.0,
        Iz: float = 2.0,
        d_lin: Sequence[float] = (5.0, 20.0, 20.0, 0.5, 2.0, 2.0),
        d_quad: Sequence[float] = (10.0, 50.0, 50.0, 0.5, 5.0, 5.0),
        h_cb: float = 0.02,
        gravity: float = 9.81,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(x_min, x_max, u_min, u_max, device=device, dtype=dtype)
        if m_t <= 0:
            raise ValueError(f"m_t must be > 0; got {m_t:.4g}.")
        for label, val in (("Ix", Ix), ("Iy", Iy), ("Iz", Iz)):
            if val <= 0:
                raise ValueError(f"{label} must be > 0; got {val:.4g}.")
        if len(d_lin) != 6 or len(d_quad) != 6:
            raise ValueError("d_lin and d_quad must have length 6.")
        if any(d < 0 for d in d_lin) or any(d < 0 for d in d_quad):
            raise ValueError("Damping coefficients must be non-negative.")
        if h_cb < 0:
            raise ValueError(f"h_cb must be >= 0; got {h_cb:.4g}.")

        self.m_t = m_t
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.gravity = gravity
        self.h_cb = h_cb
        self.B = m_t * gravity  # buoyancy magnitude (N), neutrally buoyant

        self.d_lin = torch.as_tensor(d_lin, dtype=dtype, device=device)
        self.d_quad = torch.as_tensor(d_quad, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Drift and input matrix
    # ------------------------------------------------------------------

    def f(self, x: Tensor) -> Tensor:
        """Open-loop drift dynamics.

        Args:
            x: State tensor of shape (N, 12).

        Returns:
            Drift term of shape (N, 12, 1).
        """
        phi = x[:, 3]
        theta = x[:, 4]
        psi = x[:, 5]
        u = x[:, 6]
        v = x[:, 7]
        w = x[:, 8]
        p = x[:, 9]
        q = x[:, 10]
        r = x[:, 11]

        cphi, sphi = torch.cos(phi), torch.sin(phi)
        ctheta, stheta = torch.cos(theta), torch.sin(theta)
        cpsi, spsi = torch.cos(psi), torch.sin(psi)
        ttheta = torch.tan(theta)

        # World-frame velocity = R(phi, theta, psi) * [u, v, w]
        r11 = cpsi * ctheta
        r12 = cpsi * stheta * sphi - spsi * cphi
        r13 = cpsi * stheta * cphi + spsi * sphi
        r21 = spsi * ctheta
        r22 = spsi * stheta * sphi + cpsi * cphi
        r23 = spsi * stheta * cphi - cpsi * sphi
        r31 = -stheta
        r32 = ctheta * sphi
        r33 = ctheta * cphi

        px_dot = r11 * u + r12 * v + r13 * w
        py_dot = r21 * u + r22 * v + r23 * w
        pz_dot = r31 * u + r32 * v + r33 * w

        # Euler-angle rates
        phi_dot = p + sphi * ttheta * q + cphi * ttheta * r
        theta_dot = cphi * q - sphi * r
        psi_dot = sphi / ctheta * q + cphi / ctheta * r

        # Body-frame translational drift (Coriolis + damping; gravity/buoyancy cancel).
        damp_lin = self.d_lin  # (6,)
        damp_quad = self.d_quad  # (6,)
        u_dot = (
            -(q * w - r * v)
            - (damp_lin[0] + damp_quad[0] * torch.abs(u)) * u / self.m_t
        )
        v_dot = (
            -(r * u - p * w)
            - (damp_lin[1] + damp_quad[1] * torch.abs(v)) * v / self.m_t
        )
        w_dot = (
            -(p * v - q * u)
            - (damp_lin[2] + damp_quad[2] * torch.abs(w)) * w / self.m_t
        )

        # Body-frame rotational drift (Euler + restoring + damping).
        restore_roll = self.h_cb * self.B * ctheta * sphi
        restore_pitch = self.h_cb * self.B * stheta
        p_dot = (
            -(self.Iz - self.Iy) * q * r / self.Ix
            - restore_roll / self.Ix
            - (damp_lin[3] + damp_quad[3] * torch.abs(p)) * p / self.Ix
        )
        q_dot = (
            -(self.Ix - self.Iz) * p * r / self.Iy
            - restore_pitch / self.Iy
            - (damp_lin[4] + damp_quad[4] * torch.abs(q)) * q / self.Iy
        )
        r_dot = (
            -(self.Iy - self.Ix) * p * q / self.Iz
            - (damp_lin[5] + damp_quad[5] * torch.abs(r)) * r / self.Iz
        )

        return torch.stack(
            [
                px_dot,
                py_dot,
                pz_dot,
                phi_dot,
                theta_dot,
                psi_dot,
                u_dot,
                v_dot,
                w_dot,
                p_dot,
                q_dot,
                r_dot,
            ],
            dim=1,
        ).unsqueeze(-1)

    def g(self, x: Tensor) -> Tensor:
        """Control-input matrix (constant: tau enters only the body-frame
        velocity equations via M^{-1}).

        Args:
            x: State tensor of shape (N, 12).

        Returns:
            Input matrix of shape (N, 12, 6).
        """
        N = x.shape[0]
        g_mat = torch.zeros(N, self.nx, self.nu, device=x.device, dtype=x.dtype)
        inv_diag = torch.tensor(
            [
                1.0 / self.m_t,
                1.0 / self.m_t,
                1.0 / self.m_t,
                1.0 / self.Ix,
                1.0 / self.Iy,
                1.0 / self.Iz,
            ],
            device=x.device,
            dtype=x.dtype,
        )
        # Rows 6..11 (the six velocity dimensions) get the inverse mass diagonal.
        g_mat[:, 6:12, :] = torch.diag(inv_diag)
        return g_mat
