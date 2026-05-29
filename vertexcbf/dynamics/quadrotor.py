from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine


class Quadrotor(ControlAffine):
    """Quadrotor dynamics with quaternion orientation representation.

    State: x = [px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        px, py, pz      : position (m)
        qw, qx, qy, qz  : unit quaternion (body orientation)
        vx, vy, vz      : linear velocity (m/s) in world frame
        wx, wy, wz      : angular velocity (rad/s) in body frame

    Input: u = [F, ax, ay, az]
        F   : total thrust force (N)
        ax  : angular acceleration about body x-axis (rad/s^2)
        ay  : angular acceleration about body y-axis (rad/s^2)
        az  : angular acceleration about body z-axis (rad/s^2)

    Dynamics:
        dpx/dt = vx
        dpy/dt = vy
        dpz/dt = vz
        dqw/dt = -(wx*qx + wy*qy + wz*qz) / 2
        dqx/dt = ( wx*qw + wz*qy - wy*qz) / 2
        dqy/dt = ( wy*qw - wz*qx + wx*qz) / 2
        dqz/dt = ( wz*qw + wy*qx - wx*qy) / 2
        dvx/dt = CT * (2*qw*qy + 2*qx*qz) * F / m
        dvy/dt = CT * (-2*qw*qx + 2*qy*qz) * F / m
        dvz/dt = Gz - CT * (2*qx^2 + 2*qy^2 - 1) * F / m
        dwx/dt = ax - (5/9) * wy * wz
        dwy/dt = ay + (5/9) * wx * wz
        dwz/dt = az
    """

    name = "quadrotor"
    nx = 13
    nu = 4
    periodic_states = []
    clamp_states = []

    def __init__(
        self,
        x_min: Union[list, Tensor],
        x_max: Union[list, Tensor],
        u_min: Union[list, Tensor],
        u_max: Union[list, Tensor],
        CT: float = 1.0,
        mass: float = 1.0,
        gravity: float = 9.81,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(x_min, x_max, u_min, u_max, device=device, dtype=dtype)
        if self.u_min[0] < 0:
            raise ValueError(
                f"u_min[0] (minimum thrust) must be >= 0; got {self.u_min[0].item():.4g}. "
                "A quadrotor cannot produce negative thrust."
            )
        self.CT = CT
        self.mass = mass
        self.gravity = gravity

    def f(self, x: Tensor) -> Tensor:
        """Return the open-loop drift dynamics f(x).

        Args:
            x: State tensor of shape (N, 13).

        Returns:
            Drift term of shape (N, 13, 1).
        """
        px, py, pz = x[:, 0], x[:, 1], x[:, 2]  # noqa: F841
        qw, qx, qy, qz = x[:, 3], x[:, 4], x[:, 5], x[:, 6]
        vx, vy, vz = x[:, 7], x[:, 8], x[:, 9]  # noqa: F841
        wx, wy, wz = x[:, 10], x[:, 11], x[:, 12]

        Gz = -self.gravity

        drift = torch.stack(
            [
                # position kinematics
                vx,
                vy,
                vz,
                # quaternion kinematics (no thrust)
                -(wx * qx + wy * qy + wz * qz) / 2,
                (wx * qw + wz * qy - wy * qz) / 2,
                (wy * qw - wz * qx + wx * qz) / 2,
                (wz * qw + wy * qx - wx * qy) / 2,
                # translational acceleration (gravity only; thrust goes in g)
                torch.zeros_like(vx),
                torch.zeros_like(vy),
                torch.full_like(vz, Gz),
                # angular velocity dynamics (gyroscopic coupling; inputs go in g)
                -(5.0 / 9.0) * wy * wz,
                (5.0 / 9.0) * wx * wz,
                torch.zeros_like(wz),
            ],
            dim=1,
        ).unsqueeze(
            -1
        )  # (N, 13, 1)

        return drift

    def g(self, x: Tensor) -> Tensor:
        """Return the control input matrix g(x).

        Args:
            x: State tensor of shape (N, 13).

        Returns:
            Input matrix of shape (N, 13, 4).
        """
        N = x.shape[0]
        qw, qx, qy, qz = x[:, 3], x[:, 4], x[:, 5], x[:, 6]

        # Rotation matrix columns relevant for thrust mapping
        # R * e3 components (body z-axis expressed in world frame):
        #   R_xz = 2*(qw*qy + qx*qz)
        #   R_yz = 2*(-qw*qx + qy*qz)
        #   R_zz = -(2*qx^2 + 2*qy^2 - 1)  (note: paper writes vz eqn with minus sign)
        CT_over_m = self.CT / self.mass
        R_xz = 2.0 * (qw * qy + qx * qz)
        R_yz = 2.0 * (-qw * qx + qy * qz)
        R_zz = -(2.0 * qx**2 + 2.0 * qy**2 - 1.0)

        zeros = torch.zeros(N, device=x.device, dtype=x.dtype)
        ones = torch.ones(N, device=x.device, dtype=x.dtype)

        # g is (N, 13, 4); columns correspond to [F, ax, ay, az]
        g_matrix = torch.stack(
            [
                # row for px:   [0, 0, 0, 0]
                torch.stack([zeros, zeros, zeros, zeros], dim=1),
                # row for py:   [0, 0, 0, 0]
                torch.stack([zeros, zeros, zeros, zeros], dim=1),
                # row for pz:   [0, 0, 0, 0]
                torch.stack([zeros, zeros, zeros, zeros], dim=1),
                # row for qw:   [0, 0, 0, 0]
                torch.stack([zeros, zeros, zeros, zeros], dim=1),
                # row for qx:   [0, 0, 0, 0]
                torch.stack([zeros, zeros, zeros, zeros], dim=1),
                # row for qy:   [0, 0, 0, 0]
                torch.stack([zeros, zeros, zeros, zeros], dim=1),
                # row for qz:   [0, 0, 0, 0]
                torch.stack([zeros, zeros, zeros, zeros], dim=1),
                # row for vx:   [CT/m * R_xz, 0, 0, 0]
                torch.stack([CT_over_m * R_xz, zeros, zeros, zeros], dim=1),
                # row for vy:   [CT/m * R_yz, 0, 0, 0]
                torch.stack([CT_over_m * R_yz, zeros, zeros, zeros], dim=1),
                # row for vz:   [CT/m * R_zz, 0, 0, 0]
                torch.stack([CT_over_m * R_zz, zeros, zeros, zeros], dim=1),
                # row for wx:   [0, 1, 0, 0]
                torch.stack([zeros, ones, zeros, zeros], dim=1),
                # row for wy:   [0, 0, 1, 0]
                torch.stack([zeros, zeros, ones, zeros], dim=1),
                # row for wz:   [0, 0, 0, 1]
                torch.stack([zeros, zeros, zeros, ones], dim=1),
            ],
            dim=1,
        )  # (N, 13, 4)

        return g_matrix

    # ------------------------------------------------------------------
    # Quaternion-aware overrides
    # ------------------------------------------------------------------

    def euler_step(self, x: Tensor, u: Tensor, dt: float) -> Tensor:
        """Euler step followed by quaternion re-normalization.

        Without re-normalization the quaternion drifts off the unit sphere
        under repeated integration, which corrupts the thrust-direction
        mapping in g(x).
        """
        x_next = super().euler_step(x, u, dt)
        x_next[:, 3:7] = torch.nn.functional.normalize(x_next[:, 3:7], dim=-1)
        return x_next

    def get_uniform_state_samples(
        self, num_samples: int, requires_grad: bool = False
    ) -> Tensor:
        """Sample states uniformly, drawing quaternions from the uniform
        distribution on S³ rather than a box.

        Sampling each quaternion component independently from [x_min, x_max]
        would yield off-manifold states where |q| ≠ 1 and the rotation matrix
        would be invalid.  Instead, the non-quaternion dimensions (position,
        velocity, angular velocity) are drawn from their respective boxes while
        the quaternion is drawn by normalizing a 4-D standard-normal vector,
        which gives the uniform (Haar) measure on SO(3).
        """
        # Sample all dims from the box, then overwrite the quaternion.
        samples = super().get_uniform_state_samples(num_samples, requires_grad=False)
        q = torch.randn(num_samples, 4, device=self.device, dtype=self.dtype)
        samples[:, 3:7] = torch.nn.functional.normalize(q, dim=-1)
        samples.requires_grad_(requires_grad)
        return samples

    # Indices of the 9 non-quaternion state dimensions, in state order:
    # [px, py, pz, vx, vy, vz, wx, wy, wz].
    _NON_QUAT_IDX = [0, 1, 2, 7, 8, 9, 10, 11, 12]

    def get_uniform_state_grid(
        self,
        grid_shape: Sequence[int],
        orientation: Union[list, Tensor],
        requires_grad: bool = False,
        x_min: Optional[Union[list, Tensor]] = None,
        x_max: Optional[Union[list, Tensor]] = None,
    ) -> Tensor:
        """Create a uniform grid over the 9 non-quaternion state dimensions
        with the orientation held fixed at a user-supplied unit quaternion.

        The quaternion sub-space is the 3-sphere S³, which cannot be tiled
        uniformly by a rectangular grid: independently sweeping each component
        through [x_min, x_max] would yield off-manifold states with |q| ≠ 1
        and an invalid rotation matrix in g(x).  This method therefore grids
        only the position, linear-velocity, and angular-velocity dims and
        copies the supplied orientation into every grid point, leaving the
        choice of attitude slice to the caller.

        Args:
            grid_shape: Sequence of length 9, giving the number of grid points
                along each non-quaternion dim in the order
                [px, py, pz, vx, vy, vz, wx, wy, wz].
            orientation: Unit quaternion [qw, qx, qy, qz] used at every grid
                point.  Must lie on S³ (|q| = 1).
            requires_grad: Whether the returned tensor should require gradients.
            x_min: Optional lower bounds of length 9 for the non-quaternion
                dims (same ordering as ``grid_shape``).  If omitted, the
                corresponding entries of ``self.x_min`` are used.
            x_max: Optional upper bounds of length 9 for the non-quaternion
                dims.  If omitted, the corresponding entries of ``self.x_max``
                are used.

        Returns:
            Tensor of shape (*grid_shape, 13).  Dims 3:7 are the supplied
            orientation; the remaining 9 dims sweep the requested grid.
        """
        if len(grid_shape) != 9:
            raise ValueError(
                "grid_shape must have length 9 (one entry per non-quaternion "
                f"state dim); got length {len(grid_shape)}."
            )

        q = torch.as_tensor(orientation, dtype=self.dtype, device=self.device)
        if q.shape != (4,):
            raise ValueError(
                f"orientation must be a length-4 quaternion; got shape {tuple(q.shape)}."
            )
        if not torch.isclose(
            torch.linalg.norm(q), torch.ones((), dtype=self.dtype, device=self.device),
            atol=1e-6,
        ):
            raise ValueError(
                f"orientation must be a unit quaternion (|q| = 1); got |q| = "
                f"{torch.linalg.norm(q).item():.6g}."
            )

        if x_min is None:
            lo = self.x_min[self._NON_QUAT_IDX]
        else:
            lo = torch.as_tensor(x_min, dtype=self.dtype, device=self.device)
        if x_max is None:
            hi = self.x_max[self._NON_QUAT_IDX]
        else:
            hi = torch.as_tensor(x_max, dtype=self.dtype, device=self.device)

        sub_grid = super().get_uniform_state_grid(
            grid_shape=grid_shape,
            requires_grad=False,
            x_min=lo,
            x_max=hi,
        )  # (*grid_shape, 9)

        grid = torch.empty(
            (*grid_shape, self.nx), dtype=self.dtype, device=self.device
        )
        grid[..., self._NON_QUAT_IDX] = sub_grid
        grid[..., 3:7] = q
        grid.requires_grad_(requires_grad)
        return grid
