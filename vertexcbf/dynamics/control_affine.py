from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

import torch
from torch import Tensor


class ControlAffine(ABC):
    """Abstract base class for control-affine dynamical systems.

    A control-affine system has the form:
        dx/dt = f(x) + g(x) * u

    where f(x) is the drift dynamics and g(x) is the input matrix.
    """

    name: str  # snake_case identifier used in file paths
    nx: int  # number of state dimensions
    nu: int  # number of control input dimensions
    periodic_states: list  # indices of state dimensions that are periodic
    clamp_states: list = []  # indices of state dimensions to clamp to [x_min, x_max] each euler step

    def __init__(
        self,
        x_min: Union[list, Tensor],
        x_max: Union[list, Tensor],
        u_min: Union[list, Tensor],
        u_max: Union[list, Tensor],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        if len(x_min) != self.nx or len(x_max) != self.nx:
            raise ValueError(f"x_min and x_max must have length {self.nx}.")
        if len(u_min) != self.nu or len(u_max) != self.nu:
            raise ValueError(f"u_min and u_max must have length {self.nu}.")

        self.x_min = torch.as_tensor(x_min, dtype=dtype, device=device)
        self.x_max = torch.as_tensor(x_max, dtype=dtype, device=device)
        self.u_min = torch.as_tensor(u_min, dtype=dtype, device=device)
        self.u_max = torch.as_tensor(u_max, dtype=dtype, device=device)
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def f(self, x: Tensor) -> Tensor:
        """Return the open-loop drift dynamics f(x).

        Args:
            x: State tensor of shape (N, nx).

        Returns:
            Drift term of shape (N, nx, 1).
        """

    @abstractmethod
    def g(self, x: Tensor) -> Tensor:
        """Return the control input matrix g(x).

        Args:
            x: State tensor of shape (N, nx).

        Returns:
            Input matrix of shape (N, nx, nu) or broadcastable equivalent.
        """

    def xdot(self, x: Tensor, u: Tensor) -> Tensor:
        """Compute the state derivative dx/dt = f(x) + g(x) * u.

        Args:
            x: State tensor of shape (N, nx).
            u: Control input tensor of shape (N, nu).

        Returns:
            State derivative tensor of shape (N, nx).
        """
        f = self.f(x)  # (N, nx, 1)
        g = self.g(x)  # (N, nx, nu)
        u_ = u.unsqueeze(-1)  # (N, nu, 1)
        return (f + g @ u_).squeeze(-1)  # (N, nx)

    def euler_step(self, x: Tensor, u: Tensor, dt: float) -> Tensor:
        """Advance the state one step using forward Euler integration.

        x_next = x + xdot(x, u) * dt

        Args:
            x: State tensor of shape (N, nx).
            u: Control input tensor of shape (N, nu).
            dt: Time step size.

        Returns:
            Next state tensor of shape (N, nx).
        """
        x_next = x + self.xdot(x, u) * dt

        # Clamp selected state dimensions to [x_min, x_max] before periodic wrapping.
        if self.clamp_states:
            lo = self.x_min[self.clamp_states]
            hi = self.x_max[self.clamp_states]
            x_next[:, self.clamp_states] = torch.clamp(
                x_next[:, self.clamp_states], min=lo, max=hi
            )

        # Wrap periodic state dimensions properly after integration.
        if self.periodic_states:
            lo = self.x_min[self.periodic_states]
            hi = self.x_max[self.periodic_states]
            x_next[:, self.periodic_states] = (x_next[:, self.periodic_states] - lo) % (
                hi - lo
            ) + lo
        return x_next

    def get_control_vertices(self) -> Tensor:
        """Return the vertices of the control box defined by u_min and u_max.

        Returns:
            Tensor of shape (nu, 2**nu), where each column is a vertex of the
            control input box [u_min, u_max].
        """
        nu = self.u_min.shape[0]
        indices = torch.arange(2**nu, device=self.device)
        bits = (
            indices.unsqueeze(1) >> torch.arange(nu, device=self.device).unsqueeze(0)
        ) & 1
        vertices = torch.where(
            bits.bool(), self.u_max.unsqueeze(0), self.u_min.unsqueeze(0)
        )
        return vertices.T

    def get_xdot_vertices(self, x: Tensor) -> Tensor:
        """Evaluate f(x) + g(x)*u at every vertex of the control box.

        Computes the full set of reachable state-derivative directions by
        enumerating all 2**nu extreme control inputs.  Useful for Lie-
        derivative bounds and set-membership CBF conditions.

        Args:
            x: State tensor of shape (N, nx).

        Returns:
            Tensor of shape (N, nx, 2**nu), where slice [..., k] is the
            state derivative f(x) + g(x)*u_k for the k-th control vertex.
        """
        f = self.f(x)  # (N, nx, 1)
        g = self.g(x)  # (N, nx, nu)
        u_vertices = self.get_control_vertices().to(
            dtype=self.dtype, device=self.device
        )  # (nu, 2**nu)
        return f + g @ u_vertices  # (N, nx, 2**nu)

    def get_hamiltonian(self, x: Tensor, grad_values: Tensor):
        """Compute the control-affine Hamiltonian and optimal control vertex.

        H(x, grad_V) = max_u { grad_V @ (f(x) + g(x)*u) }

        Maximized over the 2**nu vertices of the control box.

        Args:
            x: State tensor of shape (N, nx).
            grad_values: Gradient of the value function w.r.t. x, shape (N, nx).

        Returns:
            H: Hamiltonian values of shape (N,).
            optimal_u: Optimal control vertex of shape (N, nu).
        """
        xdot_vertices = self.get_xdot_vertices(x)  # (N, nx, 2**nu)
        lie_derivatives = torch.bmm(
            grad_values.unsqueeze(1), xdot_vertices
        )  # (N, 1, 2**nu)
        hamiltonian, optimal_idx = torch.max(lie_derivatives, dim=-1)  # (N, 1)
        hamiltonian = hamiltonian.squeeze(1)  # (N,)
        optimal_idx = optimal_idx.squeeze(1)  # (N,)
        u_vertices = self.get_control_vertices()  # (nu, 2**nu)
        optimal_u = u_vertices[:, optimal_idx].T  # (N, nu)
        return hamiltonian, optimal_u

    def get_uniform_state_grid(
        self,
        grid_shape: Sequence[int],
        requires_grad: bool = False,
        x_min: Optional[Union[list, Tensor]] = None,
        x_max: Optional[Union[list, Tensor]] = None,
    ) -> Tensor:
        """Create a uniform grid over the n-dimensional state space.

        Args:
            grid_shape: Sequence of length nx, where grid_shape[i] is the number
                of grid points along dimension i.
            requires_grad: Whether the returned tensor should require gradients.
            x_min: Optional lower bounds of length nx for the grid. If omitted,
                the system's global ``self.x_min`` is used.
            x_max: Optional upper bounds of length nx for the grid. If omitted,
                the system's global ``self.x_max`` is used.

        Returns:
            Tensor of shape (*grid_shape, nx), where each entry along the last
            dimension is one state coordinate vector.
        """
        lo = self.x_min if x_min is None else torch.as_tensor(
            x_min, dtype=self.dtype, device=self.device
        )
        hi = self.x_max if x_max is None else torch.as_tensor(
            x_max, dtype=self.dtype, device=self.device
        )

        if lo.shape[0] != len(grid_shape) or hi.shape[0] != len(grid_shape):
            raise ValueError("x_min, x_max, and grid_shape must have the same length.")
        if any(p <= 0 for p in grid_shape):
            raise ValueError("All entries in grid_shape must be positive integers.")
        if torch.any(hi < lo):
            raise ValueError("Each x_max must be greater than or equal to x_min.")

        axes = [
            torch.linspace(
                lo[i],
                hi[i],
                steps=grid_shape[i],
                device=self.device,
                dtype=self.dtype,
            )
            for i in range(len(grid_shape))
        ]

        meshes = torch.meshgrid(*axes, indexing="ij")
        grid = torch.stack(meshes, dim=-1)
        grid.requires_grad_(requires_grad)
        return grid

    def get_uniform_state_samples(
        self,
        num_samples: int,
        requires_grad: bool = False,
    ) -> Tensor:
        """Sample N points uniformly from the n-dimensional state space box.

        Args:
            num_samples: Number of samples.
            requires_grad: Whether the returned tensor should require gradients.

        Returns:
            Tensor of shape (num_samples, nx), where each row is one sampled state.
        """
        if not isinstance(num_samples, int):
            raise TypeError("num_samples must be an integer.")
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
        if torch.any(self.x_max < self.x_min):
            raise ValueError("Each x_max must be greater than or equal to x_min.")

        dim = self.x_min.shape[0]
        uniform = torch.rand(num_samples, dim, device=self.device, dtype=self.dtype)
        samples = self.x_min + (self.x_max - self.x_min) * uniform
        samples.requires_grad_(requires_grad)
        return samples
