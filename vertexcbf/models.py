"""Neural network model definitions for VertexCBF.

This module provides the core MLP architecture used to approximate
Control Barrier Functions (CBFs). The network supports:

- Optional per-dimension input normalisation to [-1, 1]
- Sinusoidal (sin/cos) encoding for periodic state dimensions (e.g. angles),
  with the raw value first rescaled to [-π, π] before encoding
- Configurable hidden layers with arbitrary activation functions
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------


class Sin(nn.Module):
    """Element-wise sine activation: ``out = sin(x)``."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "sin": Sin,
}


def get_activation(spec: str | tuple[str, dict]) -> nn.Module:
    """Return a fresh activation module instance by name.

    Args:
        spec: Either a plain name string, or a ``(name, kwargs)`` tuple to
              pass keyword arguments to the constructor.  For example::

                  get_activation("relu")
                  get_activation(("elu", {"alpha": 0.5}))
                  get_activation(("softplus", {"beta": 10}))

              Recognised names: ``"linear"``, ``"relu"``, ``"elu"``,
              ``"selu"``, ``"softplus"``, ``"sigmoid"``, ``"tanh"``,
              ``"sin"``.

    Returns:
        An instantiated :class:`~torch.nn.Module` for the requested activation.

    Raises:
        KeyError: If the name is not a recognised activation.
    """
    if isinstance(spec, tuple):
        name, kwargs = spec
    else:
        name, kwargs = spec, {}
    if name not in _ACTIVATIONS:
        raise KeyError(
            f"Unknown activation '{name}'. " f"Available: {sorted(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[name](**kwargs)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Multi-Layer Perceptron with structured input preprocessing.

    Before the first linear layer the raw state vector is transformed as
    follows:

    * **Periodic inputs** (e.g. angles) are first rescaled from
      ``[input_min, input_max]`` to ``[-π, π]`` and then replaced by their
      ``[cos(x), sin(x)]`` pair, making the representation invariant to
      angle wrap-around.
    * **Non-periodic inputs** are optionally linearly rescaled from
      ``[input_min, input_max]`` to ``[-1, 1]`` when ``rescale_inputs=True``
      (default).  Set ``rescale_inputs=False`` to pass them through unchanged.

    Args:
        layers_config: Sequence of ``(size, activation)`` tuples describing
            the network structure.  The **first** tuple must be
            ``(input_dim, None)`` where *input_dim* is the number of raw input features
            (the activation entry is ignored).  Each subsequent tuple defines
            one layer: *size* is the number of output features and
            *activation* is either a plain name string (e.g. ``"relu"``) or a
            ``(name, kwargs)`` tuple (e.g. ``("softplus", {"beta": 10.0})``).
            Example::

                layers_config = [
                    (2,  None),                          # input: input_dim=2
                    (64, "tanh"),
                    (64, "tanh"),
                    (1,  ("softplus", {"beta": 10.0})),  # output
                ]

        input_min: 1-D tensor of per-dimension lower bounds used for
            normalisation, shape ``(input_dim,)``.
        input_max: 1-D tensor of per-dimension upper bounds used for
            normalisation, shape ``(input_dim,)``.
        periodic_inputs: Indices of state dimensions that are periodic.
            These are rescaled to ``[-π, π]`` then encoded as ``[cos, sin]``
            pairs instead of being linearly scaled.  Defaults to ``[]``.
        rescale_inputs: Whether to rescale non-periodic states to ``[-1, 1]``.
            Defaults to ``True``.

    Example::

        model = MLP(
            layers_config=[
                (2,  None),
                (64, "tanh"),
                (64, "tanh"),
                (1,  "linear"),
            ],
            input_min=torch.tensor([-3.14, -5.0]),
            input_max=torch.tensor([ 3.14,  5.0]),
            periodic_inputs=[0],
        )
        y = model(torch.zeros(8, 2))  # batch of 8 states -> (8, 1)
    """

    def __init__(
        self,
        layers_config: list[tuple[int, str | tuple[str, dict] | None]],
        input_min: Tensor,
        input_max: Tensor,
        periodic_inputs: list[int] = [],
        rescale_inputs: bool = True,
    ) -> None:
        super().__init__()

        input_dim: int = layers_config[0][0]
        self.rescale: bool = rescale_inputs

        # Normalisation bounds — registered as buffers so they follow the
        # model to whichever device/dtype it is cast to.
        self.register_buffer(
            "input_min",
            torch.as_tensor(input_min, dtype=torch.float32),
        )
        self.register_buffer(
            "input_max",
            torch.as_tensor(input_max, dtype=torch.float32),
        )

        # Boolean mask: True for periodic states.
        periodic_mask = torch.zeros(input_dim, dtype=torch.bool)
        if periodic_inputs:
            periodic_mask[list(periodic_inputs)] = True
        self.register_buffer("periodic_mask", periodic_mask)

        # After preprocessing, each periodic state becomes *two* features
        # (cos + sin) while non-periodic states remain one feature each.
        n_periodic = int(periodic_mask.sum().item())
        preprocessed_size = input_dim + n_periodic  # each periodic dim gains 1 extra

        # Build the sequential network from layers_config[1:] (skip input spec).
        hidden_layers = layers_config[1:]
        layers: list[nn.Module] = []
        for i, (out_features, act_name) in enumerate(hidden_layers):
            in_features = preprocessed_size if i == 0 else hidden_layers[i - 1][0]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(get_activation(act_name))

        self.net = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, input: Tensor) -> Tensor:
        """Evaluate the network on a batch of states.

        Args:
            input: Raw state tensor of shape ``(batch, input_dim)``.

        Returns:
            Network output of shape ``(batch, output_size)``.
        """
        return self.net(self._preprocess(input))

    # ------------------------------------------------------------------
    # Input preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, input: Tensor) -> Tensor:
        """Normalise/encode raw states into the network's input space.

        Processing order:

        1. Periodic states are rescaled from ``[input_min, input_max]`` to
           ``[-π, π]``, then replaced by ``[cos, sin]`` pairs.
        2. Non-periodic states are rescaled to ``[-1, 1]`` when
           ``self.rescale`` is ``True``, otherwise passed through as-is.

        The two groups are concatenated in that order: periodic features
        first, then non-periodic features.

        Args:
            input: Raw state tensor of shape ``(batch, input_dim)``.

        Returns:
            Preprocessed tensor of shape ``(batch, input_dim + n_periodic)``.
        """
        parts: list[Tensor] = []

        if self.periodic_mask.any():
            input_per = input[:, self.periodic_mask]
            lo_per = self.input_min[self.periodic_mask]
            hi_per = self.input_max[self.periodic_mask]
            # Rescale to [-π, π] before encoding.
            input_per_norm = (
                2.0 * torch.pi * (input_per - lo_per) / (hi_per - lo_per) - torch.pi
            )
            parts.append(torch.cos(input_per_norm))
            parts.append(torch.sin(input_per_norm))

        input_non = input[:, ~self.periodic_mask]
        if self.rescale:
            lo = self.input_min[~self.periodic_mask]
            hi = self.input_max[~self.periodic_mask]
            input_non = 2.0 * (input_non - lo) / (hi - lo) - 1.0
        parts.append(input_non)

        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        periodic_idx = self.periodic_mask.nonzero(as_tuple=True)[0].tolist()
        return (
            f"{self.__class__.__name__}("
            f"periodic_states={periodic_idx}, "
            f"rescale={self.rescale}, "
            f"params={n_params:,}, "
            f"net={self.net})"
        )
