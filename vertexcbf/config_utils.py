"""Utilities for building VertexCBF components from config dictionaries.

Typical usage inside a training script::

    import yaml
    from vertexcbf.config_utils import build_dynamics, build_constr_fn, build_model

    with open("configs/double_integrator_1d.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dynamics  = build_dynamics(cfg["system"], device=device)
    constr_fn = build_constr_fn(cfg["constraint"])
    model     = build_model(cfg["model"], dynamics, device=device)
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from vertexcbf.dynamics import (
    AUV6DoF,
    CartPole,
    ControlAffine,
    DoubleIntegrator1D,
    DoubleIntegrator2D,
    DoubleIntegrator3D,
    DubinsCar,
    DynamicUnicycle,
    InvertedPendulum,
    KinematicBicycle,
    LandingRocket,
    Manipulator3DOF,
    Quadrotor,
    QuadrupedTrunk,
    RelativeUnicycle,
    VerticalDrone2D,
)
from vertexcbf.models import MLP
from vertexcbf.mpc import (
    beam_search,
    stochastic_beam_search,
    random_shooting,
    mppi,
    cem,
    cem_discrete,
    icem,
    branch_and_bound,
)
from vertexcbf.constraint import (
    ball_3d_sdf,
    circle_sdf,
    cylinder_sdf,
    ee_sphere_sdf,
    interval_sdf,
    landing_funnel_sdf,
    manipulator_sphere_sdf,
    rectangle_sdf,
    state_limits_sdf,
    two_disk_sdf,
)

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

DYNAMICS_REGISTRY: dict[str, type[ControlAffine]] = {
    "AUV6DoF": AUV6DoF,
    "CartPole": CartPole,
    "DoubleIntegrator1D": DoubleIntegrator1D,
    "DoubleIntegrator2D": DoubleIntegrator2D,
    "DoubleIntegrator3D": DoubleIntegrator3D,
    "DubinsCar": DubinsCar,
    "DynamicUnicycle": DynamicUnicycle,
    "InvertedPendulum": InvertedPendulum,
    "KinematicBicycle": KinematicBicycle,
    "LandingRocket": LandingRocket,
    "Manipulator3DOF": Manipulator3DOF,
    "Quadrotor": Quadrotor,
    "QuadrupedTrunk": QuadrupedTrunk,
    "RelativeUnicycle": RelativeUnicycle,
    "VerticalDrone2D": VerticalDrone2D,
}

CONSTR_REGISTRY: dict[str, Callable] = {
    "interval_sdf": interval_sdf,
    "circle_sdf": circle_sdf,
    "rectangle_sdf": rectangle_sdf,
    "state_limits_sdf": state_limits_sdf,
    "cylinder_sdf": cylinder_sdf,
    "ball_3d_sdf": ball_3d_sdf,
    "landing_funnel_sdf": landing_funnel_sdf,
    "ee_sphere_sdf": ee_sphere_sdf,
    "manipulator_sphere_sdf": manipulator_sphere_sdf,
    "two_disk_sdf": two_disk_sdf,
}

MPC_REGISTRY: dict[str, Callable] = {
    "beam_search": beam_search,
    "stochastic_beam_search": stochastic_beam_search,
    "random_shooting": random_shooting,
    "mppi": mppi,
    "cem": cem,
    "cem_discrete": cem_discrete,
    "icem": icem,
    "branch_and_bound": branch_and_bound,
}

# Acronyms used in the paper to group results by data-generation method:
#   NO_DATA — PDE/HJB residual loss only (no supervision targets).
#   FC_DATA — "full-control" data: continuous-control sampling MPC
#             (mppi, cem, icem, random_shooting).
#   VRC_DATA — "vertex-restricted control" data: discrete / control-vertex
#              search (beam_search, stochastic_beam_search, branch_and_bound,
#              cem_discrete).
METHOD_GROUPS: tuple[str, ...] = ("NO_DATA", "FC_DATA", "VRC_DATA")

_FC_METHODS: frozenset[str] = frozenset(
    {"mppi", "cem", "icem", "random_shooting"}
)
_VRC_METHODS: frozenset[str] = frozenset(
    {"beam_search", "stochastic_beam_search", "branch_and_bound", "cem_discrete"}
)


def method_group(data_cfg: dict | None, no_data_override: bool = False) -> str:
    """Return the paper-facing group (NO_DATA / FC_DATA / VRC_DATA) for a config.

    The group determines where artifacts are stored (checkpoints, precomputed
    supervision data, figures) so the three comparison conditions stay cleanly
    separated on disk.
    """
    data_cfg = data_cfg or {}
    if no_data_override or not data_cfg.get("enabled", True):
        return "NO_DATA"
    method = data_cfg.get("method", "beam_search")
    if method in _FC_METHODS:
        return "FC_DATA"
    if method in _VRC_METHODS:
        return "VRC_DATA"
    raise ValueError(
        f"Cannot map MPC method '{method}' to a paper group. "
        f"Expected one of {sorted(_FC_METHODS | _VRC_METHODS)}."
    )


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


_BEAM_METHODS = {"beam_search", "stochastic_beam_search", "branch_and_bound"}


def build_mpc_runner(
    data_cfg: dict,
    dynamics: "ControlAffine",
    constr_fn: Callable,
) -> Callable:
    """Build a callable that runs the configured MPC method on a batch of states.

    The ``data`` config section selects which sampling-based MPC method is used
    to compute supervision targets.  ``B`` is the unified budget parameter
    across all methods: beam width for beam-search methods, number of sampled
    trajectories for the others (passed internally as ``N_s``).
    Method-specific hyperparameters live under the optional ``method_params``
    sub-dict.

    Supported methods and their extra ``method_params`` keys:

    * ``beam_search``            — *(no extra params)*
    * ``stochastic_beam_search`` — ``strategy``, ``temperature``, ``epsilon``
    * ``random_shooting``        — *(no extra params)*
    * ``mppi``                   — ``sigma``, ``lam``, ``n_iter``
    * ``cem``                    — ``n_iter``, ``elite_frac``
    * ``cem_discrete``           — ``n_iter``, ``elite_frac``
    * ``icem``                   — ``n_iter``, ``elite_frac``, ``noise_beta``
    * ``branch_and_bound``       — ``n_restarts``, ``tie_noise``

    Args:
        data_cfg: The ``data`` section of a config dict.  Must contain ``B``,
            ``K``, and ``dt``.  Optional keys: ``method`` (default
            ``"beam_search"``), ``method_params`` (default ``{}``).
        dynamics:  ControlAffine instance (used only by the returned callable).
        constr_fn: Constraint function ``(M, nx) -> (M,)`` or ``(M, 1)``.

    Returns:
        A callable ``run(states) -> Tensor`` that returns the ``"values"``
        tensor (shape ``(N,)``) from the selected MPC method.

    Raises:
        KeyError: If ``method`` is not in :data:`MPC_REGISTRY`.
    """
    method_name = data_cfg.get("method", "beam_search")
    if method_name not in MPC_REGISTRY:
        raise KeyError(
            f"Unknown MPC method '{method_name}'. " f"Available: {sorted(MPC_REGISTRY)}"
        )
    fn = MPC_REGISTRY[method_name]
    B = data_cfg["B"]
    K = data_cfg["K"]
    dt = data_cfg["dt"]
    extra = data_cfg.get("method_params", {}) or {}

    # Beam-family methods take ``B`` (beam width); shooting methods take ``N_s``
    # (sample count).  The config exposes a single budget field ``B`` that gets
    # routed to whichever kwarg the chosen method expects.
    budget_kwarg = "B" if method_name in _BEAM_METHODS else "N_s"

    def run(states):
        result = fn(
            dynamics=dynamics,
            x0=states,
            **{budget_kwarg: B},
            K=K,
            dt=dt,
            constr_fn=constr_fn,
            **extra,
        )
        return result["values"]

    return run


def build_dynamics(
    system_cfg: dict,
    device: torch.device | None = None,
) -> ControlAffine:
    """Instantiate a :class:`~vertexcbf.dynamics.ControlAffine` from config.

    Args:
        system_cfg: The ``system`` section of a config dict.  Must contain:

            * ``name``  — class name (key in :data:`DYNAMICS_REGISTRY`)
            * ``x_min``, ``x_max`` — state bounds (lists of floats)
            * ``u_min``, ``u_max`` — control bounds (lists of floats)
            * ``params`` — optional dict of system-specific keyword args
              (e.g. ``{m: 2.0, l: 1.0}`` for :class:`InvertedPendulum`)

        device: Target device.

    Returns:
        Instantiated dynamics object.

    Raises:
        KeyError: If ``name`` is not in :data:`DYNAMICS_REGISTRY`.
    """
    name = system_cfg["name"]
    if name not in DYNAMICS_REGISTRY:
        raise KeyError(
            f"Unknown system '{name}'. Available: {sorted(DYNAMICS_REGISTRY)}"
        )
    cls = DYNAMICS_REGISTRY[name]
    params = system_cfg.get("params", {}) or {}
    return cls(
        x_min=system_cfg["x_min"],
        x_max=system_cfg["x_max"],
        u_min=system_cfg["u_min"],
        u_max=system_cfg["u_max"],
        device=device,
        **params,
    )


def build_constr_fn(
    constraint_cfg: dict,
) -> Callable[[Tensor], Tensor]:
    """Build the constraint function ``c(x)`` callable from config.

    Args:
        constraint_cfg: The ``constraint`` section of a config dict.  Must
            contain:

            * ``type``   — constraint function name (key in :data:`CONSTR_REGISTRY`)
            * ``params`` — keyword args forwarded to the function

            List values in ``params`` are automatically converted to
            ``torch.Tensor`` (e.g. ``center: [0.0, 0.0]``).

    Returns:
        A callable ``c(x)`` with signature ``(N, nx) -> (N, 1)``.

    Raises:
        KeyError: If ``type`` is not in :data:`CONSTR_REGISTRY`.
    """
    constr_type = constraint_cfg["type"]
    if constr_type not in CONSTR_REGISTRY:
        raise KeyError(
            f"Unknown constraint '{constr_type}'. Available: {sorted(CONSTR_REGISTRY)}"
        )
    fn = CONSTR_REGISTRY[constr_type]
    raw_params = constraint_cfg.get("params", {}) or {}

    # Convert list values to tensors (e.g. center: [0.0, 0.0])
    params = {
        k: torch.tensor(v, dtype=torch.float32) if isinstance(v, list) else v
        for k, v in raw_params.items()
    }
    return partial(fn, **params)


def build_model(
    model_cfg: dict,
    dynamics: ControlAffine,
    device: torch.device | None = None,
) -> MLP:
    """Build an :class:`~vertexcbf.models.MLP` from config.

    Args:
        model_cfg: The ``model`` section of a config dict.  Must contain:

            * ``layers`` — list of ``[size, activation]`` pairs matching the
              ``layers_config`` argument of :class:`~vertexcbf.models.MLP`.
              Activations can be a plain string or a two-element list
              ``[name, {kwargs}]``, e.g. ``[softplus, {beta: 10.0}]``.
            * ``rescale_inputs`` — optional bool (default ``True``)

        dynamics: Provides ``x_min``, ``x_max``, and ``periodic_states``
            used to configure input preprocessing.
        device: Target device.

    Returns:
        Instantiated and device-placed :class:`~vertexcbf.models.MLP`.
    """

    def _parse_activation(act):
        if act is None:
            return None
        if isinstance(act, list):
            name, kwargs = act[0], act[1] if len(act) > 1 else {}
            return (name, kwargs)
        return act  # plain string

    layers_config = [
        (int(size), _parse_activation(act)) for size, act in model_cfg["layers"]
    ]

    rescale = model_cfg.get("rescale_inputs", True)

    model = MLP(
        layers_config=layers_config,
        input_min=dynamics.x_min,
        input_max=dynamics.x_max,
        periodic_inputs=dynamics.periodic_states,
        rescale_inputs=rescale,
    )
    if device is not None:
        model = model.to(device)
    return model
