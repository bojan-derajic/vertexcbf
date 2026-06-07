"""Closed-loop validation of a learned Neural CBF.

The Neural CBF is parameterised as ``V(x) = c(x) - r_Θ(x)`` where ``c(x)`` is
the constraint function and ``r_Θ(x)`` is the learned residual network.

Given an initial state ``x_0``, the associated greedy policy is

    u*(x) = argmax_u { ∇V(x) · (f(x) + g(x) u) }  s.t. u ∈ [u_min, u_max]

Because the objective is linear in ``u``, the maximum is attained at a vertex
of the control box, so the policy reduces to picking the best of the ``2**nu``
extreme controls.  :func:`validate_cbf` rolls the closed-loop dynamics forward
with this policy and compares the **predicted** CBF value at ``x_0``,
``V(x_0)``, to the **true** minimum constraint value encountered along the
trajectory, ``min_k c(x_k)``.

Metrics are reported **per predicted-class stratum** rather than as a single
joint confusion matrix:

* the *safety-critical* question — when the CBF says "safe", how often is
  that actually true? — is the ``precision`` within the predicted-safe
  stratum ``{V(x_0) > 0}``; the complement is the **false-safe rate**;
* the *conservativeness* question — when the CBF says "unsafe", how often
  is that correct? — is the ``precision`` within the predicted-unsafe
  stratum ``{V(x_0) ≤ 0}``; the complement is the **false-unsafe rate**.

A joint confusion matrix over uniformly sampled states confounds model
quality with class imbalance: if truly-unsafe initial states are rare in the
sampling distribution the joint ``false_safe`` percentage looks small even
for a useless CBF.  Conditional rates within each stratum are invariant to
that base-rate effect.

For tight estimates of both rates regardless of the prior over the state
space, use :func:`stratified_sample_by_predicted_cbf` to rejection-sample
equal numbers of predicted-safe and predicted-unsafe initial states before
calling :func:`validate_cbf`.
"""

from __future__ import annotations

import warnings
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine
from vertexcbf.trajopt._utils import _auto_batch_size, run_in_batches, _warmup_cuda


def _peak_bytes_validate(
    n_steps: int,
    nx: int,
    nu: int,
    residual_fn: nn.Module,
    return_state_trajs: bool,
    return_control_trajs: bool,
    dtype_bytes: int,
) -> int:
    """Estimated peak GPU bytes per initial state for :func:`validate_cbf`.

    Accounts for (i) per-step autograd activations through ``residual_fn``
    needed to differentiate ``V`` w.r.t. ``x``, (ii) the ``f(x) + g(x) u_k``
    vertex buffer of shape ``(nx, 2**nu)``, and (iii) the persistent
    trajectory buffers when ``return_state_trajs`` / ``return_control_trajs``
    is enabled. A ×4 overhead covers PyTorch temporaries.
    """
    activation_elems = sum(
        m.out_features for m in residual_fn.modules() if isinstance(m, nn.Linear)
    )
    # Clip 2**nu to avoid overflow for large nu (sanity only; typically small).
    vertices = 1 << min(nu, 10)
    per_step = 2 * nx + activation_elems + nx * vertices + 4
    base = per_step
    if return_state_trajs:
        base += (n_steps + 1) * nx
    if return_control_trajs:
        base += n_steps * nu
    return 4 * base * dtype_bytes


def _validate_cbf_rollout(
    dynamics: ControlAffine,
    states: Tensor,
    constr_fn: Callable[[Tensor], Tensor],
    residual_fn: nn.Module,
    T: float,
    dt: float,
    return_state_trajs: bool,
    return_control_trajs: bool,
) -> dict:
    """Single-chunk closed-loop rollout under the Hamiltonian-greedy policy.

    Always returns ``"values"`` (``min_k c(x_k)``) and ``"initial_cbf"``
    (``V(x_0)``) so the caller can aggregate per-stratum metrics across
    chunks. Trajectory tensors are returned only when requested.
    """
    N, nx = states.shape
    nu = dynamics.nu
    n_steps = int(round(T / dt))

    was_training = residual_fn.training
    residual_fn.eval()

    try:
        x = states.detach()

        with torch.no_grad():
            running_min = constr_fn(x).reshape(N).clone()

        if return_state_trajs:
            state_traj = torch.empty(
                N, n_steps + 1, nx, device=x.device, dtype=x.dtype
            )
            state_traj[:, 0, :] = x
        if return_control_trajs:
            control_traj = torch.empty(
                N, n_steps, nu, device=x.device, dtype=x.dtype
            )

        initial_cbf: Tensor | None = None

        for k in range(n_steps):
            # Compute ∇V(x) = ∇(c(x) - r_Θ(x)) w.r.t. x.
            x_req = x.detach().requires_grad_(True)
            with torch.enable_grad():
                cbf = constr_fn(x_req) - residual_fn(x_req)  # (N, 1)
                if k == 0:
                    initial_cbf = cbf.detach().reshape(N).clone()
                grad_V = torch.autograd.grad(
                    outputs=cbf.sum(),
                    inputs=x_req,
                    create_graph=False,
                )[0]  # (N, nx)

            with torch.no_grad():
                # Argmax over control vertices of the Hamiltonian.
                _, u_opt = dynamics.get_hamiltonian(x_req.detach(), grad_V)
                x = dynamics.euler_step(x_req.detach(), u_opt, dt)

                if return_control_trajs:
                    control_traj[:, k, :] = u_opt
                if return_state_trajs:
                    state_traj[:, k + 1, :] = x

                running_min = torch.minimum(
                    running_min, constr_fn(x).reshape(N)
                )
    finally:
        if was_training:
            residual_fn.train()

    out: dict[str, Tensor] = {
        "values": running_min,
        "initial_cbf": initial_cbf,  # type: ignore[dict-item]
    }
    if return_state_trajs:
        out["state_traj"] = state_traj
    if return_control_trajs:
        out["control_traj"] = control_traj
    return out


def _stratified_metrics(initial_cbf: Tensor, values: Tensor) -> dict:
    """Per-predicted-class agreement rates as percentages.

    Each predicted class — ``predicted_safe`` (``V(x_0) > 0``) and
    ``predicted_unsafe`` (``V(x_0) ≤ 0``) — is treated as its own stratum
    and a precision is computed inside it.  The two sub-dicts are
    independent populations and their percentages do **not** sum to 100;
    rather, each ``precision`` is the fraction *within that stratum* whose
    true label matches the predicted class.  When stratified sampling is
    used, both ``n`` values are roughly equal and both precisions have
    similar variance regardless of the underlying base rate.

    Returns:
        Dict with sub-dicts ``"predicted_safe"`` and ``"predicted_unsafe"``.
        Each sub-dict has:

        * ``n`` — count of samples in the stratum.
        * ``precision`` — % of samples in the stratum whose true label
          agrees with the predicted class.
        * ``false_safe_rate`` (on ``predicted_safe`` only) —
          ``100 - precision``: how often a "safe" certificate is wrong.
        * ``false_unsafe_rate`` (on ``predicted_unsafe`` only) —
          ``100 - precision``: how often a "unsafe" label was conservative.

        When a stratum is empty the percentages are reported as ``NaN``.
    """
    predicted_safe = initial_cbf > 0
    truly_safe = values > 0

    n_ps = int(predicted_safe.sum().item())
    n_pu = int((~predicted_safe).sum().item())
    ps_correct = int((predicted_safe & truly_safe).sum().item())
    pu_correct = int((~predicted_safe & ~truly_safe).sum().item())

    ps_precision = 100.0 * ps_correct / n_ps if n_ps > 0 else float("nan")
    pu_precision = 100.0 * pu_correct / n_pu if n_pu > 0 else float("nan")

    return {
        "predicted_safe": {
            "n": n_ps,
            "precision": ps_precision,
            "false_safe_rate": (100.0 - ps_precision) if n_ps > 0 else float("nan"),
        },
        "predicted_unsafe": {
            "n": n_pu,
            "precision": pu_precision,
            "false_unsafe_rate": (100.0 - pu_precision) if n_pu > 0 else float("nan"),
        },
    }


def _volume_metrics(
    initial_cbf: Tensor,
    values: Tensor,
    *,
    predicted_safe_prior: float | None = None,
) -> dict:
    """Estimate volume ratios of the predicted and validated safe sets.

    Both quantities are reported as fractions of the state-space bounding
    box ``[x_min, x_max]``:

    * ``predicted_safe_vol_ratio = Vol{x : V(x) > 0} / Vol(state_box)`` —
      what the certificate claims is safe;
    * ``validated_safe_vol_ratio = Vol{x : trajectory from x under the
      greedy policy stays in {c > 0}} / Vol(state_box)`` — what the
      closed-loop rollout actually confirms.

    Estimation strategy depends on how ``states`` were drawn:

    * **Uniform sampling** (``predicted_safe_prior is None``).  When the
      input states are themselves uniform over the state box (``sampling``
      = ``random`` or ``grid``), both volumes are direct sample means:
      ``(initial_cbf > 0).float().mean()`` and
      ``(values > 0).float().mean()``.

    * **Stratified sampling** (``predicted_safe_prior`` supplied).  The
      input states are *not* uniform — they are balanced across the two
      predicted-class strata — so direct counts are biased.  Instead the
      prior ``p = Vol{V > 0} / Vol(state_box)`` is supplied by the caller
      (typically ``total_predicted_safe / attempts`` from
      :func:`stratified_sample_by_predicted_cbf`), and the validated
      volume is reweighted from the per-stratum rollout outcomes:

          Pr[traj safe] = p · Pr[traj safe | V > 0]
                        + (1 - p) · Pr[traj safe | V ≤ 0]

      where the two conditional probabilities are the within-stratum
      fractions of ``values > 0`` observed during the rollout.

    Args:
        initial_cbf:          ``(N,)`` predicted ``V(x_0)`` for the rolled-out
                              initial states.
        values:               ``(N,)`` per-trajectory ``min_k c(x_k)``.
        predicted_safe_prior: Externally estimated ``Vol{V > 0}/Vol(state_box)``.
                              Pass ``None`` only when ``states`` are uniform
                              draws from the state box.

    Returns:
        Dict with ``predicted_safe_vol_ratio`` and
        ``validated_safe_vol_ratio`` (both Python floats in ``[0, 1]``),
        plus an ``estimation`` tag (``"uniform"`` or ``"stratified_prior"``)
        recording which path was used.
    """
    predicted_safe = initial_cbf > 0
    truly_safe = values > 0

    if predicted_safe_prior is None:
        n = int(initial_cbf.numel())
        if n == 0:
            return {
                "predicted_safe_vol_ratio": float("nan"),
                "validated_safe_vol_ratio": float("nan"),
                "estimation": "uniform",
            }
        return {
            "predicted_safe_vol_ratio": float(predicted_safe.float().mean().item()),
            "validated_safe_vol_ratio": float(truly_safe.float().mean().item()),
            "estimation": "uniform",
        }

    p = float(predicted_safe_prior)
    n_ps = int(predicted_safe.sum().item())
    n_pu = int((~predicted_safe).sum().item())

    # When a stratum has no samples, its conditional probability is undefined.
    # We avoid the IEEE ``0 * NaN = NaN`` trap by dropping that term from the
    # mixture only when the corresponding prior weight is also zero — i.e. the
    # stratum is genuinely absent from the state box (e.g. empty predicted-safe
    # set).  If the prior weight is nonzero but the stratum is empty, the
    # estimate is irrecoverably biased and we return NaN.
    if n_ps > 0:
        safe_contrib = p * (float((predicted_safe & truly_safe).sum().item()) / n_ps)
    elif p == 0.0:
        safe_contrib = 0.0
    else:
        safe_contrib = float("nan")

    if n_pu > 0:
        unsafe_contrib = (1.0 - p) * (
            float((~predicted_safe & truly_safe).sum().item()) / n_pu
        )
    elif p == 1.0:
        unsafe_contrib = 0.0
    else:
        unsafe_contrib = float("nan")

    validated = safe_contrib + unsafe_contrib
    return {
        "predicted_safe_vol_ratio": p,
        "validated_safe_vol_ratio": float(validated),
        "estimation": "stratified_prior",
    }


def stratified_sample_by_predicted_cbf(
    dynamics: ControlAffine,
    constr_fn: Callable[[Tensor], Tensor],
    residual_fn: nn.Module,
    num_per_stratum: int,
    max_oversample: int = 100,
    chunk_size: int | None = None,
) -> dict:
    """Rejection-sample ``num_per_stratum`` predicted-safe and predicted-unsafe
    initial states, uniformly within each stratum.

    Candidates are drawn uniformly from the state box via
    :meth:`ControlAffine.get_uniform_state_samples`, classified by the sign
    of ``V(x) = c(x) - r_Θ(x)``, and accumulated into the
    ``{V > 0}`` and ``{V ≤ 0}`` buckets until both are full or the sampling
    budget ``max_oversample × num_per_stratum`` is exhausted.

    When one stratum is empty or vanishingly rare, the function returns
    fewer than ``num_per_stratum`` samples for it and emits a warning;
    callers should check the returned ``n_predicted_safe`` and
    ``n_predicted_unsafe`` counts.  As a special case, if the predicted-safe
    set is empty under the current model (zero hits in the first batch of
    candidates), the loop short-circuits and ``empty_predicted_safe = True``
    is reported, instead of burning the full oversample budget.

    Args:
        dynamics:        ControlAffine system providing the state box.
        constr_fn:       Constraint function ``c(x)``.
        residual_fn:     Learned residual network ``r_Θ``.  Put into
                         ``eval()`` for the duration of the call.
        num_per_stratum: Target sample count for each stratum.
        max_oversample:  Hard cap on total candidates drawn, expressed as a
                         multiplier of ``num_per_stratum``.
        chunk_size:      Candidates drawn per rejection step.  Defaults to
                         ``max(num_per_stratum, 1024)``.

    Returns:
        Dict containing:

        * ``"states"`` — ``(n_total, nx)`` tensor, predicted-safe samples
          stacked first, predicted-unsafe second.
        * ``"predicted_safe"`` — ``(n_total,)`` bool mask, ``True`` on the
          predicted-safe portion.
        * ``"n_predicted_safe"`` — int count of predicted-safe samples.
        * ``"n_predicted_unsafe"`` — int count of predicted-unsafe samples.
        * ``"attempts"`` — int total candidates examined.
        * ``"total_predicted_safe"`` — int count of candidates classified as
          predicted-safe across *all* ``attempts`` draws (not capped by the
          per-stratum bucket size).  The ratio
          ``total_predicted_safe / attempts`` is an unbiased estimator of
          ``Vol{V > 0} / Vol(state_box)``.
        * ``"empty_predicted_safe"`` — bool, ``True`` when the loop
          short-circuited because no predicted-safe sample was found.
    """
    if num_per_stratum <= 0:
        raise ValueError(
            f"num_per_stratum must be positive; got {num_per_stratum}."
        )

    was_training = residual_fn.training
    residual_fn.eval()
    empty_predicted_safe = False
    try:
        chunk = chunk_size if chunk_size is not None else max(num_per_stratum, 1024)
        budget = max_oversample * num_per_stratum

        ps_buf: list[Tensor] = []
        pu_buf: list[Tensor] = []
        n_ps, n_pu = 0, 0
        attempts = 0
        # Track classifications over *all* candidates drawn (independent of
        # bucket caps).  Because candidates are i.i.d. uniform over the
        # state box, ``total_predicted_safe / attempts`` is an unbiased
        # estimator of ``Vol{V > 0} / Vol(state_box)``, used downstream by
        # the volume-ratio metrics.
        total_predicted_safe = 0
        device: torch.device | None = None
        dtype: torch.dtype | None = None

        while (n_ps < num_per_stratum or n_pu < num_per_stratum) and attempts < budget:
            x = dynamics.get_uniform_state_samples(chunk, requires_grad=False)
            if device is None:
                device, dtype = x.device, x.dtype
            with torch.no_grad():
                v = (constr_fn(x) - residual_fn(x)).reshape(-1)
            ps_mask = v > 0
            total_predicted_safe += int(ps_mask.sum().item())

            if n_ps < num_per_stratum:
                x_ps = x[ps_mask]
                take = min(num_per_stratum - n_ps, x_ps.shape[0])
                if take > 0:
                    ps_buf.append(x_ps[:take])
                    n_ps += take
            if n_pu < num_per_stratum:
                x_pu = x[~ps_mask]
                take = min(num_per_stratum - n_pu, x_pu.shape[0])
                if take > 0:
                    pu_buf.append(x_pu[:take])
                    n_pu += take

            attempts += chunk

            # If the first batch of candidates contained zero predicted-safe
            # samples, conclude the predicted-safe set is empty under the
            # current model and stop — otherwise we'd spin through the entire
            # ``max_oversample * num_per_stratum`` budget for nothing.
            if total_predicted_safe == 0:
                empty_predicted_safe = True
                break
    finally:
        if was_training:
            residual_fn.train()

    nx = dynamics.nx
    if device is None:
        # Should be unreachable given num_per_stratum > 0, but be defensive.
        device = dynamics.x_min.device
        dtype = dynamics.x_min.dtype

    ps_states = (
        torch.cat(ps_buf, dim=0)
        if ps_buf
        else torch.empty(0, nx, device=device, dtype=dtype)
    )
    pu_states = (
        torch.cat(pu_buf, dim=0)
        if pu_buf
        else torch.empty(0, nx, device=device, dtype=dtype)
    )
    states = torch.cat([ps_states, pu_states], dim=0)
    predicted_safe = torch.zeros(states.shape[0], dtype=torch.bool, device=device)
    predicted_safe[: ps_states.shape[0]] = True

    if empty_predicted_safe:
        warnings.warn(
            "stratified_sample_by_predicted_cbf: predicted-safe set appears "
            f"empty under the current model (0 hits in {attempts} candidates). "
            "Skipping the safe stratum.",
            stacklevel=2,
        )
    elif n_ps < num_per_stratum or n_pu < num_per_stratum:
        warnings.warn(
            "stratified_sample_by_predicted_cbf: budget exhausted before "
            f"filling buckets (attempts={attempts}, budget={budget}). "
            f"Got n_predicted_safe={n_ps}/{num_per_stratum}, "
            f"n_predicted_unsafe={n_pu}/{num_per_stratum}. "
            "Consider raising max_oversample or check whether one stratum "
            "is empty under the current model.",
            stacklevel=2,
        )

    return {
        "states": states,
        "predicted_safe": predicted_safe,
        "n_predicted_safe": n_ps,
        "n_predicted_unsafe": n_pu,
        "attempts": attempts,
        "total_predicted_safe": total_predicted_safe,
        "empty_predicted_safe": empty_predicted_safe,
    }


def validate_cbf(
    dynamics: ControlAffine,
    states: Tensor,
    constr_fn: Callable[[Tensor], Tensor],
    residual_fn: nn.Module,
    T: float,
    dt: float,
    return_values: bool = True,
    return_initial_cbf: bool = True,
    return_state_trajs: bool = False,
    return_control_trajs: bool = False,
    batch_size: int | None = None,
    predicted_safe_prior: float | None = None,
) -> dict:
    """Roll out the greedy Neural-CBF policy and report stratified statistics.

    At every step the Hamiltonian-maximising control is the vertex of the
    control box ``u*(x) = argmax_u ∇V(x) · (f(x) + g(x) u)`` with
    ``V(x) = c(x) - r_Θ(x)``.  The state is advanced by
    :meth:`ControlAffine.euler_step` with time step ``dt`` for
    ``n_steps = round(T / dt)`` steps.  The running minimum of ``c(x_k)``
    along each trajectory is tracked as the **true** value, and is compared
    against the **predicted** value ``V(x_0)`` to compute per-stratum
    precision (see :func:`_stratified_metrics`).

    The metric is well-defined regardless of how ``states`` were drawn, but
    its statistical efficiency is best when the two predicted-class strata
    are balanced.  Use :func:`stratified_sample_by_predicted_cbf` to draw
    such a population.

    Work is done in parallel over all initial states.  When running on CUDA
    and ``batch_size`` is ``None``, the number of states processed per call
    is auto-selected from free GPU memory; :func:`run_in_batches` halves and
    retries on OOM.

    Args:
        dynamics:             ControlAffine instance.
        states:               ``(N, nx)`` initial states.
        constr_fn:            Callable ``(N, nx) -> (N, 1)`` or ``(N,)``
                              defining the safe set ``{x : c(x) ≥ 0}``.
        residual_fn:          Learned residual network outputting ``(N, 1)``.
                              Put into ``eval()`` for the duration of the call.
        T:                    Rollout horizon (seconds).
        dt:                   Euler integration step (seconds).
        return_values:        If True, include the per-trajectory running
                              minimum of ``c(x_k)`` in the output.
        return_initial_cbf:   If True, include the per-state ``V(x_0)`` in
                              the output (useful for downstream plotting
                              or recomputing per-stratum masks).
        return_state_trajs:   If True, include the full state trajectory
                              ``(N, n_steps + 1, nx)`` including ``x_0``.
        return_control_trajs: If True, include the applied control
                              sequence ``(N, n_steps, nu)``.
        batch_size:           Max number of initial states per GPU call.
                              ``None`` auto-selects from free memory.
        predicted_safe_prior: Externally supplied estimate of
                              ``Vol{V > 0} / Vol(state_box)``, used to
                              reweight the per-stratum rollout outcomes
                              into volume ratios when ``states`` were drawn
                              by stratified rejection sampling.  Leave
                              ``None`` when ``states`` are uniform over the
                              state box (``random`` / ``grid`` sampling);
                              the volume ratios are then computed as
                              direct sample means.  See
                              :func:`_volume_metrics` for the formula.

    Returns:
        Dict always containing ``"stratified_metrics"`` and
        ``"volume_metrics"`` plus any subset of tensors selected by the
        ``return_*`` flags:

            ``"stratified_metrics"`` — dict described in
                :func:`_stratified_metrics`.  ``"predicted_safe"`` reports
                the **certificate precision** (how often the CBF's "safe"
                claim was correct, in %) and the **false-safe rate**;
                ``"predicted_unsafe"`` reports the dual quantities.
            ``"volume_metrics"`` — dict described in
                :func:`_volume_metrics`: volume ratios of the predicted
                and validated safe sets, both as fractions of the state
                box.
            ``"initial_cbf"``  — ``(N,)``                ``V(x_0)``.
            ``"values"``       — ``(N,)``                running ``min_k c(x_k)``.
            ``"state_traj"``   — ``(N, n_steps+1, nx)``  full state trajectory.
            ``"control_traj"`` — ``(N, n_steps, nu)``    applied controls.
    """
    N, nx = states.shape
    nu = dynamics.nu
    n_steps = int(round(T / dt))
    if n_steps <= 0:
        raise ValueError(f"T/dt must be positive; got T={T}, dt={dt}.")

    # ------------------------------------------------------------------
    # Auto-batch: split states along dim=0 to stay within GPU memory budget.
    #
    # The rollout's autograd graph (per-step branching over control vertices
    # plus activations through ``residual_fn`` for ``∇V``) can exceed the
    # peak-memory estimate, so even when ``_auto_batch_size`` says the full
    # set fits, propagation can still OOM.  Always route through
    # ``run_in_batches``, which halves the chunk and retries on OOM.
    # ------------------------------------------------------------------
    if batch_size is None and states.is_cuda:
        _warmup_cuda(dynamics, states, dt, constr_fn)
        batch_size = _auto_batch_size(
            _peak_bytes_validate(
                n_steps, nx, nu, residual_fn,
                return_state_trajs, return_control_trajs,
                states.element_size(),
            ),
            states.device,
        )

    def _rollout_chunk(chunk: Tensor) -> dict:
        return _validate_cbf_rollout(
            dynamics, chunk, constr_fn, residual_fn, T, dt,
            return_state_trajs=return_state_trajs,
            return_control_trajs=return_control_trajs,
        )

    if batch_size is None:
        # CPU path — no GPU memory budget to enforce.
        raw = _rollout_chunk(states)
    else:
        raw = run_in_batches(_rollout_chunk, states, min(batch_size, N))

    result: dict = {
        "stratified_metrics": _stratified_metrics(raw["initial_cbf"], raw["values"]),
        "volume_metrics": _volume_metrics(
            raw["initial_cbf"],
            raw["values"],
            predicted_safe_prior=predicted_safe_prior,
        ),
    }
    if return_initial_cbf:
        result["initial_cbf"] = raw["initial_cbf"]
    if return_values:
        result["values"] = raw["values"]
    if return_state_trajs:
        result["state_traj"] = raw["state_traj"]
    if return_control_trajs:
        result["control_traj"] = raw["control_traj"]
    return result
