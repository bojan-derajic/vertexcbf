"""Training loop for VertexCBF learning.

Example usage::

    from vertexcbf.trainer import Trainer

    trainer = Trainer(
        dynamics=dynamics,
        model=model,
        constr_fn=constr_fn,
        epochs=10000,
        lr=1e-3,
        lr_milestones=[7000],
        lr_gamma=0.1,
        pde_grid_shape=(100, 100),
        pde_weight_mode="normalized",
        data_states=states_bs,
        data_values=values_bs,
        checkpoint_dir="checkpoints/my_experiment",
        log_every=20,
    )
    loss_history = trainer.train()
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine
from vertexcbf.losses import data_loss, pde_loss


def _is_oom_error(exc: BaseException) -> bool:
    """Return True for any CUDA-memory-exhaustion error, not just the strict
    ``torch.cuda.OutOfMemoryError``.

    cuBLAS, cuDNN, and the caching allocator can all surface OOM as a plain
    ``RuntimeError`` — most notably ``CUBLAS_STATUS_ALLOC_FAILED`` from
    ``cublasCreate`` — when the workspace can't be allocated.  We treat any
    of these as the signal to back off chunk sizes.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "alloc_failed" in msg
            or "cublas_status_alloc_failed" in msg
            or "cudnn_status_alloc_failed" in msg
        )
    return False


class Trainer:
    """Trains a Neural CBF of the form ``h_Θ(x) = c(x) - r_Θ(x)``.

    ``c(x)`` is the constraint function defining the safe set and ``r_Θ(x)``
    is the learned neural residual (``model``).  Training minimises a weighted
    combination of a physics term and a supervised data term:

        loss = w · pde_loss + (1 − w) · data_loss

    where ``pde_loss`` enforces the HJB optimality condition and
    ``data_loss`` is MSE against precomputed Neural CBF targets.
    When no data states/values are provided the data loss term is omitted
    and only ``pde_loss`` is used regardless of ``pde_weight_mode``.

    Three modes for computing the weight ``w`` are available via
    ``pde_weight_mode``:

    * ``"fixed"`` *(default)* — ``w = pde_weight`` throughout training.
    * ``"normalized"`` — Each loss is divided by its value at epoch 0, so
      both terms start at 1 and contribute equally without manual tuning.
      ``pde_weight`` and ``pde_weight_milestones`` are ignored in this mode.
    * ``"scheduled"`` — ``w`` starts at ``pde_weight`` and steps to new
      values at specified epochs, analogously to a learning-rate schedule.
      Provide milestone (epoch, weight) pairs via ``pde_weight_milestones``.
      If no milestones are given, the weight stays constant at ``pde_weight``.

    Args:
        dynamics: Control-affine system used to build the PDE grid.
        model: MLP that outputs the residual ``r_Θ(x)``, shape ``(N, 1)``.
        constr_fn: Constraint function ``c(x)``, callable ``(N, nx) -> (N, 1)``.
        epochs: Total number of training epochs.
        lr: Initial learning rate for Adam.
        lr_milestones: Epoch indices at which to decay the learning rate.
        lr_gamma: Multiplicative decay factor applied at each milestone.
        pde_grid_shape: Grid resolution for PDE states, one int per state dim.
            Mutually exclusive with ``pde_num_samples``.
        pde_num_samples: If given, sample PDE states randomly instead of using
            a grid.  Mutually exclusive with ``pde_grid_shape``.
        pde_weight_mode: How to combine PDE and data losses.  One of
            ``"fixed"``, ``"normalized"``, or ``"scheduled"`` (see class
            docstring for details).
        pde_weight: Initial (or fixed) PDE loss weight λ ∈ [0, 1].
            Ignored when ``pde_weight_mode="normalized"``.
        pde_weight_milestones: Used only with ``pde_weight_mode="scheduled"``.
            List of ``(epoch, new_weight)`` pairs at which to step the weight,
            e.g. ``[(5000, 0.5), (8000, 1.0)]``.
        data_states: Optional states for the data loss, shape ``(M, nx)``.
        data_values: Optional Neural CBF targets, shape ``(M,)``.
        log_every: Print a loss summary every this many epochs.
        checkpoint_dir: Directory to save checkpoints.  ``None`` disables
            checkpointing.
        checkpoint_every: Save a checkpoint every this many epochs.
        resume_from: Path to a checkpoint file to resume from.
    """

    # Auto-batching tuning knobs.  Defaults are conservative-but-effective on
    # a 24 GB consumer GPU; tune these on the class if you need finer control.
    _TARGET_MEM_FRACTION: float = 0.90  # Aim to use up to this share of VRAM.
    _OOM_BACKOFF: float = 0.75          # Multiply chunk size by this on OOM.
    _GROWTH_CHECK_EVERY: int = 25       # Steps between growth attempts.
    _GROWTH_CEILING_MARGIN: float = 0.9  # Stay this far below a known OOM size.

    def __init__(
        self,
        dynamics: ControlAffine,
        model: nn.Module,
        constr_fn: Callable[[Tensor], Tensor],
        epochs: int = 10000,
        lr: float = 1e-3,
        lr_milestones: list[int] | None = None,
        lr_gamma: float = 0.1,
        pde_grid_shape: tuple[int, ...] | None = (100, 100),
        pde_num_samples: int | None = None,
        pde_weight_mode: str = "fixed",
        pde_weight: float = 1.0,
        pde_weight_milestones: list[tuple[int, float]] | None = None,
        data_states: Optional[Tensor] = None,
        data_values: Optional[Tensor] = None,
        log_every: int = 20,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 1000,
        resume_from: Optional[str] = None,
    ) -> None:
        if pde_weight_mode not in ("fixed", "normalized", "scheduled"):
            raise ValueError(
                f"pde_weight_mode must be 'fixed', 'normalized', or 'scheduled', "
                f"got '{pde_weight_mode}'"
            )

        self.dynamics = dynamics
        self.model = model
        self.constr_fn = constr_fn
        self.epochs = epochs
        self.pde_weight_mode = pde_weight_mode
        self.pde_weight = pde_weight
        self.pde_weight_milestones: list[tuple[int, float]] = sorted(
            (tuple(m) for m in (pde_weight_milestones or [])),
            key=lambda m: m[0],
        )
        self.data_states = data_states
        self.data_values = data_values
        self.log_every = log_every
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every

        self._start_epoch = 0
        self._loss_history: list[float] = []
        self._best_loss: float = float("inf")

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=lr_milestones or [],
            gamma=lr_gamma,
        )

        # Precompute PDE states once (requires_grad=True for autograd).
        # Skip entirely when the requested sample count is zero or the PDE
        # weight is fixed at zero (no point allocating states that are never used).
        _pde_disabled = (
            (pde_num_samples is not None and pde_num_samples == 0)
            or (pde_weight_mode == "fixed" and pde_weight == 0.0)
        )
        if _pde_disabled:
            self._states_pde = None
        elif pde_num_samples is not None:
            self._states_pde = dynamics.get_uniform_state_samples(
                num_samples=pde_num_samples, requires_grad=True
            )
        else:
            self._states_pde = dynamics.get_uniform_state_grid(
                grid_shape=pde_grid_shape, requires_grad=True
            ).reshape(-1, dynamics.nx)

        self._use_pde: bool = (
            self._states_pde is not None and self._states_pde.shape[0] > 0
        )

        # Auto-batching state: stays disabled while the full set fits in GPU
        # memory.  On the first CUDA OOM (either while precomputing
        # ``_xdot_vertices`` below or during a training step) we switch to
        # gradient-accumulated chunks.  Chunk sizes then adapt during
        # training: shrink by ``_OOM_BACKOFF`` on OOM, and periodically grow
        # toward ``_TARGET_MEM_FRACTION`` of total VRAM based on measured
        # peak allocation.  Once chunks reach full N (and the precompute
        # didn't OOM), we exit chunked mode entirely.
        self._chunked: bool = False
        self._pde_chunk_size: int | None = None
        self._data_chunk_size: int | None = None
        # Last chunk size that OOMed — never auto-grow back above this.
        self._pde_oom_ceiling: int | None = None
        self._data_oom_ceiling: int | None = None
        # Steps since the last OOM; used to gate growth attempts.
        self._steps_since_oom: int = 0

        # Try to precompute xdot at control vertices (detached so not part of
        # graph).  If this OOMs we fall back to lazy per-chunk computation
        # during training — keeping ``_xdot_vertices = None`` is the signal.
        if self._use_pde:
            try:
                self._xdot_vertices = dynamics.get_xdot_vertices(
                    self._states_pde.detach()
                )
            except Exception as exc:
                if not _is_oom_error(exc):
                    raise
                torch.cuda.empty_cache()
                self._xdot_vertices = None
                self._chunked = True
                self._pde_chunk_size = max(1, self._states_pde.shape[0] // 2)
                msg = (
                    f"[Trainer] CUDA OOM precomputing xdot_vertices; switching to "
                    f"chunked lazy mode (pde_chunk_size={self._pde_chunk_size}"
                )
                if data_states is not None:
                    self._data_chunk_size = max(1, data_states.shape[0] // 2)
                    msg += f", data_chunk_size={self._data_chunk_size}"
                print(msg + ")")
        else:
            self._xdot_vertices = None

        if resume_from is not None:
            self._load_checkpoint(resume_from)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self) -> list[float]:
        """Run the training loop.

        Returns:
            List of per-epoch training losses.
        """
        use_pde = self._use_pde
        use_data = (
            self.data_states is not None
            and self.data_values is not None
            and self.data_states.shape[0] > 0
        )

        # For normalized mode, compute initial loss scales once before the loop.
        L_pde_0 = L_data_0 = None
        if use_pde and use_data and self.pde_weight_mode == "normalized":
            L_pde_0, L_data_0 = self._compute_initial_scales()
            print(
                f"Initial scales — L_pde_0: {L_pde_0:.6f}  "
                f"L_data_0: {L_data_0:.6f}  "
                f"ratio: {L_pde_0 / (L_data_0 + 1e-12):.3f}"
            )

        for epoch in range(self._start_epoch, self.epochs):
            pde_scale, data_scale = self._loss_scales(
                epoch, use_pde, use_data, L_pde_0, L_data_0
            )
            loss_pde_val, loss_data_val = self._train_step(
                epoch, use_pde, use_data, pde_scale, data_scale
            )
            total_val = pde_scale * loss_pde_val + data_scale * loss_data_val
            self._loss_history.append(total_val)

            if (epoch + 1) % self.log_every == 0:
                self._log(epoch, total_val, loss_pde_val, loss_data_val)

            if self.checkpoint_dir is not None and total_val < self._best_loss:
                self._best_loss = total_val
                self._save_checkpoint(epoch + 1, tag="best")

            if (
                self.checkpoint_dir is not None
                and (epoch + 1) % self.checkpoint_every == 0
            ):
                self._save_checkpoint(epoch + 1)

        if self.checkpoint_dir is not None:
            self._save_checkpoint(self.epochs, tag="final")

        return self._loss_history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _loss_scales(
        self,
        epoch: int,
        use_pde: bool,
        use_data: bool,
        L_pde_0: float | None,
        L_data_0: float | None,
    ) -> tuple[float, float]:
        """Multipliers applied to ``loss_pde`` / ``loss_data`` this epoch."""
        if not use_pde:
            return 0.0, 1.0
        if not use_data:
            return 1.0, 0.0
        if self.pde_weight_mode == "normalized":
            return 1.0 / L_pde_0, 1.0 / L_data_0
        if self.pde_weight_mode == "scheduled":
            w = self._get_scheduled_weight(epoch)
            return w, 1.0 - w
        return self.pde_weight, 1.0 - self.pde_weight

    def _train_step(
        self,
        epoch: int,
        use_pde: bool,
        use_data: bool,
        pde_scale: float,
        data_scale: float,
    ) -> tuple[float, float]:
        """One optimisation step.  Auto-falls back to chunked accumulation on OOM.

        Returns the unscaled scalar values of ``loss_pde`` and ``loss_data``
        for logging.
        """
        if not self._chunked:
            try:
                return self._train_step_full(use_pde, use_data, pde_scale, data_scale)
            except Exception as exc:
                if not _is_oom_error(exc):
                    raise
                torch.cuda.empty_cache()
                self.optimizer.zero_grad(set_to_none=True)
                self._chunked = True
                if use_pde:
                    self._pde_chunk_size = max(1, self._states_pde.shape[0] // 2)
                    self._pde_oom_ceiling = self._states_pde.shape[0]
                if use_data:
                    self._data_chunk_size = max(
                        1, self.data_states.shape[0] // 2
                    )
                    self._data_oom_ceiling = self.data_states.shape[0]
                # Reset the peak stat so the next chunked step measures a
                # clean high-water mark rather than the failed full attempt.
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                self._steps_since_oom = 0
                msg = (
                    f"[Trainer] CUDA OOM at epoch {epoch + 1}; switching to "
                    f"chunked gradient accumulation "
                    f"(pde_chunk_size={self._pde_chunk_size}"
                )
                if use_data:
                    msg += f", data_chunk_size={self._data_chunk_size}"
                print(msg + ")")

        return self._train_step_chunked(epoch, use_pde, use_data, pde_scale, data_scale)

    def _train_step_full(
        self,
        use_pde: bool,
        use_data: bool,
        pde_scale: float,
        data_scale: float,
    ) -> tuple[float, float]:
        """Original full-batch step — used while the data fits in GPU memory."""
        self.optimizer.zero_grad()
        total_loss = None
        pde_val = 0.0
        data_val = 0.0
        if use_pde and pde_scale != 0.0:
            loss_pde = pde_loss(
                self.model, self.constr_fn, self._states_pde, self._xdot_vertices
            )
            pde_val = loss_pde.item()
            total_loss = pde_scale * loss_pde
        if use_data:
            loss_data = data_loss(
                self.model, self.constr_fn, self.data_states, self.data_values
            )
            data_val = loss_data.item()
            scaled = data_scale * loss_data
            total_loss = scaled if total_loss is None else total_loss + scaled

        if total_loss is not None:
            total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return pde_val, data_val

    def _train_step_chunked(
        self,
        epoch: int,
        use_pde: bool,
        use_data: bool,
        pde_scale: float,
        data_scale: float,
    ) -> tuple[float, float]:
        """Gradient-accumulation step with adaptive chunk sizing.

        On OOM, only the side (pde / data) whose backward actually raised is
        backed off — by ``_OOM_BACKOFF`` (25 % drop), not a half — and the
        failing size is recorded as a ceiling so future growth never returns
        to a known-bad size.  Every ``_GROWTH_CHECK_EVERY`` successful steps,
        chunk sizes are nudged upward based on measured peak GPU allocation
        until they either saturate ``_TARGET_MEM_FRACTION`` of VRAM or reach
        the full dataset (in which case chunked mode is disabled).
        """
        while True:
            self.optimizer.zero_grad()
            loss_pde_val = 0.0
            loss_data_val = 0.0
            oom_phase: str | None = None

            try:
                if use_pde and pde_scale != 0.0:
                    loss_pde_val = self._backprop_pde_chunked(pde_scale)
            except Exception as exc:
                if not _is_oom_error(exc):
                    raise
                oom_phase = "pde"

            if oom_phase is None:
                try:
                    if use_data:
                        loss_data_val = self._backprop_data_chunked(data_scale)
                except Exception as exc:
                    if not _is_oom_error(exc):
                        raise
                    oom_phase = "data"

            if oom_phase is None:
                self.optimizer.step()
                self.scheduler.step()
                self._steps_since_oom += 1
                if (
                    torch.cuda.is_available()
                    and self._steps_since_oom >= self._GROWTH_CHECK_EVERY
                ):
                    self._maybe_grow_chunks(epoch, use_pde, use_data)
                    self._steps_since_oom = 0
                    torch.cuda.reset_peak_memory_stats()
                return loss_pde_val, loss_data_val

            # ---- OOM path: drop partial grads, back off the offending side ----
            self.optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            torch.cuda.empty_cache()

            backed_off = False
            if oom_phase == "pde" and self._pde_chunk_size > 1:
                self._pde_oom_ceiling = self._pde_chunk_size
                self._pde_chunk_size = max(
                    1, int(self._pde_chunk_size * self._OOM_BACKOFF)
                )
                backed_off = True
            elif oom_phase == "data" and self._data_chunk_size > 1:
                self._data_oom_ceiling = self._data_chunk_size
                self._data_chunk_size = max(
                    1, int(self._data_chunk_size * self._OOM_BACKOFF)
                )
                backed_off = True
            if not backed_off:
                raise RuntimeError(
                    f"[Trainer] CUDA OOM in {oom_phase} phase at epoch "
                    f"{epoch + 1} with chunk size already at 1"
                )
            self._steps_since_oom = 0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            print(
                f"[Trainer] CUDA OOM in {oom_phase} phase at epoch "
                f"{epoch + 1}; backing off "
                f"(pde_chunk_size={self._pde_chunk_size}, "
                f"data_chunk_size={self._data_chunk_size})"
            )

    def _maybe_grow_chunks(
        self, epoch: int, use_pde: bool, use_data: bool
    ) -> bool:
        """Try to enlarge chunk sizes toward ``_TARGET_MEM_FRACTION`` of VRAM.

        Uses ``torch.cuda.max_memory_allocated`` since the last peak-stats
        reset as an empirical per-step peak.  Per-row cost is overestimated
        (baseline allocations get folded into it), so growth converges from
        below over a few rounds rather than overshooting.  Growth is capped
        per call at ``2×`` the current size to avoid jumping past the OOM
        boundary in a single hop.
        """
        device: torch.device | None = None
        if use_pde:
            device = self._states_pde.device
        elif use_data and self.data_states is not None:
            device = self.data_states.device
        if device is None or device.type != "cuda":
            return False

        total_mem = torch.cuda.get_device_properties(device).total_memory
        budget = int(total_mem * self._TARGET_MEM_FRACTION)
        peak = torch.cuda.max_memory_allocated(device)
        if peak <= 0 or peak >= budget:
            return False

        active = (self._pde_chunk_size if use_pde else 0) + (
            self._data_chunk_size if use_data else 0
        )
        if active <= 0:
            return False
        bytes_per_row = peak / active
        extra_rows = int((budget - peak) / bytes_per_row)
        # Require at least a 5 % gain to be worth the OOM risk of a probe.
        if extra_rows < max(1, active // 20):
            return False

        def _capped(current: int, full: int, ceiling: int | None) -> int:
            cap = full
            if ceiling is not None:
                cap = min(cap, max(current, int(ceiling * self._GROWTH_CEILING_MARGIN)))
            share = current / active
            desired = int(extra_rows * share)
            desired = min(desired, current)  # never more than double in one go
            return min(cap, current + max(1, desired))

        grew = False
        if use_pde:
            new_size = _capped(
                self._pde_chunk_size,
                self._states_pde.shape[0],
                self._pde_oom_ceiling,
            )
            if new_size > self._pde_chunk_size:
                self._pde_chunk_size = new_size
                grew = True
        if use_data:
            new_size = _capped(
                self._data_chunk_size,
                self.data_states.shape[0],
                self._data_oom_ceiling,
            )
            if new_size > self._data_chunk_size:
                self._data_chunk_size = new_size
                grew = True

        if not grew:
            return False

        # If chunks now span the entire dataset and we have a usable
        # precomputed ``_xdot_vertices`` (i.e. the slow path isn't required),
        # return to the fast full-batch step.
        full_pde = (not use_pde) or self._pde_chunk_size >= self._states_pde.shape[0]
        full_data = (not use_data) or (
            self.data_states is not None
            and self._data_chunk_size >= self.data_states.shape[0]
        )
        precompute_ok = (not use_pde) or self._xdot_vertices is not None
        if full_pde and full_data and precompute_ok:
            self._chunked = False
            print(
                f"[Trainer] Chunks reached full N at epoch {epoch + 1}; "
                f"returning to full-batch mode"
            )
        else:
            peak_gb = peak / (1024**3)
            budget_gb = budget / (1024**3)
            print(
                f"[Trainer] Growing chunks at epoch {epoch + 1} "
                f"(peak={peak_gb:.2f} GB / target={budget_gb:.2f} GB, "
                f"pde_chunk_size={self._pde_chunk_size}, "
                f"data_chunk_size={self._data_chunk_size})"
            )
        return True

    def _backprop_pde_chunked(self, scale: float) -> float:
        """Accumulate ``scale * pde_loss`` gradients over chunks of states.

        Each chunk's mean loss is reweighted by ``chunk_size / N`` so that the
        sum across chunks equals the global mean — making this step
        mathematically identical to a single full-batch backward.

        When ``_xdot_vertices`` is ``None`` (precompute OOMed) the control-
        vertex xdots are recomputed per chunk under ``no_grad`` so that only
        one chunk's worth of vertices lives on the device at a time.
        """
        n = self._states_pde.shape[0]
        chunk = self._pde_chunk_size
        value = 0.0
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            states_c = self._states_pde[start:end]
            if self._xdot_vertices is not None:
                xdot_c = self._xdot_vertices[start:end]
            else:
                with torch.no_grad():
                    xdot_c = self.dynamics.get_xdot_vertices(states_c.detach())
            lp = pde_loss(self.model, self.constr_fn, states_c, xdot_c)
            weight = (end - start) / n
            (scale * weight * lp).backward()
            value += weight * lp.item()
        return value

    def _backprop_data_chunked(self, scale: float) -> float:
        n = self.data_states.shape[0]
        chunk = self._data_chunk_size
        value = 0.0
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            states_c = self.data_states[start:end]
            values_c = self.data_values[start:end]
            ld = data_loss(self.model, self.constr_fn, states_c, values_c)
            weight = (end - start) / n
            (scale * weight * ld).backward()
            value += weight * ld.item()
        return value

    def _get_scheduled_weight(self, epoch: int) -> float:
        """Return the scheduled pde_weight for the given epoch.

        Iterates through milestones in order and returns the weight
        associated with the latest milestone that has been reached.
        Falls back to ``self.pde_weight`` before the first milestone.
        """
        weight = self.pde_weight
        for milestone_epoch, milestone_weight in self.pde_weight_milestones:
            if epoch >= milestone_epoch:
                weight = milestone_weight
            else:
                break
        return weight

    def _compute_initial_scales(self) -> tuple[float, float]:
        """Compute initial loss values used as normalization denominators.

        Returns:
            Tuple ``(L_pde_0, L_data_0)`` — the raw loss values at the
            current model state (typically before any training steps).
        """
        n = self._states_pde.shape[0]
        pde_chunk = self._pde_chunk_size if self._chunked else n
        L_pde_0 = 0.0
        for start in range(0, n, pde_chunk):
            end = min(start + pde_chunk, n)
            # Re-detach + requires_grad so each chunk has its own graph that
            # is freed after .grad — avoids holding the full graph in memory.
            states_c = (
                self._states_pde[start:end].detach().requires_grad_(True)
            )
            constr = self.constr_fn(states_c)
            residual = self.model(states_c)
            cbf = constr - residual
            grad_cbf = torch.autograd.grad(
                outputs=cbf,
                inputs=states_c,
                grad_outputs=torch.ones_like(cbf),
                create_graph=False,
            )[0].unsqueeze(1)  # (chunk, 1, nx)
            if self._xdot_vertices is not None:
                xdot_c = self._xdot_vertices[start:end]
            else:
                with torch.no_grad():
                    xdot_c = self.dynamics.get_xdot_vertices(states_c.detach())
            hamiltonian, _ = torch.max(
                torch.bmm(grad_cbf.detach(), xdot_c), dim=-1
            )
            hamiltonian = hamiltonian.squeeze(1)
            residual_flat = residual.squeeze(1).detach()
            weight = (end - start) / n
            L_pde_0 += weight * torch.mean(
                torch.min(hamiltonian, residual_flat) ** 2
            ).item()

        m = self.data_states.shape[0]
        data_chunk = self._data_chunk_size if self._chunked else m
        L_data_0 = 0.0
        with torch.no_grad():
            for start in range(0, m, data_chunk):
                end = min(start + data_chunk, m)
                states_d = self.data_states[start:end]
                values_d = self.data_values[start:end]
                constr_d = self.constr_fn(states_d)
                residual_d = self.model(states_d)
                cbf_pred = (constr_d - residual_d).squeeze(1)
                weight = (end - start) / m
                L_data_0 += weight * torch.mean(
                    (cbf_pred - values_d) ** 2
                ).item()

        return L_pde_0, L_data_0

    def _log(self, epoch: int, total: float, pde: float, data: float) -> None:
        print(
            f"Epoch [{epoch + 1:>6}/{self.epochs}]  "
            f"loss={total:.6f}  "
            f"pde={pde:.6f}  "
            f"data={data:.6f}"
        )

    def _save_checkpoint(self, epoch: int, tag: str | None = None) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = f"epoch_{epoch}.pt" if tag is None else f"{tag}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "loss_history": self._loss_history,
                "best_loss": self._best_loss,
            },
            path,
        )
        if tag != "best":
            print(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self._start_epoch = ckpt["epoch"]
        self._loss_history = ckpt.get("loss_history", [])
        self._best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed from checkpoint: {path} (epoch {self._start_epoch})")
