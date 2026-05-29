from __future__ import annotations

import shutil
import sys
import time
from contextlib import contextmanager
from typing import Callable

import torch
from torch import Tensor

from vertexcbf.dynamics.control_affine import ControlAffine

# ---------------------------------------------------------------------------
# Progress reporting (opt-in via :func:`progress_context`)
# ---------------------------------------------------------------------------

_progress_state: dict = {"enabled": False, "desc": ""}


@contextmanager
def progress_context(desc: str):
    """Enable a progress bar inside :func:`run_in_batches` for the duration.

    ``run_in_batches`` is the natural granularity for long MPC / validation
    workloads: it iterates over chunks of initial states sized to the GPU
    memory budget.  Wrapping a top-level call (e.g. data generation) in this
    context manager makes every nested ``run_in_batches`` render a stderr
    progress bar with elapsed/ETA in HH:MM:SS.
    """
    prev = dict(_progress_state)
    _progress_state["enabled"] = True
    _progress_state["desc"] = desc
    try:
        yield
    finally:
        _progress_state.update(prev)


def _format_hms(seconds: float) -> str:
    s = max(0, int(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class _ProgressBar:
    """Stderr progress bar with HH:MM:SS elapsed/ETA and it/s rate.

    Uses unicode block characters with sub-cell resolution for the fill.
    Adapts bar width to the current terminal so the line does not wrap (a
    wrapped line breaks ``\\r``-style in-place updates and produces stacked
    bars).  Falls back to newline-per-update when stderr is not a TTY.
    """

    _BLOCKS = " ▏▎▍▌▋▊▉"  # 0/8 .. 7/8 fill
    _FULL = "█"

    def __init__(self, total: int, desc: str = "", width: int | None = None):
        self.total = max(1, int(total))
        self.desc = desc
        self.is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self.width = self._compute_width(width)
        self.n = 0
        self.start = time.time()
        self._render()

    def _compute_width(self, override: int | None) -> int:
        if override is not None:
            return override
        try:
            cols = shutil.get_terminal_size((80, 20)).columns
        except OSError:
            cols = 80
        # Reserve room for: "{desc}: 100% || N/N [hh:mm:ss<hh:mm:ss, rate it/s]"
        digits = len(str(self.total))
        overhead = len(self.desc) + 2 + 6 + 2 * digits + 1 + 32
        return max(10, min(40, cols - overhead))

    def update(self, n: int = 1) -> None:
        self.n = min(self.total, self.n + n)
        self._render()

    def close(self) -> None:
        if self.n < self.total:
            self.n = self.total
            self._render()
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _render(self) -> None:
        frac = self.n / self.total
        elapsed = time.time() - self.start
        eta = elapsed * (self.total - self.n) / self.n if self.n > 0 else 0.0
        rate = self.n / elapsed if elapsed > 0 else 0.0

        full_cells = frac * self.width
        whole = int(full_cells)
        sub = int((full_cells - whole) * 8)
        bar = self._FULL * whole
        if whole < self.width:
            bar += self._BLOCKS[sub] + " " * (self.width - whole - 1)
        bar = bar[: self.width]

        prefix = f"{self.desc}: " if self.desc else ""
        line = (
            f"{prefix}{frac * 100:3.0f}% |{bar}| {self.n}/{self.total} "
            f"[{_format_hms(elapsed)}<{_format_hms(eta)}, {rate:.1f} it/s]"
        )
        if self.is_tty:
            sys.stderr.write("\r\033[K" + line)
        else:
            sys.stderr.write(line + "\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Memory-aware auto-batching utilities
# ---------------------------------------------------------------------------


def _peak_bytes_shooting(N_s: int, K: int, nx: int, nu: int, dtype_bytes: int) -> int:
    """Estimated peak GPU bytes **per initial state** for shooting-style methods.

    Accounts for the controls tensor ``(N, N_s, K, nu)``, the flattened state
    buffer ``(N*N_s, nx)`` used inside :func:`score_sequences`, and a ×4
    overhead for PyTorch intermediate allocations.

    Applies to: ``random_shooting``, ``cem``, ``icem``, ``mppi``,
    ``cem_discrete``.
    """
    elements = N_s * (K * nu + nx + 1)
    return 4 * elements * dtype_bytes


def _peak_bytes_tree(
    B: int, M: int, K: int, nx: int, nu: int, return_trajs: bool, dtype_bytes: int
) -> int:
    """Estimated peak GPU bytes **per initial state** for tree-search methods.

    At the expansion step the frontier has up to ``B*M`` children.  The
    dominant tensors are ``states_exp (B*M, nx)``, ``states_next (B*M, nx)``,
    ``u_tiled (B*M, nu)``, and two running-min vectors ``(B*M,)``.  When
    ``return_trajs=True`` the trajectory and control-history buffers add
    ``(B*M, (K+1)*nx)`` and ``(B*M, K*nu)`` respectively.  A ×4 overhead
    covers temporaries.

    Applies to: ``beam_search``, ``stochastic_beam_search``,
    ``branch_and_bound``.
    """
    base = B * M * (2 * nx + nu + 2)
    if return_trajs:
        base += B * M * ((K + 1) * nx + K * nu)
    return 4 * base * dtype_bytes


def _auto_batch_size(
    peak_bytes_per_state: int,
    device: torch.device,
    safety: float = 0.95,
) -> int:
    """Maximum number of states that fit in currently-free GPU memory.

    Queries ``torch.cuda.mem_get_info`` for free bytes, retains ``safety``
    fraction as usable budget, and divides by ``peak_bytes_per_state``.
    Returns a large sentinel on CPU (no meaningful GPU memory constraint).

    Args:
        peak_bytes_per_state: Estimated peak bytes consumed per initial state.
        device:               The target device.
        safety:               Fraction of free GPU memory to use (default 0.8).

    Returns:
        Maximum number of states to process in a single call.
    """
    if device.type != "cuda":
        return 2**31
    # Force CUDA context initialisation so mem_get_info reflects the real
    # post-context memory (after a kernel restart the context hasn't been
    # created yet, mem_get_info returns almost all VRAM as free, and the
    # first call OOMs because the context itself consumes several hundred MB).
    if not torch.cuda.is_initialized():
        torch.zeros(1, device=device)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    budget = int(free_bytes * safety)
    return max(1, budget // peak_bytes_per_state)


def _warmup_cuda(
    dynamics: ControlAffine,
    x0: Tensor,
    dt: float,
    constr_fn,
    warmup_n: int = 1,
) -> None:
    """Warm up CUDA kernels and model weights before querying free memory.

    On the first call after a kernel restart, cuDNN workspaces, kernel JIT
    compilation, and lazy model-weight transfers have not yet happened.
    ``mem_get_info`` therefore overestimates available memory, leading
    ``_auto_batch_size`` to pick a batch that OOMs.  A dummy forward pass
    through ``dynamics.euler_step`` and ``constr_fn`` forces those one-time
    allocations to settle, after which ``empty_cache`` returns the freed
    scratch memory to the driver so ``mem_get_info`` is accurate.

    ``warmup_n`` controls how many rows the dummy pass uses.  cuBLAS/cuDNN
    bucket their kernel choice and workspace size by input shape, so warming
    up at ``n=1`` doesn't prime the larger workspaces that the actual inner
    loop (running at ``batch_size * B * M`` rows) will trigger.  Callers
    should pass a representative inner-loop size — typically ``B * M`` for
    tree-search methods — so the JIT/workspace caches are populated before
    ``mem_get_info`` is queried.
    """
    if not x0.is_cuda:
        return
    n = max(1, int(warmup_n))
    # Repeat x0 along dim=0 if we want more rows than we have.  ``expand`` is
    # zero-copy, but the euler_step forward will still materialise the right
    # shape internally, which is what triggers the cuBLAS kernel selection.
    if n <= x0.shape[0]:
        x_warm = x0[:n].contiguous()
    else:
        reps = (n + x0.shape[0] - 1) // x0.shape[0]
        x_warm = x0.repeat(reps, 1)[:n]
    u_warm = torch.zeros(n, dynamics.nu, device=x0.device, dtype=x0.dtype)
    with torch.no_grad():
        s = dynamics.euler_step(x_warm, u_warm, dt)
        constr_fn(s)
    torch.cuda.synchronize(x0.device)
    torch.cuda.empty_cache()


def run_in_batches(fn: Callable[[Tensor], dict], x0: Tensor, batch_size: int) -> dict:
    """Run ``fn(x0_chunk)`` on consecutive chunks of ``x0`` and merge results.

    Splits ``x0`` along ``dim=0`` into chunks of at most ``batch_size`` rows,
    calls ``fn`` on each chunk, and concatenates the resulting dicts along
    ``dim=0``.  All dict values must be tensors whose first dimension equals
    the chunk size.

    If a chunk triggers an out-of-memory error (e.g. because ``batch_size``
    was estimated from a formula that underestimates the dynamics internals),
    the chunk size is halved and the same chunk is retried until it fits or
    ``batch_size`` reaches 1.

    Args:
        fn:         Callable mapping ``(n, nx)`` tensor → dict of tensors with
                    leading dimension ``n``.
        x0:         ``(N, nx)`` initial states.
        batch_size: Maximum states per ``fn`` call.

    Returns:
        Merged dict with values concatenated along ``dim=0``.
    """
    N = x0.shape[0]
    results = []
    i = 0
    current_batch = batch_size
    bar = (
        _ProgressBar(total=N, desc=_progress_state["desc"])
        if _progress_state["enabled"]
        else None
    )
    try:
        while i < N:
            chunk = x0[i : i + current_batch]
            try:
                results.append(fn(chunk))
                i += current_batch
                if bar is not None:
                    bar.update(chunk.shape[0])
            except torch.cuda.OutOfMemoryError:
                if current_batch <= 1:
                    raise
                torch.cuda.empty_cache()
                current_batch = max(1, current_batch // 2)
    finally:
        if bar is not None:
            bar.close()
    if len(results) == 1:
        return results[0]
    return {k: torch.cat([r[k] for r in results], dim=0) for k in results[0]}


def score_sequences(
    dynamics: ControlAffine,
    x0: Tensor,
    controls: Tensor,
    dt: float,
    constr_fn,
) -> Tensor:
    """Score a batch of control sequences by min_{k=0..K} constr_fn(x_k).

    Args:
        dynamics:  ControlAffine instance.
        x0:        (N, nx) initial states.
        controls:  (N, S, K, nu) control sequences, where S is the number of
                   sequences per initial state and K is the horizon length.
        dt:        Integration time step.
        constr_fn: (N, nx) -> (N,) or (N, 1) constraint function.

    Returns:
        scores: (N, S) minimum constraint value along each trajectory.
    """
    N, S, K, nu = controls.shape
    nx = x0.shape[1]
    NS = N * S

    x = x0.unsqueeze(1).expand(-1, S, -1).reshape(NS, nx)
    running_min = constr_fn(x).reshape(NS)

    for k in range(K):
        u_k = controls[:, :, k, :].reshape(NS, nu)
        x = dynamics.euler_step(x, u_k, dt)
        running_min = torch.minimum(running_min, constr_fn(x).reshape(NS))

    return running_min.reshape(N, S)


def rollout(
    dynamics: ControlAffine,
    x0: Tensor,
    controls: Tensor,
    dt: float,
) -> Tensor:
    """Roll out a batch of control sequences from x0.

    Args:
        dynamics: ControlAffine instance.
        x0:       (N, nx) initial states.
        controls: (N, K, nu) control sequences.
        dt:       Integration time step.

    Returns:
        traj: (N, K+1, nx) state trajectories including x0.
    """
    traj = [x0]
    x = x0
    _, K, _ = controls.shape
    for k in range(K):
        x = dynamics.euler_step(x, controls[:, k, :], dt)
        traj.append(x)
    return torch.stack(traj, dim=1)
