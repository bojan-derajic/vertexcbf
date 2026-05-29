"""Evaluate a trained VertexCBF model and produce visualizations.

Loads a checkpoint, evaluates the predicted value function on a grid, and
optionally overlays it against a ground-truth reference (if available under
``data/true_values/<dynamics.name>/``).

For systems with more than 2 state dimensions a 2-D slice is visualised:
two dimensions are varied on a grid while all remaining dimensions are held
at fixed values (defaults to the midpoint of each dimension's range).

Usage
-----
    # Checkpoints live under checkpoints/<GROUP>/<system>/, where GROUP is one
    # of NO_DATA, FC_DATA, VRC_DATA (see README.md "Quick Start").
    python scripts/evaluate.py \\
        --config configs/double_integrator_1d.yaml \\
        --checkpoint checkpoints/VRC_DATA/double_integrator_1d/final.pt

    # Save plot to a file instead of displaying:
    python scripts/evaluate.py \\
        --config configs/inverted_pendulum.yaml \\
        --checkpoint checkpoints/VRC_DATA/inverted_pendulum/final.pt \\
        --output figures/inverted_pendulum_eval.png

    # Override evaluation grid resolution:
    python scripts/evaluate.py \\
        --config configs/double_integrator_1d.yaml \\
        --checkpoint checkpoints/VRC_DATA/double_integrator_1d/final.pt \\
        --grid 200 200

    # For a 4-D system: vary dims 0 and 2, fix dim 1 at 0.0 and dim 3 at 1.0
    python scripts/evaluate.py \\
        --config configs/some_4d_system.yaml \\
        --checkpoint checkpoints/VRC_DATA/some_4d_system/final.pt \\
        --slice-axes 0 2 \\
        --slice-fixed 0.0 1.0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml

from vertexcbf.config_utils import build_constr_fn, build_dynamics, build_model


def _load_ground_truth(name: str, base_dir: str = "data/true_values"):
    """Load ground-truth values and grid from disk, if available."""
    path = os.path.join(base_dir, name)
    values_path = os.path.join(path, "values.npy")
    grid_path = os.path.join(path, "grid.npy")
    print(f"Looking for ground-truth data at: {path}")
    if os.path.isfile(values_path) and os.path.isfile(grid_path):
        return np.load(values_path), np.load(grid_path)
    return None, None


def _predict(model, constr_fn, states, device):
    """Evaluate V(x) = h(x) - r(x) for a batch of states."""
    model.eval()
    with torch.no_grad():
        states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        constr = constr_fn(states_t)
        residual = model(states_t)
        return (constr - residual).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained VertexCBF model.")
    parser.add_argument(
        "--config", required=True, help="Path to YAML experiment config."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint .pt file."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save plot to this path (default: figures/<dynamics.name>.png).",
    )
    parser.add_argument(
        "--grid",
        nargs="+",
        type=int,
        default=None,
        help="Override evaluation grid shape, e.g. --grid 200 200.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string.  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--slice-axes",
        nargs=2,
        type=int,
        default=None,
        metavar=("I", "J"),
        help=(
            "Indices of the two state dimensions to vary on the plot grid "
            "(0-indexed).  Defaults to 0 and 1.  Ignored for nx=2 systems."
        ),
    )
    parser.add_argument(
        "--slice-fixed",
        nargs="+",
        type=float,
        default=None,
        metavar="VAL",
        help=(
            "Fixed values for the non-slice state dimensions, listed in "
            "ascending dimension order.  Must supply one value per dimension "
            "not in --slice-axes.  Defaults to the midpoint of each range."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    dynamics = build_dynamics(cfg["system"], device=device)
    constr_fn = build_constr_fn(cfg["constraint"])
    model = build_model(cfg["model"], dynamics, device=device)

    # Load weights
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # Determine grid shape
    grid_shape = args.grid or cfg.get("pde", {}).get("grid_shape", [100, 100])
    grid_shape = tuple(int(g) for g in grid_shape[:2])

    nx = dynamics.nx

    # Determine which two dimensions to slice over
    slice_axes = tuple(args.slice_axes) if args.slice_axes is not None else (0, 1)
    if len(set(slice_axes)) != 2 or any(ax < 0 or ax >= nx for ax in slice_axes):
        raise ValueError(
            f"--slice-axes must be two distinct indices in [0, {nx - 1}], "
            f"got {slice_axes}."
        )

    # Dimensions that are held fixed
    fixed_dims = [d for d in range(nx) if d not in slice_axes]

    # Determine fixed values (default: midpoint of each range)
    x_min_np = dynamics.x_min.cpu().numpy()
    x_max_np = dynamics.x_max.cpu().numpy()
    midpoints = (x_min_np + x_max_np) / 2.0

    if args.slice_fixed is not None:
        if len(args.slice_fixed) != len(fixed_dims):
            raise ValueError(
                f"--slice-fixed expects {len(fixed_dims)} value(s) "
                f"(one per non-slice dimension), got {len(args.slice_fixed)}."
            )
        fixed_values = list(args.slice_fixed)
    else:
        fixed_values = [float(midpoints[d]) for d in fixed_dims]

    # Build 2-D evaluation grid over the two slice dimensions
    ax0, ax1 = slice_axes
    x0 = np.linspace(x_min_np[ax0], x_max_np[ax0], grid_shape[0])
    x1 = np.linspace(x_min_np[ax1], x_max_np[ax1], grid_shape[1])
    g0, g1 = np.meshgrid(x0, x1, indexing="ij")  # (n0, n1)

    n_points = grid_shape[0] * grid_shape[1]
    states_flat = np.tile(midpoints, (n_points, 1))  # start from midpoints
    states_flat[:, ax0] = g0.ravel()
    states_flat[:, ax1] = g1.ravel()
    for i, d in enumerate(fixed_dims):
        states_flat[:, d] = fixed_values[i]

    values_pred = _predict(model, constr_fn, states_flat, device)
    values_pred = values_pred.reshape(grid_shape)

    # Optional ground truth (only used when nx==2 and no custom slice is set)
    load_gt = nx == 2 and args.slice_axes is None
    values_true, _ = _load_ground_truth(dynamics.name) if load_gt else (None, None)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        2 if values_true is not None else 1,
        figsize=(12 if values_true is not None else 6, 5),
    )
    if values_true is None:
        axes = [axes]

    # Build axis label strings and a description of fixed dimensions
    axis_label = [f"x[{ax0}]", f"x[{ax1}]"]
    if fixed_dims:
        fixed_desc = ", ".join(
            f"x[{d}]={v:.3g}" for d, v in zip(fixed_dims, fixed_values)
        )
    else:
        fixed_desc = None

    def _decorate_ax(a, title):
        a.set_title(title)
        a.set_xlabel(axis_label[0])
        a.set_ylabel(axis_label[1])
        a.set_xticks(
            [0, grid_shape[0] - 1],
            [f"{x_min_np[ax0]:.3g}", f"{x_max_np[ax0]:.3g}"],
        )
        a.set_yticks(
            [0, grid_shape[1] - 1],
            [f"{x_max_np[ax1]:.3g}", f"{x_min_np[ax1]:.3g}"],  # rot90 flips y
        )

    # Predicted value function
    ax = axes[0]
    im = ax.imshow(np.rot90(values_pred), aspect="auto")
    if values_true is not None:
        ax.contour(
            np.rot90(values_true),
            levels=[0],
            colors="black",
            linewidths=2,
            linestyles="-",
        )
    ax.contour(
        np.rot90(values_pred),
        levels=[0],
        colors="red",
        linewidths=1.5,
        linestyles="--",
    )
    gt_note = " (red=pred, black=GT)" if values_true is not None else ""
    _decorate_ax(ax, f"Predicted V(x){gt_note}")
    plt.colorbar(im, ax=ax)

    # Ground truth (if available)
    if values_true is not None:
        ax2 = axes[1]
        im2 = ax2.imshow(np.rot90(values_true), aspect="auto")
        ax2.contour(
            np.rot90(values_true),
            levels=[0],
            colors="black",
            linewidths=2,
            linestyles="-",
        )
        _decorate_ax(ax2, "Ground-truth V(x)")
        plt.colorbar(im2, ax=ax2)

    suptitle = dynamics.name
    if fixed_desc:
        suptitle += f"\n[{fixed_desc}]"
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()

    output_path = args.output or os.path.join("figures", f"{dynamics.name}.pdf")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # -----------------------------------------------------------------------
    # Print basic metrics (if ground truth available)
    # -----------------------------------------------------------------------
    if values_true is not None:
        # Ensure shapes match
        if values_true.shape != values_pred.shape:
            print(
                "Warning: ground-truth shape differs from predicted shape — "
                "skipping numerical metrics."
            )
        else:
            mse = float(np.mean((values_pred - values_true) ** 2))
            # Safe-set agreement: fraction of states where sign matches
            agreement = float(np.mean(np.sign(values_pred) == np.sign(values_true)))
            print(f"\nMSE vs ground truth:      {mse:.6f}")
            print(f"Safe-set sign agreement:  {agreement * 100:.2f}%")


if __name__ == "__main__":
    main()
