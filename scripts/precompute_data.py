"""Precompute supervision data via vertex-restricted tree search and cache to disk.

Generating the supervision labels is the most expensive step and rarely needs
to be repeated.  This script runs it once and saves the result so that
``train.py`` can load it instantly.

The label-generation method is controlled by ``data.method`` in the config
(default: ``beam_search``).  See ``configs/template.yaml`` for all options.

Output is auto-routed into a per-method-group sub-folder so the three paper
conditions stay separated on disk: ``data/precomputed/<GROUP>/<system>.pt``,
where ``GROUP`` is one of ``FC_DATA`` (full-control sampling MPC: ``mppi``)
or ``VRC_DATA`` ("vertex-restricted control": ``beam_search`` /
``stochastic_beam_search`` / ``branch_and_bound``).  ``NO_DATA`` configs are
skipped because supervision data is not generated when ``data.enabled: false``.

Usage
-----
    python scripts/precompute_data.py --config configs/double_integrator_1d.yaml

    # Override cache path:
    python scripts/precompute_data.py --config configs/inverted_pendulum.yaml \\
        --output data/precomputed/VRC_DATA/inverted_pendulum.pt
"""

from __future__ import annotations

import argparse
import os

import torch
import yaml

from vertexcbf.config_utils import (
    build_constr_fn,
    build_dynamics,
    build_trajopt_runner,
    method_group,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute beam-search supervision data for VertexCBF."
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML experiment config."
    )
    parser.add_argument(
        "--output",
        default=None,
        help=("Override output .pt path.  Defaults to data.cache_path from config."),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string, e.g. 'cuda' or 'cpu'.  Auto-detected if omitted.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    if not data_cfg.get("enabled", True):
        print("data.enabled is false in config — nothing to precompute.")
        return

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Build components
    dynamics = build_dynamics(cfg["system"], device=device)

    # Determine output path
    group = method_group(data_cfg)
    out_path = (
        args.output
        or data_cfg.get("cache_path")
        or f"data/precomputed/{group}/{dynamics.name}.pt"
    )
    print(f"Method group: {group}")
    constr_fn = build_constr_fn(cfg["constraint"])

    # Build states
    method = data_cfg.get("method", "beam_search")
    sampling = data_cfg.get("sampling", "grid")
    if sampling == "random":
        num_samples = data_cfg["num_samples"]
        states = dynamics.get_uniform_state_samples(
            num_samples=num_samples, requires_grad=False
        )
        print(
            f"Running {method} on {num_samples} random samples "
            f"(B={data_cfg['B']}, K={data_cfg['K']}, dt={data_cfg['dt']})..."
        )
        grid_shape = None
    else:
        grid_shape = data_cfg["grid_shape"]
        states = dynamics.get_uniform_state_grid(
            grid_shape=grid_shape, requires_grad=False
        ).reshape(-1, dynamics.nx)
        print(
            f"Running {method} on {states.shape[0]} states "
            f"(grid {grid_shape}, B={data_cfg['B']}, K={data_cfg['K']}, dt={data_cfg['dt']})..."
        )

    run_trajopt = build_trajopt_runner(data_cfg, dynamics, constr_fn)
    values = run_trajopt(states)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(
        {
            "states": states.cpu(),
            "values": values.cpu(),
            "grid_shape": grid_shape,
            "config": {
                "system": cfg["system"],
                "constraint": cfg["constraint"],
                "data": data_cfg,
            },
        },
        out_path,
    )
    print(f"Saved precomputed data to: {out_path}")


if __name__ == "__main__":
    main()
