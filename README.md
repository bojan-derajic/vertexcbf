# VertexCBF

**Learning neural Control Barrier Functions via vertex-restricted control search.**
PyTorch implementation of the method from

> B. Derajić, S. Bernhard, W. Hönig. *VertexCBF: Improving Neural Control Barrier Functions via Vertex-Restricted Control Search.* CoRL 2026.

VertexCBF learns a neural approximation `V_Θ(x)` of the stationary Hamilton–Jacobi safety
value function of a control-affine system and uses it as a CBF. Three ingredients:

- **Residual parametrisation.** `V_Θ(x) = c(x) − r_Θ(x)` with `r_Θ ≥ 0`, where `c(x)` is the
  user-specified constraint (`{c ≥ 0}` is the allowed set). The learned safe set can never
  exceed the constraint set, by construction.
- **Physics-informed + sparsely supervised training.** The loss combines the stationary
  HJB variational inequality on collocation states with a regression onto reachability
  labels at a small set of states.
- **Vertex-restricted label generation.** For control-affine dynamics and a box control set
  the Hamiltonian is maximised at a control *vertex*, so labels are computed by a GPU-parallel
  tree search over vertex sequences — no PDE solver, no continuous trajectory optimisation.

The repo ships the library, YAML configs for the 15 systems of the paper, CLI scripts for
training / evaluation, and a tutorial notebook. Full-control sampling-based baselines
(MPPI, CEM, iCEM, random shooting) are included for comparison.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Scripts](#scripts)
- [Validation metrics](#validation-metrics)
- [Available systems](#available-systems)
- [Constraint functions](#constraint-functions)
- [Adding a system or constraint](#adding-a-system-or-constraint)
- [Citation](#citation)

---

## Installation

Requires Python ≥ 3.9 and PyTorch ≥ 2.0 (CUDA strongly recommended for training).

```bash
git clone https://github.com/bojan-derajic/vertexcbf.git
cd vertexcbf
pip install -e .            # library + scripts
pip install -e ".[dev]"     # + Jupyter, for the notebook
```

`requirements.txt` pins an exact environment (CUDA 12.4 build of PyTorch). A `Dockerfile` and
`.devcontainer/` are provided; `scripts/build_image.sh` builds the image used by
`scripts/train_all.sh --docker`.

---

## Quick start

### From the command line

```bash
# 1. Train: generates (and caches) the supervision labels, trains, validates.
python scripts/train.py --config configs/inverted_pendulum.yaml

# 2. Plot a 2-D slice of the learned CBF.
python scripts/evaluate.py \
    --config configs/inverted_pendulum.yaml \
    --checkpoint checkpoints/VRC_DATA/inverted_pendulum/final.pt

# 3. Your own system: copy the fully commented template and edit it.
cp configs/template.yaml configs/my_system.yaml
```

Artifacts land in `checkpoints/<GROUP>/<system>/` (`final.pt`, `validation.{pt,json}`,
`timing.json`), cached labels in `data/precomputed/<GROUP>/<system>.pt`, figures in
`figures/<GROUP>/`. `GROUP` is derived from the `data` section of the config:

| `GROUP`    | Labels come from                                                         | `data.method`                                                              |
|------------|--------------------------------------------------------------------------|----------------------------------------------------------------------------|
| `VRC_DATA` | **vertex-restricted search** (the method of the paper)                   | `beam_search`, `stochastic_beam_search`, `branch_and_bound`, `cem_discrete` |
| `FC_DATA`  | full-control sampling-based MPC baselines                                | `mppi`, `cem`, `icem`, `random_shooting`                                   |
| `NO_DATA`  | no labels — PDE loss only                                                | `data.enabled: false` or `--no-data`                                       |

### From Python

```python
import torch
from functools import partial
from vertexcbf import (DoubleIntegrator1D, interval_sdf, MLP, Trainer, beam_search,
                       validate_cbf, stratified_sample_by_predicted_cbf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dynamics = DoubleIntegrator1D(x_min=[-1.5, -1.5], x_max=[1.5, 1.5],
                              u_min=[-0.5], u_max=[0.5], device=device)
constr_fn = partial(interval_sdf, center=0.0, d=1.0)          # allowed set {|p| <= 1}

# Supervision labels from a vertex-restricted beam search
states_d = dynamics.get_uniform_state_grid([41, 41]).reshape(-1, 2)
values_d = beam_search(dynamics, states_d, B=100, K=40, dt=0.1, constr_fn=constr_fn)["values"]

# Residual network r_Θ >= 0  (softplus output)
model = MLP(layers_config=[(2, None), (32, "sin"), (32, "sin"), (32, "sin"),
                           (1, ("softplus", {"beta": 1.0}))],
            input_min=dynamics.x_min, input_max=dynamics.x_max,
            periodic_inputs=dynamics.periodic_states).to(device)

Trainer(dynamics=dynamics, model=model, constr_fn=constr_fn,
        epochs=3000, lr=1e-3, lr_milestones=[2000],
        pde_grid_shape=(100, 100), pde_weight_mode="fixed", pde_weight=0.9,
        data_states=states_d, data_values=values_d).train()

V = lambda x: constr_fn(x) - model(x)                          # the CBF, (N, nx) -> (N, 1)

# Closed-loop check of the certificate
samples = stratified_sample_by_predicted_cbf(dynamics, constr_fn, model, num_per_stratum=2000)
print(validate_cbf(dynamics, samples["states"], constr_fn, model, T=5.0, dt=0.01)["stratified_metrics"])
```

**[`notebooks/examples.ipynb`](notebooks/examples.ipynb)** walks through this end to end with
plots, and additionally shows how to use the learned CBF in a safety filter, how to load
checkpoints produced by the scripts, and how to plug in your own system and constraint.

---

## Repository layout

```
vertexcbf/
├── vertexcbf/
│   ├── dynamics/        ControlAffine base class + 15 systems
│   ├── constraints/     constraint / signed-distance functions c(x)
│   ├── trajopt/         label generation: vertex-restricted searches + full-control baselines
│   ├── models.py        MLP with input rescaling and (cos, sin) encoding of periodic states
│   ├── losses.py        pde_loss (stationary HJB-VI) and data_loss
│   ├── trainer.py       Trainer: loss weighting, LR schedule, checkpointing, auto-batching
│   ├── validation.py    closed-loop validation + stratified sampling
│   └── config_utils.py  build_* helpers and registries for the YAML configs
├── configs/             one YAML per system (paper hyperparameters) + template.yaml
├── scripts/             train.py, evaluate.py, precompute_data.py, train_all.sh, build_image.sh
└── notebooks/           examples.ipynb (tutorial)
```

All methods in `trajopt/` solve the same finite-horizon problem
`max_{u_0..u_{K-1}} min_k c(x_k)` for a batch of initial states; the vertex-restricted ones
search over the `2^nu` control vertices, the baselines sample the full control box.

---

## Configuration

One YAML file describes an experiment; [`configs/template.yaml`](configs/template.yaml) documents
every key. The sections:

**`system`** — class name from `vertexcbf.dynamics`, state box `x_min`/`x_max`, control box
`u_min`/`u_max`, and `params` forwarded to the constructor.

**`constraint`** — `type` (a name from the [table below](#constraint-functions)) and its `params`.

**`data`** — supervision labels.
```yaml
data:
  enabled: true
  sampling: grid           # grid | random
  grid_shape: [60, 60]     # or num_samples: N for random
  method: beam_search      # see the table below
  B: 500                   # budget: beam width / number of sampled trajectories
  K: 20                    # horizon (steps)
  dt: 0.1                  # Euler step of the search rollouts
  method_params: {}        # method-specific kwargs
```

| `method`                 | Search space      | `method_params`                          |
|--------------------------|-------------------|------------------------------------------|
| `beam_search`            | control vertices  | —                                        |
| `stochastic_beam_search` | control vertices  | `strategy`, `temperature`, `epsilon`     |
| `branch_and_bound`       | control vertices  | `n_restarts`, `tie_noise`                |
| `cem_discrete`           | control vertices  | `n_iter`, `elite_frac`                   |
| `mppi`                   | full control box  | `sigma`, `lam`, `n_iter`                 |
| `cem`                    | full control box  | `n_iter`, `elite_frac`                   |
| `icem`                   | full control box  | `n_iter`, `elite_frac`, `noise_beta`     |
| `random_shooting`        | full control box  | —                                        |

**`pde`** — collocation states for the HJB-VI residual: `sampling: grid` with `grid_shape`,
or `sampling: random` with `num_samples`.

**`model`** — `layers` is a list of `[width, activation]`; the first entry is `[nx, null]`, the
last should be `[1, ["softplus", {beta: ...}]]` so that `r_Θ ≥ 0`. Activations: `linear`,
`relu`, `elu`, `selu`, `softplus`, `sigmoid`, `tanh`, `sin`. Periodic states are encoded as
`(cos, sin)` pairs automatically.

**`training`**
```yaml
training:
  epochs: 10000
  lr: 0.001
  lr_milestones: [7000]     # lr *= lr_gamma at each milestone
  lr_gamma: 0.1
  pde_weight_mode: fixed    # fixed | normalized | scheduled
  pde_weight: 0.3           # loss = w * pde_loss + (1 - w) * data_loss
  # seed: 0                 # reproducible init; also settable with --seed
  # collapse_patience: 1000 # optional early stop, see below
```
`normalized` divides each loss by its epoch-0 value instead of using `pde_weight`;
`scheduled` steps `w` at `pde_weight_milestones: [[epoch, w], ...]`.

Optional early stopping fires only when the certificate has become *trivial* — an empty
predicted safe set (`max V_Θ < collapse_max_value`) or a `V_Θ` that is constant in `x`
(spread `< collapse_const_tol`) — and has stayed so for `collapse_patience` epochs. It never
reacts to a slow loss; it is off unless `collapse_patience > 0`, and `--no-early-stop` forces it
off. `final.pt` is written either way; `timing.json` records `training_epochs` and `early_stopped`.

**`validation`** — closed-loop check run at the end of `train.py`:
```yaml
validation:
  enabled: true
  T: 5.0                   # rollout horizon [s]
  dt: 0.01                 # Euler step [s]
  sampling: stratified     # stratified (recommended) | random | grid
  num_per_stratum: 10000   # initial states per predicted class ({V > 0}, {V <= 0})
  # save_trajectories: false
```

**`output`** — `checkpoint_dir` (default `checkpoints/<GROUP>/<system>`, plus `/seed_<n>` when a
seed is set), `checkpoint_every`, `log_every`. Checkpoints hold model, optimizer, scheduler and
loss history, so training can be resumed.

---

## Scripts

```
python scripts/train.py --config <yaml> [--reuse-data] [--no-data] [--seed N]
                        [--resume <ckpt>] [--validate-only] [--no-early-stop] [--device <str>]

python scripts/evaluate.py --config <yaml> --checkpoint <ckpt>
                           [--slice-axes I J] [--slice-fixed VAL ...] [--grid N N] [--output <pdf>]

python scripts/precompute_data.py --config <yaml> [--output <path>] [--device <str>]

./scripts/train_all.sh [--device cuda:0] [--docker] [--reuse-data] [--validate-only]
                       [--n-seeds 5] [--config-dir <dir>]
```

- **`train.py`** runs the full pipeline. Labels are regenerated on every run unless
  `--reuse-data` is given (then the cache at `data/precomputed/<GROUP>/<system>.pt` is loaded).
  `--seed` fixes the RNGs and routes artifacts to `<checkpoint_dir>/seed_<n>/`, so several
  seeds can share one label cache.
- **`evaluate.py`** plots `V_Θ` on a 2-D slice of the state space (`--slice-axes` picks the two
  varying dimensions, `--slice-fixed` pins the rest; defaults to midpoints). If a reference
  solution is present at `data/true_values/<system>/{grid,values}.npy`, it is overlaid and MSE /
  sign agreement are printed.
- **`precompute_data.py`** runs only the label generation, e.g. to do it on another machine.
- **`train_all.sh`** trains every system in `configs/` sequentially (detached, logging to
  `logs/`), optionally in the Docker image built by `build_image.sh`.

---

## Validation metrics

`validate_cbf` rolls out the Hamiltonian-greedy policy
`u*(x) = argmax_u ∇V_Θ(x)·(f(x) + g(x)u)` — always a control vertex — from a batch of initial
states and compares the prediction `V_Θ(x_0) > 0` with the outcome `min_k c(x_k) > 0`. It reports

- `predicted_safe.false_safe_rate` — `P(violation | V_Θ(x_0) > 0)`, the safety-critical number;
- `predicted_unsafe.false_unsafe_rate` — `P(safe | V_Θ(x_0) ≤ 0)`, how conservative the CBF is;
- `volume_metrics` — the predicted and validated safe volumes as fractions of the state box.

Both rates are conditional on the predicted class, so they do not depend on how large the
safe set is. `stratified_sample_by_predicted_cbf` draws equally many initial states from
`{V_Θ > 0}` and `{V_Θ ≤ 0}` so that both are estimated tightly. `train.py` writes the result to
`validation.json` (metrics only) and `validation.pt` (plus per-state values and, optionally,
trajectories).

---

## Available systems

| Class                | `nx` | `nu` | State                              | Control            |
|----------------------|:----:|:----:|------------------------------------|--------------------|
| `DoubleIntegrator1D` |  2   |  1   | `[p, v]`                           | `[a]`              |
| `InvertedPendulum`   |  2   |  1   | `[θ, ω]`                           | `[τ]`              |
| `VerticalDrone2D`    |  2   |  1   | `[z, vz]`                          | `[az]`             |
| `DubinsCar`          |  3   |  1   | `[px, py, θ]`                      | `[ω]`              |
| `DoubleIntegrator2D` |  4   |  2   | `[px, py, vx, vy]`                 | `[ax, ay]`         |
| `KinematicBicycle`   |  4   |  2   | `[px, py, ψ, v]`                   | `[δ, a]`           |
| `CartPole`           |  4   |  1   | `[x, θ, v, ω]`                     | `[f]`              |
| `DynamicUnicycle`    |  5   |  2   | `[px, py, θ, v, ω]`                | `[a, α]`           |
| `RelativeUnicycle`   |  5   |  2   | `[px, py, ψ, vr, vp]`              | `[a, ω]`           |
| `DoubleIntegrator3D` |  6   |  3   | `[px, py, pz, vx, vy, vz]`         | `[ax, ay, az]`     |
| `Manipulator3DOF`    |  6   |  3   | `[q₁, q₂, q₃, q̇₁, q̇₂, q̇₃]`        | `[τ₁, τ₂, τ₃]`     |
| `LandingRocket`      |  7   |  2   | `[px, pz, vx, vz, θ, ω, m]`        | `[T, τ]`           |
| `QuadrupedTrunk`     |  9   |  4   | `[pz, φ, θ, vx, vy, vz, p, q, r]`  | contact forces     |
| `AUV6DoF`            | 12   |  6   | 6-DoF pose + body-frame velocities | `[F, τ]`           |
| `Quadrotor`          | 13   |  4   | position + quaternion + velocities | `[F, αx, αy, αz]`  |

Each class documents its exact state/control layout and parameters in its docstring
(`vertexcbf/dynamics/`); the matching paper hyperparameters are in `configs/<system>.yaml`.

---

## Constraint functions

`c(x)` is positive inside the allowed set. Available in `vertexcbf.constraints` and, by name,
in YAML configs:

| Name                     | Geometry                                       | Key params                         |
|--------------------------|------------------------------------------------|------------------------------------|
| `interval_sdf`           | interval on the first state                    | `center`, `d`                      |
| `circle_sdf`             | circular obstacle in `(x₀, x₁)`                | `center`, `radius`                 |
| `rectangle_sdf`          | axis-aligned box in `(x₀, x₁)`                 | `center`, `a`, `b`                 |
| `cylinder_sdf`           | infinite cylinder in `(x₀, x₁, x₂)`            | `center`, `direction`, `radius`    |
| `ball_3d_sdf`            | ball in `(x₀, x₁, x₂)`                         | `center`, `radius`                 |
| `two_disk_sdf`           | two-disk collision in the robot body frame     | `robot_radius`, `obstacle_radius`  |
| `state_limits_sdf`       | soft-min over per-dimension box limits         | `limits`, `alpha`                  |
| `landing_funnel_sdf`     | landing funnel for the rocket                  | `px_pad`, `slope`, `vel_weight`, … |
| `ee_sphere_sdf`          | manipulator end-effector vs. sphere            | `l1`, `l2`, `center`, `radius`     |
| `manipulator_sphere_sdf` | whole manipulator arm vs. sphere               | `l1`, `l2`, `center`, `radius`, …  |
| `composed_sdf`           | soft-min composition of several constraints    | `sdfs`, `alpha`                    |

---

## Adding a system or constraint

**System** — subclass `ControlAffine` in `vertexcbf/dynamics/`, export it from
`vertexcbf/dynamics/__init__.py`, and add it to `DYNAMICS_REGISTRY` in `vertexcbf/config_utils.py`
to make it available in YAML:

```python
from vertexcbf.dynamics.control_affine import ControlAffine

class MySystem(ControlAffine):
    name = "my_system"
    nx, nu = 2, 1
    periodic_states = []   # indices of angular states (wrapped, (cos, sin)-encoded)
    clamp_states = []      # indices clamped to [x_min, x_max] after each Euler step

    def f(self, x): ...    # (N, nx) -> (N, nx, 1)
    def g(self, x): ...    # (N, nx) -> (N, nx, nu)
```

**Constraint** — any function `(..., nx) -> (..., 1)`, positive inside the allowed set. Put it in
`vertexcbf/constraints/`, export it from `vertexcbf/constraints/__init__.py`, and add it to
`CONSTR_REGISTRY` in `vertexcbf/config_utils.py`. Section 8 of the notebook shows both, inline.

---

## Citation

```bibtex
@inproceedings{derajic2026vertexcbf,
  title     = {{VertexCBF}: Improving Neural Control Barrier Functions via Vertex-Restricted Control Search},
  author    = {Deraji{\'c}, Bojan and Bernhard, Sebastian and H{\"o}nig, Wolfgang},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```
