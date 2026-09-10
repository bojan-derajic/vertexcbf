#!/usr/bin/env bash
# Train (or validate) every system sequentially.
#
# By default runs `python scripts/train.py` directly — intended for use
# from inside a devcontainer where the env is already set up. Pass
# --docker to spin up the 'vertexcbf' container per system instead.
#
# The run is re-exec'd detached so it survives SSH disconnect; your
# terminal tails the log live. Ctrl-C only stops the tail.
#
# Usage:
#   ./scripts/train_all.sh                            # python, device=cuda
#   ./scripts/train_all.sh --device cuda:1
#   ./scripts/train_all.sh --device cpu
#   ./scripts/train_all.sh --docker                   # run inside docker
#   ./scripts/train_all.sh --validate-only            # skip training, validate only
#   ./scripts/train_all.sh --reuse-data               # reuse cached supervision data
#   ./scripts/train_all.sh --docker --validate-only --device cuda
#   ./scripts/train_all.sh --n-seeds 5                # train each system with
#                                                      # seeds 0..4 (5 experiments per
#                                                      # system); pair with --reuse-data
#                                                      # so all seeds share one supervision
#                                                      # dataset. Each seed's artifacts land
#                                                      # in checkpoints/<GROUP>/<system>/seed_<n>/
#
# --config-dir points at an alternate configs/ directory (e.g. a scratch copy
# with different hyperparameters) if you don't want to edit configs/ in place.
#
# Skip a system by commenting out its line below.

# Guard against being sourced (would kill your shell on exit).
(return 0 2>/dev/null) && { echo "Run, don't source: bash $0 [opts]"; return 1; }

set -u

# Preserve original args for the detached re-exec below (the loop shifts $@).
ORIG_ARGS=("$@")

DEVICE="cuda"
USE_DOCKER=0
VALIDATE_ONLY=0
REUSE_DATA=0
CONFIG_DIR="configs"
N_SEEDS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)        DEVICE="$2"; shift 2 ;;
    --device=*)      DEVICE="${1#*=}"; shift ;;
    --config-dir)    CONFIG_DIR="${2%/}"; shift 2 ;;
    --config-dir=*)  CONFIG_DIR="${1#*=}"; CONFIG_DIR="${CONFIG_DIR%/}"; shift ;;
    --docker)        USE_DOCKER=1; shift ;;
    --validate-only) VALIDATE_ONLY=1; shift ;;
    --reuse-data)    REUSE_DATA=1; shift ;;
    --n-seeds)       N_SEEDS="$2"; shift 2 ;;
    --n-seeds=*)     N_SEEDS="${1#*=}"; shift ;;
    -h|--help)       sed -n '2,29p' "$0"; exit 0 ;;
    *)               echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if ! [[ "$N_SEEDS" =~ ^[0-9]+$ ]] || (( N_SEEDS < 1 )); then
  echo "--n-seeds must be a positive integer, got: $N_SEEDS" >&2
  exit 2
fi

IMAGE="vertexcbf:latest"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GPU_FLAG=()
[[ "$DEVICE" == cuda* ]] && GPU_FLAG=(--gpus all)

EXTRA_ARGS=()
(( VALIDATE_ONLY )) && EXTRA_ARGS+=(--validate-only)
(( REUSE_DATA ))    && EXTRA_ARGS+=(--reuse-data)

# Ordered by nx (appendix order).
SYSTEMS=(
  inverted_pendulum       # nx=2
  double_integrator_1d    # nx=2
  vertical_drone_2d       # nx=2
  dubins_car              # nx=3
  double_integrator_2d    # nx=4
  kinematic_bicycle       # nx=4
  cart_pole               # nx=4
  dynamic_unicycle        # nx=5
  relative_unicycle       # nx=5
  double_integrator_3d    # nx=6
  manipulator_3dof        # nx=6
  landing_rocket          # nx=7
  quadruped_trunk         # nx=9
  auv_6dof                # nx=12
  quadrotor               # nx=13
)

# Re-exec detached so the run survives SSH disconnect.
if [ -z "${_DETACHED:-}" ]; then
  mkdir -p logs
  TAG=$( (( VALIDATE_ONLY )) && echo validate_all || echo train_all )
  # Keep alternate-config-dir runs' logs in their own file.
  [[ "$CONFIG_DIR" != "configs" ]] && TAG="${TAG}_$(basename "$CONFIG_DIR")"
  LOG="logs/${TAG}_$(date +%Y%m%d_%H%M%S).log"
  touch "$LOG"
  _DETACHED=1 setsid nohup "$0" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"} >"$LOG" 2>&1 </dev/null &
  BG=$!
  echo "${TAG} PID $BG   log: $LOG"
  echo "Ctrl-C detaches your view; the run keeps going. Resume: tail -F $LOG"
  exec tail -F --pid="$BG" "$LOG"
fi

ACTION=$( (( VALIDATE_ONLY )) && echo validate || echo train )

if (( USE_DOCKER )); then
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "!!! image ${IMAGE} not found — build it first: ./scripts/build_image.sh" >&2
    exit 1
  fi
  NAME="vertexcbf_${ACTION}_$(date +%Y%m%d_%H%M%S)"
  echo "=== [$(date '+%F %T')] ${ACTION} all systems in container ${NAME} (device=${DEVICE}, configs=${CONFIG_DIR}, n_seeds=${N_SEEDS}) ==="
  docker run --rm --name "$NAME" "${GPU_FLAG[@]}" \
    -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace \
    -e DEVICE="$DEVICE" -e ACTION="$ACTION" -e CONFIG_DIR="$CONFIG_DIR" \
    -e VALIDATE_ONLY="$VALIDATE_ONLY" -e REUSE_DATA="$REUSE_DATA" -e N_SEEDS="$N_SEEDS" \
    -v "$REPO":/workspace -w /workspace \
    "$IMAGE" bash -c '
      set -u
      extra=()
      (( VALIDATE_ONLY )) && extra+=(--validate-only)
      (( REUSE_DATA ))    && extra+=(--reuse-data)
      for s in "$@"; do
        for (( seed=0; seed<N_SEEDS; seed++ )); do
          seed_extra=("${extra[@]}")
          (( N_SEEDS > 1 )) && seed_extra+=(--seed "$seed")
          echo "=== [$(date "+%F %T")] ${ACTION} ${s} seed=${seed} (device=${DEVICE}) ==="
          python scripts/train.py --config "${CONFIG_DIR}/${s}.yaml" --device "$DEVICE" "${seed_extra[@]}" \
            || echo "!!! $s seed=$seed FAILED (exit $?)"
        done
      done
    ' _ "${SYSTEMS[@]}"
else
  for s in "${SYSTEMS[@]}"; do
    for (( seed=0; seed<N_SEEDS; seed++ )); do
      seed_extra=("${EXTRA_ARGS[@]}")
      (( N_SEEDS > 1 )) && seed_extra+=(--seed "$seed")
      echo "=== [$(date '+%F %T')] ${ACTION} ${s} seed=${seed} (device=${DEVICE}, mode=python, configs=${CONFIG_DIR}) ==="
      PYTHONUNBUFFERED=1 PYTHONPATH="$REPO" \
        python scripts/train.py --config "${CONFIG_DIR}/${s}.yaml" --device "$DEVICE" "${seed_extra[@]}" \
        || echo "!!! $s seed=$seed FAILED (exit $?)"
    done
  done
fi
echo "=== [$(date '+%F %T')] all done ==="
