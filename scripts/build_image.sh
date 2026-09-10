#!/usr/bin/env bash
# Build the 'vertexcbf:latest' image used by `train_all.sh --docker`.
#
# The image carries only the dependencies (see requirements.txt) — the repo
# itself is bind-mounted at /workspace at run time, so source edits take
# effect without a rebuild. Rebuild only when requirements.txt or the
# Dockerfile changes.
#
# The container user 'dev' is built with your host UID/GID so files written
# to the bind mount (checkpoints/, data/precomputed/, logs/) stay owned by
# you instead of by root.
#
# Usage:
#   ./scripts/build_image.sh                  # build vertexcbf:latest
#   ./scripts/build_image.sh --no-cache       # ignore layer cache
#   ./scripts/build_image.sh --pull           # refresh the base image first
#   ./scripts/build_image.sh --tag foo:v2     # build under another tag
#   ./scripts/build_image.sh --skip-verify    # build only, no smoke test
#
# After a successful build:
#   ./scripts/train_all.sh --docker

# Guard against being sourced (would kill your shell on exit).
(return 0 2>/dev/null) && { echo "Run, don't source: bash $0 [opts]"; return 1; }

set -u

TAG="vertexcbf:latest"
BUILD_FLAGS=()
VERIFY=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)         TAG="$2"; shift 2 ;;
    --tag=*)       TAG="${1#*=}"; shift ;;
    --no-cache)    BUILD_FLAGS+=(--no-cache); shift ;;
    --pull)        BUILD_FLAGS+=(--pull); shift ;;
    --skip-verify) VERIFY=0; shift ;;
    -h|--help)     sed -n '2,21p' "$0"; exit 0 ;;
    *)             echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "$0")/.." && pwd)"

command -v docker >/dev/null || { echo "docker not found on PATH" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "cannot talk to the docker daemon (is it running? are you in the 'docker' group?)" >&2; exit 1; }

echo "=== [$(date '+%F %T')] building ${TAG} (uid=$(id -u) gid=$(id -g)) ==="
docker build \
  ${BUILD_FLAGS[@]+"${BUILD_FLAGS[@]}"} \
  --build-arg "USER_UID=$(id -u)" \
  --build-arg "USER_GID=$(id -g)" \
  -t "$TAG" \
  -f "$REPO/Dockerfile" \
  "$REPO" || { echo "!!! build FAILED (exit $?)" >&2; exit 1; }

if (( VERIFY )); then
  echo "=== [$(date '+%F %T')] verifying ${TAG} ==="

  # Mount the repo exactly as train_all.sh does, so this exercises the real
  # import path (vertexcbf off PYTHONPATH, third-party deps from the image).
  docker run --rm \
    -e PYTHONPATH=/workspace \
    -v "$REPO":/workspace -w /workspace \
    "$TAG" python -c '
import matplotlib, numpy, torch, yaml
import vertexcbf
print(f"  torch  {torch.__version__} (built for cuda {torch.version.cuda})")
print(f"  numpy  {numpy.__version__}")
print(f"  yaml   {yaml.__version__}")
print(f"  vertexcbf imports OK from the bind mount")
' || { echo "!!! smoke test FAILED — the image is missing a dependency" >&2; exit 1; }

  # GPU check is advisory: a CPU-only host is a legitimate --device cpu setup.
  if command -v nvidia-smi >/dev/null 2>&1; then
    docker run --rm --gpus all "$TAG" \
      python -c 'import torch; assert torch.cuda.is_available(); print(f"  cuda   {torch.cuda.device_count()} device(s) visible: " + ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))' \
      || echo "  WARNING: no CUDA inside the container — 'train_all.sh --docker --device cuda' will fail. Check the NVIDIA Container Toolkit."
  else
    echo "  (no nvidia-smi on the host — skipping GPU check; use --device cpu)"
  fi
fi

echo "=== [$(date '+%F %T')] ${TAG} ready — run: ./scripts/train_all.sh --docker ==="
