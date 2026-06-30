#!/usr/bin/env bash
# Run YOLOX-X 2D object detection on your own images inside the Docker env.
#
# Usage:
#   docker/run_demo.sh <IMAGES_DIR_OR_FILE> [extra demo.py args...]
#
# Example:
#   docker/run_demo.sh ~/my_photos --conf 0.4
#
# Results are written to YOLOX_outputs/yolox_x/vis_res/<timestamp>/ in the repo
# (annotated images + per-image txt). The repo is bind-mounted, so they appear on the host.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGES="${1:?Usage: run_demo.sh <images_dir_or_file> [extra args]}"
shift || true
IMAGES="$(cd "$(dirname "$IMAGES")" && pwd)/$(basename "$IMAGES")"  # absolutize

IMAGE_TAG="edgeai-yolox-infer"
CKPT="pretrained_models/yolox_x.pth"
EXP="exps/default/yolox_x.py"

# Use sudo for docker unless the user is in the docker group.
DOCKER="docker"
if ! docker info >/dev/null 2>&1; then DOCKER="sudo docker"; fi

# Mount a directory either way (demo.py walks a dir or takes a single file).
if [ -d "$IMAGES" ]; then
  HOST_MOUNT="$IMAGES"; CONT_PATH="/data/images"
else
  HOST_MOUNT="$(dirname "$IMAGES")"; CONT_PATH="/data/images/$(basename "$IMAGES")"
fi

exec $DOCKER run --rm --gpus all \
  -v "$REPO":/workspace/edgeai-yolox \
  -v "$HOST_MOUNT":/data/images:ro \
  "$IMAGE_TAG" \
  python3 tools/demo.py image \
    -f "$EXP" \
    -c "$CKPT" \
    --path "$CONT_PATH" \
    --dataset coco --task 2dod \
    --device gpu --conf 0.3 --nms 0.45 \
    --save_result "$@"
