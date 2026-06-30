#!/usr/bin/env bash
# Run YOLOX-6D-Pose (object_pose) on a folder of images via the end-to-end ONNX model.
# No training, runs on CPU. Camera intrinsics are baked into the ONNX model (YCB-V).
#
# Usage:
#   ./run_pose.sh <image-folder-or-default-sample> [extra args to onnx_inference_object_pose.py]
#
# Notes:
#   * The script only processes *.png files and resizes each to 640x480.
#   * Recognised objects are the 21 YCB-V objects only (no retraining => fixed object set).
#   * Intrinsics are YCB-V's; translation/depth is only metrically valid for YCB-V-like
#     captures. 2D box + rotation are still meaningful on other captures.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MDIR="$REPO/pretrained_models/yolox_l_object_pose_ti_lite"
CAMERA="$REPO/camera_real.json"
# Prefer the model patched with your real intrinsics; fall back to the stock YCB-V model.
if [ -f "$MDIR/yolox_l_object_pose_ti_lite_realcam.onnx" ]; then
  MODEL="$MDIR/yolox_l_object_pose_ti_lite_realcam.onnx"
  CAM_ARG=(--camera "$CAMERA")
else
  MODEL="$MDIR/yolox_l_object_pose_ti_lite.onnx"
  CAM_ARG=()
fi
VENV="$REPO/.venv_pose_onnx"
INPUT="${1:-$REPO/sample_pose_input}"
shift || true
OUTDIR="$REPO/YOLOX_outputs/object_pose_onnx"

[ -f "$MODEL" ] || { echo "Missing model: $MODEL"; exit 1; }
[ -f "${MODEL%.onnx}.prototxt" ] || { echo "Missing prototxt next to model"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "Model: $(basename "$MODEL")"
cd "$REPO/demo/ONNXRuntime"
python onnx_inference_object_pose.py \
  --model "$MODEL" \
  --image-folder "$INPUT" \
  --output-dir "$OUTDIR" \
  --dataset ycbv --save-txt "${CAM_ARG[@]}" "$@"

echo "Done. Overlays: $OUTDIR/  (2D boxes in $OUTDIR/bbox/, raw poses in *.txt)"
