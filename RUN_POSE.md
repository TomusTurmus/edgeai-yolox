# Running YOLOX-6D-Pose (object pose) on your own images — no retraining

Runs the end-to-end **YOLOX-6D-Pose** ONNX model on a folder of images and writes a
3D-cuboid overlay + the 6D pose (rotation columns + translation) per image. CPU-only,
no training. Built on top of `README_6d_pose.md` (the "ONNXRT Inference" section).

## What it can and cannot do
- **Fixed object set:** recognises only the **21 YCB-V household objects** (mustard
  bottle, cracker/sugar box, soup/tuna cans, bowl, mug, power drill, banana, …).
  Anything else gets no detection — that's inherent to running without retraining.
- **`.png` only:** the inference script silently skips `.jpg/.jpeg`. Convert first.
- **Fixed input size:** every image is resized to **640×480** before inference.
- **Depth is approximate:** the network's depth estimate was trained under YCB-V optics.
  Using your real intrinsics (below) makes the X/Y back-projection correct, but the
  absolute depth (Z) stays only roughly valid without retraining.

## One-time setup (already done on this machine)
- venv: `.venv_pose_onnx/` (onnxruntime, opencv, numpy, tqdm, onnx).
- Model + prototxt: `pretrained_models/yolox_l_object_pose_ti_lite/` (YCB-V, ti-lite).
  The prototxt must sit next to the `.onnx` — the script asserts it exists.

To recreate from scratch:
```bash
cd ~/dipl/edgeai-yolox
python -m venv .venv_pose_onnx && . .venv_pose_onnx/bin/activate
pip install onnxruntime opencv-python-headless numpy tqdm onnx
M=pretrained_models/yolox_l_object_pose_ti_lite
BASE=http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_l_object_pose_ti_lite
mkdir -p $M && curl -fsSL -o $M/yolox_l_object_pose_ti_lite.onnx     $BASE/yolox_l_object_pose_ti_lite.onnx
              curl -fsSL -o $M/yolox_l_object_pose_ti_lite.prototxt $BASE/yolox_l_object_pose_ti_lite.prototxt
```

## Using your real camera intrinsics (recommended)
The model computes the object **translation inside the ONNX graph** by back-projecting
with camera intrinsics baked in as constants (YCB-V's `fx≈1066, cx≈313, cy≈241`). To
get geometrically correct poses for *your* camera you bake your own intrinsics in.

1. Put your **native** intrinsics in `camera_real.json` (current values are the
   RealSense from FoundationPose, captured at 640×360):
   ```json
   {"fx": 456.5, "fy": 456.5, "cx": 320.0, "cy": 180.0, "width": 640, "height": 360}
   ```
   `width`/`height` are the resolution those intrinsics were calibrated at — the patch
   tool scales them to the model's 640×480 input automatically.

2. Bake them into a patched model copy:
   ```bash
   . .venv_pose_onnx/bin/activate
   M=pretrained_models/yolox_l_object_pose_ti_lite
   python tools/patch_pose_intrinsics.py \
     --onnx $M/yolox_l_object_pose_ti_lite.onnx \
     --camera camera_real.json \
     --out   $M/yolox_l_object_pose_ti_lite_realcam.onnx
   cp $M/yolox_l_object_pose_ti_lite.prototxt $M/yolox_l_object_pose_ti_lite_realcam.prototxt
   ```
   (`tools/patch_pose_intrinsics.py` locates the four intrinsic constants by their YCB-V
   default values, scales your K to 640×480, and rewrites them.)

`run_pose.sh` automatically prefers `*_realcam.onnx` when present and passes
`--camera camera_real.json` so the **cuboid overlay uses the same K** as the model.
Delete the `_realcam.onnx` to fall back to stock YCB-V intrinsics.

> Note: a legacy "logitech webcam" reprojection in `draw_obj_pose` (which mutated the
> translation in place) was removed so the saved pose and the overlay both use one
> consistent K. See the diff in `demo/ONNXRuntime/object_pose_utils_onnx.py`.

## Run
```bash
cd ~/dipl/edgeai-yolox

# bundled sample (verifies the pipeline)
./run_pose.sh

# your own images (folder of .png, or scale your intrinsics in camera_real.json first)
./run_pose.sh /absolute/path/to/your/png_folder
```

## Output
```
YOLOX_outputs/object_pose_onnx/
  ├── <img>.png         # 3D cuboid overlay + class label
  ├── <img>.txt         # one row per detection (see layout below)
  └── bbox/<img>.png     # plain 2D detection boxes
```
Each `.txt` row: `cls  score  x1 y1 x2 y2  r1(3)  r2(3)  t(3)`
- `r1`,`r2` = first two columns of the rotation matrix (3rd = `r1 × r2`).
- `t` = translation in **mm**, in the camera frame of the baked-in intrinsics.
- Class index → name via `get_class_names('ycbv')` in
  `demo/ONNXRuntime/object_pose_utils_onnx.py` (e.g. `4 = mustard_bottle`).

## Sanity-check note
The bundled `assets/ti_mustard.png` was rendered with YCB-V optics, so with the
`_realcam.onnx` model its cuboid looks **too small** (drawn at fx≈456 over a bottle
imaged at fx≈1066). That mismatch is expected and actually confirms your intrinsics are
applied — on real captures from your own camera the cuboid will fit. For a faithful
sanity check of the stock model, delete `_realcam.onnx` and run on the sample.
