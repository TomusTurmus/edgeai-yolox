# Running YOLOX inference on your own images (no retraining)

Two tasks are set up here, both runnable on custom images without training:

- **2D object detection** (COCO, 80 classes) — Docker, GPU. See [Part A](#part-a--2d-object-detection-docker).
- **6D object pose estimation** (YCB-V, 21 household objects) — venv, CPU, end-to-end
  ONNX with 3D-cuboid output. See [Part B](#part-b--6d-pose-estimation-onnx-cpu)
  (full detail in [`RUN_POSE.md`](./RUN_POSE.md)).

---

# Part A — 2D object detection (Docker)

Run the COCO-pretrained YOLOX-X detector on custom images — no training, no GPU
dependency setup on the host. Everything runs in a container; the repo is
bind-mounted so results land back in `YOLOX_outputs/`.

## What's already in place
- **Weights:** `pretrained_models/yolox_x.pth` (COCO, 80 classes) — already downloaded.
- **Image env:** `docker/Dockerfile.infer` (torch 2.0.1+cu118; builds deps only,
  mounts this repo at runtime).
- **Runner:** `docker/run_demo.sh`.

## Host prerequisites (already satisfied on this machine)
- Docker CLI + NVIDIA Container Toolkit (`nvidia-ctk`) installed.
- GPU: RTX A2000 12 GB.
- **You are not in the `docker` group**, so docker needs `sudo`. The runner script
  auto-detects this and prepends `sudo` if needed. Because `sudo` prompts for a
  password, run the commands below yourself in the terminal (in Claude Code you can
  type `! <command>` to run them in-session).

> One-time alternative to avoid `sudo` every time:
> `sudo usermod -aG docker $USER` then log out/in.

## 1. Build the image (once, ~10–15 min first time)
```bash
cd ~/dipl/edgeai-yolox
sudo docker build -f docker/Dockerfile.infer -t edgeai-yolox-infer .
```

## 2. Run on your images
```bash
cd ~/dipl/edgeai-yolox
docker/run_demo.sh /path/to/your/images
```
- `/path/to/your/images` can be a directory (all `.jpg/.png/.jpeg/.bmp/.webp` inside,
  recursively) or a single image file.
- Pass extra `tools/demo.py` flags after the path, e.g. higher confidence:
  ```bash
  docker/run_demo.sh ~/my_photos --conf 0.4 --tsize 640
  ```

Quick sanity check on the bundled sample image:
```bash
docker/run_demo.sh assets/dog.jpg
```

## 3. Where results go
```
YOLOX_outputs/yolox_x/vis_res/<YYYY_MM_DD_HH_MM_SS>/
  ├── <image>.jpg     # annotated with boxes + COCO class labels
  └── txt/<image>.txt  # raw detections: x1 y1 x2 y2 obj_conf cls_conf ... cls_id
```
These appear directly on the host (the repo is bind-mounted).

## Manual run (without the script)
```bash
sudo docker run --rm --gpus all \
  -v "$PWD":/workspace/edgeai-yolox \
  -v /path/to/your/images:/data/images:ro \
  edgeai-yolox-infer \
  python3 tools/demo.py image \
    -f exps/default/yolox_x.py \
    -c pretrained_models/yolox_x.pth \
    --path /data/images \
    --dataset coco --task 2dod \
    --device gpu --conf 0.3 --nms 0.45 --save_result
```

## Notes / gotchas
- `--dataset coco` is **required**: `demo.py` does `_NUM_CLASSES[args.dataset]` and
  would `KeyError` on the default `None`.
- `--task 2dod` selects the standard detection path (not 6D pose).
- A one-line patch was applied to `tools/demo.py`
  (`self.cad_models = getattr(model.head, "cad_models", None)`) so the 2D-OD path
  doesn't crash — the stock line assumes the object-pose head, which the COCO model
  doesn't have.
- CPU fallback: drop `--gpus all` and use `--device cpu` (much slower).
- To try a smaller/faster model, swap `-f exps/default/yolox_s.py -c <yolox_s.pth>`
  (you'd need to download the `.pth`; only `yolox_x.pth` is present locally).
- `mmcv-full` is intentionally not installed (not needed for 2D-OD). See the note in
  `Dockerfile.infer` if a later task needs it.

---

# Part B — 6D pose estimation (ONNX, CPU)

Estimates full 6D pose (rotation + translation) and draws a 3D cuboid per object, using
the end-to-end YOLOX-6D-Pose ONNX model. No training, no Docker, runs on CPU.
Full reference: [`RUN_POSE.md`](./RUN_POSE.md).

## What's already in place
- **venv:** `.venv_pose_onnx/` (onnxruntime, opencv, numpy, tqdm, onnx).
- **Model + prototxt:** `pretrained_models/yolox_l_object_pose_ti_lite/` (YCB-V, ti-lite).
  The prototxt must sit next to the `.onnx` (the script asserts it).
- **Runner:** `run_pose.sh`. **Patch tool:** `tools/patch_pose_intrinsics.py`.

## Constraints (read first)
- **YCB-V objects only** — 21 household objects (mustard bottle, cracker/sugar box,
  soup/tuna cans, bowl, mug, power drill, banana, …). No detection for anything else;
  inherent to running without retraining.
- **`.png` only** — the script skips `.jpg/.jpeg`. Convert first.
- Every image is resized to **640×480**; absolute depth (Z) is approximate without
  retraining (X/Y are correct once real intrinsics are baked in).

## 1. (Recommended) Bake in your real camera intrinsics
The translation is back-projected with camera K baked into the ONNX graph (default:
YCB-V `fx≈1066`). Replace it with your camera's K for correct poses.

1. Edit `camera_real.json` with your **native** intrinsics + the resolution they were
   calibrated at (defaults to the RealSense capture, 640×360):
   ```json
   {"fx": 456.5, "fy": 456.5, "cx": 320.0, "cy": 180.0, "width": 640, "height": 360}
   ```
2. Bake them in (auto-scaled to the 640×480 model input):
   ```bash
   cd ~/dipl/edgeai-yolox && . .venv_pose_onnx/bin/activate
   M=pretrained_models/yolox_l_object_pose_ti_lite
   python tools/patch_pose_intrinsics.py \
     --onnx $M/yolox_l_object_pose_ti_lite.onnx \
     --camera camera_real.json \
     --out   $M/yolox_l_object_pose_ti_lite_realcam.onnx
   cp $M/yolox_l_object_pose_ti_lite.prototxt $M/yolox_l_object_pose_ti_lite_realcam.prototxt
   ```
`run_pose.sh` auto-prefers `*_realcam.onnx` and passes `--camera` so the overlay uses
the same K. Delete it to fall back to stock YCB-V intrinsics.

## 2. Run on your images
```bash
cd ~/dipl/edgeai-yolox
./run_pose.sh                                   # bundled sample (sanity check)
./run_pose.sh /absolute/path/to/your/png_folder # your own .png images
```

## 3. Where results go
```
YOLOX_outputs/object_pose_onnx/
  ├── <image>.png        # 3D cuboid overlay + class label
  ├── <image>.txt        # cls score x1 y1 x2 y2  r1(3) r2(3)  t(3)   (t in mm)
  └── bbox/<image>.png    # plain 2D detection boxes
```
`r1`,`r2` are the first two rotation-matrix columns (3rd = `r1 × r2`). Class index →
name via `get_class_names('ycbv')` (e.g. `4 = mustard_bottle`).

## Notes / gotchas
- The bundled `assets/ti_mustard.png` was rendered with YCB-V optics, so with the
  `_realcam.onnx` model its cuboid looks **too small** — expected (it confirms your K is
  applied); real captures from your own camera fit. Delete `_realcam.onnx` to sanity-check
  the stock model on the sample.
- A legacy "logitech webcam" reprojection in `draw_obj_pose` that mutated the translation
  in place was removed, so the saved pose and the overlay use one consistent K. A
  `--camera` flag was added to `demo/ONNXRuntime/onnx_inference_object_pose.py`.
- Other model sizes (s/m/l) and prototxt are linked in `README_6d_pose.md`; swap the
  download URLs in `RUN_POSE.md`'s setup block.
