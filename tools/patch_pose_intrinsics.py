#!/usr/bin/env python3
"""Bake real camera intrinsics into a YOLOX-6D-Pose ONNX model.

The end-to-end object_pose ONNX computes the object translation *inside* the graph
using camera intrinsics stored as scalar initializers (back-projection
    t = [(u-cx)/fx * Z, (v-cy)/fy * Z, Z]).
By default these hold the YCB-V test intrinsics. This script replaces them with your
real camera's intrinsics so the translation is geometrically correct for your captures.

IMPORTANT: the inference script (onnx_inference_object_pose.py) resizes every image to
the model input size (640x480 by default). Intrinsics are therefore scaled from your
native capture resolution to the model input resolution before being baked in. Pass the
same camera JSON to the inference script (--camera) so the cuboid overlay uses the same K.

Camera JSON format (native capture intrinsics):
    {"fx": 456.5, "fy": 456.5, "cx": 320.0, "cy": 180.0, "width": 640, "height": 360}

Usage:
    python tools/patch_pose_intrinsics.py \
        --onnx pretrained_models/.../model.onnx \
        --camera camera_real.json \
        --out   pretrained_models/.../model_realcam.onnx
"""
import argparse
import json
import numpy as np
import onnx
from onnx import numpy_helper

# YCB-V test-split intrinsics baked into the stock model — used to locate the
# initializers regardless of their (export-dependent) names.
YCBV = {"fx": 1066.778, "fy": 1067.487, "cx": 312.9869, "cy": 241.3109}


def scaled_K(cam, in_w, in_h):
    """Scale native intrinsics to the model input resolution (post-resize)."""
    sx, sy = in_w / cam["width"], in_h / cam["height"]
    return {
        "fx": cam["fx"] * sx,
        "fy": cam["fy"] * sy,
        "cx": cam["cx"] * sx,
        "cy": cam["cy"] * sy,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="Input ONNX model (stock).")
    ap.add_argument("--camera", required=True, help="Camera JSON with native intrinsics.")
    ap.add_argument("--out", required=True, help="Output patched ONNX path.")
    ap.add_argument("--input-w", type=int, default=640, help="Model input width.")
    ap.add_argument("--input-h", type=int, default=480, help="Model input height.")
    args = ap.parse_args()

    cam = json.load(open(args.camera))
    for k in ("fx", "fy", "cx", "cy", "width", "height"):
        if k not in cam:
            raise SystemExit(f"camera JSON missing key: {k}")
    K = scaled_K(cam, args.input_w, args.input_h)
    print(f"native K:  fx={cam['fx']} fy={cam['fy']} cx={cam['cx']} cy={cam['cy']} "
          f"@ {cam['width']}x{cam['height']}")
    print(f"scaled K:  fx={K['fx']:.3f} fy={K['fy']:.3f} cx={K['cx']:.3f} cy={K['cy']:.3f} "
          f"@ {args.input_w}x{args.input_h}  (baked into model)")

    model = onnx.load(args.onnx)
    g = model.graph

    # Map each YCB-V default value -> intrinsic name, then to the new scaled value.
    default_to_param = {v: k for k, v in YCBV.items()}
    patched = {}
    for init in g.initializer:
        a = numpy_helper.to_array(init)
        if a.size != 1:
            continue
        val = float(a.ravel()[0])
        for dv, param in default_to_param.items():
            if np.isclose(val, dv, atol=1e-2):
                new = np.array(K[param], dtype=a.dtype)
                init.CopyFrom(numpy_helper.from_array(new.reshape(a.shape), init.name))
                patched[param] = (init.name, val, K[param])
                break

    missing = set(YCBV) - set(patched)
    if missing:
        raise SystemExit(f"Could not locate intrinsic initializers for: {missing}. "
                         f"Model may already be patched or has a different structure.")
    print("patched initializers:")
    for p, (name, old, newv) in patched.items():
        print(f"  {p:3s} ({name}): {old:.4f} -> {newv:.4f}")

    onnx.save(model, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
