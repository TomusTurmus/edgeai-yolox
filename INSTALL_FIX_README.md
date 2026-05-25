# Install Fix Notes

## Why the build failed

The package setup in this repository imports `torch` inside `setup.py` while building the YOLOX extension. That is fine only when the build runs in the same environment where PyTorch is already installed.

The failure happened because pip entered the editable/build-isolated path for `setup.py develop` style installs. In that mode, the temporary build environment did not have `torch`, so the build stopped with:

`ModuleNotFoundError: No module named 'torch'`

## What was edited

### `docker/Dockerfile`

Replaced the legacy local install step:

`python3 setup.py install`

with:

`python3 -m pip install --no-cache-dir --no-build-isolation .`

This keeps the install standards-based and prevents the build from hiding the already-installed PyTorch package.

### `setup.sh`

Replaced:

`python3 setup.py develop`

with:

`pip3 install --no-input --no-build-isolation .`

### `setup_cpu.sh`

Made the same replacement as `setup.sh` so the CPU setup path behaves the same way.

### `README.md`

Updated the COCO API instructions to install `cython` and `numpy` first, then build the git-based package with `--no-build-isolation` so pip does not hide `numpy` during the build.

### `tools/eval.py` and `tools/train.py`

Replaced the bare GPU-count assertion with an explicit check that explains whether PyTorch sees no GPUs at all or fewer GPUs than requested.

## Result

The install now happens in the active environment that already has PyTorch, so the YOLOX extension build can import `torch` successfully.