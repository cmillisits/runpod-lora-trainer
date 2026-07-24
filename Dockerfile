# Runpod's official PyTorch base — bundles CUDA 12.1 + PyTorch 2.4.
# Smaller than nvidia/cuda:devel and pre-configured for serverless workers.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Explicit CUDA host-driver requirement. Our torch install below is
# 2.7.0+cu128, which needs driver 560+ (= CUDA 12.8 host runtime). Older
# Runpod pools on driver 535.x cannot run cu128 binaries — labeling this
# explicitly makes nvidia-container-cli refuse those hosts at container
# start instead of letting torch crash mid-job with the misleading
# "no kernel image" error. Mirrors Runpod_backup/Dockerfile (commit
# 12296c5 + d436a12 retrospective).
ENV NVIDIA_REQUIRE_CUDA="cuda>=12.8"

WORKDIR /workspace

# System libs Pillow / OpenCV / Kohya need
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (cached layer) ────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Kohya sd-scripts ──────────────────────────────────────────────────
# Pinned to a recent tag for reproducibility; bump after testing.
#
# We deliberately DON'T install xformers — pip auto-resolution had it
# pull a wheel that bumped torch and silently broke torchvision's C++
# ABI (RuntimeError: operator torchvision::nms does not exist), which
# cascaded into transformers/diffusers import failures. PyTorch's
# native SDPA (scaled dot product attention, since torch 2.0) does the
# same job as xformers attention with zero extra deps. Slightly slower
# (~10%) but trivially worth it for the version-mismatch resilience.
# We pass --sdpa instead of --xformers in trainer/train.py defaults.
RUN git clone https://github.com/kohya-ss/sd-scripts.git /workspace/sd-scripts \
    && cd /workspace/sd-scripts \
    && git checkout v0.9.1 \
    && pip install --no-cache-dir -r requirements.txt

# Blackwell architecture support — RTX 5090 / RTX PRO 6000 are sm_120,
# requires torch ≥2.7 + cu128 wheels. The base image ships torch 2.4.0,
# whose CUDA kernels only cover sm_50–sm_90. On a 5090 the worker fails
# Runpod's pre-launch fitness check with:
#
#     "NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not
#      compatible with the current PyTorch installation."
#
# and exits before our handler.py is ever loaded.
#
# Force-reinstall AFTER the Kohya install above — Kohya's requirements.txt
# pulls accelerate / transformers / diffusers and could otherwise drag
# torch back down. torchvision + torchaudio minor versions must match
# torch exactly. Same wheels as Runpod_backup (commit 12296c5).
#
# Host constraint: cu128 wheels need driver 560+ (= CUDA 12.8 host).
# Enforced via NVIDIA_REQUIRE_CUDA env at the top of this Dockerfile.
#
# Kohya v0.9.1 compatibility note: the pinned tag was tested upstream
# against torch 2.3–2.4. SDPA + native AdamW (our defaults) are stable
# APIs that haven't moved across the 2.4→2.7 jump. If a training run
# later fails on torch 2.7, the rollback is reverting this single RUN
# block — endpoint falls back to sm_50–90 hosts and trains as before.
# Cache-buster note (2026-07-03): the previous attempt at this layer
# (commit ac2ac5a, build 886de564) failed at Runpod buildkit's cache
# export step with "unexpected commit digest ... failed precondition"
# — the image tarball was produced successfully but the cache write
# to Runpod's registry failed, so Runpod fell back to serving the
# pre-ac2ac5a image (torch 2.4.1+cu124, no sm_120). Every worker
# since has failed the fitness check. This comment invalidates the
# layer hash so buildkit rebuilds this step and retries the cache
# commit — if the original error was transient (race condition on
# blob storage) it should now succeed. If it recurs, the fix is
# Runpod-support-side.
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Post-install version assert — hard-fails the build if torch didn't
# actually upgrade to 2.7 (e.g., a future Kohya deps bump silently
# reverts it). Cheaper to fail here at build time than ship a broken
# image and only discover the regression when workers can't boot.
RUN python -c "import torch; \
    assert torch.__version__.startswith('2.7'), \
    f'BUILD FAIL: expected torch 2.7.x for sm_120 support, got {torch.__version__}'; \
    assert torch.version.cuda == '12.8', \
    f'BUILD FAIL: expected torch built for cu128, got cu{torch.version.cuda}'; \
    print(f'torch={torch.__version__} cuda={torch.version.cuda} OK for Blackwell sm_120')"

# ── NumPy re-pin (2026-07-24) ─────────────────────────────────────────
# The `numpy<2.0` pin in requirements.txt (installed at line 22 above)
# gets trampled by either Kohya's sd-scripts/requirements.txt install
# or the torch==2.7.0 force-reinstall — one of them upgrades numpy to
# 2.x. cv2 (from libgl1 + opencv-python pulled by Kohya) is compiled
# against numpy 1.x, so at runtime every training job died at step
# 3/4 with:
#
#     A module that was compiled using NumPy 1.x cannot be run in
#     NumPy 2.4.4 as it may crash.
#     AttributeError: _ARRAY_API not found
#     ImportError: numpy.core.multiarray failed to import
#     subprocess.CalledProcessError: sd-scripts exited with code 1
#
# Reproduced 2026-07-24 in logs-lora-trainer.txt on thunderdash25's
# stuck personas (Kei + Lilliana). Every custom_persona LoRA training
# had been silently failing on this bug for an unknown duration.
#
# Fix: reassert numpy<2 as the last-word install so no downstream pin
# can trample it. Force-reinstall to be sure. Then assert at build
# time so a future regression breaks the image build instead of
# shipping a broken worker (following the torch assert pattern above).
RUN pip install --no-cache-dir --force-reinstall "numpy<2"

RUN python -c "import numpy; \
    major = int(numpy.__version__.split('.')[0]); \
    assert major < 2, \
    f'BUILD FAIL: numpy must be <2 for cv2 (compiled against numpy 1.x), got {numpy.__version__}'; \
    print(f'numpy={numpy.__version__} OK for cv2 (numpy 1.x ABI)')"

# Belt-and-suspenders — verify cv2 actually imports cleanly with the
# pinned numpy. Catches the ABI mismatch at build time so we don't
# discover it 3 minutes into a real training job.
RUN python -c "import cv2; import numpy; \
    print(f'cv2 import OK, cv2={cv2.__version__}, numpy={numpy.__version__}')"

# Pre-create accelerate default config so the launcher doesn't try
# interactive setup on first run inside the serverless worker.
RUN mkdir -p /root/.cache/huggingface/accelerate \
    && printf 'compute_environment: LOCAL_MACHINE\ndistributed_type: NO\nmixed_precision: bf16\nuse_cpu: false\nnum_processes: 1\n' \
        > /root/.cache/huggingface/accelerate/default_config.yaml

# ── Application code ──────────────────────────────────────────────────
COPY trainer /workspace/trainer
COPY configs /workspace/configs
COPY handler.py /workspace/handler.py

# Cache HF/transformers models on the network volume (mounted at runtime).
# Falls back to in-container disk if the volume isn't mounted.
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "/workspace/handler.py"]
