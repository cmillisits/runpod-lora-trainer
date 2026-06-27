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
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

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
