# Runpod's official PyTorch base — bundles CUDA 12.1 + PyTorch 2.4.
# Smaller than nvidia/cuda:devel and pre-configured for serverless workers.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

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
RUN git clone https://github.com/kohya-ss/sd-scripts.git /workspace/sd-scripts \
    && cd /workspace/sd-scripts \
    && git checkout v0.9.1 \
    && pip install --no-cache-dir -r requirements.txt

# xformers is optional in Kohya's requirements.txt (left out for users
# who don't have a CUDA build installed). Our default training config
# passes --xformers, so install it explicitly here. The runpod/pytorch
# base ships torch 2.4+cu124; pip auto-resolves a matching xformers wheel.
RUN pip install --no-cache-dir xformers

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
