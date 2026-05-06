"""Wrap Kohya `sdxl_train_network.py` with our defaults + Project Titan dataset config.

Builds the dataset TOML on the fly (Kohya's preferred input format),
merges training hyperparams (defaults + per-job overrides), and invokes
`accelerate launch` against the cloned sd-scripts checkout.
"""
import os
import subprocess
from typing import Dict, Any

import toml


# Path to the cloned Kohya repo (set in Dockerfile).
KOHYA_SCRIPT = '/workspace/sd-scripts/sdxl_train_network.py'
KOHYA_CWD = '/workspace/sd-scripts'

# Network-volume location of the base SDXL checkpoint. Operator pre-uploads
# this file to the volume; workers don't re-download 7GB on cold start.
PRETRAINED_MODEL = os.environ.get(
    'PRETRAINED_MODEL_PATH',
    '/runpod-volume/sdxl/realvisxlV40_v40Bakedvae.safetensors',
)


# Defaults tuned for character LoRAs on SDXL:
#   - dim 32 / alpha 16 captures face + body + distinctive marks without
#     overfitting on 10-15 image datasets.
#   - 2000 steps × num_repeats=10 across ~12 imgs ≈ 16-17 effective epochs;
#     usually well past convergence for character LoRAs.
#   - AdamW8bit + grad checkpointing keeps memory under 24GB so RTX 4090
#     remains a viable fallback.
#   - noise_offset=0.0357 helps low/high-key tones (Kohya's recommended
#     value for photoreal SDXL training).
DEFAULTS: Dict[str, Any] = {
    'output_name': 'v1',                     # overridden per-call
    'save_model_as': 'safetensors',
    'save_precision': 'bf16',
    'mixed_precision': 'bf16',
    'network_module': 'networks.lora',
    'network_dim': 32,
    'network_alpha': 16,
    'learning_rate': 1e-4,
    'unet_lr': 1e-3,
    'text_encoder_lr': 5e-5,
    'optimizer_type': 'AdamW8bit',
    'lr_scheduler': 'cosine',
    'lr_warmup_steps': 100,
    'max_train_steps': 2000,
    'save_every_n_steps': 500,
    'clip_skip': 2,
    'noise_offset': 0.0357,
    'gradient_checkpointing': True,
    # Use PyTorch native SDPA (since torch 2.0) instead of xformers — same
    # role (memory-efficient attention) with no extra dep, no risk of
    # torch/torchvision ABI break from xformers wheels. ~10% slower than
    # xformers in absolute throughput; trivial for our short training runs.
    'sdpa': True,
    'cache_latents': True,
    'cache_latents_to_disk': True,
    'no_half_vae': True,
    'max_data_loader_n_workers': 2,
    'persistent_data_loader_workers': True,
}


def run_training(
    slug: str,
    dataset_dir: str,
    work_dir: str,
    lora_version: int,
    config_overrides: Dict[str, Any],
) -> str:
    """Run sd-scripts SDXL LoRA training. Returns path to the final .safetensors.

    Raises FileNotFoundError if the base checkpoint isn't on the network
    volume (most common failure mode — surfaces clearly).
    """
    if not os.path.exists(PRETRAINED_MODEL):
        raise FileNotFoundError(
            f'pretrained model not found at {PRETRAINED_MODEL} — '
            f'upload RealVisXL 4.0 to the Runpod network volume before launching workers'
        )

    dataset_toml_path = _write_dataset_toml(
        dataset_dir=dataset_dir,
        work_dir=work_dir,
        num_repeats=int(config_overrides.get('num_repeats', 10)),
    )

    output_dir = os.path.join(work_dir, 'lora_out')
    os.makedirs(output_dir, exist_ok=True)
    output_name = f'v{lora_version}'

    args: Dict[str, Any] = dict(DEFAULTS)
    args.update(config_overrides or {})
    # Always-derived args (override anything the caller passes for these)
    args['pretrained_model_name_or_path'] = PRETRAINED_MODEL
    args['dataset_config'] = dataset_toml_path
    args['output_dir'] = output_dir
    args['output_name'] = output_name
    args['logging_dir'] = os.path.join(work_dir, 'logs')

    cmd = ['accelerate', 'launch', '--num_cpu_threads_per_process=4', KOHYA_SCRIPT]
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f'--{k}')
        else:
            cmd.append(f'--{k}={v}')

    print(f'[train] launching sd-scripts:\n  {" ".join(cmd)}')
    result = subprocess.run(cmd, cwd=KOHYA_CWD)
    if result.returncode != 0:
        raise RuntimeError(f'sd-scripts exited with code {result.returncode}')

    lora_path = os.path.join(output_dir, f'{output_name}.safetensors')
    if not os.path.exists(lora_path):
        raise RuntimeError(f'training completed but expected LoRA at {lora_path} is missing')
    return lora_path


def _write_dataset_toml(dataset_dir: str, work_dir: str, num_repeats: int) -> str:
    """Build the Kohya dataset config TOML for this job."""
    dataset_config: Dict[str, Any] = {
        'general': {
            # Caption shuffling helps generalization on small datasets.
            # keep_tokens=1 pins the trigger word to the front so it always
            # appears at the start of the caption regardless of shuffle.
            'shuffle_caption': True,
            'caption_extension': '.txt',
            'keep_tokens': 1,
            'keep_tokens_separator': ',',
        },
        'datasets': [{
            'resolution': 1024,
            'batch_size': 1,
            # Multi-aspect bucketing — Kohya groups images of similar aspect
            # ratio into the same training batch. Eliminates the need to
            # crop everything to 1:1.
            'enable_bucket': True,
            'bucket_reso_steps': 64,
            'min_bucket_reso': 512,
            'max_bucket_reso': 2048,
            'subsets': [{
                'image_dir': os.path.join(dataset_dir, 'images'),
                'num_repeats': num_repeats,
                'caption_extension': '.txt',
            }],
        }],
    }
    out_path = os.path.join(work_dir, 'dataset.toml')
    with open(out_path, 'w', encoding='utf-8') as f:
        toml.dump(dataset_config, f)
    return out_path
