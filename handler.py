"""Runpod serverless handler for SDXL persona LoRA training.

Receives a job describing a persona — slug, reference image URLs, optional
trigger word + caption/config overrides — runs the full pipeline (download,
caption, train, upload), and returns the trained LoRA's public URL.

Designed to live behind a Runpod serverless endpoint and called by Project
Titan's `POST /api/admin/personas/{slug}/train-lora`. See README for the
expected job payload + setup checklist.
"""
import os
import time
import traceback

import runpod

from trainer.prepare_dataset import prepare_dataset
from trainer.caption import caption_images
from trainer.train import run_training
from trainer.upload import upload_lora


def handler(job):
    started = time.time()
    job_input = (job or {}).get('input') or {}

    try:
        slug = _require(job_input, 'slug')
        ref_urls = _require(job_input, 'ref_urls')
        if not isinstance(ref_urls, list) or len(ref_urls) < 2:
            raise ValueError('ref_urls must be a list of at least 2 URLs')

        trigger_word = (job_input.get('trigger_word') or f'{slug}_v1').strip()
        lora_version = int(job_input.get('lora_version', 1))
        config_overrides = job_input.get('config_overrides') or {}
        captions_override = job_input.get('captions_override') or None
        bucket = job_input.get('lora_storage_bucket', 'persona-loras')

        work_dir = f'/tmp/jobs/{slug}-{int(started)}'
        os.makedirs(work_dir, exist_ok=True)

        print(f'[{slug}] step 1/4: prepare dataset ({len(ref_urls)} refs)')
        dataset_dir = prepare_dataset(slug=slug, ref_urls=ref_urls, work_dir=work_dir)

        print(f'[{slug}] step 2/4: auto-caption (trigger="{trigger_word}")')
        caption_images(dataset_dir=dataset_dir, trigger_word=trigger_word, captions_override=captions_override)

        print(f'[{slug}] step 3/4: train LoRA')
        lora_path = run_training(
            slug=slug,
            dataset_dir=dataset_dir,
            work_dir=work_dir,
            lora_version=lora_version,
            config_overrides=config_overrides,
        )

        print(f'[{slug}] step 4/4: upload to Supabase')
        public_url = upload_lora(slug=slug, lora_path=lora_path, lora_version=lora_version, bucket=bucket)

        duration = round(time.time() - started, 1)
        print(f'[{slug}] done in {duration}s -> {public_url}')
        return {
            'status': 'success',
            'slug': slug,
            'trigger_word': trigger_word,
            'lora_url': public_url,
            'lora_version': lora_version,
            'duration_seconds': duration,
            'image_count': len(ref_urls),
        }
    except Exception as e:
        traceback.print_exc()
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
            'duration_seconds': round(time.time() - started, 1),
        }


def _require(d, key):
    val = d.get(key)
    if val is None or val == '':
        raise ValueError(f'missing required input: {key}')
    return val


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
