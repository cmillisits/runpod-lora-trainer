"""Download persona reference images and write them into a Kohya-compatible
dataset directory. Normalizes EXIF rotation, downsamples oversize images
to keep VRAM/time predictable.
"""
import os
from typing import List

import requests
from PIL import Image, ImageOps


# Resize bounds — Kohya's bucketing handles arbitrary aspects but very
# large or very small inputs waste time / risk OOM. 2048px max is enough
# for SDXL's max bucket; 512px min ensures we don't train on thumbnails.
MAX_DIM = 2048
MIN_DIM = 512


def prepare_dataset(slug: str, ref_urls: List[str], work_dir: str) -> str:
    """Download + normalize every URL in ref_urls into a dataset dir.

    Returns the dataset root path (Kohya consumes `dataset/images/` via
    the dataset config). Raises if fewer than 2 images successfully prep
    — Kohya's text encoder fine-tune is unstable below that.
    """
    dataset_dir = os.path.join(work_dir, 'dataset')
    images_dir = os.path.join(dataset_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    saved = 0
    failures = []
    for idx, url in enumerate(ref_urls):
        filename = f'{slug}_{idx:03d}.png'
        out_path = os.path.join(images_dir, filename)
        try:
            _download_and_normalize(url, out_path)
            saved += 1
        except Exception as e:
            failures.append((url, str(e)))
            print(f'[prepare_dataset] WARN failed to fetch {url}: {e}')

    if saved < 2:
        raise RuntimeError(
            f'too few usable images after prep ({saved}/{len(ref_urls)}); '
            f'failures: {failures}'
        )

    print(f'[prepare_dataset] saved {saved}/{len(ref_urls)} images to {images_dir}')
    return dataset_dir


def _download_and_normalize(url: str, out_path: str) -> None:
    """Fetch URL, fix EXIF rotation, downscale oversized images, save as PNG."""
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()

    img = Image.open(resp.raw)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')

    w, h = img.size
    max_side = max(w, h)
    min_side = min(w, h)
    if max_side > MAX_DIM:
        scale = MAX_DIM / max_side
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    elif min_side < MIN_DIM:
        # Upscale very small refs to the minimum bucket size — better than
        # excluding them but quality is bounded by the source.
        scale = MIN_DIM / min_side
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    img.save(out_path, 'PNG', optimize=False)
