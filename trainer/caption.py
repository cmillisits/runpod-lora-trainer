"""WD14 ONNX-based booru-tag auto-captioning.

For each image in the dataset dir we run WD14 (EVA02-large v3), filter
tags by confidence, drop "character" category tags (we're TEACHING the
LoRA the persona — we don't want training captions naming a different
character that the tagger thinks the persona resembles), prepend the
trigger word, and write a `.txt` next to the image.

Caption format: `{trigger_word}, tag1, tag2, ...`

Kohya is configured with `keep_tokens=1` so the trigger word stays at the
front during caption shuffling.
"""
import csv
import os
from typing import Dict, List, Optional

import numpy as np
import onnxruntime
from PIL import Image


# Pinned tagger — EVA02-large v3 has the best tag accuracy for photoreal
# content among the WD14 family. SwinV2 is faster but loses ~5% accuracy.
WD14_MODEL_REPO = 'SmilingWolf/wd-eva02-large-tagger-v3'
WD14_MODEL_FILENAME = 'model.onnx'
WD14_TAGS_FILENAME = 'selected_tags.csv'

# Confidence threshold for keeping a tag. 0.35 is a good balance for
# character-LoRA training — strict enough to drop noise, permissive
# enough to keep distinctive features (small tattoo, specific accessory).
TAG_THRESHOLD = 0.35

# WD14 categories: 0=general, 4=character, 9=rating. We drop characters
# (don't want the tagger guessing "looks like persona X" mid-train) and
# ratings (Kohya/SDXL doesn't use them; just noise). Keep general (0).
DROP_CATEGORIES = {4, 9}

# Module-level model handle — loaded once per worker, reused across jobs.
_session: Optional[onnxruntime.InferenceSession] = None
_tag_table: List[Dict[str, object]] = []


def _ensure_loaded() -> None:
    global _session, _tag_table
    if _session is not None:
        return

    from huggingface_hub import hf_hub_download
    cache_dir = os.environ.get('HF_HOME', '/runpod-volume/huggingface')
    os.makedirs(cache_dir, exist_ok=True)

    print(f'[caption] loading WD14 from {WD14_MODEL_REPO} (cache: {cache_dir})')
    model_path = hf_hub_download(repo_id=WD14_MODEL_REPO, filename=WD14_MODEL_FILENAME, cache_dir=cache_dir)
    tags_path = hf_hub_download(repo_id=WD14_MODEL_REPO, filename=WD14_TAGS_FILENAME, cache_dir=cache_dir)

    _session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    with open(tags_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            _tag_table.append({
                'name': row['name'].replace('_', ' '),
                'category': int(row['category']),
            })
    print(f'[caption] loaded {len(_tag_table)} tag entries')


def caption_image(image_path: str) -> List[str]:
    """Return the WD14 tags above threshold for one image."""
    _ensure_loaded()
    assert _session is not None

    img = Image.open(image_path).convert('RGB')
    img = _square_pad_resize(img, 448)
    arr = np.array(img, dtype=np.float32)
    # WD14 was trained on BGR-ordered uint8 / no normalization
    arr = arr[:, :, ::-1]
    arr = np.expand_dims(arr, axis=0)

    input_name = _session.get_inputs()[0].name
    probs = _session.run(None, {input_name: arr})[0][0]

    tags: List[str] = []
    for i, p in enumerate(probs):
        if p < TAG_THRESHOLD:
            continue
        meta = _tag_table[i]
        if meta['category'] in DROP_CATEGORIES:
            continue
        tags.append(str(meta['name']))
    return tags


def caption_images(
    dataset_dir: str,
    trigger_word: str,
    captions_override: Optional[Dict[str, str]] = None,
) -> None:
    """Write a `.txt` caption next to each image in `dataset_dir/images/`.

    If `captions_override` contains a key matching the image filename, that
    override is used (verbatim, after the trigger word) instead of WD14.
    """
    images_dir = os.path.join(dataset_dir, 'images')
    fnames = sorted(f for f in os.listdir(images_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))

    for fname in fnames:
        image_path = os.path.join(images_dir, fname)
        txt_path = os.path.splitext(image_path)[0] + '.txt'

        if captions_override and fname in captions_override:
            caption = f'{trigger_word}, {captions_override[fname].strip()}'
        else:
            tags = caption_image(image_path)
            caption = ', '.join([trigger_word] + tags)

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(caption)

        preview = caption if len(caption) <= 120 else caption[:117] + '...'
        print(f'[caption] {fname}: {preview}')


def _square_pad_resize(img: Image.Image, size: int) -> Image.Image:
    """White-pad to square then resize — WD14's expected preprocessing."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new('RGB', (side, side), (255, 255, 255))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.BICUBIC)
