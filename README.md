# runpod-lora-trainer

Runpod Serverless worker that trains per-persona SDXL LoRAs for Project
Titan. Takes a list of reference image URLs + a trigger word, runs Kohya
`sd-scripts` against RealVisXL 4.0, uploads the resulting `.safetensors`
to Supabase Storage, and returns the public URL.

Built specifically for the Project Titan persona pipeline. It assumes:

- Reference images live on Supabase Storage (publicly fetchable URLs).
- Trained LoRAs go back to Supabase under bucket `persona-loras` at path
  `{slug}/v{version}.safetensors`.
- The base SDXL checkpoint (RealVisXL 4.0) lives on a Runpod **network
  volume** so workers don't re-download 7GB on every cold start.

## Repo layout

```
.
├── Dockerfile               # Container image (~2-3GB; models on volume)
├── requirements.txt
├── handler.py               # Runpod serverless entry — orchestrates the pipeline
├── trainer/
│   ├── prepare_dataset.py   # Download refs, normalize, write to dataset dir
│   ├── caption.py           # WD14 booru-tag captioning + trigger-word prefix
│   ├── train.py             # Invokes accelerate launch sdxl_train_network.py
│   └── upload.py            # Push trained LoRA → Supabase Storage
├── configs/
│   └── sdxl_default.toml    # Reference training hyperparams (informational)
└── .github/workflows/
    └── docker-publish.yml   # Builds + pushes image to GHCR on tag/push
```

## Job input / output schema

POST to the Runpod endpoint with:

```json
{
  "input": {
    "slug": "kira",
    "ref_urls": [
      "https://<supabase>.../personas/kira/photo-1.png",
      "https://<supabase>.../personas/kira/photo-2.png"
    ],
    "trigger_word": "kira_v1",
    "lora_version": 1,
    "captions_override": {
      "kira_000.png": "completely nude, full body, standing"
    },
    "config_overrides": {
      "max_train_steps": 2500,
      "network_dim": 48
    },
    "lora_storage_bucket": "persona-loras"
  }
}
```

`trigger_word` defaults to `{slug}_v1` if omitted. `captions_override` is
keyed by **filename in the dataset dir** (`{slug}_000.png`,
`{slug}_001.png`, ...) — anything not overridden is auto-captioned via
WD14. `config_overrides` is merged on top of the defaults documented in
`trainer/train.py` (`network_dim=32`, `max_train_steps=2000`, etc.).

Successful response:

```json
{
  "status": "success",
  "slug": "kira",
  "trigger_word": "kira_v1",
  "lora_url": "https://<supabase>.../persona-loras/kira/v1.safetensors",
  "lora_version": 1,
  "duration_seconds": 487.2,
  "image_count": 12
}
```

Failure response:

```json
{
  "status": "error",
  "error": "human-readable detail",
  "error_type": "ExceptionClassName",
  "duration_seconds": 12.4
}
```

## Setup checklist

### 1. Network volume

Create a Runpod network volume (50GB is plenty) and pre-upload:

```
/sdxl/realvisxlV40_v40Bakedvae.safetensors    # ~7GB — same checkpoint as inference
/huggingface/                                   # WD14 + HF cache lands here on first run
```

Same checkpoint as the inference worker keeps the LoRA "in the same world"
as the gen pipeline.

### 2. Supabase

Create a public storage bucket named `persona-loras` (or pass a custom
name via `lora_storage_bucket`). Service-role policy needs write access
on that bucket — the worker uploads with the service key.

### 3. Build + publish the image

Either tag a release (CI does the rest) or build locally:

```bash
git tag v0.1.0 && git push --tags     # triggers .github/workflows/docker-publish.yml
# or
docker build -t ghcr.io/cmillisits/runpod-lora-trainer:dev .
docker push ghcr.io/cmillisits/runpod-lora-trainer:dev
```

Make sure the GHCR package is **public** (repo Settings → Packages → Visibility)
or attach Runpod registry credentials.

### 4. Runpod serverless endpoint

Create a new serverless endpoint with:

| Setting | Value |
|---|---|
| Container Image | `ghcr.io/cmillisits/runpod-lora-trainer:main` (or pinned tag) |
| GPU | **A100 40GB** (recommended) or RTX 6000 Ada (48GB). RTX 4090 24GB works but tight |
| Container Disk | 30GB |
| Network Volume | mount the volume from step 1 at `/runpod-volume` |
| Max Workers | 5 (start) — bump for batch backfill |
| Idle Timeout | 5s (spin down quickly between jobs) |
| Execution Timeout | 1800s (30 min — generous for SDXL training) |

Endpoint env vars:

```
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SERVICE_KEY=<service-role-key>
PRETRAINED_MODEL_PATH=/runpod-volume/sdxl/realvisxlV40_v40Bakedvae.safetensors  # default
HF_HOME=/runpod-volume/huggingface                                                # default
```

### 5. Project Titan integration

In Project Titan add to env:

```
RUNPOD_TRAINING_ENDPOINT_ID=<endpoint-id>
RUNPOD_TRAINING_API_KEY=<runpod-api-key>     # can be same as inference key
```

Then wire `POST /api/admin/personas/[slug]/train-lora` to call this
endpoint with the persona's `photo_urls`.

## Cost / timing notes

| GPU | Train cost (2000 steps, ~10 imgs) | Wall time |
|---|---|---|
| A100 40GB | ~$0.30 | ~5-7 min |
| RTX 6000 Ada | ~$0.25 | ~7-10 min |
| RTX 4090 24GB | ~$0.20 | ~10-15 min (grad checkpointing forced) |

Cold start with network-volume-mounted models: ~30s. Without volume:
3-5 min on first request (re-download SDXL).

## Local development

```bash
pip install -r requirements.txt
git clone https://github.com/kohya-ss/sd-scripts.git /workspace/sd-scripts
pip install -r /workspace/sd-scripts/requirements.txt

# Test the handler locally with a fake job
export PRETRAINED_MODEL_PATH=/path/to/realvisxlV40_v40Bakedvae.safetensors
export SUPABASE_URL=https://...
export SUPABASE_SERVICE_KEY=...
python -c "
from handler import handler
result = handler({'input': {
  'slug': 'test',
  'ref_urls': ['https://example.com/a.png', 'https://example.com/b.png'],
  'trigger_word': 'test_v1',
}})
print(result)
"
```

For an HTTP-style local server, use Runpod's built-in serve mode:

```bash
python handler.py --rp_serve_api --rp_api_port 8000
```

## Versioning

Image tags map to git tags. Pin the Runpod endpoint to a specific tag in
production (e.g. `v0.1.0`) and bump only when training behavior changes —
avoids "main moved underneath you" surprises mid-batch.
