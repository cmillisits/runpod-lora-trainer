"""Upload a trained LoRA `.safetensors` to Supabase Storage."""
import os

from supabase import create_client


def upload_lora(slug: str, lora_path: str, lora_version: int, bucket: str = 'persona-loras') -> str:
    """Push the LoRA file to Supabase Storage. Returns its public URL.

    Storage layout: `{bucket}/{slug}/v{lora_version}.safetensors`. Uses
    upsert=true so re-running training for the same slug+version overwrites
    cleanly (operator-controlled overwrites only — by bumping `lora_version`
    you keep prior versions reachable).
    """
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_SERVICE_KEY')
    if not url or not key:
        raise RuntimeError(
            'SUPABASE_URL and SUPABASE_SERVICE_KEY must be set as env vars on the Runpod endpoint'
        )

    client = create_client(url, key)
    storage_path = f'{slug}/v{lora_version}.safetensors'

    size_mb = os.path.getsize(lora_path) / 1024 / 1024
    print(f'[upload] {size_mb:.1f}MB -> {bucket}/{storage_path}')

    with open(lora_path, 'rb') as f:
        client.storage.from_(bucket).upload(
            path=storage_path,
            file=f.read(),
            file_options={
                'content-type': 'application/octet-stream',
                'upsert': 'true',
            },
        )

    public_url = client.storage.from_(bucket).get_public_url(storage_path)
    return public_url
