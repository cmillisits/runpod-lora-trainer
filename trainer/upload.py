"""Upload a trained LoRA `.safetensors` to Supabase Storage, with a
network-volume fallback so a failed upload never loses the trained file.
"""
import os
import shutil
from typing import Optional

from supabase import create_client


# Network volume path where we copy the LoRA on upload failure. Persists
# across worker spin-downs (unlike /tmp), so the operator can fish the
# file off via a temp pod with the volume mounted, even after the
# serverless worker has terminated.
RECOVERY_DIR = '/runpod-volume/loras_recovery'


def upload_lora(slug: str, lora_path: str, lora_version: int, bucket: str = 'persona-loras') -> str:
    """Push the LoRA file to Supabase Storage. Returns its public URL.

    Storage layout: `{bucket}/{slug}/v{lora_version}.safetensors`. Uses
    upsert=true so re-running training for the same slug+version overwrites
    cleanly (operator-controlled overwrites only — by bumping `lora_version`
    you keep prior versions reachable).

    Failure handling: if the Supabase upload fails for any reason (auth,
    network, bucket policy, file-size limit, etc.), the trained file is
    copied to the network volume FIRST, THEN the error is re-raised with
    a pointer to the recovery path. Insurance against losing 10+ minutes
    of GPU time to upload-step misconfiguration.
    """
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_SERVICE_KEY')
    if not url or not key:
        # Even without creds we can still preserve the trained file —
        # backup to volume before raising so the operator can recover.
        backup_path = _backup_to_volume(slug, lora_path, lora_version)
        recovery_msg = (f'Trained LoRA backed up to {backup_path} for recovery via temp pod.'
                        if backup_path
                        else 'Volume backup also failed — trained LoRA is lost.')
        raise RuntimeError(
            'SUPABASE_URL and SUPABASE_SERVICE_KEY must be set as env vars on the Runpod endpoint. '
            + recovery_msg
        )

    client = create_client(url, key)
    storage_path = f'{slug}/v{lora_version}.safetensors'

    size_mb = os.path.getsize(lora_path) / 1024 / 1024
    print(f'[upload] {size_mb:.1f}MB -> {bucket}/{storage_path}')

    try:
        # Pass the path directly (not f.read()) so supabase-py 2.5+ can stream
        # the upload internally via httpx instead of loading all ~200MB into
        # memory and POSTing as a single blob — the in-memory + single-POST
        # path hangs on large files (observed at 217MB stalling 2+ min before
        # cancel). The library handles open/read/close.
        client.storage.from_(bucket).upload(
            path=storage_path,
            file=lora_path,
            file_options={
                'content-type': 'application/octet-stream',
                'upsert': 'true',
            },
        )
        # Construct the public URL directly rather than going through
        # client.storage.from_(bucket).get_public_url(). storage3 (the
        # underlying lib) has version-dependent quirks — some versions
        # return a trailing '?', some return a dict, and we've observed
        # at least one version appending a literal '$0' to the path that
        # gets baked into the stored lora_url column and later 404s when
        # the worker tries to fetch it. Hand-formatting is bulletproof.
        public_url = f"{url.rstrip('/')}/storage/v1/object/public/{bucket}/{storage_path}"
        return public_url

    except Exception as upload_err:
        # Supabase upload failed (auth / network / size / etc.). Save the
        # trained file to the network volume so it survives worker spin-down,
        # then re-raise with a recovery pointer.
        backup_path = _backup_to_volume(slug, lora_path, lora_version)
        if backup_path:
            raise RuntimeError(
                f'Supabase upload failed: {upload_err}. '
                f'Trained LoRA backed up to {backup_path} — fish it off via '
                f'a temp pod with the volume mounted.'
            ) from upload_err
        # Volume backup also failed — surface the original upload error.
        raise


def _backup_to_volume(slug: str, lora_path: str, lora_version: int) -> Optional[str]:
    """Copy the trained LoRA to the network volume's recovery directory.

    Returns the absolute backup path on success, None on failure (caller
    decides how to surface the secondary failure). Failure modes include
    the volume not being mounted (running outside Runpod) or running out
    of space — neither should mask the primary upload error.
    """
    try:
        backup_dir = os.path.join(RECOVERY_DIR, slug)
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f'v{lora_version}.safetensors')
        shutil.copy2(lora_path, backup_path)
        size_mb = os.path.getsize(backup_path) / 1024 / 1024
        print(f'[upload] backup {size_mb:.1f}MB -> {backup_path}')
        return backup_path
    except Exception as backup_err:
        print(f'[upload] WARN volume backup failed: {backup_err}')
        return None
