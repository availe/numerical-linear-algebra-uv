"""Download the course's original datasets, verifying the pinned content."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent

@lru_cache(maxsize=None)
def ensure_data(name):
    manifest = json.loads((ROOT / 'course-data.json').read_text())
    entry = manifest[name]
    path = ROOT / 'data' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    def valid(candidate):
        if candidate.stat().st_size != entry['size']:
            return False
        with candidate.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256']
    if path.exists():
        if not valid(path):
            raise ValueError(f'{path} does not match the course dataset; move it aside and retry.')
        return path
    partial = path.with_suffix(path.suffix + '.partial')
    print(f'Downloading {name} ({entry["size"] / 1e6:.1f} MB)...', flush=True)
    try:
        with urllib.request.urlopen(entry['url'], timeout=120) as response, partial.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if not valid(partial):
            raise ValueError(f'Checksum mismatch for {name}; download was not accepted.')
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path
