"""Download immutable upstream files and verify recorded SHA-256 checksums.

Large raw data and model weights are deliberately excluded from the project ZIP.
Read THIRD_PARTY_NOTICES.md before downloading or publishing derivatives.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def download(url,path,expected):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        part=path.with_suffix(path.suffix+'.part')
        with urllib.request.urlopen(url,timeout=120) as response,part.open('wb') as stream:
            while chunk:=response.read(4*1024*1024):
                stream.write(chunk)
        part.replace(path)
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=expected:
        raise RuntimeError(f'Checksum mismatch: {path}; remove it and download again')
    print(f'Verified {path.name}: {path.stat().st_size/1e6:.1f} MB',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw-dir',type=Path,default=ROOT/'raw')
    parser.add_argument('--checkpoint-dir',type=Path,default=ROOT/'checkpoints/timesfm3')
    parser.add_argument('--skip-model',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'data/source_manifest.json').read_text())
    for entry in manifest['files']:
        is_data='/datasets/' in entry['url']
        if not is_data and args.skip_model:
            continue
        destination=(args.raw_dir if is_data else args.checkpoint_dir)/entry['file']
        download(entry['url'],destination,entry['sha256'])
