"""Explicit download of one pinned public checkpoint; never executes model code."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = 'Qwen/Qwen3-4B-Instruct-2507'
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'

if __name__ == '__main__':
    destination = Path('models/qwen3-4b-instruct')
    snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=destination,
        allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', 'LICENSE'], max_workers=2)
    (destination/'source.json').write_text(json.dumps(
        {'model_id': MODEL_ID, 'revision': REVISION, 'precision': 'original BF16 weights',
         'source': 'https://huggingface.co/'+MODEL_ID}, indent=2)+'\n')
    print('Downloaded pinned checkpoint:', MODEL_ID, REVISION, flush=True)
