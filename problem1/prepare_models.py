"""Download public feature-extractor weights only; no additional datasets."""
from pathlib import Path
import time
import requests

BASE = Path(__file__).resolve().parent / 'assets'
ITEMS = [
    ('https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth', BASE / 'wav2vec2_fairseq_base_ls960_asr_ls960.pth'),
    ('https://download.pytorch.org/models/resnet18-f37072fd.pth', BASE / 'checkpoints' / 'resnet18-f37072fd.pth'),
]

def download(url, target):
    if target.exists():
        print('EXISTS', target.name, flush=True)
        return
    target.parent.mkdir(exist_ok=True, parents=True)
    partial = target.with_suffix('.download')
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with requests.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                              stream=True, timeout=(20, 35)) as response:
                response.raise_for_status()
                append = response.status_code == 206 and offset > 0
                if append:
                    assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                else:
                    offset = 0
                expected = int(response.headers.get('Content-Length', 0))
                received = 0
                last_print = time.monotonic()
                with partial.open('ab' if append else 'wb') as handle:
                    for chunk in response.iter_content(256 * 1024):
                        handle.write(chunk); received += len(chunk)
                        if time.monotonic() - last_print > 20:
                            print(target.name, offset + received, flush=True)
                            last_print = time.monotonic()
                if expected and received != expected:
                    raise IOError('Incomplete content length')
            partial.replace(target)
            print('READY', target.name, target.stat().st_size, flush=True)
            return
        except (requests.RequestException, IOError) as exc:
            print('RETRY', target.name, attempt + 1, type(exc).__name__, flush=True)
    raise RuntimeError(f'Download failed: {target.name}')

if __name__ == '__main__':
    for url, target in ITEMS:
        download(url, target)
    from huggingface_hub import snapshot_download
    snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                      revision='86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d',
                      allow_patterns=['*.json', '*.safetensors', '1_Pooling/*'])
