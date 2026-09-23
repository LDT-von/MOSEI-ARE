"""Pack Problem 1 only, leaving original videos and large public weights out."""
import hashlib
import json
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
OUTPUT = ROOT / 'problem1_submission.zip'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    quality = json.loads((BASE / 'outputs' / 'quality_report.json').read_text(encoding='utf-8'))
    assert quality['feature_samples'] == 100
    sources = []
    for path in BASE.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(BASE)
        if relative.parts[0] in {'assets', 'qa', '__pycache__'}:
            continue
        if path.suffix.lower() in {'.mp4', '.pth', '.pt', '.download', '.partial', '.pyc'}:
            continue
        if path.name in {'extraction.log', 'asr_audit.log', 'errors.jsonl', 'package_manifest.json'}:
            continue
        sources.append(path)
    assert sum(path.suffix == '.npz' for path in sources) == 100
    with zipfile.ZipFile(OUTPUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(sources):
            archive.write(path, arcname=(Path('problem1') / path.relative_to(BASE)).as_posix())
    size = OUTPUT.stat().st_size
    assert size <= 50_000_000, f'Problem 1 alone exceeds 50MB: {size}'
    manifest = {'file_name': OUTPUT.name, 'bytes': size, 'sha256': digest(OUTPUT),
                'included_file_count': len(sources), 'feature_files': 100,
                'excluded': ['original_mp4', 'pretrained_weights', 'caches', 'local_tool_logs']}
    (BASE / 'outputs' / 'package_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
