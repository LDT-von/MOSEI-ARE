"""Re-extract one original video in an isolated temporary copy; preserve delivery outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

BASE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=BASE.parent.parent)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    qa = BASE / 'qa'
    qa.mkdir(exist_ok=True)
    source_manifest = json.loads((BASE / 'input_manifest.json').read_text(encoding='utf-8'))
    source_manifest['rows'] = source_manifest['rows'][:1]
    sid = source_manifest['rows'][0]['sample_id']
    existing = BASE / 'outputs' / 'features' / f'{sid}.npz'
    before = digest(existing) if existing.exists() else None
    started = time.monotonic()
    # The temporary directory is strictly inside this problem's ignored QA folder.
    with tempfile.TemporaryDirectory(prefix='extraction_', dir=qa) as temporary:
        trial = Path(temporary).resolve()
        if qa.resolve() not in trial.parents:
            raise RuntimeError('unexpected temporary directory location')
        for name in ('pipeline.py', 'apply_quality_gate.py'):
            shutil.copy2(BASE / name, trial / name)
        (trial / 'input_manifest.json').write_text(json.dumps(source_manifest, ensure_ascii=False), encoding='utf-8')
        for weight in (BASE / 'assets').rglob('*.pth'):
            target = trial / 'assets' / weight.relative_to(BASE / 'assets')
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(weight, target)
            except OSError:
                shutil.copy2(weight, target)
        env = dict(os.environ, PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1')
        logs = []
        for command in (
            [sys.executable, str(trial / 'pipeline.py'), '--workspace', str(args.workspace.resolve()), '--device', args.device, '--limit', '1'],
            [sys.executable, str(trial / 'apply_quality_gate.py')],
        ):
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, timeout=240)
            logs.extend((result.stdout, result.stderr))
            (qa / 'smoke_check.log').write_text('\n'.join(logs), encoding='utf-8')
            if result.returncode:
                raise RuntimeError(f"extraction failed; see {qa / 'smoke_check.log'}\n{result.stderr[-3000:]}")
        with np.load(trial / 'outputs' / 'features' / f'{sid}.npz', allow_pickle=False) as arrays:
            length = len(arrays['grid_intervals'])
            for name, dim in (('text', 384), ('audio', 33), ('vision', 512)):
                if arrays[name].shape != (length, dim) or not np.isfinite(arrays[name]).all():
                    raise AssertionError(f'invalid {name} features')
                if np.any(arrays[name][~arrays[name + '_mask']] != 0):
                    raise AssertionError(f'{name} missing positions must be zero')
            report = {'status': 'passed', 'kind': 'isolated_one_video_smoke', 'sample_id': sid,
                      'aligned_length': length, 'dimensions': [384, 33, 512], 'device': args.device,
                      'pipeline_sha256': digest(BASE / 'pipeline.py'),
                      'elapsed_seconds': round(time.monotonic() - started, 2),
                      'existing_feature_unchanged': before is None or digest(existing) == before,
                      'scope': 'Extraction and quality-gate execution; not human timing accuracy.'}
    (qa / 'smoke_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
