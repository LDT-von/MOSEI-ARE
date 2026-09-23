"""Rebuild the immutable source inventory from the supplied label workbook."""
import argparse
import hashlib
import json
from pathlib import Path
from openpyxl import load_workbook

def main(workspace):
    candidates = list(workspace.rglob('label-100.xlsx'))
    if len(candidates) != 1:
        raise ValueError(f'Expected one original label-100.xlsx; found {len(candidates)}')
    source = candidates[0]
    values = list(load_workbook(source, read_only=True, data_only=True).active.values)
    rows = []
    for values_row in values[1:]:
        row = dict(zip(values[0], values_row))
        video = source.parent / str(row['video_id']) / (str(row['clip_id']) + '.mp4')
        if not video.is_file():
            raise FileNotFoundError(video)
        row['source_relpath'] = video.relative_to(workspace).as_posix()
        row['sample_id'] = str(row['video_id']) + '__' + str(row['clip_id'])
        rows.append(row)
    assert len(rows) == len({r['sample_id'] for r in rows}) == 100
    expected = {r['source_relpath'] for r in rows}
    actual = {p.relative_to(workspace).as_posix() for p in source.parent.rglob('*.mp4')}
    assert expected == actual, 'Labels and original videos differ'
    value = {'source_label_path': source.relative_to(workspace).as_posix(),
             'source_label_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'rows': rows}
    target = Path(__file__).resolve().parent / 'input_manifest.json'
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Inventory verified: 100 source rows, 100 unique samples, 100 matching videos')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parent.parent.parent)
    main(parser.parse_args().workspace.resolve())
