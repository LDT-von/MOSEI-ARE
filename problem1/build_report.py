"""Fill the 100-row chapter table and emit anonymous processing logs."""
import csv
import importlib.metadata
import json
import platform
from pathlib import Path
import cv2

BASE = Path(__file__).resolve().parent
OUT = BASE / 'outputs'

def main():
    quality = json.loads((OUT / 'quality_report.json').read_text(encoding='utf-8'))
    run = json.loads((OUT / 'run_manifest.json').read_text(encoding='utf-8'))
    assert quality['source_samples'] == quality['feature_samples'] == 100
    assert quality['samples_passing_automatic_alignment_screen'] == 79
    with (OUT / 'summary_100.csv').open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 100
    chapter_path = BASE / '问题一建模与结果.md'
    chapter = chapter_path.read_text(encoding='utf-8')
    start = '<!-- FULL_SAMPLE_TABLE -->'
    end = '<!-- END_FULL_SAMPLE_TABLE -->'
    assert start in chapter
    table = ['| 样本编号 | 播放时长 秒 | 模态维度 T/A/V | 对齐粒度 秒 | 对齐位置 | 词数 | 平均CTC分数 | 自动筛查 |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        status = '通过' if row['alignment_status'] == 'passed_automatic_screen' else '屏蔽'
        dims = f"{row['text_dim']}/{row['audio_dim']}/{row['vision_dim']}"
        table.append(f"| {row['sample_id']} | {float(row['duration_seconds']):.3f} | {dims} | {row['grid_seconds']} | {row['aligned_length']} | {row['word_count']} | {float(row['ctc_mean_score']):.3f} | {status} |")
    replacement = start + '\n\n' + '\n'.join(table) + '\n\n' + end
    if end in chapter:
        left, rest = chapter.split(start, 1)
        _, right = rest.split(end, 1)
        chapter = left + replacement + right
    else:
        chapter = chapter.replace(start, replacement)
    chapter_path.write_text(chapter, encoding='utf-8')
    environment = {
        'python': platform.python_version(),
        'imported_opencv_runtime': cv2.__version__,
        'installed_opencv_distributions': {name: importlib.metadata.version(name)
                                           for name in ['opencv-python', 'opencv-contrib-python', 'opencv-python-headless']},
        'opencv_note': 'The source environment has overlapping OpenCV distributions; reproduction requirements select one matching the imported cv2 runtime.',
        'recorded_package_versions': run['versions'],
    }
    (OUT / 'environment.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding='utf-8')
    with (OUT / 'processing_log.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            meta = json.loads((OUT / row['metadata_file']).read_text(encoding='utf-8'))
            event = {'sample_id': row['sample_id'], 'status': row['status'],
                     'alignment_status': row['alignment_status'], 'duration_seconds': float(row['duration_seconds']),
                     'runtime_seconds': meta['runtime_seconds'], 'source_sha256': row['source_sha256'],
                     'feature_sha256': row['feature_sha256']}
            handle.write(json.dumps(event, ensure_ascii=False) + '\n')
    print('Chinese chapter: 100 rows. Anonymous processing log: 100 rows.')

if __name__ == '__main__':
    main()
