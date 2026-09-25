"""Conservative, label-free screening of forced-alignment proposals.

Original words, embeddings and proposed times are always preserved. Rejected
proposals do not populate the default timed text feature tensor.
"""
import csv
import json
import re
from pathlib import Path
import numpy as np
from pipeline import save_json, sha256, overlap_pool

BASE = Path(__file__).resolve().parent

def lexical_recall(reference, hypothesis):
    a = re.findall(r"[A-Z]+(?:'[A-Z]+)*", reference.upper())
    b = re.findall(r"[A-Z]+(?:'[A-Z]+)*", hypothesis.upper())
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b):
            current.append(previous[j] + 1 if x == y else max(previous[j + 1], current[-1]))
        previous = current
    return previous[-1] / max(1, len(a))

def main():
    out = BASE / 'outputs'
    source = json.loads((BASE / 'input_manifest.json').read_text(encoding='utf-8'))
    rows = []
    gate_hash = sha256(__file__)
    # Read diagnostics output if present (produced by diagnostics.py).
    diag_path = out / 'diagnostic_summary.csv'
    health_path = out / 'alignment_health.json'
    diag_map = {}
    if diag_path.exists():
        import csv as _csv
        for r in _csv.DictReader(diag_path.open(encoding='utf-8-sig')):
            diag_map[r['sample_id']] = r
    health_doc = {}
    if health_path.exists():
        health_doc = json.loads(health_path.read_text(encoding='utf-8'))
    for original in source['rows']:
        meta_path = out / 'metadata' / (original['sample_id'] + '.json')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        row = meta['summary']
        feature_path = out / row['feature_file']
        with np.load(feature_path, allow_pickle=False) as opened:
            data = {k: opened[k] for k in opened.files}
        text = ' '.join(w['normalized'] for w in meta['words'])
        recall = lexical_recall(text, meta['ctc_greedy_transcript_for_review'])
        mean_score = float(data['raw_text_scores'].mean())
        sample_pass = mean_score >= .5 and recall >= .5
        usable = (data['raw_text_scores'] >= .5) & sample_pass
        data.setdefault('text_candidate', data['text'].copy())
        data.setdefault('text_candidate_mask', data['text_mask'].copy())
        features, valid, weights = overlap_pool(data['raw_text'].astype(np.float32),
                                                data['raw_text_intervals'], data['grid_intervals'], usable)
        data['text'] = features.astype(np.float16)
        data['text_mask'] = valid
        data['raw_text_usable'] = usable
        data['text_alignment_score'] = weights @ data['raw_text_scores']
        np.savez_compressed(feature_path, **data)
        row.update(text_valid_bins=int(valid.sum()), text_candidate_bins=int(data['text_candidate_mask'].sum()),
                   text_usable_words=int(usable.sum()), lexical_recall=round(recall, 6),
                   alignment_status='passed_automatic_screen' if sample_pass else 'unreliable_masked',
                   feature_sha256=sha256(feature_path))
        # Augment summary with diagnostic metrics if available.
        diag = diag_map.get(original['sample_id'])
        if diag:
            row.update({
                'words_per_second': round(float(diag['words_per_second']), 4),
                'speech_density': round(float(diag['speech_density']), 4),
                'mean_word_seconds': round(float(diag['mean_word_seconds']), 4),
                'n_short_pauses_gt_300ms': int(diag['n_short_pauses_gt_300ms']),
                'arousal_proxy_mean': round(float(diag['arousal_proxy_mean']), 4),
                'arousal_proxy_std': round(float(diag['arousal_proxy_std']), 4),
                'voiced_fraction': round(float(diag['voiced_fraction']), 4),
                'pitch_range_hz': round(float(diag['pitch_range_hz']), 2),
                'n_flagged_words_extended': int(diag['n_flagged']),
                'n_fallback_alignments': int(diag['n_fallback']),
                'alignment_health': diag['alignment_health'],
            })
        for word, keep in zip(meta['words'], usable):
            word['usable_for_timed_text'] = bool(keep)
            word['review_required'] = not bool(keep)
        row['review_word_count'] = int((~usable).sum())
        meta['quality_gate'] = {'sample_mean_ctc_min': .5, 'lexical_recall_min': .5,
                                'word_ctc_min': .5, 'lexical_recall': recall,
                                'sample_pass': bool(sample_pass), 'script_sha256': gate_hash,
                                'meaning': 'Heuristic automatic screening, not measured timing accuracy.',
                                'normalization_fit': 'No labels or validation outcomes used.'}
        save_json(meta_path, meta)
        rows.append(row)
    with (out / 'summary_100.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    sensitivity = []
    for threshold in [.4, .5, .6]:
        sensitivity.append({'ctc_threshold': threshold, 'lexical_recall_min': .5,
                            'samples_passing': sum(r['ctc_mean_score'] >= threshold and r['lexical_recall'] >= .5 for r in rows)})
    manifest_path = out / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['quality_gate_sha256'] = gate_hash
    manifest['quality_gate_thresholds'] = {'sample_ctc': .5, 'reference_word_recall': .5, 'word_ctc': .5}
    manifest['quality_threshold_sensitivity'] = sensitivity
    if health_doc:
        manifest['diagnostic_alignment_health'] = {
            'health_buckets': health_doc.get('health_buckets', {}),
            'flag_reason_counts': health_doc.get('flag_reason_counts', {}),
            'diagnostic_script_sha256': health_doc.get('diagnostic_script_sha256'),
        }
    save_json(manifest_path, manifest)
    print('Automatic screen passed:', sum(r['alignment_status'] == 'passed_automatic_screen' for r in rows), '/ 100')
    print('Retained but timed text masked:', sum(r['alignment_status'] == 'unreliable_masked' for r in rows))
    print(json.dumps(sensitivity))

if __name__ == '__main__':
    main()
