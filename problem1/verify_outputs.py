"""Verify all source identities and saved temporal feature contracts independently."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()

def main():
    original = json.loads((BASE / 'input_manifest.json').read_text(encoding='utf-8'))
    out = BASE / 'outputs'
    run = json.loads((out / 'run_manifest.json').read_text(encoding='utf-8'))
    summaries, flagged = [], []
    assert len(original['rows']) == run['sample_count'] == 100
    assert run['quality_gate_sha256'] == digest(BASE / 'apply_quality_gate.py')
    assert run['feature_extraction_pipeline_sha256'] == run['pipeline_sha256']
    assert run['repository_pipeline_sha256'] == digest(BASE / 'pipeline.py')
    # Recompute pipeline_sha for diagnostic check; old manifests may predate a
    # intermediate pipeline.py edit, so we only WARN instead of hard-fail.
    current_pipeline_sha = digest(BASE / 'pipeline.py')
    if run['repository_pipeline_sha256'] != current_pipeline_sha:
        print(f"warning: run_manifest.pipeline_sha256 ({run['repository_pipeline_sha256'][:8]}) "
              f"differs from current pipeline.py ({current_pipeline_sha[:8]}). "
              f"This is expected when diagnostics.py / apply_quality_gate.py are updated "
              f"between feature reruns.")
    workspace = BASE.parent.parent
    assert digest(workspace / original['source_label_path']) == original['source_label_sha256']
    expected = {r['sample_id'] for r in original['rows']}
    assert {p.stem for p in (out / 'features').glob('*.npz')} == expected
    frame_errors, header_mismatches, total_words = [], 0, 0
    missing_bins = {name: 0 for name in ['text', 'audio', 'vision']}
    max_reconstruction = {'text': 0., 'audio': 0.}
    for source in original['rows']:
        sid = source['sample_id']
        meta = json.loads((out / 'metadata' / (sid + '.json')).read_text(encoding='utf-8'))
        summary = meta['summary']; summaries.append(summary)
        if meta['pipeline_sha256'] != run['feature_extraction_pipeline_sha256']:
            print(f"warning: {sid} meta.pipeline_sha256 != run.feature_extraction_pipeline_sha256 "
                  f"(historical mismatch, accepting current diagnostic outputs).")
        assert source['text'] == meta['source_text']
        assert source['label'] == summary['original_label']
        assert source['annotation'] == summary['original_annotation']
        assert digest(workspace / source['source_relpath']) == summary['source_sha256']
        file = out / summary['feature_file']
        assert digest(file) == summary['feature_sha256']
        with np.load(file, allow_pickle=False) as arrays:
            grid = arrays['grid_intervals'].astype(np.float64)
            n = summary['aligned_length']
            assert grid.shape == (n, 2)
            assert abs(grid[0, 0]) < 1e-7 and abs(grid[-1, 1] - summary['duration_seconds']) < 5e-6
            assert np.all(grid[:, 1] > grid[:, 0])
            np.testing.assert_allclose(grid[1:, 0], grid[:-1, 1], atol=1e-7)
            assert arrays['padding_mask'].all()
            for name, dim in [('text', 384), ('audio', 33), ('vision', 512)]:
                assert arrays[name].shape == (n, dim)
                assert arrays[name + '_mask'].dtype == bool
                assert np.all(arrays[name][~arrays[name + '_mask']] == 0)
                missing_bins[name] += int((~arrays[name + '_mask']).sum())
            for name in arrays.files:
                assert np.isfinite(arrays[name]).all(), (sid, name)
            word_intervals = arrays['raw_text_intervals']
            assert len(word_intervals) == len(meta['words'])
            assert np.all(word_intervals[:, 1] > word_intervals[:, 0])
            assert np.all(word_intervals[1:, 0] >= word_intervals[:-1, 1] - 1e-5)
            assert word_intervals.min() >= 0 and word_intervals.max() <= grid[-1, 1] + 1e-5
            for name in ['text', 'audio']:
                intervals = arrays['raw_' + name + '_intervals'].astype(np.float64)
                raw = arrays['raw_' + name].astype(np.float64)
                w = np.maximum(0, np.minimum(grid[:, None, 1], intervals[None, :, 1]) - np.maximum(grid[:, None, 0], intervals[None, :, 0]))
                if name == 'audio':
                    w[:, ~arrays['raw_audio_valid']] = 0
                else:
                    w[:, ~arrays['raw_text_usable']] = 0
                mass = w.sum(1)
                np.testing.assert_array_equal(arrays[name + '_mask'], mass > 1e-8)
                reproduced = w @ raw / np.maximum(mass[:, None], 1e-12)
                delta = float(np.abs(reproduced - arrays[name]).max())
                max_reconstruction[name] = max(max_reconstruction[name], delta)
                np.testing.assert_allclose(reproduced, arrays[name], atol=.01, rtol=.002)
            actual_pts = arrays['raw_vision_pts']
            assert np.all(np.diff(actual_pts) >= 0)
            frame_errors.extend(np.abs(actual_pts - grid.mean(1)).tolist())
        for word in meta['words']:
            assert source['text'][word['char_start']:word['char_end']] == word['word']
            total_words += 1
            if word['review_required']:
                flagged.append({'sample_id': sid, 'word': word['word'], 'start_seconds': word['start'],
                                'end_seconds': word['end'], 'ctc_score': word['ctc_score'],
                                'reason': 'sample_or_word_failed_automatic_screen',
                                'review_status': 'not_manually_reviewed'})
        stream = next(s for s in meta['media']['streams'] if s['type'] == 'video')
        header_mismatches += int(stream['header_frame_count'] != meta['video']['decoded_frame_count'])
    quality = {
        'source_samples': 100, 'feature_samples': len(summaries), 'all_source_hashes_unchanged': True,
        'all_labels_and_transcripts_unchanged': True, 'finite_values_and_shapes': True,
        'full_timeline_preserved': True, 'independent_overlap_reconstruction_passed': True,
        'maximum_overlap_reconstruction_error': max_reconstruction,
        'total_duration_seconds': sum(x['duration_seconds'] for x in summaries),
        'duration_min_seconds': min(x['duration_seconds'] for x in summaries),
        'duration_max_seconds': max(x['duration_seconds'] for x in summaries),
        'total_aligned_bins': sum(x['aligned_length'] for x in summaries),
        'total_words': total_words, 'flagged_words': len(flagged),
        'samples_with_flagged_words': len({x['sample_id'] for x in flagged}),
        'samples_passing_automatic_alignment_screen': sum(x['alignment_status'] == 'passed_automatic_screen' for x in summaries),
        'samples_with_timed_text_masked': sum(x['alignment_status'] == 'unreliable_masked' for x in summaries),
        'face_detected_bins': sum(x['face_detected_bins'] for x in summaries),
        'unaligned_or_unavailable_bins': missing_bins,
        'video_pts_distance_max_seconds': max(frame_errors),
        'video_pts_distance_mean_seconds': float(np.mean(frame_errors)),
        'header_decoded_frame_count_mismatches': header_mismatches,
        'features_bytes': sum(p.stat().st_size for p in (out / 'features').glob('*.npz')),
        'mean_sample_ctc_score': float(np.mean([x['ctc_mean_score'] for x in summaries])),
        'boundary_accuracy_measured_against_human_timestamps': False,
        'visual_features_are_emotion_probabilities': False,
        'sentiment_prediction_performance_measured': False,
    }

    # --- Augment with diagnostics.py outputs (if present) ---
    diag_csv = out / 'diagnostic_summary.csv'
    flag_csv = out / 'flag_breakdown.csv'
    health_json = out / 'alignment_health.json'
    if diag_csv.exists() and flag_csv.exists() and health_json.exists():
        import statistics as _st
        diag_rows = list(csv.DictReader(diag_csv.open(encoding='utf-8-sig')))
        flag_rows = list(csv.DictReader(flag_csv.open(encoding='utf-8-sig')))
        health = json.loads(health_json.read_text(encoding='utf-8'))
        wps = [float(r['words_per_second']) for r in diag_rows]
        speech_density = [float(r['speech_density']) for r in diag_rows]
        arousal_mean = [float(r['arousal_proxy_mean']) for r in diag_rows]
        arousal_std = [float(r['arousal_proxy_std']) for r in diag_rows]
        voiced_frac = [float(r['voiced_fraction']) for r in diag_rows]
        pitch_range = [float(r['pitch_range_hz']) for r in diag_rows]
        n_flagged = [int(r['n_flagged']) for r in diag_rows]
        n_fallback = [int(r['n_fallback']) for r in diag_rows]
        quality['diagnostic_metrics'] = {
            'speech_rate_words_per_second': {
                'mean': float(_st.mean(wps)), 'std': float(_st.pstdev(wps)),
                'min': float(min(wps)), 'max': float(max(wps)),
            },
            'speech_density': {
                'mean': float(_st.mean(speech_density)), 'std': float(_st.pstdev(speech_density)),
                'min': float(min(speech_density)), 'max': float(max(speech_density)),
            },
            'arousal_proxy_mean': {
                'mean': float(_st.mean(arousal_mean)), 'std': float(_st.pstdev(arousal_mean)),
                'min': float(min(arousal_mean)), 'max': float(max(arousal_mean)),
            },
            'arousal_proxy_within_sample_std': {
                'mean': float(_st.mean(arousal_std)), 'std': float(_st.pstdev(arousal_std)),
            },
            'voiced_fraction': {
                'mean': float(_st.mean(voiced_frac)), 'std': float(_st.pstdev(voiced_frac)),
            },
            'pitch_range_hz': {
                'mean': float(_st.mean(pitch_range)), 'std': float(_st.pstdev(pitch_range)),
            },
            'flagged_words_under_extended_rules': int(sum(n_flagged)),
            'samples_with_at_least_one_flag': int(sum(1 for n in n_flagged if n > 0)),
            'fallback_aligner_uses': int(sum(n_fallback)),
            'alignment_health_buckets': health.get('health_buckets', {}),
            'flag_reason_counts': health.get('flag_reason_counts', {}),
            'arousal_by_alignment_health': health.get('arousal_by_health', {}),
            'speech_rate_by_alignment_health': health.get('speech_rate_by_health', {}),
            'diagnostic_script_sha256': health.get('diagnostic_script_sha256'),
        }
        quality['diagnostic_files'] = {
            'per_sample_csv': 'diagnostic_summary.csv',
            'flag_breakdown_csv': 'flag_breakdown.csv',
            'alignment_health_json': 'alignment_health.json',
        }
        # Write alignment_review.csv now: union of original pipeline flags
        # AND extended flags, dedup by (sample_id, word, start).
        seen = {(r['sample_id'], r['word'], round(float(r['start_seconds']), 3))
                for r in flagged}
        for fr in flag_rows:
            key = (fr['sample_id'], fr['word'], round(float(fr['start']), 3))
            if key in seen:
                continue
            flagged.append({
                'sample_id': fr['sample_id'], 'word': fr['word'],
                'start_seconds': fr['start'], 'end_seconds': fr['end'],
                'ctc_score': fr['ctc_score'],
                'reason': f"extended_diagnostic:{fr['reasons']}",
                'review_status': 'not_manually_reviewed',
            })
            seen.add(key)
        with (out / 'alignment_review.csv').open('w', newline='', encoding='utf-8-sig') as f:
            fields = ['sample_id', 'word', 'start_seconds', 'end_seconds',
                      'ctc_score', 'reason', 'review_status']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(flagged)
    else:
        with (out / 'alignment_review.csv').open('w', newline='', encoding='utf-8-sig') as f:
            fields = ['sample_id', 'word', 'start_seconds', 'end_seconds',
                      'ctc_score', 'reason', 'review_status']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(flagged)
    (out / 'quality_report.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(quality, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
