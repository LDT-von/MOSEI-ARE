"""Report playback timeline vs encoded payload without changing source videos."""
import json
from pathlib import Path
import av
import numpy as np
from pipeline import save_json

BASE = Path(__file__).resolve().parent

def main():
    original = json.loads((BASE / 'input_manifest.json').read_text(encoding='utf-8'))['rows']
    records = []
    for row in original:
        path = BASE.parent.parent / row['source_relpath']
        meta = json.loads((BASE / 'outputs' / 'metadata' / (row['sample_id'] + '.json')).read_text(encoding='utf-8'))
        with av.open(str(path), options={'ignore_editlist': '1'}) as c:
            payload_duration = c.duration / av.time_base
        with av.open(str(path)) as c:
            energy, count, max_abs = None, 0, 0.
            for frame in c.decode(audio=0):
                array = frame.to_ndarray().astype(np.float64)
                e = np.sum(array * array, axis=1)
                energy = e if energy is None else energy + e
                count += array.shape[1]; max_abs = max(max_abs, float(np.max(np.abs(array))))
        video = next(x for x in meta['media']['streams'] if x['type'] == 'video')
        records.append({'sample_id': row['sample_id'], 'presentation_duration_seconds': meta['media']['duration'],
                        'encoded_payload_duration_ignoring_edit_list_seconds': payload_duration,
                        'header_video_frames': video['header_frame_count'],
                        'presentation_decoded_video_frames': meta['video']['decoded_frame_count'],
                        'audio_channel_rms': np.sqrt(energy / max(count, 1)).tolist(),
                        'audio_max_abs': max_abs, 'audio_all_channels_exactly_zero': max_abs == 0.,
                        'note': 'Feature extraction honors the MP4 edit list. Payload-only times are diagnostic, not substituted.'})
    save_json(BASE / 'outputs' / 'media_audit.json', {'rows': records})
    print('Exactly zero audio in every decoded channel:', [r['sample_id'] for r in records if r['audio_all_channels_exactly_zero']])

if __name__ == '__main__':
    main()
