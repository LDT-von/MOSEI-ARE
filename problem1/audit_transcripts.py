"""Independent pretrained ASR spot check; it never replaces supplied transcripts."""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from pipeline import media_metadata, decode_audio, save_json

BASE = Path(__file__).resolve().parent
MODEL = 'openai/whisper-small'
REVISION = '973afd24965f72e36ca33b3055d56a652f456b4d'

def main():
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL, revision=REVISION, local_files_only=True).to(device).eval()
    ids = ['-3g5yACwYnA__13', '-iRBcNs9oI8__7', '-mJ2ud6oKI8__1', '-NFrJFQijFE__2']
    rows = json.loads((BASE / 'input_manifest.json').read_text(encoding='utf-8'))['rows']
    result = []
    for sid in ids:
        row = next(r for r in rows if r['sample_id'] == sid)
        source = BASE.parent.parent / row['source_relpath']
        signal, present = decode_audio(source, media_metadata(source))
        features = processor(signal, sampling_rate=16000, return_tensors='pt').input_features.to(device)
        with torch.inference_mode():
            tokens = model.generate(features, language='en', task='transcribe', max_new_tokens=180, do_sample=False)
        text = processor.batch_decode(tokens, skip_special_tokens=True)[0]
        result.append({'sample_id': sid, 'provided_text': row['text'], 'independent_asr_text': text,
                       'signal_rms': float(np.sqrt(np.mean(signal ** 2))), 'available_fraction': float(present.mean()),
                       'note': 'Automatic spot check only; not a manual transcription or replacement label.'})
        print(json.dumps(result[-1], ensure_ascii=False), flush=True)
    save_json(BASE / 'outputs' / 'independent_asr_audit.json', {'model': MODEL, 'revision': REVISION, 'rows': result})

if __name__ == '__main__':
    main()
