"""Example reader: padding and modality availability are different masks."""
from pathlib import Path
import numpy as np

def load_sample(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def collate_samples(paths):
    rows = [load_sample(p) for p in paths]
    lengths = np.array([len(row['grid_intervals']) for row in rows], dtype=np.int64)
    size = int(lengths.max())
    result = {'lengths': lengths, 'padding_mask': np.arange(size)[None] < lengths[:, None]}
    for key in ['grid_intervals', 'text', 'audio', 'vision', 'text_mask', 'audio_mask', 'vision_mask']:
        example = rows[0][key]
        batch = np.zeros((len(rows), size) + example.shape[1:], dtype=example.dtype)
        for i, row in enumerate(rows):
            batch[i, :lengths[i]] = row[key]
        result[key] = batch
    return result

if __name__ == '__main__':
    files = sorted((Path(__file__).resolve().parent / 'outputs' / 'features').glob('*.npz'))
    batch = collate_samples(files[:2])
    print({key: value.shape for key, value in batch.items()})
