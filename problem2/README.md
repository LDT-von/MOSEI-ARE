# Problem 2 Robust emotion prediction with local modality gaps

## Task

Train on the prescribed Attachment 2 training split and select settings on its validation split. Freeze the method before inference on the unlabeled Attachment 3 data. Handle continuous local gaps within a modality and report how gap type, position and duration affect performance.

Report polarity Accuracy and F1; report intensity MAE and Pearson correlation. Keep validation metrics separate from Attachment 3 predictions. The unlabeled special test cannot support an accuracy claim or threshold selection.

## Folder plan

- `data/`: Attachment 2/3 release and sample-ID mapping. Data stays outside Git.
- `configs/`: model, objective, masks, seed and validation-only decisions.
- `scripts/`: training, validation, missingness comparisons and Attachment 3 inference.
- `outputs/`: logs, predictions and experiment summaries. Large features and weights are Git-ignored.
- The shared model code belongs in the repository's `models/` directory.

## Status

Structure only. No model, prediction or performance result is claimed yet.
