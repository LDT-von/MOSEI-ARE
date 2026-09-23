# Problem 3 Interpretable emotion prediction

## Task

Train and select the model on the prescribed Attachment 2 training and validation splits. Freeze it before inference on the unlabeled Attachment 4 samples. For each prediction, provide polarity, intensity, modality contribution and traceable evidence that points to transcript text, audio time or an original video frame.

Evaluate Accuracy/F1 for polarity and MAE/Pearson for intensity on validation data. Define how evidence importance is measured and how selected evidence maps back to the source. Keep validation metrics separate from Attachment 4 outputs.

## Folder plan

- `data/`: Attachment 2/4 release and ID mapping. Data stays outside Git.
- `configs/`: explanation objective, evidence budget, thresholds and validation settings.
- `scripts/`: training, validation, explanation checks, evidence export and Attachment 4 inference.
- `outputs/`: explanation records, examples, plots and predictions. Large features and weights are Git-ignored.
- The shared model code belongs in the repository's `models/` directory.

## Status

Structure only. No model, prediction or performance result is claimed yet.
