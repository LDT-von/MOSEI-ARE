# Problem 3 Interpretable emotion prediction

## Task

Train and select the model on the prescribed Attachment 2 training and validation splits. Freeze it before inference on the unlabeled Attachment 4 samples. For each prediction, provide polarity, intensity, modality contribution and traceable evidence that points to transcript text, audio time or an original video frame.

Evaluate Accuracy/F1 for polarity and MAE/Pearson for intensity on validation data. Define how evidence importance is measured and how selected evidence maps back to the source. Keep validation metrics separate from Attachment 4 outputs.

## Folder plan

- `data/`: loaders for Attachment 2 (`aligned_50.pkl`) and Attachment 4 aligned-version pkl files.
- `explain.py`: counterfactual occlusion per modality and the sliding-window ablation that picks the top-k contiguous windows whose ablation drops the predicted-class logit the most.
- `scripts/run.py`: training-free entry points (`evaluate-valid`, `evaluate-test`, `evidence-valid`, `evidence-test`, `infer`) that all consume a frozen Problem 2 `best.pt`.
- `configs/`: window length and top-k defaults.
- `outputs/`: not used directly; Problem 3 writes under `problem2/outputs/<arm>/problem3_*`.

## Method

The frozen GapSlotEmo (best Arm C from Problem 2) is queried in three ways for each sample:

1. **Clean forward pass** produces the polarity / intensity predictions and the model's own per-modality weight, recorded only as auxiliary information.
2. **Modality-level counterfactual ablation**: zero the observed features of one modality at a time on originally-valid positions, re-run the model, and compute the change in the predicted-class logit and the change in regression intensity. The `normalized_magnitude` field sums to one across modalities; the modality with the largest normalized share is the `dominant_modality`.
3. **Window-level evidence scan**: for each modality, slide a fixed-length contiguous window (default 6 positions = 1.2 s) along the 50-position sequence, ablate only that window, and record the top-k windows whose ablation drops the predicted-class logit the most.

All numbers are **prediction dependencies**, not causal attributions. They answer "what would change if this modality/region were missing at inference time?" not "why does the speaker feel this way?".

## Mapping back to source

- Aligned positions 0–49 correspond to 0.1 s … 9.9 s in the original clip (50 × 0.2 s grid, centres reported in seconds).
- Each window's `start_seconds` / `end_seconds` is the aligned-grid span; it is the correct range for jumping in the original MP4 file, not a precise word- or frame-level timestamp.
- The `raw_text` snippet, the saved `video_filename`, and the per-sample evidence JSON together are the input to a human reviewer; no automatic word/frame alignment is performed here.
- The text modality skips the `[CLS]` (id=101) and `[SEP]` (id=102) positions when picking top windows, so reported windows always point at content tokens.

## Status

Implemented and tested. Frozen Problem 2 Arm C checkpoint reused. CUDA runs for:

- `evaluate-valid` (32-sample evidence), `evaluate-test` (full 727-sample metrics + 32-sample evidence), `infer` (20 Attachment 4 samples + evidence + provenance).
- See `problem2/outputs/arm_C_seed_42/problem3_test_evaluation.json` and `problem2/outputs/arm_C_seed_42/problem3_attachment4/attachment4_predictions.csv`.

赛题文献与实际数据接口审计后的建模路线见 [问题二三文献解读与建模路线](../docs/问题二三文献解读与建模路线.md)。

## Run

From the repository root with `MOSEI_DATA_ROOT` pointing at the directory that contains 附件2 and 附件4:

```powershell
$env:MOSEI_DATA_ROOT="C:\Users\栋栋\Desktop\E题\E题数据\E题数据"
python -m problem3.scripts.run evaluate-valid --checkpoint problem2/outputs/arm_C_seed_42/best.pt --sample-size 32 --device cuda
python -m problem3.scripts.run evaluate-test  --checkpoint problem2/outputs/arm_C_seed_42/best.pt --device cuda
python -m problem3.scripts.run evidence-valid  --checkpoint problem2/outputs/arm_C_seed_42/best.pt --sample-size 32 --device cuda
python -m problem3.scripts.run evidence-test   --checkpoint problem2/outputs/arm_C_seed_42/best.pt --sample-size 32 --device cuda
python -m problem3.scripts.run infer          --checkpoint problem2/outputs/arm_C_seed_42/best.pt --device cuda --window-length 6 --window-top-k 3
```

`evaluate-*` only computes polarity/intensity and metrics (fast, full split). `evidence-*` additionally performs the expensive per-sample counterfactual + window scan (default 32 samples; raise `--sample-size` to extend). Attachment 4 inference is `infer`; the script refuses to load any smoke-only checkpoint.

Unit tests:

```powershell
python -m unittest problem3.test_pipeline -v
```
