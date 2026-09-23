# Shared main model

The prediction model shared by Problems 2 and 3 belongs here. Maintain one authoritative implementation and record its revision in each experiment.

Problem-specific changes belong in the corresponding problem folder: missing-modality training and inference for Problem 2; evidence selection and explanation output for Problem 3. Store their separate checkpoints in each problem's ignored outputs directory.

`gap_slot_emo.py` is the shared Problem 2/3 aligned_50 model. It reads `text_bert` tokens, 74-dimensional audio, 35-dimensional vision and explicit time-position masks. Problem 2 controls synthetic gaps and optional same-position repair in its own folder. Problem 1's 384/33/512-dimensional features are a separate product and cannot directly replace Attachment 2 fields.
