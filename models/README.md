# Shared main model

The prediction model shared by Problems 2 and 3 belongs here. Maintain one authoritative implementation and record its revision in each experiment.

Problem-specific changes belong in the corresponding problem folder: missing-modality training and inference for Problem 2; evidence selection and explanation output for Problem 3. Store their separate checkpoints in each problem's ignored outputs directory.

The model has not been implemented yet. First fix its interface to the selected Attachment 2 release, valid-length fields, labels and metrics. Problem 1's 384/33/512-dimensional features are a separate product and cannot directly replace Attachment 2's 768/74/35-dimensional fields.
