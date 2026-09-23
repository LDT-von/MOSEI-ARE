# Repository instructions

## Organization

- Keep each contest problem's implementation, configuration, validation and results in its own top-level folder: problem1, problem2 or problem3.
- The main model shared by Problems 2 and 3 lives once in models/. Each problem owns its task-specific training, validation, inference, config and outputs.
- Shared experiment conventions live in configs/. Do not copy the shared model into each problem folder.
- Raw contest attachments remain outside the repository. Do not commit raw data, generated feature tensors, checkpoints, weights, caches or machine-specific paths.

## Evidence and reproducibility

- Treat the 2026 E题 Word document and supplied workbooks as the task authority. Preserve all supplied samples and labels.
- Keep training, validation, held-out test and unlabeled special-test evidence distinct. Never claim an unrun metric or describe a structural check as a performance result.
- Record feature release, split identity, seed, config, model version, checkpoint and inference version for every reported experiment.
- Save each special-test output in its own problem folder. Never tune settings using unlabeled special-test data.
- Record source sample IDs and time/frame coordinates for temporal evidence and explanations.

## Repository changes

- Follow .gitignore for generated files. Publish only the requested, size-checked contest attachment archive.
- Before changing SlotEmo, inspect executable model code, masks, trainer, split, metric and exact result provenance.
