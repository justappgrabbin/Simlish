# Changes

- Preserved original archive and README; separated modular and monolithic drafts.
- Matched TypeScript source layout to its existing compiler configuration.
- Added an actual trainable neural contact foundation with arbitrary feature
  channels, explicit missingness, continuous representations and channel expansion.
- Added weight checkpoints, a reproducible held-out diagnostic, numerical gradient
  checks, missing-channel invariance and checkpoint/expansion verification.
- Replaced unsupported capability claims with implementation status and a broader
  neural architecture roadmap. TypeScript runtime audit remains outstanding.

## Recurrent multimodal implementation

- Added actual backpropagation through time, per-modality encoders, learned context
  transforms, expandable named heads and next-observation training.
- Added SQLite sessions, participant separation, state restoration, corrections,
  adaptation and replay, plus schema-aware model fingerprints.
- Added raw PCM, UTF-8 and landmark adapters, training/inference/evaluation CLI and
  a runnable checkpoint quickstart.
- Retrieved and attributed public gesture, acoustic and LIBRAS movement data;
  trained a shared checkpoint and retained held-out confusion matrices.
- Added per-head evidence requirements, finite-difference temporal gradient checks,
  checkpoint/expansion/session tests and a separate synthetic temporal diagnostic.
- Documented weak movement results and remaining broad-model research requirements.
