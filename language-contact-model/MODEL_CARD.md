# Model card: shared neural contact checkpoint

Artifact: `artifacts/multimodal-contact-model.npz`.
Architecture: learned per-channel projection encoders, shared 24-unit tanh recurrence,
learned context transform, per-channel numerical forecasting and three named heads.
Implementation: NumPy; CPU; no external model/API used in inference or training.

Training: gesture pretraining for 45 epochs (seed 7), then 40 epochs of joint training
(optimization seed 27) with shuffled batches of 16. Speech/sign heads and encoders
were newly initialized. Targets and learning settings are specified in the scripts.
Training/held-out counts and all confusion matrices are retained in the reports.

Data: three attributed public UCI datasets, retrieved through GitHub mirrors. See
`data/README.md`. Speech comes from nine male speakers; movement datasets have
limited coverage. None supports broad demographic or cultural generalization.
No user-private Drive documents or actual personal interaction histories were used
as training data or included with this model.

Intended use: inspectable experiments in multimodal recurrent learning, contact
history, correction and task-specific adaptation. Current checkpoints are useful
for reproducing component evaluations and developing the broader model.

Limitations: low gesture/sign accuracy, no full signed-language translation, no
speech transcription, no conversational language training, no calibrated certainty.
Do not infer emotion, intent, health, personality or identity from unrelated task
outputs. Only the acoustic head is trained for identification among known speakers
in its small source corpus. It is not an identity-verification system.

The included context-history experiment is synthetic. Generalized relational
understanding and multi-person social reasoning are not validated by these results.

Compatibility: checkpoint restore validates parameter shapes and finite values;
serialized arrays use NumPy archives with `allow_pickle=False`.
