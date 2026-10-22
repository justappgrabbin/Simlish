# Neural Language Contact Model

A runnable neural research model of contact across movement, acoustic observations,
and interaction history. It learns modality encoders, temporal state transitions,
relational-context transforms, next-observation decoders and expandable task heads.
The current shared checkpoint is trained on real gesture, speech-feature and
LIBRAS hand-trajectory data. The neural computation and training run locally.

The design treats language broadly: words, signing, bodily expression, timing,
environment and remembered encounters can be represented through registered
channels. The present data establishes component-level learning, not general
human understanding, conversational ability or full signed-language translation.

## Use the saved model

Requires Python 3.10+ and NumPy 2.x. Dataset setup uses network access; inference and training then run locally without API keys.

```sh
cd language-contact-model
python -m pip install -r requirements.txt
python -m neural.datasets  # once, to retrieve and verify public source data
python -m neural.quickstart
python -m unittest discover -s tests -v
```

If using the standalone ZIP, enter its extracted project directory instead.
`quickstart` restores the trained shared checkpoint, reads the verified source data,
and runs persistent sessions on held-out examples from all three tasks. It prints
expected labels, predicted distributions and uncertainty flags. Example predictions
can be incorrect; full evaluation is reported below.

## Measured results

| Shared checkpoint task | Held-out accuracy | Majority baseline | Macro F1 |
| --- | ---: | ---: | ---: |
| Gesture phases, a3 recording holdout | 56.25% | 35.86% | 0.363 |
| Nine known speakers, published utterance split | 95.41% | 11.89% | 0.954 |
| 15 LIBRAS movement types, duplicate-grouped split | 36.07% | 5.74% | 0.328 |

See `artifacts/multimodal-benchmark.json` for counts and confusion matrices. Gesture
and signing performance needs substantial improvement. The gesture model fails
on the held-out Hold class. Results do not establish participant/signer-independent
performance. The datasets are independent; this run does not train simultaneous
speech/gesture alignment. See [data provenance](data/README.md) for creators,
licenses, exact splits and preprocessing. No final checkpoint was selected on
these test scores; training schedules were fixed before each run.

A separate **synthetic** history-dependent diagnostic scored 100% with prior events
and 39.06% using only the final event. This checks that recurrence can use history;
it is not evidence of human-level memory or real-world interpretation.

## Train and evaluate

```sh
# Reproduce gesture pretraining, then shared speech/sign/gesture learning.
python -m neural.gesture_benchmark
python -m neural.multimodal_benchmark

# Train your own channel schema, heads and labeled/unlabeled event sequences.
python -m neural.cli train --config data/multimodal-config.json \
  --data data/multimodal-train.jsonl --output artifacts/custom.npz --epochs 40

python -m neural.evaluate --model artifacts/multimodal-contact-model.npz \
  --data data/multimodal-test.jsonl
```

The two benchmarks regenerate JSONL files for the CLI commands. Training supports
`--resume checkpoint.npz`, `--batch-size`, `--seed` and `--learning-rate`.
A JSONL training record has this form:

```json
{"events":[{"observations":{"speech":[0.1,0.2]}}],"targets":{"intent":"request"}}
```

Feature length must match your configuration. Omitted or null channels are unknown,
not observed zeros. Missing task evidence suppresses that output head. Labels and
heads are configuration, not a closed ontology of meanings. Unlabeled multi-event
records train next-observation prediction. User feedback supplies supervised targets.

## Persistent encounters and correction

```sh
python -m neural.cli session --db contact.db --participant local-user
# Use the returned session ID in subsequent commands.
python -m neural.cli observe --model artifacts/multimodal-contact-model.npz \
  --db contact.db --participant local-user --session SESSION_ID --event event.json
python -m neural.cli feedback --model artifacts/multimodal-contact-model.npz \
  --db contact.db --participant local-user --session SESSION_ID \
  --event-id EVENT_ID --head phase --label Stroke
python -m neural.cli adapt --model artifacts/multimodal-contact-model.npz \
  --db contact.db --participant local-user --session SESSION_ID \
  --replay data/multimodal-train.jsonl --output artifacts/adapted.npz
```

`event.json` contains one normalized observation event; the quickstart and
preprocessing functions show how to construct one. Replay reduces the risk of
forgetting but is not a guarantee of retained performance. Evaluate adapted models.
Session ownership checks prevent accidental mixing of participants; this local
SQLite application is not a multi-user authenticated service. Memory stores observed
events, context features, neural state, model fingerprint and append-only corrections.
After weight changes, the runtime replays history to rebuild state under current weights.

## Extending the model

`SequenceContactModel.add_channel(name, feature_count)`, `add_head(name, labels)`,
and `expand_head(name, labels)` support additional modalities and tasks. Channels
absent from an encounter do not execute their encoders. `forward()` exposes the
continuous temporal state and next-observation predictions independently of labels.
The parameter count and hidden size are configurable; the saved 24-unit checkpoint
is an initial experiment, not an architectural ceiling.

`neural/adapters.py` includes ordered UTF-8 byte inputs, raw PCM WAV frames, and
spatial landmark sequences. These adapters preserve signals; they do not themselves
recognize words or translate signs. The saved speech checkpoint expects normalized
12-dimensional LPC features, not raw PCM. Additional backbones, attention, larger
training runs and aligned conversational data remain future work.

## Files and lineage

- `neural/sequence_model.py`: trainable recurrent multimodal core and checkpoints.
- `neural/runtime.py`, `memory.py`: persistent inference, context and adaptation.
- `neural/cli.py`, `evaluate.py`, `quickstart.py`: usable entry points.
- `artifacts/`: trained weights, normalization, benchmark evidence.
- `data/`: dataset attribution, source checksums and model configuration. The download
  command retrieves the public raw data; the standalone ZIP also includes it.
- `tests/`: numerical gradients, missingness, expansion, persistence and correction.
- The standalone ZIP additionally preserves `src/` and `legacy/`, the original
  Klein-inspired TypeScript drafts. They are not dependencies of the neural runtime.
- `docs/ORIGINAL_README.md`: original claims preserved for lineage, not current docs.

Klein's contact simulation, AUTOLING and DISEMINER motivate interaction-driven
learning and relational inference. The neural implementation is new engineering,
not a verified reproduction of his programs. [Architecture](ARCHITECTURE.md) and
[model card](MODEL_CARD.md) distinguish implementation from the broader research goal.
