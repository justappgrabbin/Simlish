# Neural architecture and research scope

## Implemented and exercised

Registered numeric observations → channel-specific learned encoders and presence
vectors → shared recurrent contact state → learned output heads and next-observation
decoders → evidence-gated predictions or clarification flags → stored encounter and
feedback → backpropagation through time with optional replay.

The update is `h_t = tanh(h_(t-1) R + sum(x_m E_m + p_m) + context M + b)`.
Each observed modality contributes its own features and presence vector. An unknown
channel contributes neither. `R`, `E`, `p`, `M`, decoders and task heads are learned.
Context messages can aggregate a participant's other explicitly selected sessions;
the runtime never automatically fetches another participant's private state.
The context transform has verified numerical gradients, but the public datasets do
not include relational contexts, so its utility on real social relations is unproven.

Gradients are implemented in NumPy and checked against finite differences for all
parameters through multiple timesteps. Classification and masked next-observation
objectives train jointly, with global gradient clipping. Feature dimensions, hidden
size, tasks and label inventories are configurable. Input channels and label heads
can expand; this does not guarantee zero forgetting after subsequent learning.

Continuous state supports more than classification. Numerical next-observation
prediction is implemented. Task heads can be added, including a byte output head,
but no conversational text generator has been trained. Predictions are feature-space
estimates and must be interpreted using each adapter's units and normalization.

SQLite stores session ownership, ordered observations, model fingerprints, state and
correction history. Feedback turns encounter prefixes into training records. Replay
can mix prior training records into adaptation. State is reconstructed after model
updates. This is persistent event memory plus learned recurrent state, not a semantic
search engine over unlimited autobiographical experience.

## Trained scope

One checkpoint combines real gesture phase, acoustic speaker, and LIBRAS hand-movement
tasks. These are separate datasets with shared recurrent parameters; no aligned
speech/sign/gesture examples were fabricated. Benchmark heads measure components,
not the entire space of contact. Explicit head requirements prevent reporting a
speaker identification without speech evidence, for example.

## Broader goals not yet established

- Raw video perception, robust ASR and complete signed-language spatial grammar.
- Learned cross-modal temporal attention and synchronized conversational grounding.
- Rich episodic retrieval, participant graph message passing beyond supplied context,
  and socially grounded relational objectives.
- Open-ended dialogue/action generation, calibrated uncertainty and out-of-distribution
  detection. A softmax threshold is only an uncalibrated clarification heuristic.
- Large-scale continual learning, forgetting guarantees and cross-community transfer.
- Human-level interpretation of thoughts. User reports, memories and observable
  expressions are usable evidence; direct thought transmission is not assumed.

These remain research/engineering requirements rather than completed capabilities.
The implementation is extensible; it is not a finished universal model.
