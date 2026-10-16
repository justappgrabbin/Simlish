# Integrated swarm: plan (written before the code)

Source: Alexis's own files only. Rules: pure JavaScript, no external dependencies, no backend, no web worker; the swarm must EXECUTE arbitrary files.

## What "execute" means (her words, her files)
"input/artifact/problem -> reduce to appropriate primitives -> determine the required transaction -> resolve/distribute through the dimensional structure -> tiny automata execute local operations <-> mesh allows interaction -> results compose -> fold/recombine upward -> higher-scale state -> resolve address -> land history/bookkeeping" (nsnf-spec execution idea; Spec §5 runtime contract).
Her Execution Pipeline Patch v0.5.1 names the stages: ingest, normalize, primitive-operator-graph, five-field-allocation, dependency-schedule-forwarding, artifact-execution, fold-recombine, state-commit, observation, ontological-address, execution-cache.

## Modules copied in as-is (integrated/, byte-identical, see MANIFEST.json)
| Folder | From |
|---|---|
| spine/execution-pipeline.mjs | Synthia-v0.5.1-EXECUTION-PIPELINE-PATCH (packages/execution-spine) |
| spine/canonical-address, dag-runtime, execution-address-resolver, pure-synthia/engine/derivation | Synthia-Universal-Execution-Spine-v0.4.0 (the pipeline's own imports) |
| automata/* (normalizer, primitive reducer, five-field compiler, ontological address signer, store, execution surface, kernel) | automata-ingestion: Synthia-Automata-Patch-v0.5.1 |
| ontological/* (ingestor + pure-JS sha256) | Synthia-Integrated-Automata-v0.5.0-ONTOLOGICAL-INGEST-PATCH |
| reference/* (not executed) | hexagram_sequences.py (INTEGRATED-TRIAL / production_ready), hexagram_sequences.js, vendor sequences.js |

## Not used, on purpose
- UniversalExecutionBridge / PaperRuntimeSandbox (v0.4.0): the sandbox runs in a Web Worker and the HTML path needs an iframe. Alexis said no web worker. Her patches' own native providers (JSON.parse, text, WebAssembly, self-contained JS) are used instead.
- The browser iframe JS provider in browser-execution-surface.mjs needs `document`; in the same thread we register a provider (her surface has `register()` for this) that runs JS the way her bridge's `runJavaScriptText` does (AsyncFunction, console capture), with a guard.

## Thin adapters (new files)
- integrated/providers.mjs: same-thread providers registered on her BrowserExecutionSurface: javascript, css (structural parse), html (structure; scripts not run without a DOM), zip (members executed one by one), generic primitive execution for everything else; non-JS language adapters are placeholders.
- integrated/wheels.mjs: spec §2.1 wheels: Fu Xi (Movement), Mawangdui (Evolution), King Wen (Space, read only). Being and Design wheels are "not yet implemented" in the spec and stay investigating.
- integrated/integrated-swarm.mjs: wires the stages together with the existing swarm (any-file-swarm, nsnf-runtime, state-space) and lands history at the address.
- integrated/demo.mjs, integrated/integrated.test.mjs.

## Flow per file
1. Ontological ingest (original bytes kept, bit-level reduction, Space derived from the four).
2. ExecutionPipeline (her 11 stages) with executeArtifactCore = automata kernel (reduce, five-field compile, execute via surface, sign address) + her ExecutionAddressResolver for the 13-place computational coordinate.
3. Swarm: registration (PSO checks), file genotype PSO packs 3-2-1, placement in state space, NSNF runtime on the file's features (existing).
4. Fold up: the pipeline's field summaries become the input of a second NSNF runtime event at the placed node (higher-scale state); runtime checkpoint = state commit.
5. History ledger entry keyed by the pipeline's ontological address key, cross-indexed by the ingest address and the automata `onto:` id.

## Known conflicts to keep visible (investigating, not decided)
- Mawangdui: hexagram_sequences.py (generated: upper trigram family order, doubled trigram first) and hexagram_sequences.js (different lower order, 57 of 64 positions differ) and vendor sequences.js (another upper order) disagree. The py file is the one spec §2.1 names as implemented. hexagram_sequences.js also has King Wen bit values that differ from the py for 50 of 64 gates; the repo's King Wen map matches the py.
- Three address schemas exist (repo 13-place, spine computational coordinate, ingest patch). No mapping chosen.
- Her pipeline allocates primitives to all five fields including Space; the NSNF state space keeps Space read-only. The state space is never written for Space.
- Same-thread JS cannot be interrupted or truly sandboxed without a worker or vm: guard is a deny-list, runaway loops are only caught by a static check.
