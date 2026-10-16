# Alexis's pipeline, taken from her own files (no invention)

Built 2026-10-16 (PT). Read-only: no code changed, nothing sent or pushed.
Her phrase (chat, t322): "Ingest analysis understand reduce to primitives do calculations scale it back up".
None of those six words appears together as a list in any file. The nearest thing she wrote is **"The execution idea"** (Rules and syntax §18). I quote that first, then take each stage from her other files.

Citation key (line numbers are in my plain-text conversions):
- **SPEC** = `/workspace/pipe/nsnf-spec.txt` (Drive doc 1a4-4qM_… "combined NSNF spec"; same text as `/workspace/nsnf-spec-combined.txt`)
- **ARCH** = `/workspace/pipe/arch-doc.txt` (Drive doc 1m4jCA-T-… "System Architecture, Ontological Addressing, and Particle Swarm Mechanics"; `/workspace/doc.txt` is the same text)
- **NS29** = `/workspace/zips/Synthia-State-Space-Foundation__4c9611a0/Synthia-State-Space-Foundation/northstar/Untitled document (29).txt` (her pasted ATO proposal, lines 1-619 are hers; lines 620-860 are an assistant's proposal she pasted)
- **PIPE-PATCH** = `/workspace/zips/_drive_x/Synthia-v0.5.1-EXECUTION-PIPELINE-PATCH/` (Drive: Synthia-v0.5.1-EXECUTION-PIPELINE-PATCH-1.zip)
- **AUTO-ING** = `/workspace/zips/automata-ingestion__152112ec/Synthia-Automata-Patch-v0.5.1/` (Drive: Synthia-v0.5.1-AUTOMATA-INGESTION-EXECUTION-PATCH.zip)
- **ONTO-ING** = `/workspace/zips/_drive_x/Synthia-Integrated-Automata-v0.5.0-ONTOLOGICAL-INGEST-PATCH/…/` (Drive zip)
- **SSSF** = `/workspace/zips/Synthia-State-Space-Foundation__4c9611a0/Synthia-State-Space-Foundation/`
- Her three Google Docs (links t299-t301) = `/workspace/alexis-doc1.txt` (17akzXCF…, "COMPLETE SENTENCE SYSTEM BREAK DOWN MESSY"), `alexis-doc2.txt` (1uQi-13ax…, "You're at the inflection point…"), `alexis-doc3.txt` (1dvrTl85…, "Consciousness Causal Graph… Master Architecture"). I re-downloaded all three from Drive (`/workspace/pipe/links/link1-3.txt`); they match the box copies (about 99% of the length, same start) and contain none of the pipeline terms (details at the end).

## 0. The spine she already wrote: "The execution idea"
SPEC 510-533 (ARCH 657-…), Rules and syntax §18:
```
input/artifact/problem
  ↓ reduce to appropriate primitives
  ↓ determine the required transaction
  ↓ resolve/distribute through the dimensional structure
  ↓ tiny automata execute local operations   ↕ mesh allows interaction
  ↓ results compose
  ↓ fold/recombine upward
  ↓ higher-scale state
  ↓ resolve address
  ↓ land history/bookkeeping
"The system does not need to destroy the original to understand or execute it."
```
Next to it, SPEC 503-509 (ARCH 649-655), §17 "Pure Cynthia runtime rule": "This one is fixed: JavaScript only. No Python. No backend. No server required as part of Pure Cynthia."

## 1. INGEST
- NS29:604-606 (hers): "**Ingest the components I provide** and produce the canonical executable state-space specification, authority map, invariants, and validation suite."
- NS29:859 (hers): "Phase 0 deliverable: **ingest my existing components** and produce the canonical executable state-space specification plus validation tests, preserving the information and relationships already present in the system."
- NS29:389-398 (hers), "Phase 0, Step 1: Inventory": "Inspect every component I provide. Do not generate replacements before understanding what already exists."
- SPEC 1202 (§46): "user contact/input → native decoder or generator → typed evidence/relational message → …"; SPEC 1137 (§31): "contact event → multimodal feature extraction → relational interpretation → …"; SPEC 1211 (§23): "1 Observe signals in their native modalities."
- Already built (zips she gave): PIPE-PATCH `EXECUTION_PIPELINE_CHANGE_RECORD.md` stage list "1. ingest 2. normalize …", described as "the execution substrate agreed from the computer-architecture work" (code: `packages/execution-spine/src/execution-pipeline.mjs:7-8,411`). AUTO-ING `docs/AUTOMATA_INGESTION_EXECUTION_PATCH.md`: "1. normalizes the artifact, 2. fingerprints exact content…". ONTO-ING `ONTOLOGICAL-INGEST-PATCH.md`: "accepts File, Blob, ArrayBuffer… keeps the original bytes for exact reconstruction".
- Gap: the word "ingest" is hers only in NS29 (about the components, not about arbitrary files). Nowhere in her own text is "ingest a file" defined as a swarm step. The file-ingest stages live in the patch zips.

## 2. ANALYSIS
She doesn't use the word "analysis" for the swarm. What her files say instead:
- SPEC 120 (ARCH §5, "The Swarm's First Learning Task"): "The swarm learns how to **examine** the system by **decomposing observed structures into their naturally occurring qualities**."
- SPEC 631 (Proportion of Perspective §II): "The practical excavation tool for finding these primitives is Who / What / Where / When / Why."
- SSSF `README.md` line 5: chat enters "the full Klein toolkit's Who/What/Where/When/Why analysis".
- NS29:287-: "Execution Model: observe → resolve → address → compose → transition → execute → observe result → update state".
- SPEC 972-981 (Pure Synthia §21, Research Sequence): "Extract linguistic primitives (bottom-up, starting below the letter). Quantify their transformations. Discover recurring operators."
- Gap: a separate "analysis" stage is not named in her files. It is the decomposition above.

## 3. UNDERSTANDING
- SPEC 533 (§18, last line): "The system does not need to destroy the original to **understand** or execute it."
- SPEC 122 (ARCH §5): "This is how larger structures become **understandable** without requiring every swarm particle to contain the complete system."
- NS29:392 (hers): "Do not generate replacements before understanding what already exists."
- SPEC 1097 (Alexis addition 2026-10-10): "Human Design is used not as established science but as **the understanding** that these qualities exist in everybody, heavier in some; the system learns when to use which mechanism to accommodate them."
- SPEC 1096 (Alexis addition): "Uncertainty should trigger investigation, not rejection."
- Gap: no file defines "understanding" as a computation. Her files tie it to keeping the original and its lower-level history (SPEC §6 "History is never flattened away").

## 4. REDUCE TO PRIMITIVES
- SPEC 510-514 (§18): "reduce to appropriate primitives".
- SPEC 631 (ARCH 1110): "Nothing is invented; things are **discovered by reduction**. Any thing, person, place, or event can be reduced to a small set of primitives — typically four, occasionally resolving to five…"
- SPEC 1089-1094 (Alexis additions, in her words): "Swarm packs are 3, 2, 1 — six in a pack." / "The packs are two bits apiece, so it takes three packs to make one hexagram." / "Swarm pieces build 2-bit, 4-bit, and 8-bit." / "**The packs make primitive values of things that need to be made.**" / "Two swarm types: GA gives the things you see on the screen; PSO gives tooling in the background." / "Overall they make up everything: language models, web pages, Klein tools, anything."
- SPEC 94-107 (ARCH §4): "Its smallest units behave more like letters, syllables, qualities, or primitive symbolic particles."
- SPEC 855-856 (Pure Synthia abstract): "from a finite set of primitive states and a finite set of reusable relational operators". SPEC 970: Pure Synthia "decomposes the input into registered primitives, generates candidate addresses…".
- Built: PIPE-PATCH stage 3 "primitive-operator-graph" (from "the existing resolver's real six-bit decomposition"); AUTO-ING `src/automata/primitive-reducer.mjs`; ONTO-ING "reduces the artifact mathematically to bit-level primitives".
- NS29:723-726 is an assistant's text, not hers: "To execute a file without a backend or workers, the runtime tokenizes and reduces the input file into the smallest operational primitives…" (with a demo `PrimitiveReducer`, `token.charCodeAt(0) % 16`).
- Gap: SPEC 1089-1094 are not in the spec yet (it says "not yet written into the spec"). The spec has no representation for packs, 4-bit or 8-bit pieces (SPEC flag before §3, "Bit widths, unresolved"). No file says how a letter's value becomes a pack's two bits.

## 5. CALCULATIONS
- SPEC 510-533 (§18): "determine the required transaction → resolve/distribute through the dimensional structure → tiny automata execute local operations ↕ mesh allows interaction → results compose".
- SPEC 1133 (§5 **Runtime execution contract**): "(1) Encode task/event into input state. (2) Resolve candidate gates and neural components using versioned topology rules. (3) Activate a sparse subset under a compute budget. (4) Execute local component computations and exchange typed messages along permitted edges. (5) Aggregate micro states into meso states; update macro coherence/state. (6) Emit output and trace. (7) Persist event, state transition, and checkpoint at a configured interval. (8) Reuse cached computations only when inputs, version, and relevant state match. No hidden fallback that pretends neural execution occurred."
- SPEC 1108 (§6 **Tiers**): "expose configurable tiers 1–5 controlling maximum active components, message-passing rounds, and time/compute budgets. Do not hard-code these tiers as the five dimensions or five centers." (The sizes 5/15/35/70/110 are not in her files.)
- SPEC 1112 (§19): "three scales: micro gate interaction, meso center-state formation, macro center competition/coherence."
- SPEC 1116-1118 (§36): six-bit hexagram with line 1 at the bottom; operators change selected lines, complement, reverse order, exchange trigrams, "extract nuclear hexagram from original lines [2,3,4,3,4,5]".
- SPEC 878-885 (Pure Synthia §4): E = f(P, R, X, C, H); SPEC 933 §13 "Complexity Ledger" L = (N_p, N_s, N_e, N_o, D_r, A_c, T_c).
- SSSF `README.md`: "The **calculation** button executes the actual spine's expression IR."
- Built: PIPE-PATCH stages 4-6 "five-field-allocation, dependency-schedule-forwarding, artifact-execution".
- Gap: the calculation rules for the nine centers are not specified (SPEC §13/§27/§59 "not yet specified"). The state-space "Hamming distance 1 at EVERY level" rule is in SPEC 787 as a locked channel-emergence rule.

## 6. SCALE BACK UP
Her exact phrase "scale it back up" is not in any file. These are her words for the same step:
- SPEC 525-533 (§18): "**fold/recombine upward** → higher-scale state → resolve address → land history/bookkeeping".
- SPEC 297-322 (Rules and syntax §5, **the recursive scale law**): "letters → words → sentences → larger expressions → glyphs / structures / composed states → still larger structures. At each level: the composed result can become a primitive of the next level. But it retains its lower history." SPEC 595-601 (§II.1a): "The structural mathematics stays the same when scale changes; the variables occupying the positions change."
- SPEC 979 (Pure Synthia §21): "Scale upward to sentence/discourse level." SPEC 890 (§5): "X_n = whole(X_{n-1}), X_n = part(X_{n+1})".
- SPEC 1133 step 5 (§5): "Aggregate micro states into meso states; update macro coherence/state."
- SPEC 208-230 (ARCH §11 **Recursive Emergence**): "Primitive qualities → Addressed particles → Relations → Mesh → Letters / syllables / partial primitives → Words / numbers / shapes → Sentences / structures → Functions / interface states → Experience → New relations → Updated mesh."
- SPEC 1091 (Alexis addition): "There are 64 macro nodes on each level, nested, broken down into 12 [she also said six], giving trillions of positions across one space."
- Built: PIPE-PATCH stages 7-11 "fold-recombine, state-commit, observation, ontological-address, execution-cache".
- Gap: the 12-or-6 split is unresolved in her own words (SPEC flag at 1100).

## No external dependencies / no backend / no web worker / executing arbitrary files
| Piece | Where | Her words or assistant's |
|---|---|---|
| JavaScript only, no Python, no backend, no server | SPEC 503-509, ARCH 649-655 | hers (the doc's wording of her rule) |
| runs "entirely client-side: JavaScript/ES modules, browser storage, browser-native computation. Explicitly excluded: backend inference, remote databases, external LLMs, MCP, cloud compute, mandatory GPU" | SPEC 955-956 (Pure Synthia §18), ARCH 5269 | hers |
| "initially runs without a backend, external language model, network service, or GPU" | SPEC 856 | hers |
| "JavaScript is the initial host … Adapters … expose external capabilities to that machine" | NS29:56-75 (§2) | hers |
| `"backendRequired": false, "modelRequired": false, "webWorkerRequired": false` | SSSF `northstar/assembly-requirements.json` | assembled from her instructions |
| "does not require Node APIs, a backend, or a Web Worker for its core operation … Workers are optional acceleration/scheduling tools, not part of Synthia's identity" | AUTO-ING `docs/AUTOMATA_INGESTION_EXECUTION_PATCH.md` | assistant-written package doc |
| "To execute a file without a backend or workers…" | NS29:725 | assistant text she pasted |
| BANTU bridge README: "No network, build tool, or backend required." | `/workspace/pipe/bridge/bantu-connected/README.md:3` | assistant-built |
| earlier design that used Web Workers | doc `72dc1fcaa93d` (the "Proportion of Perspective" attachment) and SSSF `northstar/01-Untitled-document-4-.txt` ~3372-6974 | assistant text in her pasted chats; **conflicts** with her no-worker rule |

- "No external dependencies" as a phrase is not in her own text. The closest are "JavaScript only" and "no Python".
- **"Execute arbitrary files" is not in her files as her words.** Where her files address it: SSSF `README.md` says "Foreign-artifact execution … does not establish execution of every possible language… Worker-free raw JavaScript uses the supplied iframe chamber." AUTO-ING lists native providers: JSON via `JSON.parse`, text, "self-contained JavaScript: sandboxed iframe", and WebAssembly. Her conversation request (t319) is the only source for "arbitrary files, no external dependencies, a backend or web worker".
- Conflict to settle: SPEC flag at 1188 asks whether the NSNF spec's Python/backend/worker assumptions (§8, §9, §14) are separate from the Pure Cynthia JS-only rule. She hasn't answered.

## The three wheels, nuclear, and the Ten Wings
- **King Wen / Fu Xi / Mawangdui** (her words, "Locked Architecture Spec §2.1", SPEC 787-798, ARCH 2988-2996): L1 Movement = **Fu Xi** (binary/mathematical changes); L2 Evolution = **Mawangdui** (matrix transitions); L3 Being = Eight Palaces / Taiji pivot; L4 Design = AST-style structural parsing; L5 Space = **King Wen** (received sequence, output). "Implemented: hexagram_sequences.py — Fu Xi and King Wen built from verified binary data…; Mawangdui generated from the verified upper-trigram-family-order principle rather than a separately transcribed table. Eight Palaces (L3) and AST-style parsing (L4) not yet implemented." The file exists in zips she gave: `zips/Resonance-Network-Swarm-Reality-INTEGRATED-TRIAL-2026-10-02__4e7c9fbf/resonance_integrated_trial/backend/hexagram_sequences.py`, `zips/production_ready-1__ed315739/production/backend/hexagram_sequences.py`, and a JS version `zips/content-4__90d1f5f4/synthia-semantic-state-integration/hexagram_sequences.js`. I did not open them to check that they implement Mawangdui. SPEC 1116: "Keep King Wen chapter numbering, Fu Xi/binary ordering, and Human Design gate numbering as separately named maps; never interchange indices implicitly."
- Also: SSSF `northstar/01-Untitled-document-4-.txt:834` "5 sequence buttons (Fu Xi, King Wen, 8 Palaces, Wax/Wane, Mawangdui)"; 1403-1407 "Fu Xi = read from center outward / King Wen = read as a story along the spiral / Mawangdui = read the lineage/history". SSSF `vendor/browser/docs/FRAGMENT_ALGEBRA_SPEC.md:59,281` "3 canonical (xiantian binary, houtian compass, Mawangdui familial)", with a Mawangdui entry that I did not expand.
- **Nuclear:** SPEC 1117 (§36): "extract nuclear hexagram from original lines [2,3,4,3,4,5]"; SPEC 1313 test V05-02. alexis-doc1.txt:5207 "Hu Gua ('Nuclear Hexagrams'): sixteen hexagrams generated as inner structures"; 6241-6247. FRAGMENT_ALGEBRA_SPEC.md:204-210 "T5 nuclear trigram rule … existing o_nuclear in operators.js". (ARCH 2641 `calculateNuclear` uses the middle 4 lines, which is a different rule from §36's [2,3,4,3,4,5]. Both are in her files.)
- **Ten Wings:** the phrase is **not in any of her own docs**. It appears only in the source-summary files inside zips (assistant-written corpus notes): `SSSF/vendor/browser/docs/corpus/moore-trigrams-of-han.md:20`, `adler-yijing-guide.md:156`, `zhouyi-bronze-age.md:24`, `govinda-inner-structure.md:160`, `wen-oracle.md:164`, and `docs/SOURCE_MATRIX.md:69`. Her doc1 has only "T = Tuan Zhuan (Commentary on the Judgment)" at alexis-doc1.txt:4577 (one of the Ten Wings by name) and "X = Da Xiang (Great Image)".

## Links she pasted in this conversation
I had no transcript tool, so this is from the parent conversation text I was given. It holds exactly three Google Doc links:
1. 17akzXCFksDNlA571qQwmY3-mcGAO64kMOKdHs08hJfU, "COMPLETE SENTENCE SYSTEM BREAK DOWN MESSY" = alexis-doc1.
2. 1uQi-13axCoicEZSd6LebUDQgOWY6aZAnYMXR_QxQh74, "You're at the inflection point…" = alexis-doc2. The link as shown in the chat dropped "i-" after "1uQ"; the working id has them.
3. 1dvrTl85zB43BXGK0H8n2YsJr-lZ5zFrzWl2bWvijKkk, "Consciousness Causal Graph… Master Architecture" = alexis-doc3.
Searched the pipeline terms in all three: doc1 has 1 "ingest" (a toy sentence generator that "ingests the CSVs"), 1 "scale up" ("quantum interface scaled up to biology"), 1 King Wen and 27 nuclear (Hu Gua) hits. Docs 2 and 3 have none of the stage, constraint or wheel terms.
The other Drive ids from the task (architecture doc, NSNF spec, BANTU zip) are read as above. The BANTU bridge has no stage text beyond "primitive positions grouped 3–2–1" (README:8).

## Where each of her stages is missing as a stage
- A separate "analysis" and "understanding" stage: not defined as steps anywhere. §5 "learn the system / decompose into qualities" and §18 "understand" are the only hooks.
- "Calculations" as a named step: only the runtime contract (SPEC §5) and tiers (§6).
- "Scale it back up": only "fold/recombine upward" and the recursive scale law.
- Mawangdui order and Ten Wings: no order table in her docs. Mawangdui is described by principle there; `hexagram_sequences.py/.js` in zips may hold it (unchecked). Ten Wings only in assistant-written corpus notes.
- Arbitrary-file execution with no dependencies: her conversation only; the files have the JS-only rule plus the patch zips' native-provider list.
