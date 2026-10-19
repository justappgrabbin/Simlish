# SWARM-OS inventory: every component in the ecosystem table mapped to the specification

Built 2026-10-16 (PT) by `integrated/swarm/inventory.build.mjs` from `inventory.data.mjs`; the builder refuses a row whose files are missing, whose named functions are not in the file, or whose status the listed tests do not back. Raw command results: `inventory-probes.json`. Machine-readable copy: `INVENTORY.json`. All of her files are untouched.

**Status words:** EXISTS-RUNS (a listed command passed here) · EXISTS-UNTESTED (files exist, nothing passed here) · PROPOSED (in the spec, no connection built) · UNVERIFIED (no file confirms the name or capability; candidates listed, none claimed) · MISSING (searched, nothing found).

**Counts:** EXISTS-RUNS 24 · EXISTS-UNTESTED 11 · PROPOSED 15 · UNVERIFIED 4 · MISSING 3 (PROPOSED includes the 15 research references, which are not packaged).

## Specification items

- **S1** The swarm is the code, the intelligence and the body (not a controller of conventional software)
- **S2** Particles are executable: value, operator, condition, relation, memory, instruction
- **S3** Three coherent morphing types: computational, cognitive, visual
- **S4** Intelligence without an LLM: Klein, AUTOLING, symbolic + neural + swarm
- **S5** Ingest code in many languages, export to conventional code, but EXECUTE from the swarm's own representation
- **S6** Recursive composition
- **S7** Memory, lineage, append-only history
- **S8** Photorealism connected to authoritative swarm state (existing morph/character assets first; Gaussian splats are not executable nodes)
- **S9** Local, offline, no LLM, no server, no web worker

## Components

| Component | Spec | Status | Tests (passed/listed) | Exact files | Missing connections |
|---|---|---|---|---|---|
| SWARM-OS v6 | S1 S3 | **EXISTS-UNTESTED** | 0/1 | swarm-os-v14/x/v6/swarm_os.py (class DimensionalAgent, class SwarmField, class NeighborhoodCrystallizer) | run failed here: matplotlib is not installed on this box, so v6 was not exercised (its recorded output swarm-os-v14/runs/v6out/swarm_result.json exists) |
| SWARM-OS v9 (PSO+GA+ABC hybrid) | S1 S3 | **EXISTS-RUNS** | 2/2 | swarm-os-v14/x/v9/swarm_os_full.py (class SwarmParticle, class FullScreenSwarm, def main) | Python original is not connected to the app. Its ABC roles (employed .4 / onlooker .4 / scout .2) are ported in integrated/abc.mjs (AbcV9, connected); its PSO and GA phases are not ported as such: the repo has swarm-fit.mjs (PSO, numeric only) and population.mjs (GA over primitives) |
| SWARM-OS v10 Bantu | S1 S5 | **EXISTS-RUNS** | 1/1 | swarm-os-v14/x/v10/swarm_os_bantu.py (class) | Python original not connected; the Bantu layers (binary/ascii/nato/translit/unicode) appear in repo only as primitive kind bantu-layer (6) in primitive-functions.json |
| SWARM-OS v12 cosmic engine | S3 | **EXISTS-RUNS** | 1/1 | swarm-os-v14/x/v12/cosmic_engine.py (class) | not connected. Its angular widths use 5 bases per tone (Base_Anchor 0.0052083 deg = tone/5), see the six-vs-five reconciliation |
| SWARM-OS v13 (64 Bantu gates table) | S5 | **EXISTS-UNTESTED** | – | swarm-os-v14/x/v13/bantu_64_gates.txt | data table only: no v13 code file exists in the zip |
| SWARM-OS v14 (hierarchical neural Bantu swarm) | S1 S3 | **EXISTS-UNTESTED** | – | swarm-os-v14/x/v14/expansion_summary.json; swarm-os-v14/swarm_os_v14_complete-1.zip | the zip holds v14 OUTPUTS only (5 layer PNGs, composite PNG, expansion_summary.json). No v14 generating code is in the zip, so its numbers (99,520 virtual agents) cannot be reproduced from a file |
| repair checkpoint | – | **UNVERIFIED** | – | SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/docs/REPAIR-LOG.md; zips/_drive_x/pure-synthia-pass4-step41-REPAIRED/README.md | no file is named "repair checkpoint". Candidates: Hover docs/REPAIR-LOG.md, zips/_drive_x/pure-synthia-pass3-REPAIRED, pure-synthia-pass4-step41-REPAIRED, Synthia-v0.3.0-composition-repair-01, Synthia-Compiler-Kit-v6-SELFHEALING. Which one she means is not decided here |
| Pure Synthia Swarm Body | S1 S7 S9 | **EXISTS-RUNS** | 8/8 | SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/vendor/pure-synthia-v0.4.0/src/synthia/swarm/swarmBody.mjs (SynthiaSwarmBody); SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/vendor/pure-synthia-v0.4.0/SWARM-ARCHITECTURE.json; SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/vendor/pure-synthia-v0.4.0/VERIFY-SWARM.mjs | NOT connected to this branch: nothing under integrated/ imports it. It coordinates 47 base processes with checkpoint/wake/restore; the integrated swarm here has the heartbeat loop and audit log instead. A connection (body processes <-> integrated/swarm area) is PROPOSED, not built |
| Mesh Cognition Network | S4 | **UNVERIFIED** | 3/3 | zips/_drive_x/Synthia-Relational-Mesh-v0.3.0/Synthia-Relational-Mesh-v0.3.0/package.json; SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/src/mesh/relational-mesh.mjs | no file or doc is named "Mesh Cognition Network" (searched file names and contents). Candidates by function: Relational Mesh v0.3.0 (relation algebra, source events, motif memory; its three tests pass), Hover src/mesh/relational-mesh.mjs and mesh-federation.mjs, biverse shared_cognition.py + synthia-cognitive-network.sqlite3 (Python, untested) |
| State Space Mesh | S2 S6 | **EXISTS-RUNS** | 2/2 | repo:vendor/mesh/state-space/mesh-state-space.js (StateSpace, DimensionLayer, buildGateTable); repo:state-space.mjs (hammingNeighbors, gateToMacro, macroToGate, StateSpace); repo:integrated/swarm/mesh.mjs (EmergentMesh, verifyEdges, potentialEdges) | her EmergentEdgeResolver is not used (needs amplitude/phase/coherence/zodiac values that no source defines for content primitives; uses Math.random); its 33 distinct pairs differ from her 36 field_core channels |
| ATO Core | S1 S6 S7 | **EXISTS-RUNS** | 2/2 | SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/vendor/synthia-core-v1.0.0/ato-core/src/automaton.mjs; SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/vendor/synthia-core-v1.0.0/ato-core/src/promotion.mjs; zips/synthia_os_runtime__42a15982/layer1_ato_core.js | not imported by this branch (integrated/ has its own automata kernel under integrated/automata/) |
| Tool Factory | S1 S5 | **EXISTS-RUNS** | 2/3 | SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/vendor/pure-synthia-v0.4.0/src/synthia/toolFactory.js; SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/src/pipeline/tool-synthesizer.mjs; zips/_drive_x/integrated-tool-factory-v1_6_0/package.json | REPAIRED as a working copy (repairs/repairs.json): the missing sibling ato-core-native-test/ is her ato-core from synthia-browser-continuation-v0.6.8 (where she vendored the same factory, src identical) -> 11/11. Her provenance doc names ato-mcp-v0_1-fixed.zip as the source of that ato-core; that zip is not on this box, so identity is not confirmed. The original zip alone still fails 1 of 9 files |
| Resolver (address resolution) | S5 S7 | **EXISTS-RUNS** | 4/4 | repo:integrated/spine/execution-address-resolver.mjs (ExecutionAddressResolver); SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/src/execution/integrated-address-resolver.mjs; SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/vendor/execution-spine-v0.4.0/src/pure-synthia/sovereign-entry.js | descend keeps 4096 six-bit states (spine line 87); no file of hers defines a way past it |
| Onomata | – | **MISSING** | – | – | searched: file and folder names containing "onomata"; contents of .md/.txt/.json/.mjs/.js/.py/.ts/.tsx under /workspace (excluding venvs, scratch, node_modules). The only matches are the substring inside "autonomata". No Onomata source exists in her files |
| Relational Algebra | S2 S6 | **EXISTS-RUNS** | 2/3 | SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/docs/RELATIONAL-ALGEBRA-v0.5.5.md; SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/src/mesh/relational-mesh.mjs; zips/_drive_x/Synthia-Relational-Mesh-v0.3.0/Synthia-Relational-Mesh-v0.3.0/package.json | the v0.5.5 patch zip alone fails (it is an overlay and lacks the base files); runs only inside the full Hover tree. Not connected to this branch |
| DNA / RNA / Protein patches | S2 S4 | **EXISTS-RUNS** | 5/5 | SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/docs/DNA-RNA-PROTEIN-TRANSLATION-v0.5.3.md; SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/src/state-space/biological-translation-runtime.mjs; SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/docs/AGENT-GENOME.md | not connected to this branch; the per-patch zips (v0.5.2) fail alone for the same overlay reason |
| Predictive Memory | S4 S7 | **EXISTS-RUNS** | 1/1 | zips/_drive_x/Synthia-Predictive-Memory-v0.1/Synthia-Predictive-Memory-v0.1/predictive-semantic-learning.js; zips/_drive_x/Synthia-Predictive-Memory-v0.1/Synthia-Predictive-Memory-v0.1/vqvae-generative-memory.js | not connected to this branch; the memory the branch has is the hash-chained AuditLog and HistoryLedger (append-only), not a predictive memory |
| Autopoietic Assembly | S1 S6 | **EXISTS-RUNS** | 3/3 | SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/vendor/trainable-assembly-v0.4.2/src/core/ato/automaton.mjs; swarm-search/SynthAi/computer/donors/recovered/synthai-r21.22-self-cultivation-graph-integrated/autopoiesis/AutopoieticComposer.mjs | trainable-assembly passes; AutopoieticComposer.mjs (r21.22) was not run. In this branch the self-assembly loop (swarm-loop.mjs, 73 units) is the connected counterpart |
| Generative Address Morphology | S3 | **UNVERIFIED** | 1/1 | SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/vendor/kimi-agent-automata-state-space-merge/pure-synthia-automata/src/state-space/generative-grammar.js; chassis-work/candidates/swarm-reality-trial/resonance_integrated_trial/integrations/biverse/human_design/generative_mesh.py | no file or doc is named "Generative Address Morphology". Nearest by function: generative-grammar.js (kimi merge; its suite passes), generative_mesh.py (Python, untested), Hover exact-address-recall-patch, integrated/arc-engine.mjs (address from longitude, in this branch). None is claimed to be it |
| Kimi Agent Swarm | S3 S4 | **EXISTS-RUNS** | 2/2 | SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/vendor/kimi-agent-automata-state-space-merge/pure-synthia-automata/demo.mjs; SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/vendor/kimi-agent-automata-state-space-merge/pure-synthia-automata/tests/run.mjs | matched by name to "Kimi Agent Automata State Space Merge" (866 + 42 checks pass). Other Kimi packages exist (Kimi_Agent_Deployment_* zips under swarm-search) and were not run. Not connected to this branch |
| Seed Box | – | **MISSING** | – | – | searched names and contents for "seed box"/"seedbox": no hit in her source (one hit inside a tool-output text file). Similar names that are NOT claimed: The-seed repo, Synthia-Design-Dream-Seed, paper-seed-living-tree, native-seed-server.mjs |
| Dream Swarm | S1 S3 S7 | **EXISTS-RUNS** | 2/2 | zips/_drive_x/Dream-Swarm-System-v0.1.0/Dream-Swarm-System-v0.1.0/server.mjs; zips/_drive_x/Dream-Swarm-System-v0.1.0/Dream-Swarm-System-v0.1.0/public/swarm-reality-engine.js; zips/_drive/Dream-Swarm-System-v0.1.0.zip | not connected to this branch; it has a server.mjs (the spec requires no server), its browser engine file swarm-reality-engine.js was not run in a browser |
| Resonance Network Swarm Reality Trial | S1 S3 S8 | **EXISTS-UNTESTED** | – | zips/_drive/Swarm-Reality-Trial-v0.1.0.zip; chassis-work/candidates/Resonance-Network-Swarm-Reality-INTEGRATED-TRIAL-2026-10-02.zip | neither zip was run here (74 MB integrated trial; the 17 KB Swarm-Reality-Trial zip holds 8 files) |
| Morphing Particle Swarm | S3 S8 | **EXISTS-UNTESTED** | – | zips/_drive/morphing-particle-swarm-main.zip | 3 files: README.md, index.html, assets/donut.glb. A browser page: not opened here. Not connected to this branch |
| Cynthia Morph (canonical morph runtime) | S3 S8 | **EXISTS-RUNS** | 1/1 | SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/src/morph/canonical-morph-runtime.mjs; SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/src/morph/deep-surface-morph-adapter.mjs; SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/docs/CANONICAL-MORPH-INTEGRATION-v0.5.7.md | not connected to this branch (which has photo-morph.mjs and frame-morph.mjs, tested, for image morphing) |
| Modelmaker (four-LLM position builder) | S4 | **EXISTS-RUNS** | recorded run | chassis-work/src/Stellarproximology-lab/Modelmaker-repair/app.py; chassis-work/modelmaker-run/run_movement.py; chassis-work/modelmaker-run/run-output.txt; chassis-work/modelmaker-run/movement-model/modelmaker_build.json | only the MOVEMENT model was built and run (run-output.txt); Evolution, Being and Design models were not. Her swarmdoc.md line 1342 says the dimension-to-model mapping is still open. Python only: not usable in the app |
| Trident / Trinity (the model that talks) | S4 | **EXISTS-RUNS** | 1/1 | repo:integrated/trident-js.mjs (Trident); repo:models/trident_syntia.pt; repo:integrated/reference/hf-synthia/trinity-engine.js.txt | the RAG gate of model.py is not ported (chunk embeddings come from rag.py). hdgnn is an address/trust resolver and does not talk |
| Klein mesh game engine v5.1 | S4 | **EXISTS-RUNS** | 2/2 | SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/vendor/klein-mesh-game-engine-v5.1/kmge_v5_fixed/klein-mesh-game-engine.ts; SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/vendor/klein-mesh-game-engine-v5.1/kmge_v5_fixed/core-engine.ts | her copy has TypeScript sources only; the Hover tests fail until dist is compiled (hover:* probes show the failure; hover-built:* the pass). Not connected to this branch, and not pure JS as shipped |
| AUTOLING | S4 | **EXISTS-RUNS** | 2/2 | SynthAi/computer/hover/residents/synthia17/src/runtime/autoling.js; SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/vendor/kimi-agent-automata-state-space-merge/pure-synthia-automata/src/automata/tools/tool-14-autoling.js | not connected to this branch |
| CharacterGen / CharacterFactory | S8 | **EXISTS-UNTESTED** | – | drive-drop/x-CharacterFactory-main/CharacterFactory-main/app.py; drive-drop/x-CharacterFactory-main/CharacterFactory-main/README.md | CharacterFactory is a third-party research project (arXiv 2404.15677), not her code; it needs a GPU stack and cannot run in the app. "CharacterGen" as a separate name: nothing found |
| Learned Motion Matching | S3 S8 | **MISSING** | – | – | searched names and contents for "motion matching": no source in her files. Existing motion material that is NOT claimed to be it: repo animation.mjs/rig.mjs (tested), Hover src/morph |
| YOU-N-I-VERSE (studio) | S1 S8 S9 | **EXISTS-RUNS** | 3/3 | repo:index.html; repo:studio.mjs | photorealism is not present: the studio shows rigged/morphed characters, not photoreal output |
| Simlish (this repo, integrated swarm) | S1 S5 S6 S7 S9 | **EXISTS-RUNS** | 3/3 | repo:integrated/integrated-swarm.mjs (IntegratedSwarm); repo:integrated/swarm-loop.mjs (SwarmLoop, AuditLog, Workspace, ProtectedOriginals); repo:integrated/population.mjs (Population) | see the acceptance table for what the integrated swarm does not do |
| Synthia OS | S1 S9 | **EXISTS-RUNS** | 1/1 | zips/synthia_os_runtime__42a15982/layer1_ato_core.js; zips/synthia_os_complete_v2__71b39ef3/layer1_ato_core.js | only layer 1 (ATO core) was run; layers above it were not |
| SynthAi2 | – | **EXISTS-UNTESTED** | – | swarm-search/all/Synthai2/Dockerfile | repo cloned under swarm-search/all/Synthai2; not run |
| SynthAIPRODeploy | – | **EXISTS-UNTESTED** | – | swarm-search/all/SynthAIPRODeploy/ | repo cloned; not run |
| Hover (Synthia Solo Hover v0.5.8) | S1 S4 S9 | **EXISTS-RUNS** | 14/14 | SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/src/index.mjs; SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/src/solo/solo-runtime.mjs; SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/package.json | REPAIRED as a working copy (repairs/repairs.json): her vendor/klein-mesh-game-engine-v5.1/kmge_v5_fixed had no dist/; her own donor zip klein-mesh-game-engine-v5.1-FIXED(2).zip has identical .ts sources and the dist her build script makes. With it: 56/56 tests on Node 22; 55/56 on Node 20 (19-solo-hover-shell needs the global WebSocket that Node 20 lacks; her engines say >=20). Original unrepaired: 6/27. Not connected to this branch |
| Echo (resident) | S4 | **EXISTS-RUNS** | 1/1 | SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/src/solo/synthia17-resident.mjs; oct7src/patches/echo-resident-startup-fix.patch | the file is named synthia17 resident; the name "Echo" appears in the startup patch file only. Mapping to her "Echo" is by that patch name |
| Paper family | – | **EXISTS-UNTESTED** | – | SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8/vendor/execution-spine-v0.4.0/src/pure-synthia/integration/paper-runtime-sandbox.js; zips/_drive_x/Paper-System-Source/; zips/_drive_x/Paper-Unified-Test-Hub/; swarm-search/all/Paper-Library/; swarm-search/all/Paper-seed/; swarm-search/all/Paper-synth/; swarm-search/all/Paperlibrarycloud/ | present as folders and repos; none run here |
| Resonance SCIENCE / ENTERPRISE | – | **UNVERIFIED** | – | swarm-search/all/Autopoeticagentbuild/ScienceMode.js; swarm-search/all/Resonance-Network/; swarm-search/all/Resonance_field/ | "ENTERPRISE": no file or repo found. "SCIENCE": only ScienceMode.js / src/science / scientist-loop.mjs exist; whether those are "Resonance SCIENCE" is not decided. Not run |
| Stellar Proximology | S4 | **EXISTS-UNTESTED** | – | chassis-work/src/Stellarproximology-lab/Modelmaker-repair/README.md; swarm-search/all/Stellar-proximology/; swarm-search/all/Stellarproximology-lab/; swarm-search/all/Stellarproximology.org/ | repos present; the HF model repo stellarproximology/Synthia (Trident) is the part that runs, see the trident row |
| Conway Automaton | S1 S7 | **EXISTS-UNTESTED** | – | automaton-ref/constitution.md; automaton-ref/package.json | reference only (MIT). Its patterns (recursive-timeout heartbeat, constitution, hash-chained audit log, loop detector, protected files) are re-implemented in integrated/swarm-loop.mjs, whose own tests pass; none of its code is packaged or run |

### Test evidence per component

- **SWARM-OS v6**: swarm-os:v6/swarm_os.py → fail
- **SWARM-OS v9 (PSO+GA+ABC hybrid)**: swarm-os:v9/swarm_os_full.py → pass; repo-test:integrated/abc.test.mjs → pass
- **SWARM-OS v10 Bantu**: swarm-os:v10/swarm_os_bantu.py → pass
- **SWARM-OS v12 cosmic engine**: swarm-os:v12/cosmic_engine.py → pass
- **Pure Synthia Swarm Body**: pure-synthia:VERIFY-SWARM → pass; pure-synthia:swarm-body.node.mjs → pass; pure-synthia:current-synthia-swarm.node.mjs → pass; pure-synthia:full-wiring.node.mjs → pass; pure-synthia:durable-restart.node.mjs → pass; pure-synthia:physiology-wiring.node.mjs → pass; pure-synthia:world-port.node.mjs → pass; swarm-body-v0.3.0:verify → pass
- **Mesh Cognition Network**: relational-mesh:verify → pass; relational-mesh:relations → pass; relational-mesh:life → pass
- **State Space Mesh**: repo-test:integrated/swarm/swarm.test.mjs → pass; repo-test:state-space.test.mjs → pass
- **ATO Core**: synthia-core:test → pass; synthia-os-runtime:test_ato_core → pass
- **Tool Factory**: hover-built:20-solo-tool-factory.test.mjs → pass; integrated-tool-factory:test → fail; repair:tool-factory-ato-sibling → pass
- **Resolver (address resolution)**: execution-spine:test → pass; universal-execution-spine:test → pass; repo-test:integrated/integrated.test.mjs → pass; repo-test:address-resolution.test.mjs → pass
- **Relational Algebra**: hover-built:16-relational-operator-runtime.test.mjs → pass; relational-mesh:relations → pass; relational-algebra-patch:test → fail
- **DNA / RNA / Protein patches**: hover-built:12-dna-linguistic-runtime.test.mjs → pass; hover-built:13-dna-perception-neural-runtime.test.mjs → pass; hover-built:14-proportion-perspective-dna.test.mjs → pass; hover-built:15-biological-translation-runtime.test.mjs → pass; hover-built:10-agent-genome.test.mjs → pass
- **Predictive Memory**: predictive-memory:test → pass
- **Autopoietic Assembly**: trainable-assembly:run → pass; trainable-assembly:core → pass; repo-test:integrated/app-server.test.mjs → pass
- **Generative Address Morphology**: kimi-merge:run → pass
- **Kimi Agent Swarm**: kimi-merge:run → pass; kimi-merge:smoke → pass
- **Dream Swarm**: dream-swarm:test → pass; dream-swarm:verify → pass
- **Cynthia Morph (canonical morph runtime)**: hover-built:18-canonical-morph-integration.test.mjs → pass
- **Trident / Trinity (the model that talks)**: repo-test:integrated/trident-js.test.mjs → pass
- **Klein mesh game engine v5.1**: hover-built:18-canonical-morph-integration.test.mjs → pass; hover-built:01-primitives.test.mjs → pass
- **AUTOLING**: hover-built:21-synthia17-resident.test.mjs → pass; kimi-merge:run → pass
- **YOU-N-I-VERSE (studio)**: repo-test:studio.test.mjs → pass; repo-test:rig-format.test.mjs → pass; repo-test:animation.test.mjs → pass
- **Simlish (this repo, integrated swarm)**: repo-test:integrated/integrated.test.mjs → pass; repo-test:integrated/app-server.test.mjs → pass; repo-test:integrated/population.test.mjs → pass
- **Synthia OS**: synthia-os-runtime:test_ato_core → pass
- **Hover (Synthia Solo Hover v0.5.8)**: repair:hover-klein-dist → pass; hover-built:01-primitives.test.mjs → pass; hover-built:02-live-pipelines.test.mjs → pass; hover-built:03-autonomy-governance.test.mjs → pass; hover-built:04-organism-federation.test.mjs → pass; hover-built:05-twenty-task-autonomy.test.mjs → pass; hover-built:06-cultivation-core.test.mjs → pass; hover-built:07-state-space-forms.test.mjs → pass; hover-built:08-exact-recall-and-coordination.test.mjs → pass; hover-built:09-front-screen.test.mjs → pass; hover-built:11-birth-mirror-runtime.test.mjs → pass; hover-built:17-movement-temporal-experience.test.mjs → pass; hover-built:20-hover-task-worker.test.mjs → pass; hover-built:android-hand-adapter.test.mjs → pass
- **Echo (resident)**: hover-built:21-synthia17-resident.test.mjs → pass

## The 15-row acceptance table, judged on this branch

| # | Row | Status | Evidence | Missing |
|---|---|---|---|---|
| 1 | Swarm is code | **EXISTS-UNTESTED** | integrated/integrated-swarm.mjs ingests any bytes into six-bit primitives and places them in five fields (measure.test.mjs mirror tests) · integrated/providers.mjs executes JS source through a guarded same-thread surface | execution is of the original SOURCE TEXT (AsyncFunction), not of the particle representation; no file of hers defines executing from six-bit states |
| 2 | Recursive composition | **EXISTS-RUNS** | integrated/population.mjs (composition of primitives, Population); scorecard 3 Composition PASS · repo-test:integrated/population.test.mjs | – |
| 3 | Arbitrary code ingestion | **EXISTS-UNTESTED** | any-file-swarm.mjs fileFeatures ingests any bytes; language adapters for js, py, css, json are all implemented:false | no language parser exists in the branch; Tree-sitter is only a research reference (not packaged) |
| 4 | Code creation | **EXISTS-UNTESTED** | app-finisher.mjs adds missing files from a checklist (app-finisher.test.mjs) | no creation of new program logic from intent; gameTarget and 2d adapters are not implemented stubs |
| 5 | Neuro-symbolic reasoning | **EXISTS-UNTESTED** | integrated/trident-js.mjs (neural, her weights) and integrated/guide.mjs (symbolic lookup) both run, separately · AUTOLING and Klein exist in her files and run there | they are not connected to each other or to the swarm area; no reasoning step combines them |
| 6 | Collective adaptation | **EXISTS-UNTESTED** | integrated/abc.mjs, swarm-fit.mjs (PSO), population.mjs (GA), EmergentMesh.adapt() decay/prune (swarm.test.mjs) | scorecard item 7 Adaptation is a FAIL: no search is driven by the mesh or by measured gaps yet; swarm-fit and abc do not use the shared primitive rules (registry sharesRules=false for swarm-fit) |
| 7 | Self-modification | **EXISTS-UNTESTED** | swarm-loop.mjs writes to a virtual Workspace with protected originals and a hash-chained audit log | it assembles the app from a plan; nothing rewrites the running swarm code; no push code exists (proposals only) |
| 8 | Memory continuity | **EXISTS-RUNS** | AuditLog hash chain, HistoryLedger (integrated-swarm), Workspace versions with revert; app-server.test.mjs | no predictive memory is connected (Synthia-Predictive-Memory exists and its test passes, unconnected) |
| 9 | Arbitrary visual morphing | **EXISTS-UNTESTED** | photo-morph.mjs and frame-morph.mjs (tests exist in the repo) · Hover canonical-morph-runtime passes its test, unconnected | morphing is image/skeleton morphing; arbitrary visual morphing driven by the swarm is not built |
| 10 | Photorealism | **MISSING** | – | no photoreal renderer or Gaussian-splat code exists in her files or this branch (searched for splat directories); CharacterFactory is a third-party GPU project |
| 11 | Semantic/visual coupling | **PROPOSED** | primitive kinds shape (5), sense (6) and colors/sounds exist as data in primitive-functions.json · dimension-visual/core.mjs (outside the repo) hard-maps word to dimension: scorecard FAIL | no coupling from swarm state to visuals is built; colors/sounds/shapes work is still open |
| 12 | Application creation | **EXISTS-RUNS** | app-finisher.mjs + swarm-loop.mjs assemble 73 units; CI builds the APK; guide app and Synthia app run in the emulator | creation is from a fixed plan (assembly-plan.mjs), not from a new request |
| 13 | Local operation | **EXISTS-RUNS** | pure JS, no worker, no backend; measure.test.mjs asserts the manifest has no INTERNET permission · APK emulator self-tests ran offline | trident_syntia.pt is 14.5 MB inside the APK; model load time on a phone was not measured |
| 14 | Mobile delivery | **EXISTS-RUNS** | CI runs 38885989073 and 38895484410 built debug APKs and passed emulator self-tests (GUIDE_SELFTEST, SYNTHIA_SELFTEST) | the build with the measurement view and swarm area is produced later in this step; release signing and phone install were not done |
| 15 | Reliability | **EXISTS-UNTESTED** | node --test suites on the branch (counts in the final report) | no soak/long-run test; the heartbeat loop was run for the 73-unit plan only |

## Six bases vs five bases per tone: reconciled against the source files

The five-base count is what her address specification (swarmdoc.md), Ra's quoted article, v12 code, the corrected state-space engine and this repo use. The six-base count appears in the v14 OUTPUT summary and in one raw older text; no v14 code is in the zip, so the 6 cannot be traced to a function. Her own degree_recursion.py calls it "the still-open 5-vs-6 question". This repo keeps five; v14's six is recorded here and in the guide's "To confirm" list as unresolved, and nothing was changed.

**Five:**
- `drive/swarmdoc.md`:54: 5 bases per tone (line 3827: "64 gates, 6 lines, 6 colors, 6 tones, 5 bases")
- `research/hd-math-verification.md`:106: Ra Uru Hu's own article states the structure: "Six Lines, six Colors, six Tones, five Bases." ... 6 x 6 x 6 x 5 = 1,080 bases per gate, 69,120 on the wheel
- `swarm-os-v14/x/v12/cosmic_engine.py`:32: Base_Anchor width_deg 0.0052083 (= 5.625/6/6/6/5, i.e. tone divided by 5)
- `synthai-read/computer/donors/recovered/you-n-i-verse-corrected/state-space-engine-v2.ported.mjs`:219: FIXED: 5 bases (1-5), was clamped to 6
- `swarm-search/Back-up-/vendor/pure-synthia-v0.4.0/src/synthia/semantic-mesh/state-space/constants.js`: BASE_ARCSECONDS = 18.75; // 5 bases/tone
- `chassis-work/candidates/swarm-reality-trial/resonance_integrated_trial/integrations/biverse/human_design/degree_recursion.py`:40: BASE_SPAN = TONE_SPAN / 5  # 5 bases per tone -- the still-open 5-vs-6 question
- `integrated/primitive-functions.json (this repo)`: base.1 .. base.5; arc-engine.mjs DEGREES_PER_BASE = DEGREES_PER_TONE/5

**Six:**
- `swarm-os-v14/x/v14/expansion_summary.json`:8: "bases_per_tone": 6 — its totals agree with six: 99,520 virtual agents = 64 x 1,555 and 1,555 = 1+6+36+216+1,296 (six bases); five bases would give 1+6+36+216+1,080 = 1,339 per gate
- `swarm-search/Back-up-/vendor/kimi-agent-automata-state-space-merge/pure-synthia-automata/docs/corpus/raw-sentence-breakdown.txt`:3594: Source: 6 bases within each tone (also "Base (1-6)")

## Research references (not packaged: no donor code without a purpose, a connection and a test)

- Gaussian splatting repos: PROPOSED reference only; no local copy used
- Scallop: PROPOSED reference only; no local copy used
- DeepProbLog: PROPOSED reference only; no local copy used
- LNN: PROPOSED reference only; no local copy used
- Chemlambda: PROPOSED reference only; no local copy used
- ScaFi: PROPOSED reference only; no local copy used
- Protelis: PROPOSED reference only; no local copy used
- Mesa: PROPOSED reference only; no local copy used
- PySwarms: PROPOSED reference only; no local copy used
- MEALPY: PROPOSED reference only; no local copy used
- DreamCoder: PROPOSED reference only; no local copy used
- PonyGE2: PROPOSED reference only; no local copy used
- DEAP: PROPOSED reference only; no local copy used
- Tree-sitter: PROPOSED reference only; no local copy used
- Comby: PROPOSED reference only; no local copy used

## The 68-entry swarm registry

`integrated/swarm/registry.json`: 68 swarms by reference with sha256 (referenced 6, not-connected 42, in-repo-reference 8, connected 11, connected-copy 1), plus 216 further swarm-named files found and listed so none is dropped.
