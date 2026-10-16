# Primitive population: what is built, what is not (honest status)

Governing text: `docs/SWARM-OS-FOUNDATIONAL-ARCHITECTURE.md` (statement + "Critical Correction: Preserve Primitive Diversity").
Run: `node integrated/population-demo.mjs`. Tests: `node --test you-n-i-verse/integrated/*.test.mjs` (Node 20 and 22).

## Unity (one ontology, shared rules) — `population.mjs`
Every primitive is a node of the existing nested state space (`state-space.mjs`: five dimensions, 64 macro slots per level) with a six-bit hexagram state (packs 3-2-1). Same address resolution (her `ExecutionAddressResolver`: descend to six-bit states, resolve the 13-place address), same composition rule (a composite of cells acts as ONE primitive of the next level, with an emergent dimension and six-bit code), same tick rule (a signal read from a later position is the previous tick's value; no scheduler or controller), state held as six-bit atoms, remembered at the resolved address.

## Diversity (differentiated primitives, not one generic one)
Roles come from her ontological-ingest patch. Four writable kinds, each with its own properties, constraints, state and transformation rule (`PRIMITIVE_KINDS`):
Movement (change field, stateless transition table) · Evolution (change-of-change: state advances by the change applied) · Being (persistence: holds a value until enabled) · Design (organization: routes, and instantiates composed organization). Space has no cell and no execution: it is read-only, emergent. The same hexagram and wiring give four different behaviours (tested).

## Three unrelated targets (`targets.mjs`, data only) — all through `lifecycle.mjs`
| target | output | primitives used |
|---|---|---|
| running total (number) | 5, 15, 42 | Movement + Evolution + Design instantiating HalfAdder inside FullAdder (3 nested levels) |
| word case (text) | "Swarm Os Is Alive" | Movement + Being (boundary memory) + Design |
| pattern (picture) | 8x8 x AND y = 0 | Movement + Evolution (counter) |
Each: reduced (her descend) · resolved to an address (deterministic) · composed (definitions as primitives) · executed · state snapshotted as atoms, restored from a fresh resolution, continues identically · one atom corrupted and the stream reorganized back (Hamming-1 neighbourhood; the existing PSO `fitParticles` is run when local change is stuck).

## Existing primitives: inventory by live probe (`inventory.mjs`)
24 of 24 probes execute (ABC probe added after it was found in v9/v10; earlier text said 23 of 24 with ABC absent): hexagram state/packs, 64-gate map, operators (flip/complement/reverse/exchange/nuclear), Hamming-1, five dimensions + wheels, TinyMLP cells, NSNF runtime, PSO on cells, PSO numeric fit, PSO/GA swarm pair, Bantu codec, five execution fields, address resolver, ontological ingestor. **ABC: absent from her files; not implemented, not claimed.**
Bantu (her `bridge.js` lines 5-9) is kept verbatim (`bantu.mjs`, tested against the reference copy).
Originals untouched: `integrated/MANIFEST.json` holds SHA-256 of every copied file; the earlier modules are unchanged except the small, tested `any-file-swarm.mjs` option additions.

## Failures and limitations (reported, not passes)
1. **Targets are authored, not derived.** The three organizations were written by hand as netlists (`assemble.mjs` is an authoring tool). Existing programs (JS, Python, HTML ...) are NOT reduced to behaviour-preserving primitive organizations. Reducing arbitrary files to executable primitives that mean what the file means is NOT achieved. Any byte stream does decode and run (total decode), but only well-formed streams mean what an author intended.
2. **File engines exist and violate the correction.** `providers.mjs` (JavaScript via same-thread AsyncFunction, CSS/HTML/text/zip/generic parsers) are file-type engines, not derived from primitive organization. JSON.parse, WebAssembly and the JS provider mirror her patches' own providers; the CSS/HTML/text/zip/generic ones are mine. They were added before the correction; they are not part of the Unity/Diversity claim and should be replaced or removed. Only `.prim` files go through the population.
3. **Presentation is outside the primitives.** Turning output bits into a number, text or picture happens in the demo/tests.
4. **Executable semantics of the four kinds are an engineering reading** of one-line role descriptions (see `POPULATION_INVESTIGATING`). Not confirmed by Alexis.
5. **Reorganization:** the repair is done by Hamming-1 local search; the existing PSO was invoked when stuck and gave no improvement in tested double-corruption cases. GA is not used for reorganization (it exists only as the file-swarm phenotype). ABC EXISTS (corrected: swarm-os-v14/x/v9/swarm_os_full.py abc_phase, v10, oct7src morph-search.mjs), is ported in `abc.mjs`, and alone repaired 0 of 6 test corruptions (`dynamics-experiment.mjs`); see `SCORECARD.md`. Success was shown for single and some double corruptions only, not open-ended reorganization.
6. **Not connected to the population's execution:** dimension wheels (annotation only), TinyMLP cells and NSNF tiers (they run the file swarm), Klein/Autoling tools, Space readout (no emergence rule in her spec).
7. **Round trip (`roundtrip.mjs`, measured per file type):** bytes -> six-bit primitives -> bytes exact for every tested type; Bantu tokens exact; her ingestor's reconstruct exact because it STORES the original bytes; reconstruction from the address alone is impossible; source-level reconstruction not attempted. Her `descend` keeps at most 4096 six-bit states, so files over 3072 bytes are not fully reduced by it (our own `bytesToAtoms` is complete).
8. Same-thread JS guard is a deny-list, not a security boundary (no worker/vm allowed). Three address schemas (repo 13-place, spine, ingest) are not mapped to each other.
