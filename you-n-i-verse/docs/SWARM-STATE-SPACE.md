# Swarm + nested state space (pure JavaScript)

Built from Alexis's rules and the NSNF spec v0.1 (Drive: "Synthia: Architecture and Nested State-Space Neural Field (combined)").
Her words are kept; anything she or the spec has not settled is marked **investigating** in the code (each module exports an `INVESTIGATING` list) and below.

## Run it

```
node swarm-demo.mjs                       # built-in sample files: js, json, css, py, binary, zip
node swarm-demo.mjs path/to/any-file ...  # your own files (read only, never executed)
node --test *.test.mjs                    # full suite (this folder), also run in CI
```
The demo address is the repo's test fixture, not Alexis's. Use `SIMLISH_ADDRESS='{"planetary":...}'` to pass a real one.

## What a file does

`processFile({name, bytes}, options)` in `any-file-swarm.mjs`:

1. **Registration**: the existing registration swarm checks the full 13-place address (incomplete is kept as `investigating`).
2. **Genotype (PSO)**: six checks in packs of 3, 2, 1 -> three 2-bit pack values -> a six-bit hexagram (`swarm-component.mjs`).
3. **Placement**: dimension, level-1 gate (King Wen -> Fu Xi map) and sub-position (line) come from the registered address; deeper levels from the file's SHA-256. No address, or a Space-led address -> the file is **held**, not placed (nothing guesses a dimension; Space cannot be written).
4. **Execution (`nsnf-runtime.mjs`)**: Spec §5 contract: encode -> resolve candidates -> activate a sparse subset under the tier budget -> local compute + typed messages -> micro -> meso -> macro -> output + trace -> persist event/snapshot/transition/checkpoint -> cache only on an exact state match. Uncertainty starts at tier 2 (Spec §17).
5. **Finisher** (optional, on by default): additive files under `simlish/`, translation target, original bytes to Supabase (or the outbox, `pending-supabase`).
6. **Phenotype (GA)**: what is shown (title, placement, six lines, status), expressed from the genotype.

The uploaded file is only read. It is never run or changed.

## State space (`state-space.mjs`)

Five dimensions x 64 slots; six-bit hexagram per slot (line 1 bottom). Nested: 64 macro nodes per level, each broken into 12 (Alexis also said six: both allowed, 12 is only the default), five levels -> (64x12)^5 = 267,181,325,549,568 positions per dimension (with six: 8,349,416,423,424). Nodes exist only once touched. Space is non-primitive: its 64 slots are reported but never written; its value is read through an injectable emergence rule (none by default). Spec §36 operators (line flip, complement, reverse, trigram exchange, nuclear) are included; King Wen, Fu Xi and gate numbers are separate named maps.

## Components and learning

Spec §7 interface (`initialize, forward, update, snapshot, restore`) on a tiny trainable cell. Parameters are stored apart from events (`parameter_ref`). `learn()` updates only the components an event activates. PSO (`trainWithParticleSwarm`) is the background tool that trains a component gradient-free and reports held-out loss; it only replaces parameters if held-out loss does not get worse. Cross-component transfer is **not** implemented or claimed.

## Still investigating (needs Alexis; none of these was decided silently)

- Split: 12 or six. Levels: Spec §6 calls "five levels" tiers whose meaning is not fixed.
- Space readout: no rule exists (Spec §70 forbids silently deciding it).
- Nine emergent centers: not implemented (rule unspecified, no modulo-nine). Meso groups are connected groups of the active graph instead.
- Tier numbers (5/15/35/70/110 components, 0-2 rounds, Hamming radius 0-2) are the v0.1 prototype defaults, configurable.
- Which edges are permitted (Spec §36 flag: her "Hamming distance 1 at every level" rule is not confirmed for NSNF). Edges here emerge from co-activation.
- How a file maps below its address, which file features matter, pack count -> two-bit value and pack order.
- Language adapters (js, py, css, json) are placeholders; only generic byte features run.
- Supabase bucket name `simlish-originals` is a placeholder default.
- `reality-execution.mjs` `STATE_SPACE` (9 centers per dimension, 2880) conflicts with the spec; kept, flagged, not used by the swarm.
