# SWARM-OS — Pass 05: literal implementation of the specified dimensional key & relational-position law

**Source explicitly supplied by user:** https://docs.google.com/document/d/1pRXTJGn5gwZvYuX2NQwKinM72HM2pU8arUS7KKK-nh0/edit
Read on 2026-10-17. The source contains repeated earlier drafts. This pass uses the explicitly labeled **current fixed key** and **final recap** in the last section, not older assistant-authored guesses.

## Where the mathematical source rules are located in the supplied document

- “WHO/WHAT/WHERE/WHEN/WHY” dimensional/ax-word key: final recap §1, around source lines 5938–5970 and repeated at end (lines 6336–6343).
- Intrinsic identity is distinct from relational placement: §2, around lines 5979–6035.
- “Dominating structure” establishes the local frame, recursively: §3, around lines 6041–6092.
- **Only** the Being-led traversal is explicitly ordered: §4, around lines 6100–6141.
- Landed relationships remain in append-only history: §5, around lines 6154–6202.
- Larger compositions retain lower constituents and can themselves become the next primitive: §6, around lines 6210–6243.
- Exact dimension-to-degree/minute/second and color/tone/base arc-placement equation **not finished**: around lines 5489–5535 and 5781–5795.

## What existing code had wrong

- Earlier `semantic-dna-swarm` `WH_MAP` assigned WHERE to Movement and WHEN to Being. The submitted document says **WHERE = Being / Matter / Arc-second**, **WHEN = Movement / Energy / Axiom**. This pass does NOT mutate the old donor file or silently retain its incorrect routing. It adds a source-locked table and tests against all 5 exact mappings.
- Previous `positioned-state-space.mjs` validated gate/line/color/tone/base, relative views, and history of *a position* but had no intrinsic-vs-relational identity representation or dominating-frame relationships.
- Prior passing arithmetic tests did NOT show that those two rules were implemented.

## Actual implementation (no replacement of old files)

New additive source: `source/spec-locked-dimensional-relations.mjs`:

- `fixedDimensionalKey()` returns the explicit source table with all 5 questions, dimensions, natures, ax-words.
- `dimensionalIdentity()` looks up a fixed intrinsic quality. It is **never renamed** by the role/slot it occupies inside another frame.
- `frameReading()` resolves explicitly supplied subject + dominating frame + relative slot. Distinguishes intrinsic identity from relational placement; annotates source. It supplies Being-led order exactly; **does not invent an order for the four other leaders**.
- `PositionalRelationLedger` retains explicitly supplied 13-field positions by calling the *existing* `resolvePosition` validator, has append-only registrations/placements, allows contained things to function as higher-scale frames, preserves constituent references, and supports deterministic checkpoint/replay with tamper checks.
- If no positional data is provided, the address is `null`, never fabricated.

New direct spec tests: `tests/spec-locked-dimensional-relations.test.mjs`

They verify: 5 exact fixed correspondences, Being-led traversal, example where a Movement/Axiom identity occupies a different frame's Axis slot without becoming Axis, multiple placements with both histories retained, recursively nested frames and constituent references, restart/checkpoint replay, rejection of invalid addresses and invented frame references.

## Local result

`node --test tests/*.test.mjs` => **38 passed / 0 failed**, including **7 new tests** tied directly to this document. These tests establish the **local implementation of the specified bookkeeping rules**, not scientific generality or full integration with all SWARM-OS repos/Drive files.

## Still not claimed / not invented

- Exact dimension-dependent arc formula, placement of ax-word terms by derived angular coordinates, complete shape/sound/color equivalence across fields, authoritative 64-gate semantic content integration, proof of universal correspondences, and actual wiring into all SWARM-OS apps remain unverified or outside this narrowly sourced pass.
- `arc-second` as the structural term for Being is preserved as a named role. It is **not** silently equated with numeric absolute longitude `arcSecond`, or automatically assigned to the current position.
- The source document includes alternate/older formulations and excerpts. This pass deliberately relies on the final, explicitly **fixed** table for these five terms.
- Original Pass 04 (and donor source archives) are untouched; this is a new additive ZIP for reproducible inspection.
