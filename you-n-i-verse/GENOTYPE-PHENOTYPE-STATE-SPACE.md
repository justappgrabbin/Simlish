# Simlish Genotype/Phenotype State Space Integration

Simlish CONSUMES Generative-address-system. The original source state space is not copied, rewritten or replaced. The Simlish world, 3D rig, sprite tools, relationship memory and original game remain in Simlish.

## Direct executable contract

* Simlish exports connectGenerativeStateSpace() from you-n-i-verse/generative-state-space-plug.mjs.
* Boot with bootSimlishStateSpace() in you-n-i-verse/generative-state-space-runtime.mjs. Set SYNTHIA_STATE_SPACE_ROOT to the local checkout of Synthia-Generative-Address-Morphology-Integrated-v0.2.1-RUNNABLE.
* The live source assembly supplies structural analysis, Klein evaluation, canonical address validation and the owner-gated swarm capability registry.
* Simlish supplies the real world via world.morphSelected(), retaining its current phenotype states, addressed relationship evidence and world execution ledger.
* A caller-supplied, provenance-tagged complete genotype address is distinct from Simlish's HASH-GENERATED demo event addresses. No invented or hash-derived semantics are promoted to canonical genotype.
* Permitted phenotype effects: transform, move and duplicate. Destructive remove is not exposed. The owner policy decides whether side effects execute.
* The resulting event preserves genotype coordinates, phenotype before/after, structural observation and the existing swarm execution receipt.

## Source checkout and CI

The workflow .github/workflows/state-space-plug.yml fetches the existing state-space repository at exact commit 4c5ba5560c14aab654c2d6e7070a422fd13c54ea using a sparse source checkout, without making a second 800 MB Simlish history. A sparse checkout reduces the worktree; it does not guarantee minimal Git network transfer.

To run locally from you-n-i-verse:

    SYNTHIA_STATE_SPACE_ROOT=/checkout/Generative-address-system/Synthia-Generative-Address-Morphology-Integrated-v0.2.1-RUNNABLE node --test generative-state-space-plug.test.mjs

## Semantic invariants and limits

There are two linked state spaces, not one flattened address: genotype is the source-controlled positional semantic structure, phenotype is its expressed and observed game-world state. The swarm is the executable formation of the four existing primitive families, not a fifth factory.

Simlish's arc coordinate (0..99), string planetary names, and angle rules are not interchangeable with Synthia's arcAxis and numbered planetary schema without a verified mapping. This integration does not invent that mapping; it requires a complete source-side canonical address.

This stage is a direct **Node execution plug**. A browser/Android runtime, semantic cloud overlay UI, spritesheet builder, automatic high-fidelity skins and personalized birth data initialization are later milestones, not claimed delivered here. The original Simlish world and animation files are unmodified.
