# Simlish embodied mesh integration

This is an incremental physics bridge, not a completed organism or a replacement for the supplied system.

Open `swarm/index.html` through a static HTTP host. The JavaScript modules run client-side without Python, remote inference, or a runtime backend. Choose a flavor and apply an impulse by clicking the world.

Implemented: finite-mass particles, gravity, opposing Hooke spring forces, damping, restitution at world boundaries, speed constraints, strain-dependent bond rupture, nested ordered/unordered composition references, persistent constituent identity during morphs, and causal event provenance. Flavor parameters are explicit simulation settings, not asserted ontological laws. The demonstration's colors identify bits; they are not the canonical address-derived color mapping.

The scene grammar is preserved from the supplied State Space Foundation at `vendor/browser/src/merged/scene-grammar.js`. Bond conditions use its actual rule-selection interface. The simulation parameters and bond rules are new, disclosed implementation choices.

Validation: `node --test swarm/world.test.mjs`.

Still required for the full integration: executable per-particle automata and capability constraints, the character needs/emotions/skills adapter, host-game authoritative action routing, self-cultivation and human-success feedback, canonical addresses, source-defined sound/shape/color resolution, GA composition and PSO task algorithms. Phase exclusion is implemented; it does not itself implement GA or PSO. No claims of complete source integration are made.

The supplied CharacterEngine is a C# prototype with traits, emotions and skills; its README explicitly says its decision model is incomplete. The agent archive includes C# FSO interfaces and JavaScript astronomia. These are preserved in the source inventory; they have not been silently replaced with the particle simulation.

`sources.json` identifies all sixteen downloaded source pieces and their hashes. Original Simlish files remain present. Downloaded archives remain in the review workspace while selected modules are integrated.
