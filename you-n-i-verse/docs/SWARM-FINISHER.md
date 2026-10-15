# Register, finisher and personalize as swarm components

Alexis's rules used: packs are 3, 2, 1; each pack is two bits; three packs make one six-bit hexagram; pieces are 2, 4 or 8 bits;
PSO is the genotype (background tooling, the mechanical side); GA is the phenotype (what is seen, the physical trait);
uncertainty triggers investigation, not rejection. NSNF v0.1 terms are used as written (component record, event_log, transition, hexagram_code).

Flow: `personalize` -> `registerSimlish` (swarm: PSO genotype checks the full 13-place address, GA phenotype shows it) -> `finishApp`
(PSO genotype assesses the checklist, additive files are added under `simlish/`, target adapter translates, GA phenotype is the report) ->
original sent to Supabase when the app was not finished (outbox + `pending-supabase` when env is missing).

Marked INVESTIGATING (not chosen by Alexis or the spec; see `swarm-nsnf.mjs` INVESTIGATING): pack-count-to-two-bits mapping and pack order,
gate_id binding, architecture_type names, no learning/update, no GA/PSO search run yet, default Supabase bucket name `simlish-originals`.
Targets: `reality` implemented (scene event + reality-readiness, nothing rendered); `game` and `2d` are stubs (not yet implemented).
