# SWARM-OS preservation branch

This branch is reserved for additive, non-destructive backup of SWARM-OS repair checkpoints.

## Rules
- Preserve each numbered pass under its original identity and never overwrite historical archives.
- Never delete or hide existing Simlish application source.
- Keep an index showing present, missing, verified, and integrated checkpoints separately.
- A backup is only marked **uploaded** after its bytes are committed and independently verified.
- Do not confuse archived passes with cumulative runtime integration.
- Do not merge into main without separate review.

## Initial checkpoint status (2026-10-16)
- Branch created from Simlish main.
- Local working environment contains repair archives for passes 1–25 (first checkpoint named `resonance-first-verification-and-repair.zip`).
- Archive binaries are **NOT YET UPLOADED** to this branch.
- Next step: commit binary checkpoints and checksums; verify remote objects.
