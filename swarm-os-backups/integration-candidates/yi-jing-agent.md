# Yi-Jing Agent integration candidate — additive SWARM-OS research

Upstream: https://github.com/yayasasapig/yi-jing-agent
License reported by upstream: MIT (retain upstream copyright/license on any vendored source).

## Proposed connections
- Six-stage task lifecycle → existing state-space and execution spine
- Six-bit state mutations → existing cube invariants and failure transitions
- Reflection → existing Klein/Situation Router
- Authorization → existing explicit consent gate
- Memory → append-only trace

## Current status
- Upstream repository reviewed at README level; upstream source not cloned or tested.
- Independent local six-stage adapter prototyped against existing Pass 25 Python port.
- 10 local adapter tests passed.
- Prototype ZIP is **NOT uploaded to GitHub** in this commit.
- No replacement of Simlish or existing SWARM-OS modules.
- Future integration requires actual source audit, license preservation, interface mapping, and regression tests.
