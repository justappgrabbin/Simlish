# Human Design path audit: birth data → chart → guide → morph/state-space → what the user sees

Checked on 2026-10-19 (PT). Statuses: EXISTS-RUNS = has a passing executable test or self-test; UNTESTED = code exists, nothing runs it; MISSING = no code connects it.

| # | Step | Before this commit | Now | Evidence |
|---|------|--------------------|-----|----------|
| 1 | Birth data → `integrated/chart-engine.mjs` `calculateBirthChart` | EXISTS-RUNS | EXISTS-RUNS | `integrated/chart-engine.test.mjs`; GUIDE_SELFTEST PASS (headless Chrome + emulator, earlier CI runs) |
| 2 | Chart → guide (`integrated/guide.mjs`, `guide.html`) | EXISTS-RUNS | EXISTS-RUNS | `integrated/guide.test.mjs`; GUIDE_SELFTEST |
| 3 | Chart → state-space / morph / world | MISSING (only guide.mjs, app-server.mjs and tests import chart-engine; state-space.mjs, studio.mjs, game-path.mjs and world-ui.mjs do not) | EXISTS-RUNS through the new play layer | `integrated/play/adaptive.mjs`, `birth-record.mjs`; `integrated/play/play.test.mjs` |
| 4 | Chart → start address (Personality Sun gate.line → Fu Xi state) | MISSING | EXISTS-RUNS | play.test: "two charts get different programming…" (the I Ching world opens on the user's Sun gate) |
| 5 | Interactions → Evolution memory transitions (append-only, hash-chained, Hamming-1 moves) | MISSING | EXISTS-RUNS | play.test: "relational rule and memory…" and "one user's programming changes…" |
| 6 | State → content/tools/views (PSO genotype → GA phenotype, SwarmPair) | MISSING | EXISTS-RUNS | `program()` in adaptive.mjs; play.test |
| 7 | Chart → her youniverse MorphEngine/CharacterRenderer (chart-amplified embodiment) | MISSING (her app was only on Drive) | EXISTS-RUNS | games-host.mjs wraps her code; play.test "her three worlds run unchanged with the real chart"; the page shows her rendered SVG |
| 8 | Chart → repo Studio morph (studio.mjs / photo-morph / frame-morph) | MISSING | MISSING (not changed) | those modules have their own tests, but nothing feeds them the chart; the Studio morph image is offered as the avatar in play.html |
| 9 | `state-space.mjs` (bitsOf/flipLines) in the user path | UNTESTED in the user path (module tests only) | UNTESTED in the user path | the play layer uses `vendor/kingwen.mjs` (gateToFuXiDecimal, the same one-line toggle game-path.mjs uses), not state-space.mjs |
| 10 | What the user sees: guide.html | EXISTS-RUNS | EXISTS-RUNS | GUIDE_SELFTEST |
| 11 | What the user sees: world.html / world-ui.mjs | UNTESTED (no test or self-test opens it) | UNTESTED | — |
| 12 | What the user sees: play.html (new) | — | EXISTS-RUNS locally (headless Chrome PLAY_SELFTEST PASS); CI headless + emulator added | play.mjs self-test |

Investigating (not decided here): five vs six bases, Mawangdui version, EmergentEdgeResolver vs the 36 channels, the 302°/272° wheel, gate 22, and Draconic relevance (§1.7 gives no defined-center rule for it).
