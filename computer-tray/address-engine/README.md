# State space fill (from Alexis's Google Doc 1pRXTJGn5gwZvYuX2NQwKinM72HM2pU8arUS7KKK-nh0)
Run (Node 22, pure JS): `node build.mjs && node resolver.test.mjs && node coverage.mjs && node questions.mjs`
- source.docx / doc-items.json — the doc, every paragraph + table row numbered in doc order (extract.py)
- state-space.data.json — filled state space: gates 1-64 (+ 6 line slots each), lines 1-6, colors 1-6, tones 1-6, bases 1-5 (Base 5 Space = non-primitive), axes, zodiac 12, houses 12, context, placement map (doc item -> where it went)
- resolver.mjs — state("gate.line.color.tone.base") or state({degree}) -> full state + B H A N T U G H encodings; decode.* for round trips
- resolver.test.mjs / test-output.txt — round-trip test
- coverage.mjs / coverage.json — doc items vs placed (0 skipped)
- questions.mjs / open-questions.json — every disagreement/unclear spot, not decided
- vendor/ — copied unchanged from you-n-i-verse (state-space, dodeca-face-axes, axis-order) and dimension-visual/core.mjs (letter layers)
