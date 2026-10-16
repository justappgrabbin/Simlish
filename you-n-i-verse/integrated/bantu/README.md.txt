# B.A.N.T.U. × 3–2–1 Swarm Bridge (experimental)

Open `index.html` in a modern browser, including Android Chrome. No network, build tool, or backend required.

## Implemented
- Actual UTF-8 bytes, ASCII code points for ASCII characters, NATO spelling alphabet, Unicode code points.
- Reversible 6-bit token encoding of UTF-8 bytes: 0..62 directly; 63 is an escape followed by two 6-bit tokens holding a byte's high 2 and low 6 bits. The escape makes this a custom transport, not ASCII and not a canonical hexagram-to-word semantic mapping.
- Each six-bit token displayed as six independently simulated primitive positions grouped 3–2–1 (two bits per level).
- Live particle motion influenced by neighbor distances and a selectable experimental attraction/repulsion/orbit rule.
- Explicit text round-trip verification and manual stepping.
- Original prior Encoding Lab preserved intact as `encoding-lab-original.html`.

## NOT implemented
No original Synthia particle runtime imported, no full causal graph traversal, no sound physics, no generic transliteration, no automatic tool assembly, no word generation, no persistent swarm learning. The motion rule is an isolated demonstration, not a validated source-derived law.

## Tests
`node test.js` validates Unicode round trips and structural invariants.
