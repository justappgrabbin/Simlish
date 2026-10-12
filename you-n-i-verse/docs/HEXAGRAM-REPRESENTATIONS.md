# Embodied and action representations

The user described the supplied hexagram CSV as embodied and the preceding causal specification as action (2026-10-12). The user further clarified these are the paired embodiment/action sides, like yin and yang. Both projections retain the same King Wen gate identity. `hexagram-representations.mjs` joins them explicitly and rejects a mismatched gate. Neither admits an incompletely addressed resident or supplies a geometric template for the swarm.

`hexagram-encodings.json` retains all 64 supplied Chinese names, English labels, pinyin and illustrative RGB values. `hexagram-encodings.mjs` derives the supplied glyph/Unicode, ASCII, NATO, trigram labels and illustrative frequency using the validated gate mapping. The supplied frequency column equals 220 + 5 × display decimal for every row. Those examples remain illustrative; they do not become causal tone/color categories or physical placement rules.

The CSV's binary string is ordered bottom-to-top, but its decimal column parses that displayed string with the first digit most significant. Existing execution uses line 1 as bit 0. Both conventions remain named and distinct: Gate 3 is displayed `100010`, display decimal 34, execution decimal 17. Reversing bit order bridges the representations. Line transitions keep the existing bottom-bit-zero convention.

Pinyin `lü` remains intact for gates 10 and 56; the supplied ASCII/NATO columns transliterate it to `lu`. Repeated names/pronunciations (qian, kun, bi, yi, jian) are not identifiers: gate number and Unicode remain distinct.

The action projection contains the authored compatibility/anchor/resonance evaluation. It does not infer body shape, avatar coordinates, or world geometry from the CSV's example color, sound, or the trigram phrase “shape.” Self-assembly still requires the placement and interaction rules documented in CAUSAL-CONDITIONS.md.
