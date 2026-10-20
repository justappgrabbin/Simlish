# Author charts — design-side reading (Alexis's simplified method, 2026-10-20). NOT applied to code.

Method: birth DATE only. Tropical Sun gate/line from her chart engine (integrated/chart-engine.mjs, wheel "gate 41 starts at 302°"),
evaluated at 00:00 and 23:59 UTC of the date (line given as a range when it changes). Gate → Fu Xi decimal → upper/lower trigram →
Govinda p.152 (one circle, abstract order: each line = two movements, e.g. KAN-KUN and KUN-KAN; doubled hexagrams outside the circle)
and p.67 (each trigram's Fu Hi → King Wen move). Dimension = her repo mapping DOC3_GATES[gate].base → DOC3_BASE_DIM
(integrated/reference/dimension-visual-core.mjs.txt:992, base 1 movement, 2 evolution, 3 being, 4 design, 5 space).

| Author | Birth date (source) | Sun gate.line | Hexagram trigrams (upper / lower) | p.152 line | p.67 moves (upper; lower) | Dimension (her doc3 map) |
|---|---|---|---|---|---|---|
| Lama Anagarika Govinda | 1898-05-17, Waldheim, Saxony (Wikipedia "Anagarika Govinda" — verified) | 8.2–8.3 | KAN / KUN | KAN-KUN / KUN-KAN (gates 8 / 7) | KAN W→N; KUN N→SW | **Movement** (base 1) |
| Richard Rutt | 1925-08-27 (Wikipedia "Richard Rutt"); born Bedfordshire (book's "About the author") | 59.4–59.5 | SUN / KAN | SUN-KAN / KAN-SUN (59 / 6) | SUN SW→SE; KAN W→N | **Being** (base 3) |
| Richard Wilhelm | 1873-05-10, Stuttgart (Wikipedia "Richard Wilhelm (sinologist)") | 23.1–23.2 | GEN / KUN | GEN-KUN / KUN-GEN (23 / 15) | GEN NW→NE; KUN N→SW | **Evolution** (base 2) |
| Bradford Hatcher | 1949 only (castiching.com "Bradford Hatcher (1949-2020)"); full date NOT found | — | — | — | — | not computable |
| Martin (Maria) Schönberger | 1912 only (Univ. Zürich "China und der Westen": Schönberg 1912-2005; archive.org record) | — | — | — | — | not computable |
| Shao Yong | 1011–1077 (no verified day) | — | — | — | — | not computable |
Time unknown for all. Sun line ranges are within one gate for each computed author (gate is stable across the day).

## Which dimension each disputed attribute belongs to
Her files do not assign "element colour", "trigram→element", "organ", "number" or "codon" to one of the five dimensions → **investigating** for every conflict. The only placement her repo gives in this area is: Movement = Fu Xi / the p.67 Fu Hi→King Wen movement (wheels.mjs L1; bhantugh Movement diagonal).

## Proposed resolutions (for Alexis to approve; nothing applied)
| Conflict | Authors in conflict | Aligned author IF the attribute is placed in… | Proposal |
|---|---|---|---|
| Trigram→element (Zhen Fire vs Wood; Dui Water vs Metal; Xun, Kun, Gen) | Govinda(derived)/Std vs Hatcher | Movement → Govinda (Movement). Hatcher not computable | If trigram→element is a Movement attribute (it comes from the p.67 directional move), use Govinda's derivation + p.54 print; Hatcher kept as an alternate layer |
| Five-element organs | Govinda p.57 vs Std | Movement → Govinda | Keep Govinda's printed row as primary if Movement; Std as science-layer note |
| Element numbers (Wood 7/8, Metal 8/9, Fire 9/7) | Govinda vs Hatcher | Movement → Govinda | Govinda if Movement |
| Mawangdui order | Rutt vs her versions A–E | Evolution → Wilhelm (no Mawangdui table of his in the set); Being → Rutt | Mawangdui is her Evolution wheel; Rutt's chart is Being → not aligned; stays UNRESOLVED |
| Binary string orientation | Rutt vs Hatcher | Rutt Being; Hatcher n/a | No change: repo already equals Rutt |
| Codon UAG/UGA/AUA | Govinda/Schönberger text vs table vs standard code | Schönberger not computable | Science layer (NCBI) stays separate; no resolution by chart |
