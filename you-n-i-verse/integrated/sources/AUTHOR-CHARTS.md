# Author charts — design-side reading (Alexis's method, corrected 2026-10-20 1:32 AM PT). NOT applied to code.

Birth DATE only; time unknown (computed at 00:00 and 23:59 UTC of the date). Tropical positions from her chart engine
(integrated/chart-engine.mjs, wheel "gate 41 starts at 302°"). The Govinda diagrams are used as a REFERENCE to infer the
Sun gate's dimension — the gate is NOT drawn onto the diagram.

## Birth data
| Author | Birth date | Source |
|---|---|---|
| Lama Anagarika Govinda | 1898-05-17, Waldheim, Saxony | Wikipedia "Anagarika Govinda" (verified) |
| Richard Rutt | 1925-08-27, Bedfordshire | Wikipedia "Richard Rutt"; place from the book's "About the author" |
| Richard Wilhelm | 1873-05-10, Stuttgart | Wikipedia "Richard Wilhelm (sinologist)" |
| Bradford Hatcher | 1949 (year only) | castiching.com "Bradford Hatcher (1949-2020)" — not computable |
| Martin Maria Schönberger | 1912 (year only) | Univ. Zürich "China und der Westen" (Schönberg 1912-2005) — not computable |
| Shao Yong | 1011 (year only) | — not computable |

## Step-by-step Sun reading (reference: Govinda p.152 "Projection of all hexagrams on one circle according to the abstract order"; p.67 "Transformation from the abstract to the temporal order")
1. **Sun gate** (stable all day): Govinda 8 (line 2–3), Rutt 59 (line 4–5), Wilhelm 23 (line 1–2).
2. **Where it sits per the diagram** (p.152 reads a hexagram as the line joining its two trigrams; each line = two movements; p.67 gives each trigram's Fu Hi → King Wen move):
   - Govinda 8 = KAN over KUN → p.152 line KAN–KUN / KUN–KAN (gates 8 and 7); p.67: KAN W→N, KUN N→SW.
   - Rutt 59 = SUN over KAN → line SUN–KAN / KAN–SUN (59 and 6); p.67: SUN SW→SE, KAN W→N.
   - Wilhelm 23 = GEN over KUN → line GEN–KUN / KUN–GEN (23 and 15); p.67: GEN NW→NE, KUN N→SW.
3. **Its dimension**: the diagrams carry no dimension labels, and her files define no rule from a p.152/p.67 position to a dimension → **INVESTIGATING**. The one dimension the repo ties to this diagram family is Movement (wheels.mjs L1 = Fu Xi/abstract order, the order p.152 draws), which applies to the diagram as a whole, not to a single gate. Her per-gate map (DOC3_GATES[gate].base → DOC3_BASE_DIM, integrated/reference/dimension-visual-core.mjs.txt:992) gives, for reference only: 8 → Movement, 59 → Being, 23 → Evolution.
4. **What the author represents when speaking**: follows from step 3, so it is INVESTIGATING until Alexis gives the position→dimension rule. Read via her doc3 map (reference only): Govinda speaks from Movement, Rutt from Being, Wilhelm from Evolution.

## Mercury (raw data only, no interpretation)
| Author | Tropical Mercury gate.line (00:00 → 23:59 UTC) |
|---|---|
| Govinda | 27.5 (35.87° → 36.10°) |
| Rutt | 59.1 (150.23°) → 29.6 (149.41°): Mercury retrograde, crosses the 59/29 boundary that day |
| Wilhelm | 42.4 (23.78°) → 42.5 (25.05°) |

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
