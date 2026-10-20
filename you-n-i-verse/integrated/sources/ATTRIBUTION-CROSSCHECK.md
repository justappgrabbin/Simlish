# Attribution cross-check (2026-10-20, PT) — no side picked; disagreements marked ⚠

Books (box copies in /workspace/sources/books/, Drive ids):
- Govinda, *The Inner Structure of the I Ching* (1981) — Drive `1unb8il81Anv1REcSCdrpEzMEyRvgYwGP`. Printed page = PDF page − 20.
- Rutt, *Zhouyi: The Book of Changes* (1996/2002) — Drive `1OgSb-EljAqAhM5m-uNtcP-w_V5WhgZSy`, `1uiKe_YkJApeEo6s_hU65-0C_ep86IBYL`.
- Hatcher, *Yijing, Word by Word* Vol. 1 (2009) — Drive `1A3iHHeqtcxjhGYNvLhFUcPvby-jjAawK`, `16HwWv3JDjr7HN0X9qiyMXV7to2BGYQTt`; also https://www.hermetica.info/Yijing1+2.pdf (same size, 4,355,407 bytes). The t431u paste is from this book (title confirmed).
- Standard public Wu Xing reference: the common Chinese-medicine / Huangdi Neijing correspondence (Wikipedia "Wuxing (Chinese philosophy)", correspondence table). Shown as "Std".

## Five elements
| Attribute | Govinda (p.) | Std | Hatcher | Agreement |
|---|---|---|---|---|
| Wood colour / direction / season | Green / East / Spring (p.57) | Green(cyan) / East / Spring | Wood=Shao Yang: Spring, East (p.456) | ✓ |
| Fire | Red / South / Summer (p.57) | Red / South / Summer | Fire=Tai Yang: Summer, South (p.460) | ✓ |
| Earth | Yellow / Center / last 18 days (p.57); "between the seasons" (p.60) | Yellow / Center / late summer or 18 days per season | Earth left out of Si Xiang (p.447) | ✓ (Hatcher silent) |
| Metal | "Iron", White / West / Autumn (p.57) | Metal, White / West / Autumn | Metal=Shao Yin: Autumn, West (p.458) | name ⚠ Iron vs Metal (translation) |
| Water | (Blue) Black / North / Winter (p.57) | Black / North / Winter | Water=Tai Yin: Winter, North (p.454) | ✓ |
| Organ — Wood | Stomach, Spleen (p.57) | Liver (Gallbladder) | — | ⚠ |
| Organ — Fire | Lungs (p.57) | Heart (Small intestine) | — | ⚠ |
| Organ — Earth | Heart (p.57) | Spleen (Stomach) | — | ⚠ |
| Organ — Metal | Liver (p.57) | Lungs (Large intestine) | — | ⚠ |
| Organ — Water | Kidneys (p.57) | Kidneys | — | ✓ |
| Taste | Sour, Bitter, Sweet, Hot, Salty (p.57) | Sour, Bitter, Sweet, Pungent, Salty | — | ✓ (Hot = pungent) |
| Planet | Jupiter, Mars, Saturn, Venus, Mercury (p.57) | same | — | ✓ |
| Number | Wood 8, Fire 7, Earth 5, Iron 9, Water 6 (p.57 planet row) | He Tu: Wood 3/8, Fire 2/7, Earth 5/10, Metal 4/9, Water 1/6 | Si Xiang number: Water 6, Wood 7, Metal 8, Fire 9 (pp.454-461); He Tu: Water 6&1, Wood 8&3, Metal 9&4, Fire 7&2 | ⚠ Hatcher's Si Xiang number (Wood 7, Metal 8, Fire 9) ≠ Govinda (Wood 8, Iron 9, Fire 7); He Tu pairs agree |
| Five tones | named only as a count (p.51) | Jue Wood, Zhi Fire, Gong Earth, Shang Metal, Yu Water | — | Govinda gives no mapping → stays INVESTIGATING (not wired from Std) |
| Shape | — (not in Govinda) | Wood rectangle, Fire triangle, Earth square, Metal circle, Water wave | — | not wired (no book source in her set) |
| Generating / destroying cycles | p.44, p.57 | same | — | ✓ |

## Animals (Govinda pp.52-53, p.60)
Bull, Dragon, Sheep, Dog = Earth (yellow); Tiger, Hare = Wood; Serpent, Horse = Fire; Monkey, Bird = Iron; Boar, Mouse = Water. Std "fixed elements" of the branches agree (Ox, Dragon, Goat, Dog Earth; Tiger, Rabbit Wood; Snake, Horse Fire; Monkey, Rooster Metal; Pig, Rat Water) ✓. p.60 names "Bird" (Cock / pheasant note) ✓.

## Trigram → element
| Trigram | Govinda (derived: direction pp.46-47 = place p.57) | Std (Later Heaven) | Hatcher "Trigrams by Element" (Si Xiang, pp.454-461) | Hatcher Ba Gua "Elements" (pp.464-467) |
|---|---|---|---|---|
| Li / KAN / JEN / DUI | Fire / Water / Wood / Iron | Fire / Water / Wood / Metal | Li Fire, Kan Water, **Zhen Fire ⚠**, **Dui Water ⚠** | — |
| KIAN (Qian) | not stated (INVESTIGATING) | Metal | Metal | — |
| SUN (Xun) | not stated | Wood | **Metal ⚠** | — |
| KUN | not stated | Earth | **Wood ⚠** | Greater Earth |
| GEN | not stated | Earth | **Wood ⚠** | Lesser Earth |
| p.54 colour print | inner petals: south (Li) red, north (Kan) grey-blue, east green, west white; intercardinal petals washed pale/yellowish — not legible enough to assign | | | |
Hatcher's Si Xiang grouping is by the emblem formed by two lines (Xian Tian), not the Later-Heaven Wu Xing; Hatcher himself flags the Wu Xing has "no home base in the Yijing" (p.447).

## Binary / numbering conventions
| Item | Rutt Table 4 (p.93) | Hatcher | Repo |
|---|---|---|---|
| line string | right digit = bottom line, 1 whole | written bottom line FIRST (Gen = 00I, Ba Gua no. 1; Zhen = 4) ⚠ | bitsOf: line 1 = bit 0 = Rutt ✓; Hatcher's numbering is the bit-reverse |

## Codons (Govinda p.64, Schönberger) — see you-n-i-verse/integrated/sources/codon-comparison.json
AUA printed in the "Met" bracket (standard: Ile) ⚠; p.65 text "UAG = Abstraction (12)" vs table UGA = 12 ⚠ (typo); text "start UAG" vs real start AUG (= 41 in the table) ⚠.

## Mawangdui
Rutt Table 9 / 10 (pp.116-117) vs her versions A, C, D, E, B-literal, B-internal: 40 / 54 / 58 / 54 / 61 / 61 positions differ — none of hers equals Rutt ⚠.
