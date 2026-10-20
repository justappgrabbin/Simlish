// Each sourced function reproduces the printed table it was ported from (page cited in the source file).
import test from 'node:test'; import assert from 'node:assert/strict'; import fs from 'node:fs';
import * as G from './govinda.mjs'; import * as R from './rutt.mjs'; import * as S from './schonberger.mjs'; import { standardAminoAcid } from './science.mjs';
import { bitsOf } from '../../state-space.mjs'; import { mawangduiRuttMacros } from '../mawangdui-rutt.mjs'; import { mawangduiVersions, differ } from '../mawangdui-versions.mjs';
test('Govinda p. ix: word-symbols and abbrev signs per trigram; trigram lines', () => {
  assert.deepEqual(G.TRIGRAMS.map((t) => G.TRANSLITERATION_P_IX[t].wordSymbol), ['Heaven', 'Earth', 'Thunder', 'Wind', 'Fire', 'Water', 'Lake', 'Mountain']);
  assert.deepEqual(G.TRIGRAMS.map(G.trigramShape), ['○', '□', '∪', '⌅', '△', '▽', '⊻', '∩']);
  assert.equal(new Set(Object.values(G.TRIGRAM_VALUE)).size, 8);
});
test('Govinda pp.46-47: every plane row has all eight trigrams; values spot-checked against the page', () => {
  for (const p of G.PLANES) assert.equal(Object.keys(G.PLANES_PP46_47[p]).length, 8, p);
  assert.equal(G.trigramQuality('KIAN', 'partsOfBody'), 'HEAD'); assert.equal(G.trigramQuality('GEN', 'functions'), 'RELIABILITY');
  assert.equal(G.trigramQuality('KAN', 'qualitiesOfNature'), 'COLD'); assert.deepEqual(G.PLANES_PP46_47.shuoGuaSequence, { KIAN: 6, KUN: 4, JEN: 1, SUN: 2, LI: 3, KAN: 7, DUI: 5, GEN: 8 });
});
test('Govinda p.42: one plane per call; hexagrams read through upper/lower trigrams', () => {
  assert.throws(() => G.trigramQuality('KIAN', 'partsOfBody', 'psychologicalQualities')); assert.throws(() => G.hexagramQuality(63, 'images', 'functions'));
  const q = G.hexagramQuality(0b101010, 'images'); assert.equal(q.upper.trigram, 'LI'); assert.equal(q.lower.trigram, 'KAN');
});
test('Govinda p.43 groups agree with the pp.46-47 "states of being" row; p.44 polarities', () => {
  const map = { universal: 'UNIVERSAL', organic: 'ORGANIC', elemental: 'ELEMENTARY', inorganic: 'INORGANIC' };
  for (const t of G.TRIGRAMS) assert.equal(map[G.groupOf(t)], G.PLANES_PP46_47.statesOfBeing[t], t);
  assert.equal(G.complementTrigram('JEN'), 'SUN'); assert.equal(G.reverseTrigram('JEN'), 'GEN'); assert.equal(G.reverseTrigram('LI'), 'LI');
});
test('Govinda pp.44/57 element cycles close; p.53 colours = p.57 colours; animals by element', () => {
  for (const c of [G.GENERATES, G.DESTROYS]) { let e = 'WOOD'; for (let i = 0; i < 5; i++) e = c[e]; assert.equal(e, 'WOOD'); }
  assert.equal(G.DESTROYS.WATER, 'FIRE'); assert.equal(G.GENERATES.FIRE, 'EARTH');
  assert.deepEqual(G.ANIMALS_PP51_53.filter((a) => a.element === 'EARTH').map((a) => a.direction).sort(), ['North-East', 'North-West', 'South-East', 'South-West']);
  assert.equal(G.SIXTY_CYCLE_P52.length, 60); assert.equal(G.FIVE_TONES_P51.status, 'INVESTIGATING');
  assert.deepEqual(G.TRIGRAMS.map((t) => G.trigramElement(t).element), [null, null, 'WOOD', null, 'FIRE', 'WATER', 'IRON', null]);
});
test('Govinda p.67: Fu Hi -> King Wen changes; printed movement types vs diagram geometry (KAN-twice flag kept)', () => {
  assert.deepEqual(G.FU_HI_TO_KING_WEN_P67, { KIAN: 'LI', LI: 'JEN', JEN: 'GEN', GEN: 'KIAN', KUN: 'KAN', KAN: 'DUI', DUI: 'SUN', SUN: 'KUN' });
  assert.equal(G.FU_HI_PLACE.KIAN, 'SOUTH'); assert.equal(G.FU_HI_PLACE.KUN, 'NORTH'); assert.equal(G.FU_HI_PLACE.LI, 'EAST'); assert.equal(G.FU_HI_PLACE.KAN, 'WEST');
  const [s1, s2, s3, s4] = G.MOVEMENT_STATEMENTS_P67;
  assert.equal(G.classifyPair(...s2.pair), s2.kind); assert.equal(G.classifyPair(...s3.pair), s3.kind); assert.equal(G.classifyPair(...s4.pair), s4.kind);
  assert.notEqual(G.classifyPair(...s1.pair), s1.kind, 'printed KIAN/KAN does not give the diagonal on the diagram'); assert.equal(G.classifyPair('KIAN', 'KUN'), 'diagonal-divergent');
});
test('Govinda p.66 DURATION quote is on the Evolution wheel as a phrase', async () => { const W = await import('../wheels.mjs'); assert.equal(W.WHEELS.Evolution.principle.quote, G.DURATION_P66); });
test('Schönberger p.64: printed decimals, stops, AUG; differs from the standard code only at AUA', () => {
  for (const [c, n] of [['UUU', 0], ['UUC', 16], ['UUG', 32], ['UUA', 48], ['CUU', 1], ['GUU', 2], ['AUU', 3], ['AUG', 35], ['UAA', 60], ['UAG', 44], ['UGA', 56], ['AAA', 63]]) assert.equal(S.schonbergerDecimal(c), n, c);
  assert.equal(S.schonbergerCodonToHexagram('UAA').kingWen, 33); assert.equal(S.schonbergerCodonToHexagram('UAG').kingWen, 56); assert.equal(S.schonbergerCodonToHexagram('UGA').kingWen, 12); assert.equal(S.schonbergerCodonToHexagram('AUG').kingWen, 41);
  for (let g = 1; g <= 64; g++) assert.equal(S.schonbergerCodonToHexagram(S.schonbergerHexagramToCodon(g)).kingWen, g);
  const diff = Object.keys(S.SCHONBERGER_AMINO_P64).filter((c) => S.PRINTED_TO_ONE[S.SCHONBERGER_AMINO_P64[c]] !== standardAminoAcid(c)); assert.deepEqual(diff, ['AUA']);
});
test('codon comparison snapshot exists and records differences (no winner)', () => { const j = JSON.parse(fs.readFileSync(new URL('./codon-comparison.json', import.meta.url))); assert.equal(j.rows.length, 64); assert.equal(j.mismatchCounts.lcm, 1); });
test('Rutt Table 4: right digit = bottom line, 1 = whole; repo bitsOf agrees for all 64', () => { assert.equal(R.repoMatchesTable4(), true); assert.equal(bitsOf(1), '000001'); assert.deepEqual(R.fuXiOrder().slice(0, 3), [0, 1, 2]); });
test('Rutt pp.99-100 Diagram 6: bagua numerals decode to the printed Zhouyi numbers', () => { for (const d of R.DIAGRAM_6_P99) assert.equal(R.decodeBaguaNumerals(d.numerals).kingWen, d.zhouyi, d.numerals); });
test('Rutt pp.115-117: generated Mawangdui = Table 9 grid = Table 10 = mawangdui-rutt.mjs; differs from every version of hers', async () => {
  const g = R.mawangduiGenerated(); assert.deepEqual([...g], [...R.table9Macros()]); assert.deepEqual(R.mawangduiGeneratedKingWen(), [...R.TABLE_10_P117]); assert.deepEqual([...g], mawangduiRuttMacros());
  const V = await mawangduiVersions(fs.readFileSync(new URL('../reference/hexagram_sequences.js.txt', import.meta.url), 'utf8'));
  assert.deepEqual(Object.fromEntries(Object.entries(V).map(([k, v]) => [k, differ({ macros: g }, v)])), { A: 40, C: 54, D: 58, E: 54, Bliteral: 61, Binternal: 61 });
});
