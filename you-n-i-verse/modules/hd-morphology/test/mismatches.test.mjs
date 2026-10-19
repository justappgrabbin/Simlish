import test from 'node:test';
import assert from 'node:assert/strict';
import { mismatches, consistencyChecks, dataset, computeMismatches, factById } from '../src/index.mjs';

test('mismatch list is non-empty and every entry points at real facts and sources', async () => {
  const { SOURCE_BY_ID } = await import('../data/sources.mjs');
  const mm = mismatches();
  assert.ok(mm.length > 0);
  assert.equal(new Set(mm.map((m) => m.id)).size, mm.length);
  for (const m of mm) {
    assert.ok(['science-mismatch', 'science-partial', 'source-conflict', 'rule-conflict', 'observation'].includes(m.kind), m.id);
    for (const b of m.basis) assert.ok(factById(b), `${m.id} basis ${b}`);
    for (const c of m.cites) assert.ok(SOURCE_BY_ID[c], `${m.id} cite ${c}`);
  }
  assert.ok(consistencyChecks().length > 0);
});
test('mismatches are generated from data, not hardcoded: changing data changes the list', () => {
  const base = computeMismatches(dataset()).mismatches.map((m) => m.id);
  // 1) move the Root center into the abdomen -> the adrenal region mismatch must disappear
  const d1 = dataset(); d1.centerRegion = { ...d1.centerRegion, Root: 'abdomen' };
  const m1 = computeMismatches(d1).mismatches.map((m) => m.id);
  assert.ok(base.includes('mm.region.Root.adrenal')); assert.ok(!m1.includes('mm.region.Root.adrenal'));
  // 2) pretend the genetic code had a single start codon -> the "only start codon" partial flag must disappear
  const d2 = dataset(); d2.codons = d2.codons.map((c) => ({ ...c, possibleStart: c.dna === 'ATG' }));
  assert.ok(!computeMismatches(d2).mismatches.some((m) => m.id === 'mm.hd.rave.41.start'));
  // 3) give one Gene Keys gate a different amino acid -> a degeneracy mismatch must appear
  const d3 = dataset(); d3.hdGates = d3.hdGates.map((g) => g.gate === 1 ? { ...g, aminoAcid: 'W' } : g);
  const m3 = computeMismatches(d3).mismatches.map((m) => m.id);
  assert.ok(m3.includes('mm.degeneracy.genekeys.W') && m3.includes('mm.degeneracy.genekeys.K'));
  // 4) drop the project note -> the rule conflict disappears
  const d4 = dataset(); d4.projectNotes = [];
  assert.ok(!computeMismatches(d4).mismatches.some((m) => m.kind === 'rule-conflict'));
});
test('Gene Keys rings agree with codon counts (all 21 groups) and HD start/stop gates are 41 and 12/33/56', () => {
  const checks = consistencyChecks().filter((c) => c.text.startsWith('Gene Keys codon rings'));
  assert.equal(checks.length, 21);
  const d = dataset();
  assert.deepEqual(d.hdGates.filter((g) => g.aminoAcid === 'M').map((g) => g.gate), [41]);
  assert.deepEqual(d.hdGates.filter((g) => g.aminoAcid === '*').map((g) => g.gate), [12, 33, 56]);
});
