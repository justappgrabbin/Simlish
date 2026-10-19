import test from 'node:test';
import assert from 'node:assert/strict';
import { CODONS, translateCodon, codonsFor, AMINO_ACIDS } from '../data/genetic-code.mjs';

test('standard code has 64 unique codons', () => {
  assert.equal(CODONS.length, 64);
  assert.equal(new Set(CODONS.map((c) => c.dna)).size, 64);
  assert.equal(new Set(CODONS.map((c) => c.rna)).size, 64);
});
test('exactly three stops: TAA, TAG, TGA (UAA, UAG, UGA)', () => {
  assert.deepEqual(CODONS.filter((c) => c.stop).map((c) => c.rna).sort(), ['UAA', 'UAG', 'UGA']);
});
test('61 sense codons cover exactly the 20 amino acids', () => {
  const sense = CODONS.filter((c) => !c.stop);
  assert.equal(sense.length, 61);
  assert.deepEqual([...new Set(sense.map((c) => c.aa))].sort(), Object.keys(AMINO_ACIDS).sort());
});
test('spot checks against the printed NCBI table 1 rows', () => {
  const printed = { TTG: 'Leu', AGA: 'Arg', TGG: 'Trp', ATA: 'Ile', ATG: 'Met', GAT: 'Asp', CAA: 'Gln', AAG: 'Lys', TGT: 'Cys', CCG: 'Pro' };
  for (const [codon, three] of Object.entries(printed)) assert.equal(translateCodon(codon).aminoAcid.three, three, codon);
  assert.deepEqual(CODONS.filter((c) => c.possibleStart).map((c) => c.dna).sort(), ['ATG', 'CTG', 'TTG']);
});
test('degeneracy matches the table (e.g. Leu/Ser/Arg 6, Met/Trp 1)', () => {
  assert.equal(codonsFor('L').length, 6); assert.equal(codonsFor('S').length, 6); assert.equal(codonsFor('R').length, 6);
  assert.equal(codonsFor('M').length, 1); assert.equal(codonsFor('W').length, 1); assert.equal(codonsFor('*').length, 3);
});
