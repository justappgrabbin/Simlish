// ESTABLISHED SCIENCE layer (separate from the books): the Standard Genetic Code, NCBI transl_table=1.
// Same compact NCBI form as Trinity's hd-morphology data/genetic-code.mjs (branch feat/hd-morphology-body-layer), base order T,C,A,G.
export const NCBI_TABLE_1 = Object.freeze({
  base1: 'TTTTTTTTTTTTTTTTCCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGG',
  base2: 'TTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGG',
  base3: 'TCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAG',
  aas:   'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG',
});
const IDX = new Map(Array.from({ length: 64 }, (_, i) => [(NCBI_TABLE_1.base1[i] + NCBI_TABLE_1.base2[i] + NCBI_TABLE_1.base3[i]).replace(/T/g, 'U'), i]));
export const standardAminoAcid = (rna) => { const i = IDX.get(String(rna).toUpperCase().replace(/T/g, 'U')); if (i === undefined) throw new RangeError('codon ' + rna); return NCBI_TABLE_1.aas[i]; };
export const STANDARD_STOPS = Object.freeze(['UAA', 'UAG', 'UGA']);
export const STANDARD_START = 'AUG';
