// ESTABLISHED SCIENCE: the Standard Genetic Code, NCBI transl_table=1 (source id 'ncbi-gc1').
// Encoded in NCBI's own compact form (base order T,C,A,G for positions 1,2,3), transcribed
// verbatim from the table at the cited URL; every row below expands to the printed table.
export const NCBI_TABLE_1 = Object.freeze({
  source: 'ncbi-gc1',
  base1: 'TTTTTTTTTTTTTTTTCCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGG',
  base2: 'TTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGG',
  base3: 'TCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAG',
  aas:   'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG',
  // 'M' marks codons NCBI lists as possible initiators in table 1 (printed as "i": TTG, CTG, ATG).
  starts:'---M------**--*----M---------------M----------------------------',
});

// IUPAC-IUB JCBN 3AA-1 Table 1 names and symbols (source id 'iupac-aa').
export const AMINO_ACIDS = Object.freeze({
  A: { one: 'A', three: 'Ala', name: 'Alanine' },        R: { one: 'R', three: 'Arg', name: 'Arginine' },
  N: { one: 'N', three: 'Asn', name: 'Asparagine' },     D: { one: 'D', three: 'Asp', name: 'Aspartic acid' },
  C: { one: 'C', three: 'Cys', name: 'Cysteine' },       Q: { one: 'Q', three: 'Gln', name: 'Glutamine' },
  E: { one: 'E', three: 'Glu', name: 'Glutamic acid' },  G: { one: 'G', three: 'Gly', name: 'Glycine' },
  H: { one: 'H', three: 'His', name: 'Histidine' },      I: { one: 'I', three: 'Ile', name: 'Isoleucine' },
  L: { one: 'L', three: 'Leu', name: 'Leucine' },        K: { one: 'K', three: 'Lys', name: 'Lysine' },
  M: { one: 'M', three: 'Met', name: 'Methionine' },     F: { one: 'F', three: 'Phe', name: 'Phenylalanine' },
  P: { one: 'P', three: 'Pro', name: 'Proline' },        S: { one: 'S', three: 'Ser', name: 'Serine' },
  T: { one: 'T', three: 'Thr', name: 'Threonine' },      W: { one: 'W', three: 'Trp', name: 'Tryptophan' },
  Y: { one: 'Y', three: 'Tyr', name: 'Tyrosine' },       V: { one: 'V', three: 'Val', name: 'Valine' },
});
export const STOP = Object.freeze({ one: '*', three: 'Ter', name: 'Stop (termination)' });

/** All 64 codons, DNA (T) and mRNA (U) spellings. */
export const CODONS = Object.freeze(Array.from({ length: 64 }, (_, i) => {
  const t = NCBI_TABLE_1;
  const dna = t.base1[i] + t.base2[i] + t.base3[i];
  const aa = t.aas[i];
  return Object.freeze({
    index: i, dna, rna: dna.replace(/T/g, 'U'), aa,
    aminoAcid: aa === '*' ? STOP : AMINO_ACIDS[aa],
    stop: aa === '*', possibleStart: t.starts[i] === 'M', canonicalStart: dna === 'ATG',
    source: 'ncbi-gc1',
  });
}));
const BY = new Map(CODONS.flatMap((c) => [[c.dna, c], [c.rna, c]]));

/** Translate one codon (DNA or RNA spelling, case-insensitive). */
export function translateCodon(codon) {
  const c = BY.get(String(codon).toUpperCase());
  if (!c) throw new RangeError(`not a codon: ${codon}`);
  return c;
}
/** Codons for a one-letter amino acid code ('*' for stop). */
export function codonsFor(one) { return CODONS.filter((c) => c.aa === one); }

/** Look up an amino acid by name as printed in a source ("ASPARAGINIC ACID", "NONE", "Met", ...). */
export function aminoAcidFromPrinted(printed) {
  if (printed == null) return null;
  const s = String(printed).trim().toUpperCase().replace(/\s*\(.*$/, '');
  if (s === '' ) return null;
  if (['NONE', 'STOP', 'STOP CODON', 'TER', 'NONE (TERMINATOR'].includes(s) || s.startsWith('NONE')) return STOP;
  const alias = { 'ASPARAGINIC ACID': 'ASPARTIC ACID' }[s] ?? s;
  return Object.values(AMINO_ACIDS).find((a) => a.name.toUpperCase() === alias || a.three.toUpperCase() === alias || a.one === alias) ?? null;
}
