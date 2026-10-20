// SOURCED: Govinda, The Inner Structure of the I Ching, p. 64 (PDF p. 84): "THE GENETIC CODE REPRESENTED BY HEXAGRAMS /
// THE TWO CODES ACCORDING TO THE BINARY ORDER, According to Martin Schönberger". Ported from the page image.
// Printed numbers: decimal = b1 + 4*b2 + 16*b3 with U=0, C=1, G=2, A=3 (e.g. UUG 32, AUG 35, UAA 60). The printed 6-digit strings
// put the top line first (UUG 32 = 100000, top line whole), i.e. the decimal is the Fu Xi decimal = repo macro (line 1 = bit 0).
import { FUXI_DECIMAL_TO_KING_WEN as FX2KW, KING_WEN_TO_FUXI_DECIMAL as KW2FX } from '../../vendor/kingwen.mjs';
const BASE = { U: 0, C: 1, G: 2, A: 3 };
export const schonbergerDecimal = (rna) => { const s = String(rna).toUpperCase().replace(/T/g, 'U'); if (!/^[UCGA]{3}$/.test(s)) throw new RangeError('codon ' + rna); return BASE[s[0]] + 4 * BASE[s[1]] + 16 * BASE[s[2]]; };
export const schonbergerCodonToHexagram = (rna) => { const macro = schonbergerDecimal(rna); return { macro, kingWen: FX2KW[macro] }; };
export const schonbergerHexagramToCodon = (kingWen) => { const m = KW2FX[kingWen]; const B = 'UCGA'; return B[m & 3] + B[(m >> 2) & 3] + B[(m >> 4) & 3]; };
// Amino-acid brackets AS PRINTED (abbreviations of the page legend). Two differ from the standard code: AUA is inside the "Met" bracket
// (standard: Ile) and UGA is printed "Stop?". UAG is printed "Amber", UAA "Ochre".
const P = { UUU: 'Phe', UUC: 'Phe', UUG: 'Leu', UUA: 'Leu', UCU: 'Ser', UCC: 'Ser', UCG: 'Ser', UCA: 'Ser', UGU: 'Cys', UGC: 'Cys', UGG: 'Tryp', UGA: 'Stop?', UAU: 'Tyr', UAC: 'Tyr', UAG: 'Amber', UAA: 'Ochre',
  CUU: 'Leu', CUC: 'Leu', CUG: 'Leu', CUA: 'Leu', CCU: 'Pro', CCC: 'Pro', CCG: 'Pro', CCA: 'Pro', CGU: 'Arg', CGC: 'Arg', CGG: 'Arg', CGA: 'Arg', CAU: 'His', CAC: 'His', CAG: 'GluN', CAA: 'GluN',
  GUU: 'Val', GUC: 'Val', GUG: 'Val', GUA: 'Val', GCU: 'Ala', GCC: 'Ala', GCG: 'Ala', GCA: 'Ala', GGU: 'Gly', GGC: 'Gly', GGG: 'Gly', GGA: 'Gly', GAU: 'Asp', GAC: 'Asp', GAG: 'Glu', GAA: 'Glu',
  AUU: 'Ileu', AUC: 'Ileu', AUG: 'Met', AUA: 'Met', ACU: 'Thr', ACC: 'Thr', ACG: 'Thr', ACA: 'Thr', AGU: 'Ser', AGC: 'Ser', AGG: 'Arg', AGA: 'Arg', AAU: 'AspN', AAC: 'AspN', AAG: 'Lys', AAA: 'Lys' };
export const SCHONBERGER_AMINO_P64 = Object.freeze(P);
export const PRINTED_TO_ONE = Object.freeze({ Phe: 'F', Leu: 'L', Ser: 'S', Cys: 'C', Tryp: 'W', Tyr: 'Y', Pro: 'P', Arg: 'R', His: 'H', GluN: 'Q', Val: 'V', Ala: 'A', Gly: 'G', Asp: 'D', Glu: 'E', Ileu: 'I', Met: 'M', Thr: 'T', AspN: 'N', Lys: 'K', Amber: '*', Ochre: '*', 'Stop?': '*' });
// p. 65 text, verbatim: "the two stop-codons UAA and UAG of the genetic code mean in the language of the I Ching: RETREAT (Chapter 33)
// and ABSTRACTION (Chapter 12), whereas the starting signal of the genetic code UAG means the wanderer (Chapter 56)".
export const P65_CLAIM = Object.freeze({ stops: [{ codon: 'UAA', hexagram: 33 }, { codon: 'UAG', hexagram: 12 }], start: { codon: 'UAG', hexagram: 56 },
  note: 'LIKELY SOURCE TYPO: the real start codon is AUG; by the p.64 table UAG -> 56 and UGA -> 12, so "UAG = Abstraction" fits UGA. Text kept as printed.' });
