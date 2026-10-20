// PENDING SOURCE: pasted pages 444-464 of a Zhouyi/Yixue book with "Wai Guang" segments, cited as 'Hatcher(?)' (Bradford Hatcher,
// The Book of Changes / Yijing — NOT confirmed; no copy of the book was found in Drive). Alexis's paste is message t431u; its verbatim text
// was NOT available to this pass (no /workspace/sources/2026-10-20/t431u.txt). Only the structure the dispatch message names is recorded
// here; every table cell that would need the pasted text is null + 'source pending'. Nothing is rendered from this file.
export const HATCHER_Q = Object.freeze({ cite: 'Hatcher(?)', pages: '444-464', paste: 't431u', verbatim: false, status: 'SOURCE PENDING' });
export const SCALES = Object.freeze([1, 2, 3, 4, 5, 6, 8, 12, 16, 28, 32].map((n) => Object.freeze({ n, name: null, page: null, status: 'source pending' })));
export const SCALES_NOTES = Object.freeze(['colour scale six beats five', 'Liu Fu six treasuries = Wu Xing + Grain', 'scales 7 and 9 are absent from the Yi'].map((t) => ({ note: t, status: 'as summarised by the dispatch; verbatim pending' })));
export const ER_YAO = Object.freeze({ 0: { name: 'Rou', keyWords: null }, 1: { name: 'Gang', keyWords: null }, waiGuang: null, status: 'source pending' });
export const SI_XIANG = Object.freeze({
  '00': { element: 'Water', xiang: 'Tai Yin' }, '01': { element: 'Wood', xiang: 'Shao Yang' }, '10': { element: 'Metal', xiang: 'Shao Yin' }, '11': { element: 'Fire', xiang: 'Tai Yang' },
  otherRows: null, dimensions: ['Depth', 'Length', 'Breadth', 'Time'], dimensionAssignment: null, status: 'element/xiang names as given in the dispatch; all other rows source pending' });
export const BA_GUA = Object.freeze({ Kun: { bits: '000', rows: null }, Gen: { bits: '001', rows: null }, others: 'source pending (text cut off after Gen)' });
// Conflict to check once verbatim: Si Xiang Metal vs Govinda p.57 "Iron"; Hatcher(?) Wood/Shao Yang etc. vs any element derivation in govinda.mjs.
export const LINE_TRIPLICITY = Object.freeze({ bottom: 'beginning', middle: 'while going', top: 'after going', candidateFor: '3-2-1 pack', status: 'INVESTIGATING' });
