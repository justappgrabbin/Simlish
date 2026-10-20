// Builds codon-comparison.json: Schönberger p.64 (Govinda) vs her codon_mapping (LCM-State-space- fill/lcm-20261019 codonMapping.js,
// Tinker's port) vs Trinity's hd-morphology her-sources (Simlish feat/hd-morphology-body-layer). Reports differences; picks no winner.
import { execFileSync } from 'node:child_process'; import { writeFileSync, mkdtempSync } from 'node:fs'; import { createHash } from 'node:crypto'; import { join } from 'node:path'; import { tmpdir } from 'node:os';
import { schonbergerHexagramToCodon, SCHONBERGER_AMINO_P64, PRINTED_TO_ONE } from './schonberger.mjs';
import { standardAminoAcid } from './science.mjs';
const show = (repo, ref) => execFileSync('git', ['-C', repo, 'show', ref], { encoding: 'utf8', maxBuffer: 1 << 26 });
const LCM = ['/workspace/apps/LCM-State-space-', 'fill/lcm-20261019:extracted/session_build-1/codonMapping.js'];
const HDM = ['/workspace/simlish-audit', 'origin/feat/hd-morphology-body-layer:you-n-i-verse/modules/hd-morphology/data/her-sources.mjs'];
const d = mkdtempSync(join(tmpdir(), 'cc-')), sha = (s) => createHash('sha256').update(s).digest('hex');
const lt = show(...LCM), ht = show(...HDM); writeFileSync(join(d, 'l.mjs'), lt); writeFileSync(join(d, 'h.mjs'), ht);
const { CODON_TABLE } = await import(join(d, 'l.mjs')); const HS = (await import(join(d, 'h.mjs'))).default;
const NAME1 = { alanine: 'A', arginine: 'R', asparagine: 'N', 'aspartic acid': 'D', cysteine: 'C', glutamine: 'Q', 'glutamic acid': 'E', glycine: 'G', histidine: 'H', isoleucine: 'I', leucine: 'L', lysine: 'K', methionine: 'M', phenylalanine: 'F', proline: 'P', serine: 'S', threonine: 'T', tryptophan: 'W', tyrosine: 'Y', valine: 'V', terminator: '*', terminators: '*', stop: '*', 'asparaginic acid': 'D' /* Ra's spelling, per codonMapping.js note */, none: '*' /* Gene Keys prints NONE for the terminator gates */ };
const THREE = { ala: 'A', arg: 'R', asn: 'N', asp: 'D', cys: 'C', gln: 'Q', glu: 'E', gly: 'G', his: 'H', ile: 'I', leu: 'L', lys: 'K', met: 'M', phe: 'F', pro: 'P', ser: 'S', thr: 'T', trp: 'W', tyr: 'Y', val: 'V', stop: '*', ter: '*' };
const one = (s) => { if (!s) return null; const k = String(s).toLowerCase().trim(); return NAME1[k] ?? THREE[k] ?? (k.length === 1 ? k.toUpperCase() : 'UNPARSED:' + s); };
const lcmGate = {}; for (const [aa, v] of Object.entries(CODON_TABLE)) for (const g of v.gates || []) (lcmGate[g] ||= []).push(one(aa));
const rows = []; const count = { lcm: 0, geneKeys: 0, donorHdJs: 0, dnaHdCompilerCodon: 0, dnaHdCompilerAA: 0, synthaiRuleCodon: 0 };
for (let g = 1; g <= 64; g++) {
  const codon = schonbergerHexagramToCodon(g), printed = SCHONBERGER_AMINO_P64[codon], s1 = PRINTED_TO_ONE[printed], h = HS.gates.find((x) => x.gate === g) || {};
  const r = { gate: g, schonberger: { codon, aminoPrinted: printed, one: s1 }, standardOfThatCodon: standardAminoAcid(codon),
    lcmCodonMapping: lcmGate[g] || null, geneKeys: one(h.geneKeys?.aminoAcidAsPrinted), donorHdJs: one(h.donorHdJs?.aminoAcid),
    dnaHdCompiler: { codon: h.dnaHdCompiler?.codon ?? null, aa: one(h.dnaHdCompiler?.aminoAcid) }, synthaiRule: { rnaCodon: h.synthaiRule?.rnaCodon ?? null } };
  r.mismatch = {
    lcm: !(r.lcmCodonMapping || []).includes(s1), geneKeys: r.geneKeys !== s1, donorHdJs: r.donorHdJs !== s1,
    dnaHdCompilerCodon: (r.dnaHdCompiler.codon || '').replace(/T/g, 'U') !== codon, dnaHdCompilerAA: r.dnaHdCompiler.aa !== s1, synthaiRuleCodon: (r.synthaiRule.rnaCodon || '') !== codon };
  for (const k in count) if (r.mismatch[k]) count[k]++; rows.push(r);
}
const out = { note: 'Differences only; no source is preferred. Schönberger = Govinda p.64 table with the repo King Wen<->Fu Xi table.', sources: { lcm: { repo: LCM[0], ref: LCM[1], sha256: sha(lt) }, hdMorphology: { repo: HDM[0], ref: HDM[1], sha256: sha(ht) } }, mismatchCounts: count, rows };
writeFileSync(new URL('./codon-comparison.json', import.meta.url), JSON.stringify(out, null, 1) + '\n'); console.log(JSON.stringify(count));
