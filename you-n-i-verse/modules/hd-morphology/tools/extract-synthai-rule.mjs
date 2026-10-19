// Read-only: runs Alexis's SynthAi gate->codon translation rule (v0.5.3) and snapshots it.
// Source: justappgrabbin/SynthAi computer/hover/src/state-space/biological-translation-runtime.mjs
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
const repo = '/workspace/SynthAi';
const file = `${repo}/computer/hover/src/state-space/biological-translation-runtime.mjs`;
const m = await import(file);
const commit = execSync('git rev-parse HEAD', { cwd: repo }).toString().trim();
const gates = {};
for (let g = 1; g <= 64; g++) {
  const t = m.translateGate(g);
  gates[g] = { bitsAsUsed: t.bits.join(''), dnaCodon: t.dna.codon, rnaCodon: t.rna.codon, aminoAcidAsComputed: t.aminoAcid?.code ?? t.aminoAcid?.symbol ?? null,
               start: Boolean(t.start), stop: Boolean(t.stop) };
}
const out = { source: { repo: 'justappgrabbin/SynthAi', commit, path: 'computer/hover/src/state-space/biological-translation-runtime.mjs',
  sha256: createHash('sha256').update(readFileSync(file)).digest('hex'), alphabet: m.DNA_DIGRAM_ALPHABET }, gates };
writeFileSync(new URL('./synthai-rule.raw.json', import.meta.url), JSON.stringify(out, null, 1));
console.log('synthai gates', Object.keys(gates).length, 'commit', commit);
