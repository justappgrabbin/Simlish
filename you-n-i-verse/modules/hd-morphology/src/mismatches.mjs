// Mismatch generator. Every entry is computed from the data passed in — no prose list is
// hardcoded. Change the data and the list changes (see test/mismatches.test.mjs).
// kind: 'science-mismatch'   HD claim disagrees with established science
//       'science-partial'    HD claim agrees in part
//       'source-conflict'    two of Alexis's sources (or HD texts) disagree with each other
//       'rule-conflict'      a source disagrees with Alexis's locked rules
//       'observation'        a comparison worth seeing that is NOT evidence either way
//   basis: fact ids that produced the entry; `dependsOnUnsourced` is true when an unsourced fill-in was used.

const factId = (gate, key) => `hd.gate.${gate}.${key}`;

export function computeMismatches(d) {
  const out = [];
  const checks = [];
  const push = (e) => out.push(Object.freeze({ ...e, basis: Object.freeze(e.basis), cites: Object.freeze([...new Set(e.cites)]) }));
  const codonsOf = (one) => d.codons.filter((c) => c.aa === one);
  const aaName = (one) => one === '*' ? 'stop' : d.aminoAcids[one]?.name ?? one;

  // 1. HD center → organ: does the organ's anatomical region match where the center is placed?
  for (const c of d.hdCenters) {
    const cr = d.centerRegion[c.id];
    for (const o of c.organs) {
      const org = d.organs[o];
      if (!org || org.region === 'whole-body') continue;
      const ok = org.region === cr;
      const e = { subject: `center:${c.id}`, basis: [`hd.center.${c.id}.biology`, org.regionFact, `uns.center.${c.id}.region`], cites: ['hd-blackbook', ...d.factById(org.regionFact).cites], dependsOnUnsourced: true };
      if (ok) checks.push({ ...e, text: `${c.label} ↔ ${org.name}: both in the ${cr}.` });
      else push({ ...e, id: `mm.region.${c.id}.${o}`, kind: 'science-mismatch',
        text: `HD maps the ${c.label} center to the ${org.name}. Science places the ${org.name} in the ${org.region} (${d.factById(org.regionFact).text}) while the center is placed in the ${cr}.` });
    }
  }

  // 2. Codon rings: does the number of gates given an amino acid equal that amino acid's codon count (NCBI)?
  const ringCheck = (label, source, assign, kindIfBad) => {
    const by = {};
    for (const [gate, one] of assign) if (one) (by[one] ||= []).push(gate);
    const unassigned = assign.filter(([, one]) => !one).map(([g]) => g);
    const bad = [];
    for (const [one, gs] of Object.entries(by)) {
      const n = codonsOf(one).length;
      const e = { subject: `aa:${one}`, basis: gs.map((g) => factId(g, source.fact)), cites: [source.id, 'ncbi-gc1'] };
      if (gs.length === n) checks.push({ ...e, text: `${label}: ${aaName(one)} — ${gs.length} gate(s), ${n} codon(s).` });
      else bad.push({ one, gs, n, e });
    }
    if (bad.length && !source.aggregate) for (const b of bad) push({ ...b.e, id: `mm.degeneracy.${source.key}.${b.one}`, kind: kindIfBad,
      text: `${label} gives ${aaName(b.one)} to ${b.gs.length} gate(s) (${b.gs.join(', ')}), but the standard code has ${b.n} codon(s) for ${aaName(b.one)}.` });
    if (bad.length && source.aggregate) push({ id: `mm.degeneracy.${source.key}`, kind: kindIfBad, subject: 'scale:amino-acid',
      basis: bad.flatMap((b) => b.e.basis), cites: [source.id, 'ncbi-gc1'],
      text: `${label}: for ${bad.length} amino acids the number of gates does not equal the number of codons in the standard code — ` +
        bad.map((b) => `${aaName(b.one)} ${b.gs.length} gates vs ${b.n} codons`).join('; ') + '.' });
    const missing = Object.keys(d.aminoAcids).filter((one) => !by[one]);
    if (missing.length || unassigned.length) push({ id: `mm.coverage.${source.key}`, kind: kindIfBad, subject: 'scale:amino-acid', basis: [], cites: [source.id, 'ncbi-gc1'],
      text: [missing.length ? `${label} assigns no gate to ${missing.map(aaName).join(', ')}` : '', unassigned.length ? `${label} leaves gates ${unassigned.join(', ')} with no amino acid` : ''].filter(Boolean).join('; ') + '.' });
  };
  ringCheck('Gene Keys codon rings', { id: 'hd-gene-keys', fact: 'aminoAcid', key: 'genekeys' }, d.hdGates.map((g) => [g.gate, g.aminoAcid]), 'science-mismatch');
  for (const v of d.variants) ringCheck(v.label, { ...v, fact: `variant.${v.key}`, aggregate: true }, d.hdGates.map((g) => [g.gate, d.factById(factId(g.gate, `variant.${v.key}`)).data.aminoAcid]), 'source-conflict');

  // 3. Rave I'Ching codon statements vs NCBI table 1
  for (const c of d.raveClaims) {
    if (c.claim === 'only-start') {
      const starts = d.codons.filter((x) => x.possibleStart);
      const e = { id: `mm.${c.id}`, subject: `gate:${c.gate}`, basis: [c.id, factId(c.gate, 'aminoAcid'), 'sci.codon.table'], cites: [...c.cites, 'ncbi-gc1'] };
      if (starts.length > 1) push({ ...e, kind: 'science-partial', text: `"${c.quote}": AUG (methionine) is the standard initiation codon, which fits; but "only" is too strong — NCBI table 1 also lists ${starts.filter((x) => !x.canonicalStart).map((x) => x.rna).join(' and ')} as rare starts, and AUG also codes methionine inside proteins.` });
      else checks.push({ ...e, text: c.quote });
    }
    if (c.claim === 'stop') {
      const gaa = d.hdGates.find((g) => g.gate === c.gate)?.aminoAcid;
      const e = { subject: `gate:${c.gate}`, basis: [c.id, factId(c.gate, 'aminoAcid')], cites: [...c.cites, 'hd-gene-keys', 'ncbi-gc1'] };
      if (gaa === '*') checks.push({ ...e, text: `Gate ${c.gate} stop claim agrees with its Gene Keys ring (no amino acid).` });
      else push({ ...e, id: `mm.${c.id}`, kind: 'source-conflict', text: `Rave I'Ching calls gate ${c.gate} a stop codon; Gene Keys gives it ${aaName(gaa)}.` });
    }
  }

  // 4. Project variants: start/stop gates and per-gate amino acid vs HD literature
  const hdStart = d.hdGates.filter((g) => g.aminoAcid === 'M').map((g) => g.gate);
  const hdStop = d.hdGates.filter((g) => g.aminoAcid === '*').map((g) => g.gate);
  for (const v of d.variants) {
    const vf = (g) => d.factById(factId(g, `variant.${v.key}`)).data;
    const differ = d.hdGates.filter((g) => vf(g.gate).aminoAcid && g.aminoAcid && vf(g.gate).aminoAcid !== g.aminoAcid).map((g) => g.gate);
    if (differ.length) push({ id: `mm.variant.${v.key}.aa`, kind: 'source-conflict', subject: 'scale:amino-acid', basis: differ.map((g) => factId(g, `variant.${v.key}`)), cites: [v.id, 'hd-gene-keys'],
      text: `${v.label} disagrees with Gene Keys on the amino acid of ${differ.length} of 64 gates.` });
    const vStart = d.hdGates.filter((g) => vf(g.gate).codon === 'AUG').map((g) => g.gate);
    const vStop = d.hdGates.filter((g) => vf(g.gate).aminoAcid === '*').map((g) => g.gate);
    if (vf(1).codon !== null && (vStart.join() !== hdStart.join() || vStop.join() !== hdStop.join()))
      push({ id: `mm.variant.${v.key}.startstop`, kind: 'source-conflict', subject: 'scale:codon', basis: [...vStart, ...vStop].map((g) => factId(g, `variant.${v.key}`)), cites: [v.id, 'hd-gene-keys', 'hd-rave-iching'],
        text: `${v.label}: start codon AUG falls on gate ${vStart.join(', ') || 'none'} and stops on gates ${vStop.join(', ') || 'none'}; HD literature puts the start on gate ${hdStart.join(', ')} and stops on ${hdStop.join(', ')}.` });
  }

  // 5. Gate → center: Black Book vs her project tables
  const merged = {};
  for (const [c, gs] of Object.entries(d.sourceTables.mergedCenters)) for (const g of gs) (merged[g] ||= []).push(c === 'Solar' ? 'SolarPlexus' : c);
  const donorNorm = (s) => ({ 'Solar Plexus': 'SolarPlexus' })[s] ?? s;
  const unassigned = [];
  for (const g of d.hdGates) {
    const m = merged[g.gate] || [];
    if (!m.length) unassigned.push(g.gate);
    else if (m.length > 1 || m[0] !== g.center) push({ id: `mm.center.merged.${g.gate}`, kind: 'source-conflict', subject: `gate:${g.gate}`, basis: [factId(g.gate, 'center')], cites: ['hd-blackbook', 'synthia-centers'],
      text: `Gate ${g.gate}: Black Book center ${g.center}; centers-channels.js CENTERS lists ${m.join(' + ')}.` });
    const dn = donorNorm(d.sourceTables.donorCenters[g.gate]);
    if (dn && dn !== g.center) push({ id: `mm.center.donor.${g.gate}`, kind: 'source-conflict', subject: `gate:${g.gate}`, basis: [factId(g.gate, 'center')], cites: ['hd-blackbook', 'synthia-hdjs'],
      text: `Gate ${g.gate}: Black Book center ${g.center}; human-design.js GATE_CENTER says ${dn}.` });
  }
  if (unassigned.length) push({ id: 'mm.center.merged.unassigned', kind: 'source-conflict', subject: 'scale:body', basis: unassigned.map((g) => factId(g, 'center')), cites: ['hd-blackbook', 'synthia-centers'],
    text: `centers-channels.js CENTERS leaves ${unassigned.length} gates without a center: ${unassigned.join(', ')}.` });

  // 6. Gene Keys internal: a gate's ring name differs from the ring its printed member list belongs to
  for (const g of d.hdGates) {
    const f = d.factById(factId(g.gate, 'aminoAcid')).data;
    const peers = (f.ringMembers || []).filter((x) => x !== g.gate).map((x) => d.hdGates.find((y) => y.gate === x)?.ring);
    if (peers.length && peers.every((r) => r && r !== g.ring) && new Set(peers).size === 1)
      push({ id: `mm.gk.ringname.${g.gate}`, kind: 'source-conflict', subject: `gate:${g.gate}`, basis: [factId(g.gate, 'aminoAcid')], cites: ['hd-gene-keys'],
        text: `Gene Keys names gate ${g.gate}'s ring "${g.ring}" but lists it with ${f.ringMembers.join(', ')}, which the book calls "${peers[0]}".` });
  }

  // 7. Two HD texts, one gate: Gene Keys physiology organ vs Black Book center organs
  let notComparable = 0;
  for (const g of d.hdGates) {
    const gk = d.factById(factId(g.gate, 'physiology')).data.organs;
    const center = d.hdCenters.find((c) => c.id === g.center);
    if (!gk.length) { notComparable++; continue; }
    if (!gk.some((o) => center.organs.includes(o))) push({ id: `mm.hdtexts.${g.gate}`, kind: 'source-conflict', subject: `gate:${g.gate}`, basis: [factId(g.gate, 'physiology'), `hd.center.${g.center}.biology`], cites: ['hd-gene-keys', 'hd-blackbook'],
      text: `Gate ${g.gate}: Gene Keys physiology ${gk.map((o) => d.organs[o].name).join(', ')}; Black Book puts the gate in ${center.label} (${center.organs.map((o) => d.organs[o].name).join(', ')}).` });
  }
  checks.push({ subject: 'scale:organ', basis: [], cites: ['hd-gene-keys'], text: `${notComparable} Gene Keys physiology entries name body parts outside this module's organ list (e.g. sternum, neocortex) and were not compared.` });

  // 8. Locked rules vs project notes
  for (const n of d.projectNotes) {
    const extra = n.dimensionsListed.filter((x) => !d.lockedRules.primitives.includes(x));
    if (extra.length) push({ id: `mm.rule.${n.id}`, kind: 'rule-conflict', subject: 'scale:amino-acid', basis: [], cites: [n.source],
      text: `${n.text} Your locked rule has four primitives and Space is not one; this module uses that rule's codon arithmetic only and does not use its ${extra.join(', ')} mapping.` });
  }

  // 9. Observation: amino acids science names as hormone precursors vs amino acids HD gives the matching center's gates
  for (const [organ, p] of Object.entries(d.hormonePrecursors)) {
    const center = d.hdCenters.find((c) => c.organs.includes(organ));
    if (!center) continue;
    const gateAas = center.gates.map((g) => d.hdGates.find((x) => x.gate === g).aminoAcid);
    const hit = p.aminoAcids.filter((a) => gateAas.includes(a));
    const elsewhere = d.hdGates.filter((g) => p.aminoAcids.includes(g.aminoAcid)).map((g) => `${g.gate} (${g.center})`);
    push({ id: `obs.precursor.${organ}`, kind: 'observation', subject: `organ:${organ}`, basis: [p.fact, `hd.center.${center.id}.biology`, ...center.gates.map((g) => factId(g, 'aminoAcid'))], cites: [...d.factById(p.fact).cites, 'hd-blackbook', 'hd-gene-keys'],
      text: `${d.organs[organ].name}: its hormones are made from ${p.aminoAcids.map(aaName).join(', ')} (science). HD's ${center.label} gates carry ${[...new Set(gateAas)].map(aaName).join(', ')} — ${hit.length ? 'overlap: ' + hit.map(aaName).join(', ') : 'no overlap'}; ${p.aminoAcids.map(aaName).join(', ')} gate(s): ${elsewhere.join(', ') || 'none'}. A precursor link is chemistry, not a codon link, so this neither supports nor refutes HD.` });
  }
  return { mismatches: Object.freeze(out), checks: Object.freeze(checks.map(Object.freeze)) };
}
