// Browser view. Importing this file touches no DOM; call mount(document).
import * as M from './index.mjs';

const SCALE_LABEL = { body: 'Body', organ: 'Organ / gland', cell: 'Cell', dna: 'DNA', codon: 'Codon', 'amino-acid': 'Amino acid' };
const CENTER_COLOR = { Head: '#e8d44d', Ajna: '#6cbf6c', Throat: '#b0835a', G: '#e8d44d', Heart: '#d65a5a', SolarPlexus: '#b0835a', Spleen: '#b0835a', Sacral: '#d65a5a', Root: '#b0835a' };
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const LAYER_BADGE = (f) => f.layer === 'science' ? '<span class="badge layer-science">SCIENCE</span>'
  : f.layer === 'unsourced' ? '<span class="badge layer-unsourced">UNSOURCED</span>'
  : f.status === 'variant' ? '<span class="badge layer-hd variant">PROJECT</span>' : '<span class="badge layer-hd">HD</span>';
const cites = (f) => f.cites.map((id) => `<a class="cite" href="#src-${esc(id)}">[${esc(id)}${f.locator ? ' ' + esc(f.locator) : ''}]</a>`).join('');
export const factHTML = (f) => `<li class="fact layer-${f.layer}" data-fact="${esc(f.id)}">${LAYER_BADGE(f)} ${esc(f.text)}${cites(f)}${f.reason ? ` <span class="status">why: ${esc(f.reason)}</span>` : ''}</li>`;
const list = (fs, empty = 'Nothing sourced at this scale.') => fs.length ? `<ul class="facts">${fs.map(factHTML).join('')}</ul>` : `<p class="hint">${esc(empty)}</p>`;
const hexHTML = (bits) => `<span class="hex" aria-label="hexagram lines bottom-first ${bits}">${[...bits].map((b) => `<i class="${b === '1' ? 'yang' : 'yin'}"></i>`).join('')}</span>`;
const kindBadge = (k) => `<span class="badge kind ${k}">${esc(k.replace('-', ' ').toUpperCase())}</span>`;

const SHAPES = {
  'tri-up': 'M0,-16 L16,12 L-16,12 Z', 'tri-down': 'M-16,-12 L16,-12 L0,16 Z', square: 'M-15,-15 H15 V15 H-15 Z', diamond: 'M0,-18 L18,0 L0,18 L-18,0 Z',
  'tri-small': 'M0,-10 L11,8 L-11,8 Z', 'tri-right': 'M16,0 L-12,-16 L-12,16 Z', 'tri-left': 'M-16,0 L12,-16 L12,16 Z',
};
const SILHOUETTE = 'M150,18 C172,18 186,36 186,62 C186,88 172,104 162,110 L162,130 C190,136 222,146 236,170 L262,300 C266,318 252,322 246,306 L222,206 L214,330 C218,380 222,420 218,450 L206,610 C204,628 182,628 182,610 L170,460 L150,460 L130,460 L118,610 C118,628 96,628 94,610 L82,450 C78,420 82,380 86,330 L78,206 L54,306 C48,322 34,318 38,300 L64,170 C78,146 110,136 138,130 L138,110 C128,104 114,88 114,62 C114,36 128,18 150,18 Z';

export function mount(doc) {
  const $ = (id) => doc.getElementById(id);
  const svg = $('body'), NS = 'http://www.w3.org/2000/svg';
  const state = { center: null, organ: null, scale: 'body', gate: null };
  const reduce = doc.defaultView?.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  const el = (tag, attrs = {}, parent = svg) => { const e = doc.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); parent.appendChild(e); return e; };
  const activate = (node, fn) => { node.addEventListener('click', fn); node.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn(); } }); };

  // ---- body map
  el('path', { d: SILHOUETTE, class: 'silhouette', 'aria-hidden': 'true' });
  for (const c of M.centers()) {
    const p = M.LAYOUT.centers[c.id];
    const g = el('g', { transform: `translate(${p.x},${p.y})`, tabindex: '0', role: 'button', 'data-center': c.id,
      'aria-label': `${c.label} center, ${c.gates.length} gates. HD links it to ${c.organs.map((o) => M.organ(o).name).join(' and ')}.` });
    el('path', { d: SHAPES[p.shape], class: 'center-shape' }, g);
    const t = el('text', { class: 'label', 'text-anchor': 'middle', y: '4' }, g); t.textContent = c.id === 'SolarPlexus' ? 'SP' : c.id;
    activate(g, () => select({ center: c.id, organ: null, scale: 'organ' }));
  }
  for (const [id, p] of Object.entries(M.LAYOUT.organs)) {
    const o = M.organ(id);
    const g = el('g', { tabindex: '0', role: 'button', 'data-organ': id, 'aria-label': `${o.name} (science organ marker)` });
    el('circle', { cx: p.x, cy: p.y, r: 4.5, class: 'organ-dot' }, g);
    const t = el('text', { class: 'olabel', x: p.x + 7, y: p.y + 2.5 }, g); t.textContent = o.name.split(' ')[0];
    activate(g, () => { const c = M.centers().find((x) => x.organs.includes(id)); select({ center: c?.id ?? null, organ: id, scale: 'organ' }); });
  }

  let vb = [0, 0, 300, 640];
  function zoomTo(target) {
    const from = vb.slice();
    if (reduce || !doc.defaultView?.requestAnimationFrame) { vb = target; svg.setAttribute('viewBox', vb.join(' ')); return; }
    const t0 = performance.now();
    const step = (t) => { const k = Math.min(1, (t - t0) / 350); vb = from.map((v, i) => v + (target[i] - v) * k); svg.setAttribute('viewBox', vb.join(' ')); if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  }

  // ---- scale stepper
  const scalesOl = $('scales');
  for (const s of M.SCALES) {
    const li = doc.createElement('li'); const b = doc.createElement('button'); b.type = 'button'; b.textContent = SCALE_LABEL[s]; b.dataset.scale = s;
    b.addEventListener('click', () => select({ scale: s })); li.appendChild(b); scalesOl.appendChild(li);
  }
  doc.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    const i = M.SCALES.indexOf(state.scale);
    if (i > 0) select({ scale: M.SCALES[i - 1] }); else select({ center: null, organ: null, scale: 'body' });
  });

  function select(patch) {
    Object.assign(state, patch);
    for (const n of svg.querySelectorAll('[data-center]')) n.querySelector('path').classList.toggle('sel', n.dataset.center === state.center);
    for (const n of svg.querySelectorAll('[data-organ]')) n.querySelector('circle').classList.toggle('sel', n.dataset.organ === state.organ);
    for (const b of scalesOl.querySelectorAll('button')) { b.toggleAttribute('aria-current', false); if (b.dataset.scale === state.scale) b.setAttribute('aria-current', 'step'); b.disabled = !state.center && b.dataset.scale !== 'body'; }
    const p = state.organ ? M.LAYOUT.organs[state.organ] : state.center ? M.LAYOUT.centers[state.center] : null;
    const z = { body: 1, organ: 2.2, cell: 3, dna: 3.6, codon: 4.2, 'amino-acid': 4.8 }[state.scale];
    zoomTo(p && state.scale !== 'body' ? [p.x - 150 / z, p.y - 320 / z, 300 / z, 640 / z] : [0, 0, 300, 640]);
    renderPanel();
  }

  // ---- panel
  function renderPanel() {
    const d = $('detail');
    if (!state.center) {
      d.innerHTML = `<h2>Whole body</h2><p>Nine Human Design centers (shapes) and thirteen organs/glands named by science (blue dots). Choose one to zoom in.</p>
        ${list(M.FACTS.filter((f) => f.subject === 'scale:body'))}
        <h3>Centers</h3><ul class="facts">${M.centers().map((c) => `<li><span class="badge layer-hd">HD</span> <button type="button" data-go="${c.id}">${esc(c.label)}</button> → ${c.organs.map((o) => esc(M.organ(o).name)).join(', ')} <a class="cite" href="#src-hd-blackbook">[hd-blackbook]</a></li>`).join('')}</ul>`;
      for (const b of d.querySelectorAll('[data-go]')) b.addEventListener('click', () => select({ center: b.dataset.go, scale: 'organ' }));
      return;
    }
    const c = M.describeCenter(state.center);
    const organs = state.organ ? c.organs.filter((o) => o.id === state.organ) : c.organs;
    const organFacts = (scale) => organs.flatMap((o) => M.factsFor(`organ:${o.id}`, { scale }));
    const centerFacts = (scale) => M.factsFor(`center:${c.id}`, { scale });
    const gateFacts = (key) => c.gates.flatMap((g) => M.factsFor(`gate:${g}`).filter((f) => f.id === `hd.gate.${g}.${key}`));
    const mm = M.mismatches().filter((m) => m.subject === `center:${c.id}` || organs.some((o) => m.subject === `organ:${o.id}`));
    let body = '';
    switch (state.scale) {
      case 'body': body = `${list([...centerFacts('body'), ...organFacts('body')])}`; break;
      case 'organ': body = `<h3>What science says</h3>${list(organFacts('organ').concat(organFacts('body')))}
        <h3>What Human Design says</h3>${list(centerFacts('organ'))}<h4>Per-gate biology / physiology</h4>${list([...gateFacts('biology'), ...gateFacts('physiology')])}`; break;
      case 'cell': body = `${schematic('cell')}<h3>Cells of ${organs.map((o) => esc(o.name)).join(', ')}</h3>${list(organFacts('cell'), 'No cell-level fact sourced for this organ yet.')}${list(M.factsFor('scale:cell'))}${list(M.factsFor('scale:cell', { layer: 'unsourced' }))}`; break;
      case 'dna': body = `${schematic('dna')}${list(M.factsFor('scale:dna'))}<h3>Gates of this center as hexagrams <span class="badge layer-hd">HD</span></h3>
        <p class="hint">The six lines are I Ching structure. Whether lines correspond to DNA bases is a project model, not science.</p>
        <ul class="facts">${c.gates.map((g) => `<li>${hexHTML(M.gate(g).lines)} ${factHTML(M.factById(`hd.gate.${g}.hexagram`)).replace(/^<li[^>]*>|<\/li>$/g, '')}</li>`).join('')}</ul>`; break;
      case 'codon': body = `${schematic('codon')}${list(M.factsFor('scale:codon'))}<h3>Codons for the amino acids HD gives these gates</h3>
        <table><thead><tr><th>Gates <span class="badge layer-hd">HD</span></th><th>Amino acid <span class="badge layer-hd">HD</span></th><th>Its codons <span class="badge layer-science">SCIENCE</span> <a class="cite" href="#src-ncbi-gc1">[ncbi-gc1]</a></th></tr></thead>
        <tbody>${c.gateAminoAcids.map((a) => `<tr><td>${a.gates.join(', ')}</td><td>${esc(a.name)}</td><td>${a.codons.join(' ')}</td></tr>`).join('')}</tbody></table>
        <h4>Exact triplet per gate</h4>${list(gateFacts('codonTriplet').slice(0, 1))}<h4>Project variants (your tables)</h4>${list(c.gates.flatMap((g) => ['dnahd', 'synthai'].map((k) => M.factById(`hd.gate.${g}.variant.${k}`))))}`; break;
      case 'amino-acid': body = `${schematic('aa')}${list(M.factsFor('scale:amino-acid'))}${list(organFacts('amino-acid'), 'Science names no amino-acid precursor for this organ\u2019s main hormones here.')}
        <h3>Codon rings HD gives these gates</h3>${list(gateFacts('aminoAcid'))}`; break;
    }
    const mmAll = [...mm, ...M.mismatches().filter((m) => m.kind === 'observation' && organs.some((o) => m.subject === `organ:${o.id}`) && !mm.includes(m))];
    d.innerHTML = `<h2>${esc(c.label)} center${state.organ ? ' · ' + esc(M.organ(state.organ).name) : ''} <small>— ${esc(SCALE_LABEL[state.scale])}</small></h2>
      <p class="hint">HD links this center to: ${c.organs.map((o) => `<button type="button" data-organ-btn="${o.id}">${esc(o.name)}</button>`).join(' ')} · gates ${c.gates.join(', ')}</p>
      ${body}${mmAll.length ? `<h3>Flags here</h3>${mmAll.map(mmHTML).join('')}` : ''}`;
    for (const b of d.querySelectorAll('[data-organ-btn]')) b.addEventListener('click', () => select({ organ: b.dataset.organBtn }));
  }
  const schematic = (kind) => {
    const inner = {
      cell: '<circle cx="75" cy="55" r="48" fill="none" stroke="#3fa7d6"/><circle cx="75" cy="55" r="18" fill="#3fa7d633" stroke="#3fa7d6"/><text x="58" y="58">nucleus</text>',
      dna: Array.from({ length: 9 }, (_, i) => `<line x1="${20 + i * 14}" y1="${55 + 30 * Math.sin(i / 1.4)}" x2="${20 + i * 14}" y2="${55 - 30 * Math.sin(i / 1.4)}" stroke="#3fa7d6"/>`).join('') + '<text x="40" y="104">double helix</text>',
      codon: '<rect x="15" y="35" width="38" height="30" fill="#1d2130" stroke="#3fa7d6"/><rect x="56" y="35" width="38" height="30" fill="#1d2130" stroke="#3fa7d6"/><rect x="97" y="35" width="38" height="30" fill="#1d2130" stroke="#3fa7d6"/><text x="30" y="55">A</text><text x="71" y="55">U</text><text x="112" y="55">G</text><text x="35" y="90">3 bases = 1 codon</text>',
      aa: Array.from({ length: 6 }, (_, i) => `<circle cx="${18 + i * 23}" cy="55" r="10" fill="#3fa7d633" stroke="#3fa7d6"/>`).join('') + '<text x="30" y="90">amino-acid chain</text>',
    }[kind];
    return `<svg class="schematic" viewBox="0 0 150 110" role="img" aria-label="schematic, not to scale (unsourced drawing)">${inner}</svg>`;
  };
  const mmHTML = (m) => `<div class="mm">${kindBadge(m.kind)} ${esc(m.text)} ${m.cites.map((id) => `<a class="cite" href="#src-${esc(id)}">[${esc(id)}]</a>`).join('')}${m.dependsOnUnsourced ? ' <span class="badge layer-unsourced">USES UNSOURCED PLACEMENT</span>' : ''}<div class="basis">from: ${esc(m.basis.slice(0, 8).join(', '))}${m.basis.length > 8 ? ' …' : ''}</div></div>`;

  // ---- gates
  const gatesDiv = $('gates');
  for (const g of M.gates()) {
    const b = doc.createElement('button'); b.type = 'button'; b.textContent = g.gate; b.setAttribute('role', 'listitem'); b.setAttribute('aria-pressed', 'false');
    b.setAttribute('aria-label', `Gate ${g.gate}, ${g.keynote}, ${g.center} center`); b.style.setProperty('--c', CENTER_COLOR[g.center]);
    b.addEventListener('click', () => showGate(g.gate)); gatesDiv.appendChild(b);
  }
  function showGate(n) {
    state.gate = n;
    for (const b of gatesDiv.children) b.setAttribute('aria-pressed', String(Number(b.textContent) === n));
    const g = M.describeGate(n);
    $('gate-detail').innerHTML = `<div class="card"><h3>${hexHTML(g.lines)} Gate ${n} · ${esc(g.name)} · ${esc(g.keynote)}</h3>
      <p><button type="button" id="gate-center">Show ${esc(g.centerLabel)} center on the body</button></p>
      <h4>Human Design says <span class="badge layer-hd">HD</span></h4>${list(g.hd)}
      <h4>Science for the amino acid HD names</h4>${g.aminoAcidInfo ? `<ul class="facts"><li><span class="badge layer-science">SCIENCE</span> ${esc(g.aminoAcidInfo.name)} (${esc(g.aminoAcidInfo.three)}) is encoded by ${g.scienceCodonsForClaimedAminoAcid.join(', ')}.<a class="cite" href="#src-ncbi-gc1">[ncbi-gc1]</a><a class="cite" href="#src-iupac-aa">[iupac-aa]</a></li></ul>` : '<p class="hint">UNRESOLVED</p>'}
      <h4>Your project tables <span class="badge layer-hd variant">PROJECT</span></h4>${list(g.variants)}
      ${g.mismatches.length ? `<h4>Flags</h4>${g.mismatches.map(mmHTML).join('')}` : ''}</div>`;
    $('gate-center').addEventListener('click', () => { select({ center: g.center, organ: null, scale: 'organ' }); $('panel').focus(); });
  }

  // ---- mismatches, code table, sources
  const order = ['science-mismatch', 'science-partial', 'rule-conflict', 'source-conflict', 'observation'];
  $('mismatches').innerHTML = order.map((k) => { const ms = M.mismatches().filter((m) => m.kind === k); return ms.length ? `<h3>${kindBadge(k)} ${ms.length}</h3>${ms.map(mmHTML).join('')}` : ''; }).join('');
  $('checks').innerHTML = M.consistencyChecks().map((c) => `<li>${esc(c.text)} ${c.cites.map((id) => `<a class="cite" href="#src-${esc(id)}">[${esc(id)}]</a>`).join('')}</li>`).join('');
  $('code').innerHTML = `<p class="hint">mRNA spelling. Green: AUG start (UUG/CUG can rarely start). Red: stop. <a class="cite" href="#src-ncbi-gc1">[ncbi-gc1]</a> <a class="cite" href="#src-iupac-aa">[iupac-aa]</a></p><div class="code" role="list">${M.CODONS.map((c) => `<div role="listitem" class="${c.stop ? 'stop' : c.canonicalStart ? 'start' : ''}">${c.rna} ${c.stop ? 'Stop' : c.aminoAcid.three}</div>`).join('')}</div>`;
  const kinds = { science: 'Established science', hd: 'Human Design literature', 'her-project': 'Your project tables and code', 'checked-not-used': 'Checked, not used (no body/codon content)' };
  $('sources').innerHTML = Object.entries(kinds).map(([k, label]) => `<h3>${label}</h3><table><tbody>${M.SOURCES.filter((s) => s.kind === k).map((s) => `<tr id="src-${esc(s.id)}"><td><code>${esc(s.id)}</code></td><td>${esc(s.author ?? '')}${s.year ? ' (' + s.year + ')' : ''}. <em>${esc(s.title)}</em>${s.publisher ? '. ' + esc(s.publisher) : ''}. ${s.url ? `<a href="${esc(s.url)}" rel="noopener">${esc(s.url)}</a> (accessed ${esc(s.accessed)})` : ''}${s.localPath ? `<br><small>local: ${esc(s.localPath)}</small>` : ''}${s.drive ? `<br><small>Drive: ${esc(s.drive)}</small>` : ''}${s.repo ? `<br><small>${esc(s.repo)} @ ${esc(s.commit)} ${esc(s.path)}</small>` : ''}${s.note ? `<br><small>${esc(s.note)}</small>` : ''}</td></tr>`).join('')}</tbody></table>`).join('');

  select({});
  return { state, select, showGate };
}
