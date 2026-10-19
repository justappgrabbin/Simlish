import test from 'node:test';
import assert from 'node:assert/strict';
import { FACTS, SOURCE_BY_ID, LAYERS, centers, gates, describeGate, describeCenter, CENTER_IDS, LOCKED_RULES } from '../src/index.mjs';
import { SCIENCE_FACTS, ORGANS } from '../data/science.mjs';
import { HD_FACTS } from '../data/hd.mjs';
import { UNSOURCED_FACTS } from '../data/unsourced.mjs';

test('every science fact cites at least one existing SCIENCE source', () => {
  assert.ok(SCIENCE_FACTS.length > 30);
  for (const f of SCIENCE_FACTS) {
    assert.ok(f.cites.length >= 1, f.id);
    for (const id of f.cites) { assert.ok(SOURCE_BY_ID[id], `${f.id} cites missing ${id}`); assert.equal(SOURCE_BY_ID[id].kind, 'science', `${f.id} cites non-science ${id}`); }
  }
});
test('every HD fact cites existing HD or project sources only; unsourced facts cite nothing and give a reason', () => {
  for (const f of HD_FACTS) {
    assert.ok(f.cites.length >= 1, f.id);
    for (const id of f.cites) { assert.ok(SOURCE_BY_ID[id], `${f.id} → ${id}`); assert.ok(LAYERS.hd.sourceKinds.includes(SOURCE_BY_ID[id].kind), `${f.id} → ${id}`); }
    assert.ok(['claim', 'variant'].includes(f.status));
    if (f.status === 'variant') assert.equal(SOURCE_BY_ID[f.cites[0]].kind, 'her-project', f.id);
  }
  for (const f of UNSOURCED_FACTS) { assert.equal(f.cites.length, 0, f.id); assert.ok(f.reason, f.id); }
});
test('no fact sits in two layers; ids are unique; layer field matches its array', () => {
  const ids = FACTS.map((f) => f.id);
  assert.equal(new Set(ids).size, ids.length, 'duplicate fact id');
  const sets = { science: SCIENCE_FACTS, hd: HD_FACTS, unsourced: UNSOURCED_FACTS };
  for (const [layer, arr] of Object.entries(sets)) for (const f of arr) assert.equal(f.layer, layer, f.id);
  const texts = new Map();
  for (const f of FACTS) { const k = f.text.toLowerCase(); assert.ok(!texts.has(k) || texts.get(k) === f.layer, `same text in two layers: ${f.text}`); texts.set(k, f.layer); }
  for (const f of SCIENCE_FACTS) assert.doesNotMatch(f.text, /\bhuman design\b|\bHD\b|\bgates?\b|\bcenters?\b|\bcodon rings?\b/i, `science fact mentions HD: ${f.id}`);
});
test('all 9 centers and all 64 gates have entries; every gate in exactly one center', () => {
  assert.deepEqual(centers().map((c) => c.id), [...CENTER_IDS]);
  assert.equal(gates().length, 64);
  const all = centers().flatMap((c) => c.gates).sort((a, b) => a - b);
  assert.deepEqual(all, Array.from({ length: 64 }, (_, i) => i + 1));
  for (let n = 1; n <= 64; n++) {
    const g = describeGate(n);
    assert.ok(g.keynote && g.center && g.lines.length === 6 && g.aminoAcid, `gate ${n}`);
    assert.ok(g.hd.length >= 5, `gate ${n} hd facts`);
    assert.equal(g.variants.length, 4);
  }
  for (const c of centers()) {
    const d = describeCenter(c.id);
    assert.ok(d.organs.length >= 1 && d.organs.every((o) => ORGANS[o.id]), c.id);
    assert.ok(d.byScale.organ.length >= 1 && d.byScale.cell.length >= 1, c.id);
  }
});
test('locked rule respected: four primitives, Space not one, none assigned to body parts', () => {
  assert.deepEqual([...LOCKED_RULES.primitives], ['Movement', 'Evolution', 'Being', 'Design']);
  assert.equal(LOCKED_RULES.spaceIsPrimitive, false);
  for (const f of FACTS) assert.doesNotMatch(JSON.stringify(f.data ?? {}), /"primitive"/);
});
