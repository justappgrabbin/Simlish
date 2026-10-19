import test from 'node:test';
import assert from 'node:assert/strict';

test('model and view import with no DOM present', async () => {
  assert.equal(typeof globalThis.document, 'undefined');
  assert.equal(typeof globalThis.window, 'undefined');
  const m = await import('../src/index.mjs');
  const v = await import('../src/view.mjs');
  assert.equal(typeof m.describeGate, 'function');
  assert.equal(typeof v.mount, 'function');
  assert.match(v.factHTML(m.factById('sci.thyroid.body')), /SCIENCE.*osx-17\.4/);
});
test('index.html references only local files (works offline)', async () => {
  const { readFile } = await import('node:fs/promises');
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  assert.doesNotMatch(html, /(src|href)="https?:/);
  for (const f of ['view.mjs', 'index.mjs', 'mismatches.mjs']) {
    const js = await readFile(new URL(`../src/${f}`, import.meta.url), 'utf8');
    assert.doesNotMatch(js, /\bfetch\(|import\s+[^'"]*['"]https?:/);
  }
});
