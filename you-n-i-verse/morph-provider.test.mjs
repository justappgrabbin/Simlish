import test from 'node:test';
import assert from 'node:assert/strict';
import { planCharacterMorph, createSimlishMorphProvider, registerWithComputer } from './morph-provider.mjs';
const img = 'data:image/png;base64,YWJj';
const base = { photo: img, body: img, morphedImage: null, sprites: null, morphedSprites: null };

test('no user photo prompts once; no invented character', () => {
  assert.equal(planCharacterMorph(null).status, 'needs_input');
  assert.equal(planCharacterMorph({ body: img }).reason, 'portrait-missing');
  assert.equal(planCharacterMorph(base, { intent: 'rigged' }).reason, 'prepared-rig-required');
});
test('uses cached personalized results without image work', async () => {
  const provider = createSimlishMorphProvider({ decodeImage: () => { throw Error('should not decode cached'); } });
  const p = { ...base, morphedImage: img };
  const result = await provider.prepare({ project: p, intent: 'portrait' });
  assert.equal(result.status, 'ready');
  assert.equal(result.output.dataUrl, img);
  assert.equal(result.receipt.reused, true);
});
test('requires consent before reading stored portraits', async () => {
  let reads = 0;
  const provider = createSimlishMorphProvider({ loadProject: async () => { reads++; return base; } });
  assert.equal((await provider.prepare({ intent: 'auto' })).status, 'needs_consent');
  assert.equal(reads, 0);
});
test('runs the existing portrait warp when no completed image is saved', async () => {
  const provider = createSimlishMorphProvider({
    decodeImage: async () => ({ width: 10, height: 12 }),
    warp: async () => ({ canvas: { toDataURL: () => img }, receipt: { method: 'existing-triangle-warp' } }),
  });
  const result = await provider.prepare({ project: base, intent: 'portrait' });
  assert.equal(result.status, 'ready');
  assert.equal(result.receipt.engine, 'existing-triangle-warp');
  assert.equal(result.output.dataUrl, img);
});
test('animated path uses existing pose personalize and interpolation', async () => {
  const provider = createSimlishMorphProvider({
    decodeImage: async () => ({ width: 40, height: 10 }),
    personalize: async () => ({ canvas: { width: 40, height: 10 } }),
    interpolate: async () => ({ canvas: { toDataURL: () => img }, grid: { count: 24 }, method: 'existing-interpolator' }),
  });
  const result = await provider.prepare({ project: { ...base, sprites: img, sheet: { cols: 4, rows: 1 } }, intent: 'animated' });
  assert.equal(result.kind, 'animated-2d');
  assert.equal(result.receipt.frames, 24);
  assert.equal(result.receipt.keyframes, 4);
});
test('registers with Computer as a callable service', () => {
  const calls = [];
  const computer = { morph: { registerProvider(...args) { calls.push(args); return args[0]; } } };
  assert.equal(registerWithComputer(computer), 'simlish-local-morph-v1');
  assert.deepEqual(calls[0][2].formats, ['portrait', 'animated']);
});
