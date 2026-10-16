import test from 'node:test';
import assert from 'node:assert/strict';
import { createMorphSearch, solveMorph, solveMotionMorph, solveWorldMorph } from '../runtime/morph-search.mjs';
function seeded() { let seed = 19; return () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296); }
test('ABC PSO GA improve bounded candidates and retain the best across rounds', () => {
  const search = createMorphSearch({ bounds: [[-10, 10], [-10, 10]], initial: [8, -8], random: seeded(),
    evaluate: ([x, y]) => (x - 2) ** 2 + (y + 3) ** 2 });
  let last = search.snapshot().score;
  for (let i = 0; i < 12; i++) {
    const result = search.step();
    assert.ok(result.score <= last); last = result.score;
    assert.ok(result.position.every(value => value >= -10 && value <= 10));
    assert.equal(result.algorithms.ABC, i + 1);
    assert.equal(result.algorithms.PSO, i + 1);
    assert.equal(result.algorithms.GA, i + 1);
  }
  assert.ok(last < .02);
});
test('action changes optimize actual animation controls across five fields', () => {
  const walk = solveMotionMorph('walk', [5, 1, .45], seeded());
  const sit = solveMotionMorph('sit', walk.position, seeded());
  assert.ok(walk.position[0] > 10);
  assert.ok(sit.position[0] < 7);
  assert.ok(sit.position[1] < .8);
  assert.equal(sit.fields.length, 5);
  assert.ok(sit.evaluations <= 400);
});
test('invalid search controls and nonfinite fitness fail explicitly', () => {
  assert.throws(() => solveMorph({ bounds: [[0, 1]], evaluate: () => NaN }));
  assert.throws(() => solveMorph({ bounds: [[1, 0]], evaluate: () => 0 }));
  assert.throws(() => solveMorph({ bounds: [[0, 1]], evaluate: () => 0 }, 100));
});

test('world layout changes geometry to separate overlapping swarm structures without editing addresses', () => {
  const pieces = [1, 2, 3].map(id => ({ id: String(id), address: { gate: 12, line: 3, arc: 1000 } }));
  const before = JSON.stringify(pieces);
  const result = solveWorldMorph(pieces, seeded());
  const distances = result.anchors.flatMap((a, i) => result.anchors.slice(i + 1).map(b => Math.hypot(a.position[0] - b.position[0], a.position[2] - b.position[2])));
  assert.ok(distances.every(distance => distance > 3));
  assert.equal(JSON.stringify(pieces), before);
  assert.ok(result.algorithms.ABC && result.algorithms.PSO && result.algorithms.GA);
});
