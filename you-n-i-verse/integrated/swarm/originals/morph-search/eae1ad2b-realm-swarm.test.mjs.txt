import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveRealmSwarm } from '../runtime/realm-swarm.mjs';
import { expressAddress } from '../runtime/address-expression.mjs';

test('swarm spans five fields and retains weighted observer relationships without rewriting addresses', () => {
  const actors = [1, 32].map((gate, i) => ({ id: `owner${i}`, addresses: [{ id: 'Sun', expression: expressAddress({ planetary: 'Sun', gate, line: 2, color: 3, tone: 4, base: 5 }) }] }));
  const movement = resolveRealmSwarm(actors, 0);
  const being = resolveRealmSwarm(actors, 8);
  assert.deepEqual(Object.keys(being.pieces[0].fields), ['Movement', 'Evolution', 'Being', 'Design', 'Space']);
  assert.deepEqual(movement.pieces, being.pieces);
  assert.equal(being.pieces[0].fields.Space.scale, 'observer');
  assert.equal(being.pieces[0].fields.Space.macro, null);
  assert.equal(being.pieces[0].fields.Evolution.microStates.length, 2);
  assert.equal(being.pieces[0].fields.Evolution.micro.expression.name, 'The Mind');
  assert.equal(being.pieces[0].fields.Movement.macro.chain[0].value, 'Movement is Energy');
  assert.notEqual(being.pieces[0].fields.Movement.qualities.amplitude, being.pieces[0].fields.Evolution.qualities.amplitude);
  assert.ok(being.pieces[0].fields.Design.relationships.some(edge => edge.from.startsWith('owner0:') && edge.to.startsWith('owner1:')));
  assert.equal(being.hexagrams.length, 64);
  assert.deepEqual(being.pieces[0].address, actors[0].addresses[0].expression.address);
  assert.ok(being.edges.some(edge => edge.from.startsWith('owner0:') && edge.to.startsWith('owner1:')));
  assert.equal(being.pieces[0].expression.structure.hexagram, '䷀');
  assert.deepEqual(being.pieces[0].expression.structure.trigrams, [[1,1,1],[1,1,1]]);
  assert.equal(being.mode, 'local-mesh');
});
