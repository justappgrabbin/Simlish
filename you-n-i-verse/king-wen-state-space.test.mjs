import test from 'node:test';
import assert from 'node:assert/strict';
import { gateToLines } from './vendor/klein/king-wen.mjs';
import { gateState, canonicalAddress, normalizeAddress } from './vendor/klein/address-space.mjs';
import { kingWenGateForBinaryValue } from './vendor/klein/gate-address.mjs';
import { StateSpaceKernel } from './vendor/klein/state-space-kernel.mjs';

const address = gate => ({mode:'macro',gate,line:1,color:1,tone:1,base:1});

test('all 64 King Wen gates map bijectively to six-bit hexagrams', () => {
  const indices = new Set();
  for (let gate=1;gate<=64;gate++) {
    const state=gateState(address(gate));
    const expected=gateToLines(gate).reverse().join('');
    assert.equal(state.bits,expected,`gate ${gate}`);
    assert.equal(state.index,Number.parseInt(expected,2));
    assert.equal(kingWenGateForBinaryValue(state.index),gate);
    indices.add(state.index);
  }
  assert.equal(indices.size,64);
  assert.equal(gateState(address(1)).bits,'111111');
  assert.equal(gateState(address(2)).bits,'000000');
});

test('mounted state-space preserves canonical gate identity', () => {
  const kernel=new StateSpaceKernel();
  assert.equal(kernel.states.size,64);
  for (let gate=1;gate<=64;gate++) {
    const a=address(gate);
    const state=kernel.get(a);
    assert.equal(state.addressKey,canonicalAddress(a));
    assert.equal(state.value.metadata.hexagram,gateState(a).bits);
    assert.deepEqual(state.value.vector.slice(0,6),[...gateState(a).bits].map(Number));
    assert.deepEqual(normalizeAddress(state.address),normalizeAddress(a));
  }
});

test('invalid gate addresses fail closed', () => {
  for (const gate of [0,65,-1,1.5]) assert.throws(()=>gateState(address(gate)));
});
