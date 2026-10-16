import test from 'node:test';
import assert from 'node:assert/strict';
import { gateToLines } from './vendor/klein/king-wen.mjs';
import { gateState, canonicalAddress, normalizeAddress, macroProjection, AddressSpace, transition } from './vendor/klein/address-space.mjs';
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

test('macro projection removes all micro-only coordinates', () => {
  const micro = { ...address(6), mode:'micro', planetary:'Sun', dimension:'movement', degree:25, minute:59, second:31, arc:99, zodiac:6, house:5, scope:'test', originContext:'source' };
  const macro = macroProjection(micro);
  assert.equal(macro.mode,'macro');
  for (const field of ['planetary','dimension','degree','minute','second','arc','zodiac','house','scope','originContext']) assert.equal(macro[field],undefined);
  assert.equal(macro.gate,6);
  assert.equal(gateState(macro).bits,gateState(address(6)).bits);
});

test('address-space storage never trusts a supplied forged key', () => {
  const space = new AddressSpace();
  const a = address(6);
  const state = space.put({address:a,addressKey:'macro:gate=1',value:{ok:true}});
  assert.equal(state.addressKey,canonicalAddress(a));
  assert.equal(space.resolve(a),state);
  assert.equal(space.resolve(address(1)),null);
});

test('transitions canonicalize both endpoints even when supplied keys are stale', () => {
  const source={address:address(6),addressKey:'forged-source'};
  const target={address:address(59),addressKey:'forged-target'};
  const edge=transition(source,target,'transmit');
  assert.equal(edge.source,canonicalAddress(source.address));
  assert.equal(edge.target,canonicalAddress(target.address));
  const space=new AddressSpace();
  space.put(source); space.put(target);
  space.connect(edge);
  assert.equal(space.reachable(source.address,target.address),true);
});

test('micro-address resolves without optional provenance metadata', () => {
  const micro={...address(6),mode:'micro',planetary:'Sun',dimension:'movement',degree:25,minute:59,second:31,arc:99,zodiac:6,house:5};
  const resolved=normalizeAddress(micro);
  assert.equal(resolved.unresolved.length,0);
  assert.equal(resolved.gate,6);
  assert.equal(resolved.scope,undefined);
  assert.equal(resolved.originContext,undefined);
});

test('arc-second coordinate rejects values outside 0 through 99', () => {
  const micro={...address(6),mode:'micro',planetary:'Sun',dimension:'movement',degree:25,minute:59,second:31,arc:99,zodiac:6,house:5};
  assert.equal(normalizeAddress(micro).arc,99);
  assert.throws(()=>normalizeAddress({...micro,arc:100}));
  assert.throws(()=>normalizeAddress({...micro,arc:-1}));
});

test('unsupported address modes cannot masquerade as macro states', () => {
  assert.throws(()=>normalizeAddress({...address(6),mode:'other'}),error=>error.code==='INVALID_MODE');
  assert.throws(()=>canonicalAddress({...address(6),mode:'other'}),error=>error.code==='INVALID_MODE');
});

test('identity metadata is typed and canonical serialization escapes delimiters', () => {
  const a={...address(6),scope:'user|scope',originContext:'a=b'};
  const key=canonicalAddress(a);
  assert.match(key,/scope=user%7Cscope/);
  assert.match(key,/originContext=a%3Db/);
  assert.throws(()=>normalizeAddress({...a,scope:{value:'bad'}}));
  assert.throws(()=>normalizeAddress({...a,originContext:' '}));
});

test('address input must be an object rather than an array or primitive', () => {
  assert.throws(()=>normalizeAddress('gate=6'));
  assert.throws(()=>normalizeAddress([6,4,4,4,2]));
  assert.throws(()=>normalizeAddress(42));
  assert.throws(()=>normalizeAddress(0));
  assert.throws(()=>normalizeAddress(false));
  assert.throws(()=>normalizeAddress(''));
});

test('canonical keys reject incomplete addresses', () => {
  assert.throws(()=>canonicalAddress({mode:'macro',gate:6}));
  assert.throws(()=>canonicalAddress({mode:'micro',gate:6,line:4,color:4,tone:4,base:2}));
});

import { addressKey as coreAddressKey, normalizeMacroAddress, normalizeMicroAddress } from './vendor/klein/index.mjs';

test('core and state-space serializers agree on macro and micro addresses', () => {
  const macro=normalizeMacroAddress(address(6));
  assert.equal(coreAddressKey(macro),canonicalAddress(macro));
  const micro=normalizeMicroAddress({...address(6),planetary:'Sun',dimension:'movement',degree:25,minute:59,second:31,arc:99,zodiac:6,house:5});
  assert.equal(coreAddressKey(micro),canonicalAddress(micro));
});

test('core serializer rejects unsupported address modes', () => {
  assert.throws(()=>coreAddressKey({...address(6),mode:'unsupported'}));
});
