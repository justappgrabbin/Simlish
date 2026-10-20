import test from 'node:test';import assert from 'node:assert/strict';
import {probeDualities,dualityOf,ASSIGNMENT} from './morph-duality-layer.mjs';
test('only 5:3 and 3:1 (unequal) and 8:8 (equal) close on all wholes',()=>{const ok=probeDualities().filter(r=>r.closesAll).map(r=>r.scheme);assert.deepEqual(ok,['dodeca:icosa 20:12 = 5:3','yarrow old yang:old yin 3:1','yarrow all yin:all yang 8:8'])});
test('phi is rejected everywhere',()=>{const r=probeDualities().find(r=>r.scheme.startsWith('phi'));assert.equal(r['64 gates'],'no');assert.equal(r['69,120 slices'],'no')});
test('Space is the one, never split',()=>{assert.equal(dualityOf('Space','Lo Shu 9:1').kind,'one');assert.equal(ASSIGNMENT.Space,'one')});
test('a non-closing scheme is refused',()=>{assert.throws(()=>dualityOf('Being','Lo Shu 9:1',64),/rejects/)});
test('closing splits land on whole gates: 40:24 and 48:16',()=>{assert.deepEqual(dualityOf('Movement','dodeca:icosa 20:12 = 5:3',64).halves,[40,24]);assert.deepEqual(dualityOf('Design','yarrow old yang:old yin 3:1').halves.map(x=>x/1080),[48,16])});
