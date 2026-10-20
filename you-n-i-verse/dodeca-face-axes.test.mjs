import test from 'node:test';import assert from 'node:assert/strict';
import {AXES,analyse,kingWenPairs,axisAngles} from './dodeca-face-axes.mjs';
test('12 faces = 6 opposite axes; every axis pair at 63.4349°',()=>{assert.equal(AXES.length,6);assert.deepEqual(axisAngles(),['63.4349'])});
test('64 lit sets; complement hexagram = antipodal face set in all 64',()=>{const a=analyse();assert.equal(a.sets,64);assert.equal(a.complementIsAntipode,64)});
test('12 caps, 52 connected, shape classes 12/20/20/12, complement always same shape',()=>{const a=analyse();assert.equal(a.caps,12);assert.equal(a.connected,52);assert.deepEqual(Object.values(a.shapes).sort((x,y)=>x-y),[12,12,20,20]);assert.equal(a.complementSameShape,64)});
test('King Wen pairs: 24 inversion-only, 4 complement-only (1-2,27-28,29-30,61-62), 4 both',()=>{const k=kingWenPairs();assert.equal(k.inversionOnly,24);assert.equal(k.complementOnly,4);assert.equal(k.both,4);assert.equal(k.none,0)});
