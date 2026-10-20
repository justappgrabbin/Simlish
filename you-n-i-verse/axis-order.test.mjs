import test from 'node:test';import assert from 'node:assert/strict';
import {ORDER,facesForHexagram,buildOrder} from './axis-order.mjs';import {adjacent} from './dodeca-face-axes.mjs';
test('6 lines on 6 distinct axes',()=>{assert.equal(new Set(ORDER.map(o=>o.axis)).size,6)});
test('Qian (all yang) is a cap centred on line 5; Kun is the opposite cap',()=>{const q=facesForHexagram(63),k=facesForHexagram(0);const c=ORDER[4].yangFace;assert.equal(q.filter(f=>adjacent(c,f)).length,5);assert.ok(k.every(f=>!q.includes(f)))});
test('ring lines 1,2,3,4,6 are consecutive neighbours and close',()=>{const r=[1,2,3,4,6].map(l=>ORDER[l-1].yangFace);for(let i=0;i<5;i++)assert.ok(adjacent(r[i],r[(i+1)%5]))});
test('any centre line builds; out of range is refused',()=>{for(let l=1;l<=6;l++)assert.equal(buildOrder(l).length,6);assert.throws(()=>buildOrder(7))});
