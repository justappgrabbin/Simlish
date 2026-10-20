import test from 'node:test';import assert from 'node:assert/strict';
import {divides,yarrow} from './morph-48-49-probe.mjs';
test('48 divides the wheel (1,440), arc-minutes, colors, tones, lines; 49 divides none',()=>{assert.equal(divides(48)['69,120 slices'],1440);assert.ok(Object.values(divides(49)).every(v=>v==='no'))});
test('neither divides 1,080, 64, 12 or 120',()=>{for(const n of[48,49])for(const k of['1,080 slices/gate','64 gates','12 caps','120 dodeca symmetries'])assert.equal(divides(n)[k],'no')});
test('49 stalks reproduce the book odds ≈1:5:7:3 /16; 48 does not',()=>{const y=yarrow(49),z=yarrow(48);assert.ok(Math.abs(y['9']*16-3)<0.2&&Math.abs(y['6']*16-1)<0.1);assert.ok(Math.abs(z['9']*16-3)>0.8)});
