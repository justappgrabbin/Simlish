import test from 'node:test';import assert from 'node:assert/strict';
import {closures,SECTOR,sectorOf,sectorSpan,FACTS} from './morph-48-layer.mjs';
const c=Object.fromEntries(closures().map(r=>[r.count,r]));
test('48 closes on slices (1,440), tones (288), colors (48), lines (8), 21,600′ (450′ = 7.5°)',()=>{for(const k of['wheel slices (base)','tones','colors','lines','arc-minutes'])assert.ok(c[k].exact,k);assert.equal(SECTOR,1440)});
test('48 fails on 64 gates, 1,080 per gate (22.5), 120 and 60',()=>{for(const k of['gates','slices per gate','dodecahedron symmetries (full)','dodecahedron rotations'])assert.equal(c[k].exact,false,k)});
test('48 sectors tile the wheel; one sector = 1 gate + 2 lines',()=>{assert.equal(SECTOR,1080+2*180);assert.deepEqual(sectorSpan(47).degrees,[352.5,360]);assert.equal(sectorOf(69119),47)});
test('5 cubes × 24 kept symmetries = 120',()=>{assert.ok(FACTS.check)});
