import test from 'node:test';import assert from 'node:assert/strict';
import * as A from './arc-engine.mjs';
test('her degree constants and the 69,120 lattice',()=>{assert.equal(A.DEGREES_PER_GATE,5.625);assert.equal(A.DEGREES_PER_LINE,0.9375);assert.equal(A.DEGREES_PER_COLOR,0.15625);assert.ok(Math.abs(A.DEGREES_PER_TONE-0.0260416667)<1e-9);assert.ok(Math.abs(A.DEGREES_PER_BASE-0.00520833333)<1e-9);assert.equal(A.LATTICE_SIZE,69120)});
test('the wheel table equals the numbers printed in sentence-engine.ts (spot checks) and matches the Black Book gate-1 start 13°15\'',()=>{
 const T={41:272,19:277.625,3:356.375,27:2,24:7.625,1:193.25,60:266.375,22:317,44:187.625};for(const [g,d] of Object.entries(T))assert.equal(A.GATE_START_DEGREES[g],d,'gate '+g);
 assert.equal(new Set(Object.values(A.GATE_START_DEGREES)).size,64)});
test('longitude -> gate/line/color/tone/base round-trips at the start of every gate and line',()=>{
 for(let g=1;g<=64;g++)for(let l=1;l<=6;l++){const lon=A.gateLineToLongitude(g,l)+1e-6;assert.equal(A.longitudeToGate(lon),g);assert.equal(A.longitudeToLine(lon),l);assert.equal(A.longitudeToColor(lon),1)}});
test('every longitude lands in exactly one gate and the five sub-values stay in range (10,000 samples)',()=>{let s=7;const r=()=>(s=(s*16807)%2147483647)/2147483647;
 for(let i=0;i<10000;i++){const lon=r()*360,a=A.addressFromLongitude(lon);assert.ok(a.gate>=1&&a.gate<=64&&a.line>=1&&a.line<=6&&a.color>=1&&a.color<=6&&a.tone>=1&&a.tone<=6&&a.base>=1&&a.base<=5);assert.ok(a.latticePosition>=1&&a.latticePosition<=69120)}});
test('lattice position is a bijection with her formula, and the inverse restores the address (all 69,120)',()=>{const seen=new Set();
 for(let b=1;b<=5;b++)for(let t=1;t<=6;t++)for(let c=1;c<=6;c++)for(let l=1;l<=6;l++)for(let g=1;g<=64;g++){const p=A.calculateLatticePosition(g,l,c,t,b);seen.add(p);if(p%977===0)assert.deepEqual(A.latticePositionToAddress(p),{gate:g,line:l,color:c,tone:t,base:b})}
 assert.equal(seen.size,69120);assert.equal(A.calculateLatticePosition(1,1,1,1,1),1);assert.equal(A.calculateLatticePosition(64,6,6,6,5),69120)});
test('base = tone/5: five bases fill a tone',()=>{const lon=A.GATE_START_DEGREES[1]+1e-9,seen=new Set();for(let i=0;i<5;i++)seen.add(A.longitudeToBase(lon+(i+0.5)*A.DEGREES_PER_BASE));assert.equal(seen.size,5)});
import {readFileSync} from 'node:fs';
test('all 64 start degrees equal the explicit table in her sentence-engine.ts (parsed from the reference copy)',()=>{
 const t=readFileSync(new URL('./reference/her/sentence-engine.ts.txt',import.meta.url),'utf8'),blk=t.slice(t.indexOf('GATE_START_DEGREES: Record'),t.indexOf('HD_WHEEL_GATES: number[]')),m=[...blk.matchAll(/(\d+):\s*([\d.]+)/g)];
 assert.equal(m.length,64);for(const [,g,d] of m)assert.equal(A.GATE_START_DEGREES[g],Number(d),'gate '+g)});
