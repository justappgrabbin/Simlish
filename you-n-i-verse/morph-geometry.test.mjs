import test from 'node:test';import assert from 'node:assert/strict';import {morphGeometry,radialScale} from './morph-geometry.mjs';
const source=[{x:0,y:0},{x:1,y:0},{x:0,y:1},{x:.2,y:.4}],target=[{x:10,y:20},{x:30,y:20},{x:10,y:40},{x:22,y:32}],oval=[0,1,2];
test('shape endpoints preserve fitted photo and exact target geometry',()=>{assert.deepEqual(morphGeometry(source,target,oval),target);assert.deepEqual(morphGeometry(source,target,oval,{shape:0})[3],{x:14,y:28});});
test('radial sculpt is finite at center and localized',()=>{const p=radialScale([{x:0,y:0},{x:1,y:0},{x:10,y:0}],{x:0,y:0},1,1);assert.deepEqual(p[0],{x:0,y:0});assert.ok(p[1].x>1);assert.ok(p[2].x-10<p[1].x-1);});
test('invalid controls are rejected before warp',()=>assert.throws(()=>morphGeometry(source,target,oval,{shape:NaN}),/range/));
