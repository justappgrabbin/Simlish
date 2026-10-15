import test from 'node:test';
import assert from 'node:assert/strict';
import {stateCapacity,inspectRealityOrder,executeRealityOrder} from './reality-execution.mjs';
test('state space has 2880 primary states and reports capacity',()=>{assert.equal(stateCapacity().primary,2880);assert.equal(stateCapacity().emergentCenterStates,81)});
test('missing scene assets block execution',async()=>{const r=await executeRealityOrder({});assert.equal(r.status,'blocked');assert.ok(r.inspection.missing.length>0)});
test('approved complete order invokes renderer once',async()=>{let count=0;const order={semanticAddress:{leadingDimension:'Design',evidence:{source:'fixture'}},scene:{objects:[{id:'a'}]},assets:[{geometry:{},material:{}}],camera:{},lighting:{},renderer:{render:async()=>{count++;return {output:'fixture.png'}}},quality:{referenceImages:['fixture.png']}};const r=await executeRealityOrder(order,{approve:()=>true});assert.equal(r.status,'rendered-unverified');assert.equal(count,1)});
