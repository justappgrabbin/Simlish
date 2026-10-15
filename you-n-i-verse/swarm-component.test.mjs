import test from 'node:test';import assert from 'node:assert/strict';
import {SwarmComponent,hexagramFromPacks,piece,packValue,PACK_SHAPE} from './swarm-component.mjs';
const yes=()=>true,no=()=>false;
const mk=(checks,kind='PSO')=>new SwarmComponent({id:'t',kind,members:checks.map((c,i)=>({id:'m'+i,check:c}))});
test('packs are 3,2,1 and three two-bit packs make a six-bit hexagram',async()=>{
 assert.deepEqual(PACK_SHAPE,[3,2,1]);
 const r=await mk([yes,yes,yes,yes,yes,yes]).run({});
 assert.deepEqual(r.packs.map(p=>p.value),[3,2,1]);assert.equal(r.hexagram.bits,'111001');assert.equal(r.status,'complete');
 assert.equal(hexagramFromPacks([0,0,0]).bits,'000000');
 assert.throws(()=>packValue(4));assert.throws(()=>hexagramFromPacks([1,2]));
});
test('failures become investigating with named members, never a throw or drop',async()=>{
 const r=await mk([yes,no,yes,yes,()=>{throw new Error('boom')},yes],'GA').run({});
 assert.equal(r.status,'investigating');assert.deepEqual(r.failed,['m1','m4']);assert.match(r.notes.join(' '),/boom/);assert.equal(r.role,'visible-on-screen');
});
test('pieces build 2, 4 and 8 bit sizes only',()=>{
 assert.equal(piece(2,3).bits,'11');assert.equal(piece(4,9).bits,'1001');assert.equal(piece(8,200).bits,'11001000');
 assert.throws(()=>piece(3,1));assert.throws(()=>piece(2,4));
 assert.throws(()=>new SwarmComponent({id:'x',kind:'XX',members:[]}));
});
