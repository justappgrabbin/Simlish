import test from 'node:test';
import assert from 'node:assert/strict';
import {registerSimlish,readSimlishRegistrations,pendingRegistrations,syncRegistrations,REGISTRY_KEY} from './synthia-registration.mjs';
function storage(){const m=new Map();return {getItem:k=>m.has(k)?m.get(k):null,setItem:(k,v)=>m.set(k,v)}}
test('version updates preserve address and capabilities',()=>{
 const s=storage();
 registerSimlish(s,{id:'simlish',version:'1',address:{gate:6},capabilities:['world']});
 registerSimlish(s,{id:'simlish',version:'2'});
 assert.equal(readSimlishRegistrations(s).simlish.version,'2');
 assert.deepEqual(readSimlishRegistrations(s).simlish.address,{gate:6});
 assert.deepEqual(readSimlishRegistrations(s).simlish.capabilities,['world']);
});
test('unreadable registry cannot be overwritten',()=>{
 const s=storage();s.setItem(REGISTRY_KEY,'corrupted');
 assert.throws(()=>registerSimlish(s,{id:'simlish',version:'1'}),/Unreadable/);
 assert.equal(s.getItem(REGISTRY_KEY),'corrupted');
});
test('pending update retries and clears only after successful acknowledgement',async()=>{
 const s=storage();registerSimlish(s,{id:'simlish',version:'2'});
 let count=0;
 const fetchImpl=async()=>({ok:++count>1,status:count>1?200:503});
 assert.deepEqual((await syncRegistrations(s,{endpoint:'https://example.test/register',fetchImpl})).map(x=>x.ok),[false]);
 assert.ok(pendingRegistrations(s).simlish);
 assert.deepEqual((await syncRegistrations(s,{endpoint:'https://example.test/register',fetchImpl})).map(x=>x.ok),[true]);
 assert.deepEqual(pendingRegistrations(s),{});
});
