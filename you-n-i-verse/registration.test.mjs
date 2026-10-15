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
import './vendor/kernel.js';
const full={planetary:'Sun',dimension:'Being',zodiac:'Virgo',gate:6,line:4,color:4,tone:4,base:2,degree:25,minute:59,second:31,arcSecond:92,house:5};
test('incomplete addresses are kept as investigating with the missing places listed, never dropped',()=>{
 const s=storage();
 const r=registerSimlish(s,{id:'a',version:'1',address:{gate:6}});
 assert.equal(r.status,'investigating');assert.deepEqual(r.address,{gate:6});
 assert.ok(r.investigation.missingPlaces.includes('arcSecond'));assert.ok(r.investigation.missingPlaces.includes('planetary'));
 const r2=registerSimlish(s,{id:'b',version:'1',address:{foo:'bar'}});assert.equal(r2.status,'investigating');assert.deepEqual(readSimlishRegistrations(s).b.address,{foo:'bar'});
 const r3=registerSimlish(s,{id:'c',version:'1'});assert.equal(r3.status,'investigating');
 assert.ok(pendingRegistrations(s).a&&pendingRegistrations(s).c);
});
test('full 13-place address is marked registered; legacy arc alias still works; conflicts are investigated',()=>{
 const s=storage();
 const {arcSecond,...legacy}=full;
 assert.equal(registerSimlish(s,{id:'ok',version:'1',source:'test',address:full}).status,'registered');
 assert.equal(registerSimlish(s,{id:'old',version:'1',source:'test',address:{...legacy,arc:92}}).status,'registered');
 const bad=registerSimlish(s,{id:'bad',version:'1',source:'test',address:{...full,arc:5}});assert.equal(bad.status,'investigating');assert.match(bad.investigation.notes.join(' '),/arc/i);
 const noSrc=registerSimlish(s,{id:'ns',version:'1',address:full});assert.equal(noSrc.status,'investigating');assert.ok(noSrc.investigation.failedChecks.includes('provenance'));
 const wrong=registerSimlish(s,{id:'w',version:'1',source:'t',address:{...full,gate:99}});assert.equal(wrong.status,'investigating');
});
test('registration is a swarm result: packs 3,2,1 and a six-bit hexagram',()=>{
 const r=registerSimlish(storage(),{id:'ok',version:'1',source:'t',address:full});
 assert.deepEqual(r.swarm.genotype.packs,[3,2,1]);assert.equal(r.swarm.genotype.hexagram_code,'111001');assert.equal(r.swarm.genotype.kind,'PSO');assert.equal(r.swarm.phenotype.kind,'GA');assert.equal(r.swarm.phenotype.status,'complete');assert.equal(r.swarm.phenotype.visible.lines.length,6);
});
