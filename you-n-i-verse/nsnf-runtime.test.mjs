import test from 'node:test';import assert from 'node:assert/strict';
import {mkdtempSync} from 'node:fs';import {tmpdir} from 'node:os';import {join} from 'node:path';
import {NsnfRuntime,TinyMLP,trainWithParticleSwarm,MemoryStore,fileStore,storageStore,DEFAULT_TIERS,hashOf,INVESTIGATING} from './nsnf-runtime.mjs';
import {StateSpace} from './state-space.mjs';
const X=[.1,.9,.3,.5,.7,.2,.8,.4],Y=[.9,.1,.6,.2,.3,.8,.1,.5];
const seeds=[{dimension:'Being',path:[{macro:5,sub:2}],role:'placement'}];
test('component interface: initialize, forward, update, snapshot, restore (Spec §7); same seed gives same weights',()=>{
 const a=new TinyMLP({seed:1}),b=new TinyMLP({seed:1});assert.deepEqual(a.params(),b.params());assert.notDeepEqual(a.params(),new TinyMLP({seed:2}).params());
 const y=[1,0,0,0],l0=a.loss(X,y);for(let i=0;i<50;i++)a.update({input:X,target:y,lr:0.1});assert.ok(a.loss(X,y)<l0*.5);assert.equal(a.version,50);
 const s=a.snapshot(),c=new TinyMLP({seed:9});c.restore(s);assert.deepEqual(c.forward(X).out,a.forward(X).out);
 assert.throws(()=>a.forward([1]),/finite/);assert.throws(()=>new TinyMLP({hidden:3}).restore(s),/differs/);
});
test('execution is sparse: tier 1 activates at most its budget, never the whole 5x64 field',()=>{
 const rt=new NsnfRuntime(),r=rt.execute({input:X,seeds});
 assert.equal(r.status,'executed');assert.ok(r.n_active<=DEFAULT_TIERS[1].maxActive);assert.ok(rt.space.size<20);assert.equal(rt.componentRecords().length,r.n_active);
 assert.equal(r.trace.active.filter(a=>a.why==='placement').length,1);assert.ok(r.trace.active.some(a=>a.why==='encoded'));
});
test('tiers are nested and bounded; forcing a higher tier recruits more (never fewer) components',()=>{
 const rt=new NsnfRuntime(),ids=t=>rt.execute({input:X,seeds,tier:t,log:false}).trace.active.map(a=>a.node);
 const t1=ids(1),t2=ids(2),t3=ids(3);
 assert.ok(t2.length>t1.length&&t3.length>=t2.length);assert.ok(t1.every(i=>t2.includes(i))&&t2.every(i=>t3.includes(i)));
 for(const t of [1,2,3,4,5])assert.ok(ids(t).length<=DEFAULT_TIERS[t].maxActive);
});
test('escalation is recorded with its reason and reuses lower-tier component outputs',()=>{
 const rt=new NsnfRuntime({confidenceThreshold:0.99}),r=rt.execute({input:X,seeds});
 assert.equal(r.startTier,1);assert.equal(r.tier,5);assert.equal(r.escalation.length,4);assert.match(r.escalation[0],/tier 1->2: confidence/);
 const forced=rt.execute({input:X,seeds,tier:2,log:false});assert.equal(forced.tier,2);assert.equal(forced.escalation.length,0);
});
test('no hidden fallback: bad input or no candidates is a blocked result, not a pretend execution',()=>{
 const rt=new NsnfRuntime();
 for(const bad of [[],[1,2],null,[...X.slice(1),NaN]]){const r=rt.execute({input:bad});assert.equal(r.status,'blocked');assert.equal(r.executed,false)}
 assert.equal(rt.execute({input:X,tier:9}).status,'blocked');
 const none=new NsnfRuntime({tiers:{1:{radius:0,maxActive:0,rounds:0,ancestors:false}}}).execute({input:X});assert.equal(none.status,'blocked');assert.match(none.reason,/no neural execution/);
});
test('edges emerge from activity and each topology update is a versioned transition',()=>{
 const rt=new NsnfRuntime(),before=rt.topologyVersion;assert.equal(rt.dataModel().connection.length,0);
 const r=rt.execute({input:X,seeds,tier:2});assert.ok(r.trace.newEdges>0);assert.ok(rt.topologyVersion>before);
 const dm=rt.dataModel();assert.equal(dm.connection.length,r.trace.edges);assert.ok(dm.connection.every(c=>c.rule_version==='edge-coactivation-v0'&&c.weight>0));
 assert.ok(dm.transition.some(t=>t.event_type==='topology-update'&&/edge-coactivation-v0/.test(t.reason)));
 const again=rt.execute({input:X,seeds,tier:2,log:false});assert.equal(again.trace.newEdges,0);
 assert.ok(r.trace.messages.count>0&&r.trace.meso.length>=1);
});
test('cache reuses a result only when input, seeds, tier, versions and state match',()=>{
 const rt=new NsnfRuntime();rt.execute({input:X,seeds});const hit=rt.execute({input:X,seeds});assert.equal(hit.cache_hit,true);assert.equal(hit.macs,0);
 assert.equal(rt.execute({input:Y,seeds}).cache_hit,false);assert.equal(rt.execute({input:X,seeds,tier:2}).cache_hit,false);
 rt.learn({input:X,target:[1,0,0,0],seeds,tier:1});assert.equal(rt.execute({input:X,seeds}).cache_hit,false,'learning changes version, so no stale reuse');
});
test('learning changes behaviour and parameter_ref; parameters are stored apart from events (Spec §21)',()=>{
 const rt=new NsnfRuntime();rt.execute({input:X,seeds});const refs=rt.componentRecords().map(c=>c.parameter_ref),outBefore=rt.execute({input:X,seeds,tier:1,log:false}).macro;
 let last;for(let i=0;i<30;i++)last=rt.learn({input:X,target:[1,0,0,0],seeds,tier:1,lr:0.1});
 assert.ok(last.lossAfter<last.lossBefore);assert.notDeepEqual(rt.execute({input:X,seeds,tier:1,log:false}).macro,outBefore);
 const now=rt.componentRecords().map(c=>c.parameter_ref);assert.notDeepEqual(now,refs);assert.ok(rt.parameterBlob(now[0]).params.length>0);
 assert.ok(rt.store.list('event_log').every(e=>!('params' in e)),'event_log carries hashes only');
});
test('PSO as background tooling lowers held-out loss, or leaves parameters unchanged',async()=>{
 const m=new TinyMLP({seed:3}),mk=i=>({input:X.map((v,j)=>(v+i*.037*(j+1))%1),target:[i%2,0,0,1-i%2]}),data=Array.from({length:24},(_,i)=>mk(i));
 const r=await trainWithParticleSwarm(m,data.slice(0,16),{heldOut:data.slice(16),particles:20,iterations:30});
 assert.equal(r.evaluatedOn,'held-out');assert.ok(r.heldOutAfter<=r.heldOutBefore);assert.ok(r.accepted);assert.ok(r.heldOutAfter<r.heldOutBefore);
 await assert.rejects(()=>trainWithParticleSwarm(m,[]),/required/);
});
test('event_log, transition and state_snapshot rows follow the Spec §4 shapes; log is append-only by interface',()=>{
 const rt=new NsnfRuntime();rt.execute({input:X,seeds});rt.execute({input:Y,seeds});const dm=rt.dataModel();
 assert.deepEqual(Object.keys(dm.event_log[0]),['id','actor','payload_hash','timestamp']);assert.match(dm.event_log[0].payload_hash,/^[0-9a-f]{64}$/);
 assert.deepEqual(Object.keys(dm.transition[0]),['id','before_id','after_id','event_type','reason']);
 for(const k of ['id','timestamp','dimension_codes','active_masks','center_states','checkpoint_ref'])assert.ok(k in dm.state_snapshot[0]);
 assert.equal(dm.dimension.length,5);assert.equal(dm.gate.count,320);assert.equal(dm.dimension[4].emergent,true);assert.equal(dm.gate.center_binding,null);
 assert.deepEqual(Object.keys(dm.component[0]),['id','gate_id','architecture_type','artifact_hash','parameter_ref','version','param_version']);
 assert.equal(typeof rt.store.append,'function');assert.equal(rt.store.remove,undefined);assert.equal(rt.store.update,undefined);
});
test('snapshot rows carry the 64-bit active masks per dimension, with Space always 0 and its code investigating',()=>{
 const rt=new NsnfRuntime(),r=rt.execute({input:X,seeds}),s=rt.store.list('state_snapshot')[0];
 assert.match(s.active_masks.Being,/^[0-9a-f]{16}$/);assert.notEqual(s.active_masks.Being,'0000000000000000');assert.equal(s.active_masks.Space,'0000000000000000');
 assert.equal(s.dimension_codes.Space,null);assert.equal(r.space.status,'investigating');
});
test('checkpoint and restore into a fresh runtime reproduce outputs; shape mismatch is refused',()=>{
 const rt=new NsnfRuntime();rt.execute({input:X,seeds,tier:3});rt.learn({input:X,target:[0,1,0,0],seeds,tier:2,lr:.1});rt.checkpoint();
 const f=new NsnfRuntime();f.restore(rt.store.loadCheckpoint());
 for(const t of [1,2,3])assert.deepEqual(f.execute({input:X,seeds,tier:t,log:false}).macro,rt.execute({input:X,seeds,tier:t,log:false}).macro);
 assert.equal(f.space.size,rt.space.size);assert.equal(f.globalVersion,rt.globalVersion);
 assert.throws(()=>new NsnfRuntime({hidden:5}).restore(rt.store.loadCheckpoint()),/differs/);assert.throws(()=>new NsnfRuntime().restore(null),/No checkpoint/);
});
test('replay of the event log reproduces every output; a tampered payload is detected',()=>{
 const rt=new NsnfRuntime();rt.execute({input:X,seeds});rt.learn({input:X,target:[1,0,0,0],seeds,tier:1});rt.execute({input:Y,seeds,tier:2});rt.execute({input:X,seeds});
 const rp=rt.replayLog();assert.equal(rp.ok,true);assert.equal(rp.events,4);
 const payloads=rt.store.list('event_payload').map(p=>p.event_id===3?{...p,payload:{...p.payload,input:X}}:p);
 const bad=NsnfRuntime.replay(rt.store.list('event_log'),payloads,{space:new StateSpace(),checkpointEvery:0});assert.equal(bad.ok,false);assert.match(bad.mismatches[0].why,/hash differs/);
});
test('state persists across sessions through a file store (JSONL logs plus checkpoint)',async()=>{
 const dir=mkdtempSync(join(tmpdir(),'nsnf-t-')),a=new NsnfRuntime({store:await fileStore(dir)});a.execute({input:X,seeds,tier:2});a.learn({input:X,target:[1,0,0,0],seeds,tier:1});a.checkpoint();
 const b=new NsnfRuntime({store:await fileStore(dir)});b.restore();assert.equal(b.store.list('event_log').length,2);
 assert.deepEqual(b.execute({input:X,seeds,tier:2,log:false}).macro,a.execute({input:X,seeds,tier:2,log:false}).macro);
});
test('browser storage store works with a localStorage-shaped object',()=>{
 const m=new Map(),ls={getItem:k=>m.has(k)?m.get(k):null,setItem:(k,v)=>m.set(k,v)},a=new NsnfRuntime({store:storageStore(ls)});a.execute({input:X,seeds});a.checkpoint();
 const b=new NsnfRuntime({store:storageStore(ls)});b.restore();assert.deepEqual(b.execute({input:X,seeds,log:false}).macro,a.execute({input:X,seeds,log:false}).macro);
});
test('encoder: one six-bit code per primitive dimension, deterministic, Space never encoded',()=>{
 const rt=new NsnfRuntime(),c=rt.encode(X);assert.deepEqual(Object.keys(c),['Movement','Evolution','Being','Design']);assert.ok(Object.values(c).every(v=>v>=0&&v<64));
 assert.deepEqual(new NsnfRuntime().encode(X),c);assert.notDeepEqual(rt.encode(Y),c);
});
test('open items are listed, not hidden',()=>{assert.ok(INVESTIGATING.length>=6);assert.ok(INVESTIGATING.some(x=>/nine/.test(x)))});
