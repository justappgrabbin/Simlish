// Runnable demo: a file goes registration -> state-space placement -> swarm execution (-> finisher), then learning, checkpoint, restore, replay.
// Usage: node swarm-demo.mjs            (built-in sample files)
//        node swarm-demo.mjs path/to/any-file [more files...]
// DEMO FIXTURE ONLY: the address below is the repo's test fixture, NOT Alexis's address. Pass your own with SIMLISH_ADDRESS='{"planetary":...}'.
import './vendor/kernel.js';
import {readFile,mkdtemp} from 'node:fs/promises';import {tmpdir} from 'node:os';import {join,basename} from 'node:path';
import {processFile,fileFeatures} from './any-file-swarm.mjs';
import {NsnfRuntime,fileStore,trainWithParticleSwarm,hashOf} from './nsnf-runtime.mjs';
import {StateSpace,capacityTable,PRIMITIVE_DIMENSIONS} from './state-space.mjs';
import {memoryOutbox} from './supabase-original.mjs';
import {writeZip} from './zip-lite.mjs';
const te=new TextEncoder();
const FIXTURE={planetary:'Sun',dimension:'Being',zodiac:'Virgo',gate:6,line:4,color:4,tone:4,base:2,degree:25,minute:59,second:31,arcSecond:92,house:5};
const address=process.env.SIMLISH_ADDRESS?JSON.parse(process.env.SIMLISH_ADDRESS):FIXTURE;
const say=(t,o)=>console.log('\n== '+t+(o===undefined?'':'\n'+JSON.stringify(o,null,1)));
const files=process.argv.length>2?await Promise.all(process.argv.slice(2).map(async p=>({name:basename(p),bytes:new Uint8Array(await readFile(p)),address})))
 :[{name:'app.js',bytes:te.encode('export const hi = () => "hello";\nconsole.log(hi());\n'),address},
   {name:'data.json',bytes:te.encode('{"a":1,"b":[1,2,3]}\n'),address},
   {name:'style.css',bytes:te.encode('body{color:#333}\n'),address:{...FIXTURE,dimension:'Movement',gate:1,line:1}},
   {name:'tool.py',bytes:te.encode('print("hi")\n'),address:null},
   {name:'blob.bin',bytes:Uint8Array.from({length:512},(_,i)=>(i*131+7)&255),address:{...FIXTURE,dimension:'Space'}},
   {name:'site.zip',bytes:writeZip([{path:'index.html',data:'<h1>hi</h1>'}]),address}];
say('state space capacity (Alexis said 12 and also six; both shown, neither chosen)',capacityTable().map(c=>({split:c.split,levels:c.levels,perDimension:c.positionsPerDimensionAtFullDepth})));
const dir=await mkdtemp(join(tmpdir(),'nsnf-demo-')),store=await fileStore(dir),outbox=memoryOutbox();
const runtime=new NsnfRuntime({store,checkpointEvery:5});
for(const f of files){
 const r=await processFile(f,{address:f.address,personalization:{name:'demo-uploader',worldTheme:'forest',address:f.address,target:'reality'},source:'demo',runtime,env:{},outbox});
 if(f===files[0]){const r2=await processFile(f,{address:f.address,source:'demo',runtime,finish:false});say('same file straight away again: swarm output reused only because input, seeds, tier, version and state all match',{cache_hit:r2.execution.cache_hit,macs:r2.execution.macs})}
 say(f.name+' -> '+r.status,{kind:r.file.kind,language:r.file.language,registration:r.registration.status,
  genotype_PSO:{packs:r.genotype_pso.packs,hexagram:r.genotype_pso.hexagram_code},placement:r.placement.label,
  execution:r.execution.executed===false?r.execution:{tier:r.execution.tier,startTier:r.execution.startTier,active:r.execution.n_active,macs:r.execution.macs,confidence:+r.execution.confidence.toFixed(3),coherence:r.execution.coherence,space:r.execution.space.status},
  phenotype_GA:{title:r.phenotype_ga.visible.title,lines:r.phenotype_ga.visible.lines.join(' '),status:r.phenotype_ga.status},finisher:{status:r.finisher.status,supabase:r.finisher.supabase.status},reasons:r.reasons.slice(0,3)});
}
const f0=files[0],ft=fileFeatures(f0.name,f0.bytes);
const again=await processFile(f0,{address:f0.address,source:'demo',runtime,finish:false});
// learning: feedback changes behavior (local per-component update), then PSO as background tooling
const target=[1,0,0,0],first=runtime.learn({input:ft.vector,target,seeds:[{dimension:'Being',path:again.placement.path,role:'placement'}],tier:2,lr:0.1});
say('learning (spec §7 update): loss before -> after on the activated components',{before:+first.lossBefore.toFixed(4),after:+first.lossAfter.toFixed(4),updated:first.updated});
const comp=new (await import('./nsnf-runtime.mjs')).TinyMLP({seed:3}),data=Array.from({length:24},(_,i)=>({input:ft.vector.map((v,j)=>((v+i*0.037*(j+1))%1)),target:[i%2,0,0,1-(i%2)]}));
const pso=await trainWithParticleSwarm(comp,data.slice(0,16),{heldOut:data.slice(16),particles:20,iterations:30});
say('PSO (background tooling) trains one component; held-out loss reported honestly',{accepted:pso.accepted,heldOutBefore:+pso.heldOutBefore.toFixed(4),heldOutAfter:+pso.heldOutAfter.toFixed(4)});
// checkpoint -> restore into a fresh runtime -> same output; replay of the append-only event log
runtime.checkpoint();
const fresh=new NsnfRuntime({store:await fileStore(await mkdtemp(join(tmpdir(),'nsnf-restore-')))});fresh.restore(runtime.store.loadCheckpoint());
const a=runtime.execute({input:ft.vector,seeds:[{dimension:'Being',path:again.placement.path,role:'placement'}],tier:1,log:false}),b=fresh.execute({input:ft.vector,seeds:[{dimension:'Being',path:again.placement.path,role:'placement'}],tier:1,log:false});
say('restore: fresh runtime from checkpoint gives the same macro state',{same:hashOf(a.macro)===hashOf(b.macro),nodes:fresh.space.size});
const rp=runtime.replayLog();say('replay of the event log on a fresh runtime',{events:rp.events,ok:rp.ok,mismatches:rp.mismatches});
const dm=runtime.dataModel();
say('spec §4 data model rows (counts)',{dimensions:dm.dimension.length,gates:dm.gate.count,components:dm.component.length,connections:dm.connection.length,state_snapshots:dm.state_snapshot.length,transitions:dm.transition.length,event_log:dm.event_log.length,space:runtime.space.readSpace().status});
console.log('\nPersisted to '+dir+' (event_log.jsonl, event_payload.jsonl, state_snapshot.jsonl, transition.jsonl, checkpoint.json)');
