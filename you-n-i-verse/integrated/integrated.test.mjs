import test from 'node:test';import assert from 'node:assert/strict';
import '../vendor/kernel.js';
import {readFileSync,readdirSync} from 'node:fs';import {createHash} from 'node:crypto';import {fileURLToPath} from 'node:url';import {dirname,join} from 'node:path';
import {IntegratedSwarm,HistoryLedger,INTEGRATED_INVESTIGATING} from './integrated-swarm.mjs';
import {WHEELS,WHEELS_INVESTIGATING,TEN_WINGS,wheelRank,wheelReadout,mawangduiKingWen} from './wheels.mjs';
import {deriveAppCode,verifyAppCode,APP_CODE_INVESTIGATING} from './app-code.mjs';
import {guardJavaScript,runJavaScript,parseCss,parseHtml,genericExecute,createSurface} from './providers.mjs';
import {normalizeArtifact} from './automata/artifact-normalizer.mjs';
import {EXECUTION_PIPELINE_STAGES} from './spine/execution-pipeline.mjs';
import {readSimlishRegistrations} from '../synthia-registration.mjs';
import {writeZip} from '../zip-lite.mjs';
import {bitsOf,gateToMacro} from '../state-space.mjs';
const here=dirname(fileURLToPath(import.meta.url)),te=new TextEncoder();
const mem=()=>({d:{},getItem(k){return this.d[k]??null},setItem(k,v){this.d[k]=v}});
const F=(name,s)=>({name,bytes:typeof s==='string'?te.encode(s):s});
const FIXTURE={planetary:'Sun',dimension:'Being',zodiac:'Virgo',gate:6,line:4,color:4,tone:4,base:2,degree:25,minute:59,second:31,arcSecond:92,house:5};

test('her modules are copied in byte-for-byte (manifest) and the originals are not edited',()=>{
 const m=JSON.parse(readFileSync(join(here,'MANIFEST.json'),'utf8'));
 assert.ok(Object.keys(m.files).length>=18);
 for(const [p,v] of Object.entries(m.files))assert.equal(createHash('sha256').update(readFileSync(join(here,p))).digest('hex'),v.sha256,p+' must stay as in the zip');
 for(const need of ['spine/execution-pipeline.mjs','automata/ingestion-execution-kernel.mjs','ontological/ingestion/ontological-file-ingestor.mjs','reference/hexagram_sequences.py'])assert.ok(m.files[need],need);
});
test('no worker, backend, network or external dependency in the integration (adapters and copied modules)',()=>{
 const scan=dir=>readdirSync(dir,{withFileTypes:true}).flatMap(e=>e.isDirectory()?scan(join(dir,e.name)):[join(dir,e.name)]);
 const files=scan(here).filter(f=>/\.(mjs|js)$/.test(f)&&!f.endsWith('.test.mjs')&&!f.includes('/reference/')&&(!f.includes('/swarm/originals/')||f.endsWith('-morph-search.mjs'))); // swarm/originals holds her untouched files by reference; only the one the app imports (morph-search) must satisfy these rules
 assert.ok(files.length>=20);
 for(const f of files){const t=readFileSync(f,'utf8').replace(/\/\/.*$/gm,'').replace(/\/\*[\s\S]*?\*\//g,'');
  assert.ok(!/new\s+Worker\s*\(|new\s+SharedWorker|\bfetch\s*\(|XMLHttpRequest\s*\(|new\s+WebSocket|\brequire\s*\(/.test(t),f);
  for(const m of t.matchAll(/(?:^|\n)\s*(?:import|export)\b[^;\n]*?\bfrom\s*['"]([^'"]+)['"]/g))assert.ok(m[1].startsWith('.')||((f.endsWith('demo.mjs')||f.endsWith('.build.mjs'))&&m[1].startsWith('node:')),f+' must import only local files (demo may use node: built-ins), got '+m[1])}
});
test('wheels per spec §2.1: Fu Xi Movement, Mawangdui Evolution, King Wen Space (read only); Being/Design investigating, nothing invented',()=>{
 assert.deepEqual(WHEELS.Movement.order,Array.from({length:64},(_,i)=>i));
 const mw=mawangduiKingWen();assert.deepEqual(mw.slice(0,10),[1,33,6,25,12,10,13,44,52,26]); // = mawangdui_order() of hexagram_sequences.py
 assert.deepEqual([...mw].sort((a,b)=>a-b),Array.from({length:64},(_,i)=>i+1));
 assert.equal(new Set(WHEELS.Evolution.order).size,64);assert.equal(WHEELS.Evolution.status,'investigating');
 assert.equal(WHEELS.Space.readOnly,true);assert.equal(WHEELS.Space.order[0],63);assert.equal(WHEELS.Space.order[1],0);
 for(const d of ['Being','Design']){assert.equal(WHEELS[d].order,null);assert.equal(WHEELS[d].status,'investigating');assert.equal(wheelRank(d,5),null);assert.equal(wheelReadout(d,[1,2]).meanPosition,null)}
 assert.equal(TEN_WINGS.status,'investigating');assert.ok(WHEELS_INVESTIGATING.length>=4);
});
test('her zips hold conflicting Mawangdui versions: the js reference disagrees with the py port and is flagged, not used',async()=>{
 const src=readFileSync(join(here,'reference/hexagram_sequences.js.txt'),'utf8');
 const mod=await import('data:text/javascript;base64,'+Buffer.from(src).toString('base64'));
 const ours=WHEELS.Evolution.order;let diff=0;for(let i=0;i<64;i++)if(mod.MD_TO_BITS[i+1]!==ours[i])diff++;
 assert.ok(diff>0,'versions differ ('+diff+' of 64 positions)');
 let kw=0;for(let k=1;k<=64;k++)if(mod.KW_TO_BITS[k]!==WHEELS.Space.order[k-1])kw++;assert.ok(kw>0);
});
test('same-thread JavaScript runs; escape attempts, module syntax, endless loops are blocked; executeCode:false never runs it',async()=>{
 const r=await runJavaScript('console.log("a",1);return [1,2,3].reduce((a,b)=>a+b)');assert.equal(r.ok,true);assert.equal(r.result.returnValue,6);assert.equal(r.result.stdout[0].args[0],'a');
 const bad=await runJavaScript('return typeof process+typeof require+typeof fetch+typeof globalThis');assert.equal(bad.result.returnValue,'undefinedundefinedundefinedundefined');
 for(const src of ['process.exit(1)','require("fs")','(()=>{}).constructor("return 1")()','eval("1")','import("node:fs")','while(true){}','globalThis.x=1'])assert.equal(guardJavaScript(src).ok,false,src);
 globalThis.__ran=false;const sw=new IntegratedSwarm({executeCode:false});
 const off=await sw.execute(F('x.js','globalThis.__ran=true;return 1'));assert.equal(globalThis.__ran,false);assert.equal(off.executedUserCode,false);
 const mod=await new IntegratedSwarm().execute(F('m.mjs','export const a=1'));assert.equal(mod.execution.ok,false);assert.deepEqual(mod.execution.unresolved,['module-loader']);
 const t=await runJavaScript('await new Promise(r=>setTimeout(r,200));return 1',{timeoutMs:20});assert.equal(t.ok,false);assert.match(t.error,/exceeded/);
 const e=await runJavaScript('throw new Error("boom")');assert.equal(e.ok,false);assert.equal(e.error,'boom');
});
test('execute means decompose -> pipeline (11 stages, her order) -> run -> fold up -> commit -> address -> history',async()=>{
 const sw=new IntegratedSwarm(),orig=te.encode('const n=[2,3,4].map(x=>x*2);console.log(n.join());return n.length;');
 const r=await sw.execute({name:'f.js',bytes:orig});
 assert.deepEqual(r.stages.map(s=>s.stage),[...EXECUTION_PIPELINE_STAGES]);assert.equal(EXECUTION_PIPELINE_STAGES.length,11);
 assert.equal(r.stages.every(s=>s.ok),true);
 assert.equal(r.executedUserCode,true);assert.equal(r.execution.returnValue,3);assert.equal(r.execution.stdout[0].args[0],'4,6,8');
 assert.equal(r.ingest.status,'ingested');assert.equal(r.ingest.exactReconstruction,true);assert.ok(r.ingest.dimensions.Space,'Space derived by her ingest patch');
 assert.ok(r.fold.vector.length===8&&r.fold.vector.every(Number.isFinite));assert.equal(r.swarm.foldUp.tier>=1,true);
 assert.deepEqual(r.swarm.genotype_pso.packs,[3,2,1]);assert.equal(r.swarm.genotype_pso.hexagram_code.length,6);assert.equal(r.swarm.phenotype_ga.visible.lines.length,6);
 assert.equal(r.swarm.phenotype_ga.visible.userCodeExecuted,true);assert.equal(r.safety.originalBytesModified,false);assert.equal(r.safety.workerUsed,false);assert.equal(r.safety.backendUsed,false);
 assert.equal(r.history.addressKey,r.address.resolved.key);assert.equal(sw.history.recall(r.address.resolved.key).length,1);
 assert.equal(sw.pipeline.snapshot().runSequence,1);assert.ok(sw.mesh.edges.length>0);assert.deepEqual([...orig],[...te.encode('const n=[2,3,4].map(x=>x*2);console.log(n.join());return n.length;')]);
 const again=await sw.execute({name:'f.js',bytes:orig});
 assert.equal(again.address.resolved.key,r.address.resolved.key,'same file, same resolved address');assert.equal(again.app.code,r.app.code);
 assert.equal(sw.history.recall(r.address.resolved.key).length,2);assert.equal(sw.pipeline.snapshot().runSequence,2);
});
test('each kind of file is executed by its own provider; non-JS languages are placeholders; Space is never written',async()=>{
 const sw=new IntegratedSwarm(),wasm=Uint8Array.from([0,0x61,0x73,0x6d,1,0,0,0,1,7,1,0x60,2,0x7f,0x7f,1,0x7f,3,2,1,0,7,7,1,3,0x61,0x64,0x64,0,0,10,9,1,7,0,0x20,0,0x20,1,0x6a,0x0b]);
 const zip=writeZip([{path:'a.js',data:'return 2*21'},{path:'b.json',data:'{"k":1}'},{path:'c.css',data:'a{b:c}'},{path:'d.py',data:'print(1)'}]);
 const cases=[['d.json','{"a":[1,2]}','JSON.parse'],['x.prim','\u0001\u0002\u0003','population-tick'],['s.css','a{b:c}','css-structural-parse'],['p.html','<h1>x</h1><script>globalThis.__ranHtml=true</script>','html-structural-parse'],['n.md','# t\nwords','TextDecoder'],
  ['t.py','print(1)','generic-primitive-execution'],['x.ts','const a:number=1','generic-primitive-execution'],['m.wasm',wasm,'WebAssembly'],['b.bin',Uint8Array.from({length:300},(_,i)=>(i*37+11)&255),'generic-primitive-execution'],['z.zip',zip,'zip-member-execution']];
 globalThis.__ranHtml=false;
 for(const [n,b,engine] of cases){const r=await sw.execute(F(n,b));assert.equal(r.execution.engine,engine,n);assert.equal(r.execution.ok,true,n+' '+r.execution.error);
  const act=[r.swarm.execution?.n_active];assert.ok(act[0]===undefined||act[0]>=1)}
 assert.equal(globalThis.__ranHtml,false,'html scripts are not run without a DOM');
 const py=await sw.execute(F('t.py','print(1)'));assert.deepEqual(py.execution.unresolved,['adapter:python (placeholder)']);assert.equal(py.execution.returnValue.placeholder,true);
 const z=await sw.execute(F('z.zip',zip));const members=z.execution.returnValue.members;assert.equal(members.length,4);assert.equal(members.find(m=>m.path==='a.js').engine,'same-thread-async-function');assert.ok(z.execution.unresolved.includes('adapter:python (placeholder)'));
 const wa=await sw.execute(F('m.wasm',wasm));assert.deepEqual(wa.execution.returnValue.exports,['add']);
 assert.equal(sw.runtime.space.readSpace().status,'investigating','Space stays read-only: no emergence rule invented');
});
test('parsers and generic primitive execution are deterministic',()=>{
 assert.deepEqual(parseCss('/*c*/a,b{x:y;z:w}@media x{c{d:e}}').rules,2);assert.equal(parseCss('a{').errors.length,1);
 assert.equal(parseHtml('<script src=x></script><script>1</script>').externalScripts,1);
 const a=genericExecute(Uint8Array.from([1,2,3,250,251,252])),b=genericExecute(Uint8Array.from([1,2,3,250,251,252]));assert.deepEqual(a,b);assert.equal(a.sixBitStates,8);
});
test('app code is DERIVED from the resolved code (resolved address): no per-app table, no storage, re-derivable',async()=>{
 const resolved={address:{...FIXTURE,gate:6},key:'k1',basis:'creator-supplied'};
 const d=deriveAppCode(resolved);assert.equal(d.status,'derived');assert.equal(d.derived,true);assert.equal(d.stored,false);assert.equal(d.spaceWritten,false);
 assert.equal(d.hexagram_code,bitsOf(gateToMacro(6)));assert.match(d.code,/^app-[01]{6}-B6\.4\.4\.4\.2-[0-9a-f]{12}$/);
 assert.equal(deriveAppCode(resolved).code,d.code,'pure function');assert.ok(verifyAppCode(d.code,resolved));
 assert.equal(verifyAppCode(d.code,{...resolved,key:'k2'}),false,'different resolved code, different app code');
 assert.notEqual(deriveAppCode({...resolved,address:{...resolved.address,line:5}}).code,d.code);
 for(const bad of [null,{},{address:{dimension:'Being',gate:6}},{address:{...FIXTURE,gate:70},key:'x'},{address:{...FIXTURE,dimension:'Nope'},key:'x'}]){const x=deriveAppCode(bad);assert.equal(x.code,null);assert.equal(x.status,'investigating')}
 assert.ok(APP_CODE_INVESTIGATING.some(s=>/resolved code/.test(s)));
 // through the swarm: the registration id IS the derived code; the same bytes under two names get the same code; names/ids are not inputs
 const storage=mem(),sw=new IntegratedSwarm();
 const r=await sw.execute(F('one.js','return 6*7;'),{address:FIXTURE,storage});
 assert.equal(r.address.resolved.basis,'creator-supplied');assert.ok(verifyAppCode(r.app.code,{address:FIXTURE,key:r.address.resolved.key,basis:'creator-supplied'}));
 assert.deepEqual(Object.keys(readSimlishRegistrations(storage)),[r.app.code],'registered under the derived code, not a per-app id');
 const r2=await sw.execute(F('two-other-name.js','return 6*7;'),{address:FIXTURE});assert.equal(r2.app.code,r.app.code,'same resolved code -> same app code');
 const r3=await sw.execute(F('three.js','return 6*8;'),{address:FIXTURE});assert.equal(r3.app.code,r.app.code,'a registered address is authoritative: the code follows the address, not the file');
 // no address given: resolved by her ExecutionAddressResolver after the first execution
 const e=await sw.execute(F('e.js','return 1;'));assert.equal(e.address.resolved.basis,'first-execution-derived');assert.equal(e.address.resolved.key,e.address.executionDerived.key);
 assert.ok(verifyAppCode(e.app.code,{address:e.address.executionDerived.address,key:e.address.resolved.key,basis:e.address.resolved.basis}));
 assert.notEqual(e.app.code,r.app.code);
 const src=readFileSync(join(here,'app-code.mjs'),'utf8');assert.ok(!/\bnew Map\(|localStorage|setItem\(/.test(src),'nothing stored per app');
});
test('history is appended at the resolved address, cross-indexed, append-only',async()=>{
 const l=new HistoryLedger();const a=l.append('K',{x:1});l.append('K',{x:2});l.append('J',{x:3});
 assert.equal(l.recall('K').length,2);assert.equal(l.size,3);assert.equal(a.seq,1);assert.throws(()=>{'use strict';a.x=9});
 const sw=new IntegratedSwarm();const r=await sw.execute(F('h.json','{"h":1}'));
 const [e]=sw.history.recall(r.address.resolved.key);assert.equal(e.addresses.automata,r.address.automataId);assert.equal(e.addresses.ingest,r.address.ingest);assert.equal(e.appCode,r.app.code);
 assert.ok(sw.runtime.store.list('history').length===1,'mirrored into the runtime store');
});
test('unknown choices stay marked investigating and her words are kept',()=>{
 const sw=new IntegratedSwarm(),all=sw.investigating.join('\n');
 for(const w of ['Mawangdui','Ten Wings','resolved code','Space','sandbox','placeholder'])assert.ok(all.includes(w),w);
 assert.ok(INTEGRATED_INVESTIGATING.length>=5);
});
test('the file is never modified; empty and large inputs are handled; bad input is rejected',async()=>{
 const sw=new IntegratedSwarm({maxPipelineBytes:256});
 const big=Uint8Array.from({length:2000},(_,i)=>i*7&255),copy=big.slice();const r=await sw.execute(F('big.bin',big));
 assert.equal(r.fold.truncatedPipelineInput,true);assert.deepEqual([...big],[...copy]);
 const e=await sw.execute(F('empty.txt',new Uint8Array(0)));assert.ok(e.app.code||e.app.status==='investigating');
 await assert.rejects(()=>sw.execute({name:'x'}),/bytes required/);
 const n=await normalizeArtifact(F('q.js','1'));assert.equal(createSurface().providers.length,8);assert.equal(n.kind,'javascript');
});
