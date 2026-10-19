import test from 'node:test';import assert from 'node:assert/strict';
import './vendor/kernel.js';
import {adaptLanguage,processFile,fileFeatures,placeFile,languageOf,LANGUAGE_ADAPTERS,FEATURE_NAMES,INVESTIGATING} from './any-file-swarm.mjs';
import {NsnfRuntime} from './nsnf-runtime.mjs';
import {StateSpace,gateToMacro} from './state-space.mjs';
import {writeZip} from './zip-lite.mjs';
import {memoryOutbox} from './supabase-original.mjs';
import {sha256} from './sha256.mjs';
const te=new TextEncoder();
const full={planetary:'Sun',dimension:'Being',zodiac:'Virgo',gate:6,line:4,color:4,tone:4,base:2,degree:25,minute:59,second:31,arcSecond:92,house:5};
const SAMPLES={'app.js':te.encode('export const a=1;\n'),'x.py':te.encode('print(1)\n'),'s.css':te.encode('a{b:c}'),'d.json':te.encode('{"k":[1]}'),
 'notes.txt':te.encode('plain words\n'),'blob.bin':Uint8Array.from({length:300},(_,i)=>(i*37+11)&255),'z.zip':writeZip([{path:'index.html',data:'<p>x</p>'}]),
 'empty.dat':new Uint8Array(0),'unicode.txt':te.encode('héllo wörld ✓\n'),'noext':te.encode('no extension here')};
test('features: 8 finite numbers in [0,1] for any file; kinds and languages are detected',()=>{
 assert.equal(FEATURE_NAMES.length,8);
 for(const [n,b] of Object.entries(SAMPLES)){const f=fileFeatures(n,b);assert.equal(f.vector.length,8);assert.ok(f.vector.every(v=>Number.isFinite(v)&&v>=0&&v<=1),n);assert.equal(f.sha256,sha256(b));assert.equal(f.size,b.length)}
 assert.equal(fileFeatures('z.zip',SAMPLES['z.zip']).kind,'zip');assert.equal(fileFeatures('b',SAMPLES['blob.bin']).kind,'binary');assert.equal(fileFeatures('e',new Uint8Array(0)).kind,'empty');assert.equal(fileFeatures('t',te.encode('hi')).kind,'text');
 assert.equal(languageOf('a.MJS'),'js');assert.equal(languageOf('a.py'),'py');assert.equal(languageOf('a.css'),'css');assert.equal(languageOf('a.json'),'json');assert.equal(languageOf('a.rs'),null);
 assert.throws(()=>fileFeatures('x',123),/bytes/);
});
test('language adapters: js and json reduce through her PrimitiveReducer; css and py stay placeholders, honestly marked',async()=>{
 assert.deepEqual(Object.keys(LANGUAGE_ADAPTERS),['js','py','css','json']);
 assert.deepEqual(Object.entries(LANGUAGE_ADAPTERS).map(([k,a])=>[k,a.implemented,a.placeholder]),[['js',true,false],['py',false,true],['css',false,true],['json',true,false]]);
 const js=await adaptLanguage('a.mjs',te.encode("import x from './x.mjs';export function add(a,b){return a+b}"));assert.equal(js.implemented,true);assert.deepEqual(js.structure.dependencies,['./x.mjs']);assert.deepEqual(js.structure.definitions,['add']);assert.ok(js.structure.moduleSyntax);
 const j=await adaptLanguage('a.json',te.encode('{"a":{"b":[1,2]}}'));assert.equal(j.implemented,true);assert.ok(j.primitives.length>1);
 for(const f of ['a.py','a.css'])assert.equal((await adaptLanguage(f,te.encode('x'))).implemented,false);
 assert.match(fileFeatures('a.py',te.encode('x')).adapter.note,/placeholder/);assert.match(fileFeatures('a.js',te.encode('x')).adapter.note,/PrimitiveReducer/);assert.equal(fileFeatures('a.rs',te.encode('x')).adapter,null);assert.ok(INVESTIGATING.some(x=>/placeholders/.test(x)));
});
test('ANY file goes registration -> placement -> execution -> phenotype without throwing or being rejected',async()=>{
 const runtime=new NsnfRuntime();
 for(const [name,bytes] of Object.entries(SAMPLES)){
  const r=await processFile({name,bytes},{address:full,personalization:{name:'T',worldTheme:'forest',address:full,target:'reality'},source:'test',runtime,outbox:memoryOutbox()});
  assert.ok(['complete','investigating'].includes(r.status),name);assert.equal(r.file.sha256,sha256(bytes));
  assert.ok(r.registration.status);assert.deepEqual(r.genotype_pso.packs.length,3);assert.match(r.genotype_pso.hexagram_code,/^[01]{6}$/);
  assert.equal(r.execution.executed===false?false:true,true,name+' executed');assert.equal(r.phenotype_ga.visible.lines.length,6);assert.equal(r.phenotype_ga.visible.userCodeExecuted,false);
  assert.deepEqual(r.safety,{userCodeExecuted:false,originalBytesModified:false});assert.ok(r.finisher,name+' finisher ran');
 }
});
test('registered address places the file: dimension from address, level 1 from gate (King Wen->Fu Xi) and sub from line',async()=>{
 const r=await processFile({name:'app.js',bytes:SAMPLES['app.js']},{address:full,source:'test',finish:false});
 assert.equal(r.status,'complete');assert.equal(r.placement.status,'placed');assert.equal(r.placement.dimension,'Being');
 assert.equal(r.placement.path[0].macro,gateToMacro(6));assert.equal(r.placement.path[0].sub,3);assert.equal(r.placement.path.length,5);
 assert.equal(r.placement.level1_gate,6);assert.equal(r.placement.provenance[0].macro,'address.gate (King Wen -> Fu Xi map)');assert.equal(r.placement.provenance[1].macro,'content hash');
 assert.equal(r.execution.startTier,1);assert.ok(r.execution.trace.active.some(a=>a.why==='placement'));assert.equal(r.execution.space.status,'investigating');
});
test('placement is deterministic per content; different content lands at different deeper positions',async()=>{
 const a=await processFile({name:'a.js',bytes:te.encode('one')},{address:full,source:'t',finish:false}),b=await processFile({name:'a.js',bytes:te.encode('one')},{address:full,source:'t',finish:false}),c=await processFile({name:'a.js',bytes:te.encode('two')},{address:full,source:'t',finish:false});
 assert.deepEqual(a.placement.path,b.placement.path);assert.notDeepEqual(a.placement.path.slice(1),c.placement.path.slice(1));assert.deepEqual(a.placement.path[0],c.placement.path[0]);
});
test('no address: file is held (nothing guesses a dimension), kept as investigating, still executes, starts at tier 2',async()=>{
 const r=await processFile({name:'tool.py',bytes:SAMPLES['x.py']},{source:'t',finish:false});
 assert.equal(r.status,'investigating');assert.equal(r.placement.status,'held');assert.equal(r.placement.nodeId,null);assert.equal(r.registration.status,'investigating');
 assert.equal(r.execution.startTier,2);assert.match(r.execution.reasons[0],/ambiguity/);assert.ok(r.execution.n_active>5);
 assert.ok(r.reasons.some(x=>/registration investigating/.test(x))&&r.reasons.some(x=>/placement held/.test(x)));
});
test('partial address: dimension known but gate missing is placed-investigating with content-derived level 1',async()=>{
 const r=await processFile({name:'a.js',bytes:SAMPLES['app.js']},{address:{dimension:'Design'},source:'t',finish:false});
 assert.equal(r.placement.status,'placed-investigating');assert.equal(r.placement.dimension,'Design');assert.equal(r.placement.provenance[0].macro,'content hash');assert.equal(r.status,'investigating');
});
test('Space-led address: file is held because Space is emergent, never written',async()=>{
 const r=await processFile({name:'a.js',bytes:SAMPLES['app.js']},{address:{...full,dimension:'Space'},source:'t',finish:false});
 assert.equal(r.placement.status,'held');assert.match(r.placement.notes[0],/Space is emergent/);assert.equal(r.registration.status,'registered');
 const s=new StateSpace();assert.throws(()=>s.ensure('Space',[{macro:1,sub:0}]));
});
test('state persists and accumulates in the runtime across files; second identical file reuses swarm output',async()=>{
 const rt=new NsnfRuntime(),a=await processFile({name:'a.js',bytes:SAMPLES['app.js']},{address:full,source:'t',runtime:rt,finish:false}),n=rt.space.size;
 const b=await processFile({name:'a.js',bytes:SAMPLES['app.js']},{address:full,source:'t',runtime:rt,finish:false});
 assert.equal(a.execution.cache_hit,false);assert.equal(b.execution.cache_hit,true);assert.equal(rt.space.size,n);assert.ok(rt.store.loadCheckpoint());
 await processFile({name:'b.js',bytes:te.encode('other')},{address:full,source:'t',runtime:rt,finish:false});assert.ok(rt.space.size>n);assert.ok(rt.store.list('event_log').length>=2);
});
test('finisher: unfinished upload keeps the original byte-for-byte and queues it for Supabase when env is missing',async()=>{
 const outbox=memoryOutbox(),bytes=SAMPLES['blob.bin'],copy=bytes.slice();
 const r=await processFile({name:'blob.bin',bytes},{address:full,source:'t',outbox,env:{}});
 assert.equal(r.finisher.supabase.status,'pending-supabase');assert.deepEqual(bytes,copy);assert.equal(r.finisher.supabase.sha256,sha256(bytes));
 const q=await outbox.list();assert.equal(q.length,1);assert.deepEqual(q[0].bytes,bytes);assert.ok(r.finisher.added.every(p=>p.startsWith('simlish/')));
});
test('the uploaded code is never run: a file that would throw or set a global is only read',async()=>{
 globalThis.__ran=false;
 const r=await processFile({name:'evil.js',bytes:te.encode('globalThis.__ran=true;throw new Error("x")')},{address:full,source:'t',finish:false});
 assert.equal(globalThis.__ran,false);assert.equal(r.safety.userCodeExecuted,false);
});
test('placeFile can be used on its own and never touches Space',()=>{
 const space=new StateSpace(),f=fileFeatures('a.js',SAMPLES['app.js']),g={hexagram_code:'111001'};
 const p=placeFile({features:f,genotype:g,address:{dimension:'Movement',gate:1,line:6},space});assert.equal(p.path[0].macro,63);assert.equal(p.path[0].sub,5);assert.equal(space.size,1);
 assert.equal(placeFile({features:f,genotype:g,address:{dimension:'Nonsense'},space}).status,'held');
});
