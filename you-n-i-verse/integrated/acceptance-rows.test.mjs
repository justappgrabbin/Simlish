// Executable tests for the 8 acceptance rows that were EXISTS-UNTESTED (1, 3, 4, 5, 6, 7, 9, 15). Each test measures what the
// code on this branch actually does and asserts that measurement. The row verdict (PASS/FAIL) recorded in INVENTORY/SCORECARD
// follows from what is asserted here; a FAIL row's test passes when the measured limitation is exactly as reported.
import test from 'node:test';import assert from 'node:assert/strict';
import {readFileSync,readdirSync,mkdtempSync} from 'node:fs';import {tmpdir} from 'node:os';import {join,dirname} from 'node:path';import {fileURLToPath} from 'node:url';
import '../vendor/kernel.js';
import {TARGETS} from './targets.mjs';
import {Population,decode,run,atomsToBytes,bytesToAtoms,fitness,reorganize} from './population.mjs';
import {runJavaScript} from './providers.mjs';
import {IntegratedSwarm} from './integrated-swarm.mjs';
import {finishApp} from '../app-finisher.mjs';
import {TARGET_ADAPTERS} from '../finisher-targets.mjs';
import {createAppServer} from './app-server.mjs';
import {ProtectedError} from './swarm-loop.mjs';
import {affineTriangle} from '../photo-morph.mjs';
import {calculateBirthChart} from './chart-engine.mjs';
import {createPlaySession} from './play/session.mjs';
import {avatarIdentity,rgbaToBmp} from './play/avatar.mjs';
const HERE=dirname(fileURLToPath(import.meta.url)),ROOT=join(HERE,'..'),te=new TextEncoder(),td=new TextDecoder();
const bits=(v,n)=>Array.from({length:n},(_,i)=>(v>>i)&1),num=a=>a.reduce((s,b,i)=>s+(b<<i),0);
const loadBytes=async p=>new Uint8Array(readFileSync(join(ROOT,p))),loadJson=async p=>JSON.parse(readFileSync(join(ROOT,p),'utf8'));
const scan=d=>readdirSync(d,{withFileTypes:true}).flatMap(e=>e.isDirectory()?(['vendor','node_modules','reference','originals','.git'].includes(e.name)?[]:scan(join(d,e.name))):[join(d,e.name)]);
test('row 1 Swarm is code — PASS (limit: JS source runs after decoding the atoms back to text)',async()=>{
 // (a) the population state itself is the program: her six-bit atoms decode to an instance that computes
 const p=decode(TARGETS.number.atoms),out=run(p,[5,10,27].map(v=>bits(v,8))).map(num);assert.deepEqual(out,[5,15,42]);
 // (b) a JS program carried as six-bit atoms is lossless and runs with the same result as its source
 const src='let s=0;for(let i=1;i<=10;i++)s+=i;console.log("sum",s);return s*2;',b=te.encode(src),atoms=bytesToAtoms(b);assert.ok(atoms.every(a=>a>=0&&a<64));
 const back=td.decode(atomsToBytes(atoms).slice(0,b.length));assert.equal(back,src);
 const r1=await runJavaScript(src),r2=await runJavaScript(back);assert.equal(r1.ok,true);assert.deepEqual(r2.result.returnValue,r1.result.returnValue);assert.equal(r1.result.returnValue,110)});
test('row 3 Arbitrary code ingestion — FAIL (js/json/css/html get structure; python and other languages are placeholders)',async()=>{
 const sw=new IntegratedSwarm(),F=(name,s)=>({name,bytes:te.encode(s)});
 const js=await sw.execute(F('a.js','return 1+1;')),json=await sw.execute(F('a.json','{"a":[1,2]}')),css=await sw.execute(F('a.css','a{color:red}')),html=await sw.execute(F('a.html','<p>x</p>')),py=await sw.execute(F('a.py','print(1)')),rs=await sw.execute(F('a.rs','fn main(){}'));
 assert.equal(js.execution.returnValue,2);assert.deepEqual(json.execution.returnValue,{a:[1,2]});assert.equal(css.execution.returnValue.rules,1);assert.ok(html.execution.returnValue);
 assert.deepEqual(py.execution.unresolved,['adapter:python (placeholder)']);assert.equal(rs.execution.returnValue.placeholder,true)});
test('row 4 Code creation — FAIL (finishApp creates scaffolding files from a checklist; no program logic is created; game/2d targets not implemented)',async()=>{
 const r=await finishApp({name:'x.html',bytes:te.encode('<!doctype html><h1>x</h1>')},{env:{}});
 const added=r.report.addedFiles;assert.ok(added.length>=4);assert.ok(added.every(p=>/^simlish\/(entry\.html|manifest\.json|registration\.json|scene\.json|reality\.json|personalization\.json|finisher-report\.json)$/.test(p)),added.join(','));
 assert.ok(!added.some(p=>/\.m?js$/.test(p)),'no program file is created');for(const t of ['game','2d'])assert.equal(TARGET_ADAPTERS[t].implemented,false)});
test('row 5 Neuro-symbolic reasoning — FAIL (neural Trident and symbolic guide both exist; no module combines them)',()=>{
 const files=scan(ROOT).filter(f=>/\.m?js$/.test(f)&&!/\.test\.m?js$/.test(f));const uses=re=>files.filter(f=>re.test(readFileSync(f,'utf8')));
 const neural=uses(/from\s*['"][^'"]*trident-js\.mjs['"]/),symbolic=uses(/from\s*['"][^'"]*(?:integrated\/|\.\/)guide\.mjs['"]/);
 assert.ok(neural.length>=1&&symbolic.length>=1);
 const both=neural.filter(f=>symbolic.includes(f)).map(f=>f.slice(ROOT.length+1));
 // app-server.mjs hosts both as separate tools (chat vs guide.ask); no function passes one's output into the other
 assert.deepEqual(both,['integrated/app-server.mjs']);const t=readFileSync(join(ROOT,'integrated/app-server.mjs'),'utf8');assert.ok(!/guide\.[a-z]+\([^)]*generate|generate[A-Za-z]*\([^)]*guide/.test(t))});
test('row 6 Collective adaptation — FAIL (PSO alone does not repair a corrupted primitive organisation; the Hamming-1 refinement does)',async()=>{
 const ex=[{inputs:[5,10,27].map(v=>bits(v,8)),expect:[5,15,42].map(v=>bits(v,8))}],a=[...TARGETS.number.atoms];a[3]^=1;const bad=fitness(a,ex);assert.ok(bad>0);
 const pso=await reorganize(a,ex,{refineRounds:0}),full=await reorganize(a,ex);
 assert.ok(fitness(pso.atoms,ex)>0,'PSO alone left wrong bits');assert.equal(pso.repaired,false);assert.equal(fitness(full.atoms,ex),0);assert.equal(full.repaired,true)});
test('row 7 Self-modification — FAIL (the loop writes only its virtual workspace; the running code and protected originals cannot be rewritten)',async()=>{
 const s=await createAppServer({loadJson,loadBytes,loadModel:false,schedule:()=>0,cancel:()=>{}});
 for(let i=0;i<120&&s.loop.progress().done<s.loop.plan.length;i++)await s.handle('POST','/api/loop/cycle');
 const files=(await s.handle('GET','/api/files')).body.files.map(f=>f.path);assert.ok(files.length>0);
 assert.ok(files.every(p=>/^(app|user)\//.test(p)),'loop wrote outside app/: '+files.filter(p=>!/^(app|user)\//.test(p)).slice(0,3));
 assert.throws(()=>s.workspace.write('integrated/primitive-functions.json','x'),ProtectedError);
 assert.equal((await s.handle('POST','/api/tool/workspace.write',{path:'integrated/swarm-loop.mjs',text:'x'})).body.error,'only user/ paths can be written by tools')});
test('row 9 Arbitrary visual morphing — FAIL (image morph and her chart-driven embodiment run; nothing lets swarm state drive the morph)',async()=>{
 const m=affineTriangle([[0,0],[1,0],[0,1]],[[0,0],[2,0],[0,2]]);assert.ok(m);
 const chart=calculateBirthChart({date:'1990-05-17',time:'08:30',timeZone:'America/Los_Angeles',latitude:34.05,longitude:-118.24}),w=24,px=new Uint8Array(w*w*4).map((_,i)=>(i*5)%256);
 const s=createPlaySession({chart,dataDir:mkdtempSync(join(tmpdir(),'row9-')),visualIdentity:avatarIdentity(rgbaToBmp(w,w,px),'avatar:row9')});
 const e=await s.enter('hd-rpg'),r=await s.choose({type:'use-ability',abilityGate:e.lesson.need.gate});await s.exit();
 assert.notEqual(e.entry.morph.embodimentId,r.result.consequence.embodimentId,'her morph changes with the game choice');
 const src=readFileSync(join(HERE,'play/games-host.mjs'),'utf8')+readFileSync(join(HERE,'play/session.mjs'),'utf8');assert.ok(!/morphEngine\.morph\(|hexagram_code[^;]*morph/.test(src),'no swarm output is fed into the morph')});
test('row 15 Reliability — PASS for a soak of 400 loop cycles + 400 heartbeats + 120 play turns (no failures, audit chain intact, originals unchanged)',async()=>{
 let k=0;const s=await createAppServer({loadJson,loadBytes,loadModel:false,schedule:()=>0,cancel:()=>{},now:()=>new Date(Date.UTC(2026,9,19,14,0,0,k++)).toISOString()});
 for(let i=0;i<400;i++)await s.handle('POST','/api/loop/cycle');for(let i=0;i<400;i++)await s.heartbeat.beat();
 const st=(await s.handle('GET','/api/status')).body;assert.equal(st.loop.failed,0);assert.equal(st.loop.done,st.loop.units);assert.equal(st.audit.chain.ok,true);assert.equal(st.protected.ok,true);
 const chart=calculateBirthChart({date:'1984-11-02',time:'22:10',timeZone:'Europe/London',latitude:51.5,longitude:-0.12}),w=24,px=new Uint8Array(w*w*4).map((_,i)=>(i*3)%256);
 const p=createPlaySession({chart,dataDir:mkdtempSync(join(tmpdir(),'row15-')),visualIdentity:avatarIdentity(rgbaToBmp(w,w,px),'avatar:row15')});
 for(let i=0;i<30;i++)for(const wd of ['hd-rpg','iching-story','oracles-thread','line-walk']){const e=await p.enter(wd);await p.choose(e.lesson.suggested)}
 await p.exit();assert.equal(p.memory.transitions.length,120)});
