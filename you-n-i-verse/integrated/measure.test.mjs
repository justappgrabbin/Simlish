import test from 'node:test';import assert from 'node:assert/strict';
import '../vendor/kernel.js';
import {readFileSync,readdirSync,statSync} from 'node:fs';import {join} from 'node:path';import {fileURLToPath} from 'node:url';
import {measureContent,sixBitStatesAll,fieldIndexForPrimitive,nestingDepth,references,sandboxReport,toJson,DESCEND_CAP} from './measure.mjs';
import {ExecutionAddressResolver} from './spine/execution-address-resolver.mjs';
import {IntegratedSwarm} from './integrated-swarm.mjs';
import {writeZip} from '../zip-lite.mjs';
import {DIMENSIONS} from '../state-space.mjs';
const root=fileURLToPath(new URL('..',import.meta.url)),te=new TextEncoder();
const readTree=()=>{const out=[];const walk=rel=>{for(const n of readdirSync(join(root,rel))){if(['node_modules','.git','models','vendor','android','apk'].includes(n)&&!rel)continue;const r=rel?rel+'/'+n:n;if(statSync(join(root,r)).isDirectory())walk(r);else if(statSync(join(root,r)).size<400000)out.push({path:r,bytes:new Uint8Array(readFileSync(join(root,r)))})}};walk('');return out};
const PRIM=JSON.parse(readFileSync(join(root,'integrated/primitive-functions.json'),'utf8'));
const SMALL={ingestLimitBytes:200000,ingestBudgetBytes:1500000,primitives:PRIM};
const sum=a=>a.reduce((x,y)=>x+y,0);
test('mirror: six-bit packing equals her descend on the first 4096 states, and continues past the cap',()=>{const r=new ExecutionAddressResolver();
 for(const n of [0,1,5,100,3071,3072,3073,9000]){const b=new Uint8Array(n).map((_,i)=>(i*131+7)&255);const mine=sixBitStatesAll(b),her=r.descend({name:'x',bytes:b});assert.equal(mine.length,her.sixBitStateCount);assert.deepEqual(mine.slice(0,4096),[...her.sixBitStates]);assert.equal(mine.length,Math.ceil(n*8/6))}});
test('mirror: five-field allocation equals her pipeline fold on a real file',async()=>{
 const b=te.encode('const n=[2,3,4].map(x=>x*2);\nconsole.log(n.join());\nreturn n.length;');const r=await new IntegratedSwarm({executeCode:false}).execute({name:'f.js',bytes:b});
 const states=sixBitStatesAll(b),cnt=DIMENSIONS.map(()=>0);states.forEach((s,i)=>cnt[fieldIndexForPrimitive(s,i)]++);
 const her=Object.fromEntries(r.fold.fields.map(f=>[f.field,f.primitives]));
 const m=await measureContent({name:'f.js',bytes:b});
 for(const [i,d] of DIMENSIONS.entries()){assert.equal(m.perDimension[d].primitives,cnt[i]);assert.equal(her[d],cnt[i],'pipeline field '+d)}});
test('input A, the repo itself: totals reconcile, five dimensions measured, JSON-safe, deterministic',async()=>{
 const files=readTree();assert.ok(files.length>60);const a=await measureContent({name:'you-n-i-verse',files},SMALL),b=await measureContent({name:'you-n-i-verse',files},SMALL);
 assert.equal(a.input.files,files.length);assert.equal(a.input.bytes,sum(files.map(f=>f.bytes.length)));assert.equal(a.totals.sixBitStates,sum(files.map(f=>Math.ceil(f.bytes.length*8/6))));
 assert.equal(sum(DIMENSIONS.map(d=>a.perDimension[d].primitives)),a.totals.sixBitStates);assert.deepEqual(Object.keys(a.perDimension),DIMENSIONS);
 assert.equal(a.perDimension.Space.ingestDerived,true);for(const d of DIMENSIONS){const p=a.perDimension[d];assert.ok(p.statesUsed<=64&&p.stateCoverage<=1&&p.ingestScore>=0&&p.ingestScore<=1);assert.ok(p.wheel)}
 assert.ok(a.coverage.states.distinct>=60&&a.coverage.transitionGrid.of===4096);assert.ok(a.structure.importDepth>=3,'import chain depth '+a.structure.importDepth);assert.ok(a.structure.maxNestingDepth>=5);
 assert.ok(a.totals.languages.js>50);assert.ok(a.huge.cappedFiles>10&&a.huge.statesBeyondCap>0);assert.ok(a.findings.length>=2);
 const strip=r=>{const c=JSON.parse(toJson(r));delete c.elapsedMs;return c};assert.deepEqual(strip(a),strip(b));assert.ok(a.gaps.references.missingCount>=0);assert.ok(a.gaps.primitiveGapTags.length>0&&a.gaps.primitiveGapTags[0].gatesTouched>0,'her primitive gap tags are read when the caller passes primitive-functions.json')});
test('input B, one file over 3 KB: split over five fields past the 4096 cap, ingestor reduces every bit',async()=>{
 const b=new Uint8Array(readFileSync(join(root,'integrated/chart-engine.mjs')));assert.ok(b.length>3072);const m=await measureContent({name:'chart-engine.mjs',bytes:b});
 const states=Math.ceil(b.length*8/6);assert.equal(m.totals.sixBitStates,states);assert.equal(m.huge.statesBeyondCap,states-DESCEND_CAP);assert.equal(m.huge.cappedFiles,1);assert.equal(sum(Object.values(m.huge.splitAcrossDimensions)),states);
 assert.equal(m.totals.ingestPrimitiveBits,b.length*8,'ingestor covers every bit');assert.ok(m.huge.rule.includes('4096'));
 assert.ok(m.gaps.references.missingCount===2&&m.gaps.references.missing.every(x=>x.wants.includes('isohuman')),'vendor imports named as absent from a lone file');
 assert.equal(m.gaps.finisher.skipped,'not an app-shaped input (single file)');
 const small=await measureContent({name:'s.txt',bytes:te.encode('hello')});assert.equal(small.huge.cappedFiles,0)});
test('input C, a zip of a small app: references the structure implies but lacks, finishApp assessment, nothing written',async()=>{
 const zip=writeZip([{path:'app/index.html',data:te.encode('<!doctype html><title>t</title><link rel="stylesheet" href="style.css"><script src="main.js"></script><script src="missing.js"></script><img src="logo.png">')},
  {path:'app/main.js',data:te.encode("import {a} from './lib.js';import {z} from './gone.js';import x from 'three';console.log(a)")},{path:'app/lib.js',data:te.encode('export const a=1')},{path:'app/style.css',data:te.encode('body{background:url(bg.png)}')}]);
 const m=await measureContent({name:'app.zip',bytes:zip});assert.equal(m.input.zip.entries,4);assert.equal(m.input.files,4);
 const want=m.gaps.references.missing.map(x=>x.resolved).sort();assert.deepEqual(want,['app/bg.png','app/gone.js','app/logo.png','app/missing.js']);assert.deepEqual(m.gaps.references.externalSpecifiers,['three']);
 assert.equal(m.structure.importDepth,2,'index.html -> main.js -> lib.js');assert.equal(m.structure.directoryDepth,1);
 assert.ok(Array.isArray(m.gaps.finisher.missingBefore)&&m.gaps.finisher.missingBefore.length>0,JSON.stringify(m.gaps.finisher));assert.equal(m.gaps.finisher.originalFilesUntouched,4,'all four original files preserved');assert.ok(m.gaps.finisher.note.includes('nothing was written'));
 for(const d of DIMENSIONS)assert.ok(m.perDimension[d].hamming1.impliedAbsent>=0)});
test('input D, a binary blob and a folder of different shapes (unrelated to the repo): kinds, binary files, empty file',async()=>{
 const blob=new Uint8Array(70000).map((_,i)=>(i*i+(i>>3))&255);const files=[{path:'blob.bin',bytes:blob},{path:'empty.txt',bytes:new Uint8Array(0)},{path:'doc/readme.md',bytes:te.encode('# hi\n'.repeat(900))},{path:'data.json',bytes:te.encode(JSON.stringify({a:[{b:[{c:[1]}]}]}))}];
 const m=await measureContent(files,SMALL);assert.equal(m.input.files,4);assert.equal(m.totals.sixBitStates,Math.ceil(70000*8/6)+0+Math.ceil(4500*8/6)+Math.ceil(te.encode(JSON.stringify({a:[{b:[{c:[1]}]}]})).length*8/6));
 assert.equal(m.huge.cappedFiles,2);assert.ok(m.files.find(f=>f.path==='blob.bin').ingest);assert.equal(m.structure.maxNestingDepth,6);assert.equal(m.structure.directoryDepth,1)});
test('over the device limits: ingestor is skipped honestly, counts still cover every byte by her rule',async()=>{
 const m=await measureContent({name:'big.bin',bytes:new Uint8Array(300000).map((_,i)=>i&255)},{ingestLimitBytes:100000});assert.equal(m.totals.ingestedFiles,0);assert.deepEqual(m.huge.ingestorSkipped,['big.bin']);assert.equal(sum(Object.values(m.huge.splitAcrossDimensions)),Math.ceil(300000*8/6));assert.equal(m.perDimension.Movement.ingestScore,null)});
test('structure helpers',()=>{assert.equal(nestingDepth('a{b[c(d)]}"{{{{"'),3);assert.deepEqual(references('a.mjs',"import a from './x.js';export * from '../y.mjs';const z=await import('./w.js')",'text'),['./x.js','../y.mjs','./w.js'])});
test('sandbox: measurement never executes input; execution is off by default and the surface blocks network and eval',async()=>{
 globalThis.__measureRan=false;await measureContent({name:'x.js',bytes:te.encode('globalThis.__measureRan=true;fetch("http://example.com")')});assert.equal(globalThis.__measureRan,false);
 const s=sandboxReport();assert.equal(s.execution.probes.find(p=>p.code.startsWith('fetch')).allowed,false);assert.equal(s.execution.probes.find(p=>p.code==='eval("1")').allowed,false);assert.equal(s.execution.probes.find(p=>p.code==='1+1').allowed,true);assert.ok(s.network.includes('no INTERNET')||s.network.includes('INTERNET'));
 assert.equal(JSON.parse(toJson(s)).github.includes('proposals only'),true)});
test('no INTERNET permission in the Android manifest',()=>{const mf=readFileSync(join(root,'../android/app/src/main/AndroidManifest.xml'),'utf8');assert.ok(!/android\.permission\.INTERNET/.test(mf))});
