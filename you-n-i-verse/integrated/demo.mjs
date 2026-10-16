// Runnable demo: several different files go through ingest -> execute -> fold up -> state commit -> address (-> history at that address).
// Usage: node integrated/demo.mjs                 (built-in sample files)
//        node integrated/demo.mjs path/a path/b   (your own files; code files are run in this thread, see PROVIDER_INVESTIGATING)
import '../vendor/kernel.js';
import {readFile} from 'node:fs/promises';import {basename} from 'node:path';
import {IntegratedSwarm} from './integrated-swarm.mjs';
import {writeZip} from '../zip-lite.mjs';
import {verifyAppCode} from './app-code.mjs';
const te=new TextEncoder();
// DEMO FIXTURE ONLY: this address is the repo's test fixture, not Alexis's address.
const FIXTURE={planetary:'Sun',dimension:'Being',zodiac:'Virgo',gate:6,line:4,color:4,tone:4,base:2,degree:25,minute:59,second:31,arcSecond:92,house:5};
const WASM_ADD=Uint8Array.from([0,0x61,0x73,0x6d,1,0,0,0,1,7,1,0x60,2,0x7f,0x7f,1,0x7f,3,2,1,0,7,7,1,3,0x61,0x64,0x64,0,0,10,9,1,7,0,0x20,0,0x20,1,0x6a,0x0b]);
const SAMPLES=[
 {name:'squares.js',bytes:te.encode('const sq=[1,2,3,4].map(x=>x*x);console.log("squares",sq.join(","));return sq.reduce((a,b)=>a+b,0);')},
 {name:'registered.js',bytes:te.encode('return 6*7;'),address:FIXTURE},
 {name:'escape.js',bytes:te.encode('return process.exit(1);')},
 {name:'module.mjs',bytes:te.encode('export const a = 1;\n')},
 {name:'config.json',bytes:te.encode('{"name":"demo","tiers":[1,2,3]}')},
 {name:'style.css',bytes:te.encode(':root{--c:#333}\nbody{color:var(--c);margin:0}\n@media (min-width:600px){a{color:red}}\n')},
 {name:'page.html',bytes:te.encode('<!doctype html><title>t</title><h1>hi</h1><script>document.title="x"</script>')},
 {name:'notes.md',bytes:te.encode('# Notes\nplain words go here\n')},
 {name:'tool.py',bytes:te.encode('print("hello")\n')},
 {name:'add.wasm',bytes:WASM_ADD},
 {name:'blob.bin',bytes:Uint8Array.from({length:600},(_,i)=>(i*131+7)&255)},
 {name:'site.zip',bytes:writeZip([{path:'main.js',data:'return 1+1;'},{path:'data.json',data:'[1,2]'},{path:'a.css',data:'a{b:c}'}])},
 {name:'squares.js',bytes:te.encode('const sq=[1,2,3,4].map(x=>x*x);console.log("squares",sq.join(","));return sq.reduce((a,b)=>a+b,0);')}];
const files=process.argv.length>2?await Promise.all(process.argv.slice(2).map(async p=>({name:basename(p),bytes:new Uint8Array(await readFile(p))}))):SAMPLES;
const swarm=new IntegratedSwarm();
const rows=[];
for(const f of files){
 const r=await swarm.execute(f,{address:f.address});
 const resolved={address:r.address.resolved.basis==='creator-supplied'?f.address:r.address.executionDerived.address,key:r.address.resolved.key,basis:r.address.resolved.basis};
 rows.push({file:f.name,executedVia:r.execution.engine||r.execution.path,ok:r.execution.ok,result:JSON.stringify(r.execution.returnValue)?.slice(0,34)??'',unresolved:r.execution.unresolved.length,
  appCode:r.app.code,codeVerifies:verifyAppCode(r.app.code,resolved),addressBasis:r.address.resolved.basis,placement:r.swarm.placement.status,foldUpTier:r.swarm.foldUp.tier,historyAtAddress:r.history.entriesAtAddress});
}
console.log('stages (her Execution Pipeline Patch):',(await swarm.execute({name:'x.txt',bytes:te.encode('x')})).stages.map(s=>s.stage).join(' > '));
console.table(rows);
console.log('history addresses:',swarm.history.addresses().length,'entries:',swarm.history.size,'| mesh edges:',swarm.mesh.edges.length,'| runtime events+snapshots committed via checkpoint');
