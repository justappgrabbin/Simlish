import test from 'node:test';import assert from 'node:assert/strict';
import './vendor/kernel.js';
import {finishApp} from './app-finisher.mjs';
import {writeZip,readZip,isZip} from './zip-lite.mjs';
import {sha256} from './sha256.mjs';
import {personalize} from './personalize.mjs';
import {memoryOutbox} from './supabase-original.mjs';
import {TARGET_ADAPTERS} from './finisher-targets.mjs';
const te=new TextEncoder();
const address={planetary:'Sun',dimension:'Being',zodiac:'Virgo',gate:6,line:4,color:4,tone:4,base:2,degree:25,minute:59,second:31,arcSecond:92,house:5};
const mem=()=>({d:{},getItem(k){return this.d[k]??null},setItem(k,v){this.d[k]=v}});
const pers=()=>personalize({name:'Alexis',worldTheme:'forest',address,target:'reality'},mem());
const fixtures={
 'html zip':{name:'site.zip',bytes:writeZip([{path:'index.html',data:'<h1>hi</h1>'},{path:'style.css',data:'h1{}'}])},
 'node zip':{name:'tool.zip',bytes:writeZip([{path:'package.json',data:'{"name":"t","main":"lib/x.js"}'},{path:'lib/x.js',data:'console.log(1)'}])},
 'no-entry zip':{name:'data.zip',bytes:writeZip([{path:'notes.txt',data:'just notes'},{path:'img.bin',data:new Uint8Array([1,2,3,255])}])},
 'single html':{name:'page.html',bytes:te.encode('<p>x</p>')},
 'single binary':{name:'blob.dat',bytes:new Uint8Array([0,255,1,2])},
 'folder manifest':{name:'folder-app',files:[{path:'main.js',data:'export default 1'},{path:'README.md',data:'r'}]}};
for(const [label,input] of Object.entries(fixtures)){
 test('finish '+label+': originals untouched, additive files, original queued when unfinished',async()=>{
  const outbox=memoryOutbox(),p=pers(),before=new Map((input.files||[]).map(f=>[f.path,String(f.data)]));
  const r=await finishApp(input,{personalization:p,address,kernel:globalThis.AgentomagotchiKernel,env:{},outbox});
  assert.ok(['finished','finished-with-investigating'].includes(r.report.status));
  assert.ok(r.report.addedFiles.every(f=>f.startsWith('simlish/')));
  assert.ok(r.report.addedFiles.includes('simlish/finisher-report.json')&&r.report.addedFiles.includes('simlish/personalization.json'));
  assert.deepEqual(r.report.missingAfter.filter(x=>x!=='original-preserved'),[]);
  const zipFiles=await readZip(r.zip);assert.equal(zipFiles.length,r.files.length);
  if(input.files)for(const [path,data] of before)assert.equal(new TextDecoder().decode(zipFiles.find(f=>f.path===path).data),data);
  if(input.bytes&&isZip(input.bytes)){for(const o of await readZip(input.bytes))assert.deepEqual(zipFiles.find(f=>f.path===o.path).data,o.data)}
  else if(input.bytes)assert.deepEqual(zipFiles.find(f=>f.path===input.name).data,input.bytes);
  assert.equal(r.report.supabase.status,'pending-supabase');assert.equal((await outbox.list()).length,1);
  assert.equal(sha256(r.originalBytes),r.report.originalSha256);
  assert.equal(r.report.registration.status,'registered');assert.equal(r.report.translation.scene.status,'mapped');
  assert.equal(r.report.translation.execution.status,'blocked');
  assert.equal(r.report.swarm.genotype_pso.after.packs.join(),'3,2,1');
 });
}
test('no address given: registration stays investigating and nothing is invented',async()=>{
 const r=await finishApp(fixtures['html zip'],{personalization:personalize({name:'A'},mem()),env:{},outbox:memoryOutbox()});
 assert.equal(r.report.registration.status,'investigating');assert.equal(r.report.translation.scene.status,'investigating');assert.equal(r.report.status,'finished-with-investigating');
});
test('an existing simlish/ file is never overwritten',async()=>{
 const r=await finishApp({name:'x',files:[{path:'index.html',data:'a'},{path:'simlish/manifest.json',data:'MINE'},{path:'manifest.json',data:'{}'},{path:'simlish/scene.json',data:'SCENE'}]},{address,env:{},outbox:memoryOutbox()});
 assert.equal(new TextDecoder().decode(r.files.find(f=>f.path==='simlish/scene.json').data),'SCENE');
});
test('game and 2d targets are stubs that say not-implemented; reality is implemented',async()=>{
 assert.equal(TARGET_ADAPTERS.reality.implemented,true);
 for(const t of ['game','2d']){assert.equal(TARGET_ADAPTERS[t].implemented,false);const r=await finishApp(fixtures['single html'],{target:t,address,env:{},outbox:memoryOutbox()});assert.equal(r.report.translation.status,'not-implemented')}
});
test('unthemed or unsupported world theme is investigating, not rejected',async()=>{
 const r=await finishApp(fixtures['html zip'],{personalization:personalize({name:'A',worldTheme:'Dungeons and Dragons',address},mem()),address,env:{},outbox:memoryOutbox()});
 assert.equal(r.report.translation.scene.status,'investigating');assert.match(r.report.translation.scene.reason,/Dungeons and Dragons/);
});
