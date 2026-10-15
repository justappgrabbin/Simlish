// App finisher: take a put-in app (zip, folder manifest, or single file), assess it against a checklist drawn from
// what this repo already has, fill the gaps with ADDITIVE files under simlish/, translate through a target adapter,
// and send the untouched original to Supabase when the app was not finished. Originals are never modified.
// Swarm pair: PSO genotype (the assessment/mechanics) -> GA phenotype (the finished-app report that is shown).
import {SwarmComponent,SwarmPair} from './swarm-nsnf.mjs';
import {sha256} from './sha256.mjs';
import {isZip,readZip,writeZip} from './zip-lite.mjs';
import {registerSimlish} from './synthia-registration.mjs';
import {TARGET_ADAPTERS} from './finisher-targets.mjs';
import {sendOriginal} from './supabase-original.mjs';
const td=new TextDecoder(),te=new TextEncoder();
const has=(files,re)=>files.some(f=>re.test(f.path));
const ENTRY=/(^|\/)(index\.html?|main\.m?js|index\.m?js|app\.m?js|main\.py|index\.php)$/i;
const MANIFEST=/(^|\/)(manifest\.json|package\.json|AndroidManifest\.xml|app\.json|pubspec\.yaml|pyproject\.toml)$/i;
function packageMain(files){const p=files.find(f=>/(^|\/)package\.json$/i.test(f.path));try{return JSON.parse(td.decode(p.data)).main||null}catch{return null}}
export const CHECKLIST=Object.freeze(['entry file','manifest','address registration','scene mapping','reality mapping','original preserved']);
const genotype=new SwarmComponent({id:'finisher',kind:'PSO',purpose:'Assess whether a put-in app is finished against the Simlish checklist',
 members:[
  {id:'entry-file',check:i=>{const own=i.files.filter(f=>!/^simlish\//.test(f.path));return has(i.files,ENTRY)||has(i.files,/(^|\/)simlish\/entry\.html$/)||(own.length===1&&/\.html?$/i.test(own[0].path))||!!packageMain(i.files)}},
  {id:'manifest',check:i=>has(i.files,MANIFEST)},
  {id:'address-registration',check:i=>has(i.files,/(^|\/)simlish\/registration\.json$/)},
  {id:'scene-mapping',check:i=>has(i.files,/(^|\/)simlish\/scene\.json$/)},
  {id:'reality-mapping',check:i=>has(i.files,/(^|\/)simlish\/reality\.json$/)},
  {id:'original-preserved',check:i=>!!i.originalSha256}]});
const phenotypeComponent=new SwarmComponent({id:'finisher-view',kind:'GA',purpose:'Show the finished-app report',members:[
 {id:'title',check:i=>!!i.visible.title},{id:'summary',check:i=>!!i.visible.summary},{id:'lists-added',check:i=>Array.isArray(i.visible.added)},
 {id:'original-hash',check:i=>/^[0-9a-f]{64}$/.test(i.visible.originalSha256||'')},{id:'status-matches',check:i=>i.visible.assessment===i.genotype.status},{id:'targets-listed',check:i=>Array.isArray(i.visible.targets)}]});
const trait=(g,i)=>({title:(i.personalization?.name?i.personalization.name+' · ':'')+'Finished '+i.appName,
 summary:g.status==='complete'?'The app already had every piece; nothing was added.':'The app was missing: '+g.failed.join(', ')+'. Pieces were added beside your files, never over them.',
 assessment:g.status,added:i.addedPaths||[],originalSha256:i.originalSha256,targets:Object.values(TARGET_ADAPTERS).map(t=>({id:t.id,implemented:t.implemented}))});
export const finisherPair=()=>new SwarmPair({id:'finisher',genotype,phenotype:{component:phenotypeComponent,trait}});
const slug=s=>String(s||'app').toLowerCase().replace(/\.[a-z0-9]+$/,'').replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'')||'app';
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function intake(input){
 const bytes=input.bytes instanceof Uint8Array?input.bytes:input.bytes?new Uint8Array(input.bytes):null;
 if(input.files)return {files:input.files.map(f=>({path:f.path,data:typeof f.data==='string'?te.encode(f.data):f.data})),originalBytes:bytes,kind:'folder-manifest',
   originalForUpload:bytes??writeZip(input.files.map(f=>({path:f.path,data:f.data})))};
 if(!bytes)throw new TypeError('Provide bytes (zip or single file) or files (folder manifest)');
 if(isZip(bytes))return {files:await readZip(bytes),originalBytes:bytes,kind:'zip',originalForUpload:bytes};
 return {files:[{path:input.name||'upload',data:bytes}],originalBytes:bytes,kind:'single-file',originalForUpload:bytes};
}
// input: {name, bytes|files}; options: {personalization, address, source, kernel, env, client, fetchImpl, outbox, target}
export async function finishApp(input,{personalization=null,address=null,source=null,kernel=globalThis.AgentomagotchiKernel,env={},client=null,fetchImpl,outbox,target='reality'}={}){
 const name=input.name||'upload',taken=await intake(input),files=taken.files;
 const originalSha256=sha256(taken.originalForUpload),pair=finisherPair();
 const before=pair.genotype.forward({files,originalSha256});
 const fixedAddress=address??personalization?.address??null,uploaderSource=source||(personalization?.name?'uploader:'+personalization.name:'uploaded-app:'+name);
 const taken2=new Set(files.map(f=>f.path)),added=[];
 const add=(path,obj)=>{let p=path;if(taken2.has(p))p=path.replace(/(\.[a-z]+)$/,'.simlish$1');taken2.add(p);added.push({path:p,data:typeof obj==='string'?te.encode(obj):te.encode(JSON.stringify(obj,null,2))});return p};
 const failed=new Set(before.failed),surfaces=files.filter(f=>/\.(html?|m?js|py|php)$/i.test(f.path)).map(f=>f.path).slice(0,50);
 const entryGuess=(files.find(f=>ENTRY.test(f.path))||files.find(f=>/\.html?$/i.test(f.path)))?.path||packageMain(files);
 if(failed.has('entry-file')){add('simlish/entry.html','<!doctype html><meta charset="utf-8"><title>'+esc(name)+'</title><h1>'+esc(name)+'</h1><p>Generated launcher stub: this upload had no entry file. Original files are listed below, unchanged.</p><ul>'+files.map(f=>'<li>'+esc(f.path)+'</li>').join('')+'</ul>')}
 if(failed.has('manifest')){add('simlish/manifest.json',{name,entry:entryGuess||'simlish/entry.html',generatedBy:'simlish-app-finisher',generated:true,fileCount:files.length,personalization:personalization?{name:personalization.name,worldTheme:personalization.worldTheme,target:personalization.target}:null})}
 // registration: the uploader's full 13-place address, or investigating with what is missing (no invented address)
 const mem={d:{},getItem(k){return this.d[k]??null},setItem(k,v){this.d[k]=v}};
 const reg=registerSimlish(mem,{id:slug(name)+'-'+originalSha256.slice(0,8),version:'uploaded-'+originalSha256.slice(0,8),source:uploaderSource,address:fixedAddress,capabilities:['uploaded-app'],kernel});
 if(failed.has('address-registration'))add('simlish/registration.json',{registration:reg,note:reg.status==='registered'?'Full 13-place address registered.':'Kept as investigating; the uploader\'s address is not complete yet.'});
 const adapter=TARGET_ADAPTERS[target]||TARGET_ADAPTERS.reality,app={name,files,surfaces,address:fixedAddress,personalization,source:uploaderSource,kernel};
 const translation=await adapter.translate(app);
 if(failed.has('scene-mapping'))add('simlish/scene.json',translation.scene??{status:translation.status,note:translation.note});
 if(failed.has('reality-mapping'))add('simlish/reality.json',{target:translation.target,readiness:translation.readiness??null,execution:translation.execution??null,status:translation.status??'mapped-not-rendered',note:translation.note});
 if(personalization)add('simlish/personalization.json',personalization);
 const finishedFiles=[...files,...added];
 const after=pair.genotype.forward({files:finishedFiles,originalSha256});
 const out=pair.phenotype.express(after,{files:finishedFiles,appName:name,personalization,originalSha256,addedPaths:added.map(a=>a.path)});
 const unfinished=before.status!=='complete';
 let supabase={status:'not-needed',note:'App was already finished'};
 if(unfinished)supabase=await sendOriginal({name:input.name||'upload',bytes:taken.originalForUpload,contentType:taken.kind==='zip'?'application/zip':'application/octet-stream',meta:{kind:taken.kind,missing:before.failed,uploader:personalization?.name??null}},{env,client,fetchImpl,outbox});
 const report={app:name,intakeKind:taken.kind,originalSha256,assessed:CHECKLIST,missingBefore:before.failed,missingAfter:after.failed,
  swarm:{genotype_pso:{before:{hexagram_code:before.hexagram_code,packs:before.packs},after:{hexagram_code:after.hexagram_code,packs:after.packs}},phenotype_ga:{visible:out.visible,hexagram_code:out.hexagram_code}},
  addedFiles:[...added.map(a=>a.path),'simlish/finisher-report.json'],originalFilesUntouched:files.length,registration:{status:reg.status,investigation:reg.investigation},translation,personalization:personalization?{name:personalization.name,worldTheme:personalization.worldTheme,target:personalization.target,status:personalization.status}:null,supabase:{status:supabase.status,sha256:supabase.sha256,path:supabase.path,note:supabase.note},
  status:unfinished?(after.status==='complete'&&reg.status==='registered'?'finished':'finished-with-investigating'):'already-finished'};
 added.push({path:'simlish/finisher-report.json',data:te.encode(JSON.stringify(report,null,2))});
 const all=[...files,...added];
 return {report,files:all,zip:writeZip(all.map(f=>({path:f.path,data:f.data}))),originalBytes:taken.originalForUpload,swarm:pair.snapshot()};
}
