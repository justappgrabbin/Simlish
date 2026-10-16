// Measurement by dimension: step 1 of the content swarm. A continuation of the existing pipeline, nothing new beside it.
// EXISTING, used as-is: OntologicalFileIngestor (exact-byte reduction, five dimension scores, Space derived), fileFeatures (kind/language/adapter), ExecutionAddressResolver.descend (six-bit states),
//   the pipeline's five-field placement rule (state+index)%5 (spine/execution-pipeline.mjs fieldIndexForPrimitive, mirrored below and tested against the pipeline), wheelReadout, hammingNeighbors/macroToGate (state-space.mjs),
//   primitive-functions.json gap tags, finishApp (assess only; its added files are not written anywhere), providers.guardJavaScript (sandbox).
// NEW, derived (marked in the report): reading the packing without descend's 4096 slice, the 64x64 transition grid coverage from her doc ("4096-state interaction grid"), nesting depth, static reference resolution, import-graph depth.
// Measurement is read-only: bytes are never executed and never written.
import {OntologicalFileIngestor} from './ontological/ingestion/ontological-file-ingestor.mjs';
import {fileFeatures} from '../any-file-swarm.mjs';
import {ExecutionAddressResolver} from './spine/execution-address-resolver.mjs';
import {DIMENSIONS,macroToGate,hammingNeighbors} from '../state-space.mjs';
import {wheelReadout} from './wheels.mjs';
import {isZip,readZip} from '../zip-lite.mjs';
import {finishApp} from '../app-finisher.mjs';
import {sha256} from '../sha256.mjs';
import {guardJavaScript} from './providers.mjs';
export const SCHEMA='synthia-measurement/1';
export const DESCEND_CAP=4096; // inherited from ExecutionAddressResolver.descend (line 87); her files define no way past it
export const DEFAULTS=Object.freeze({ingestLimitBytes:1<<20,ingestBudgetBytes:6<<20,finishLimitBytes:5<<20,finishLimitFiles:400,filesListed:60});
const td=new TextDecoder('utf-8',{fatal:false});
// same MSB-first six-bit packing as ExecutionAddressResolver.descend, without its slice(0,4096) (tested equal on the first 4096)
export function sixBitStatesAll(bytes){const out=[];let buf=0,n=0;for(const b of bytes){buf=(buf<<8)|b;n+=8;while(n>=6){const s=n-6;out.push((buf>>>s)&63);n-=6;buf&=n?((1<<n)-1):0}}if(n)out.push((buf<<(6-n))&63);return out}
export const fieldIndexForPrimitive=(state,index)=>(Number(state)+Number(index))%DIMENSIONS.length; // her pipeline rule, "runtime placement rule only"
const ext=p=>{const m=/\.([A-Za-z0-9]+)$/.exec(p);return m?m[1].toLowerCase():''};
const norm=p=>{const out=[];for(const s of String(p).replace(/\\/g,'/').split('/')){if(!s||s==='.')continue;if(s==='..')out.pop();else out.push(s)}return out.join('/')};
const dirOf=p=>{const i=p.lastIndexOf('/');return i<0?'':p.slice(0,i)};
// ---------- input -> file list ----------
export async function expandInput(input){
 if(Array.isArray(input))return {name:'folder',files:input.map(f=>({path:norm(f.path||f.name),bytes:f.bytes}))};
 if(input&&Array.isArray(input.files))return {name:input.name||'folder',files:input.files.map(f=>({path:norm(f.path||f.name),bytes:f.bytes}))};
 const bytes=input.bytes instanceof Uint8Array?input.bytes:new Uint8Array(input.bytes);
 if(isZip(bytes)){const z=await readZip(bytes);return {name:input.name||'archive.zip',zip:{bytes:bytes.length,entries:z.length},files:z.map(e=>({path:norm(e.path),bytes:e.data}))}}
 return {name:input.name||'upload',files:[{path:norm(input.name||'upload'),bytes}]};
}
// ---------- structure (NEW, derived) ----------
export function nestingDepth(text){let d=0,m=0,q=null,esc=false;for(let i=0;i<text.length;i++){const c=text[i];if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'||c==='['||c==='(')m=Math.max(m,++d);else if(c==='}'||c===']'||c===')')d=Math.max(0,d-1)}return m}
export function references(path,text,kind){const refs=[],e=ext(path);
 if(['js','mjs','cjs','ts','tsx','jsx'].includes(e)){for(const m of text.matchAll(/(?:^|[\s;])(?:import|export)\s+(?:[^'"`;]*?\s+from\s+)?['"]([^'"]+)['"]/g))refs.push(m[1]);for(const m of text.matchAll(/(?:\bimport\s*\(|\brequire\s*\()\s*['"]([^'"]+)['"]/g))refs.push(m[1])}
 else if(['html','htm'].includes(e)){for(const m of text.matchAll(/\b(?:src|href)\s*=\s*["']([^"'#?]+)["']/gi))refs.push(m[1])}
 else if(e==='css'){for(const m of text.matchAll(/(?:@import\s+(?:url\()?|url\()\s*["']?([^"')]+)["']?/gi))refs.push(m[1])}
 return refs}
export function resolveRef(from,spec,paths){if(/^(?:[a-z][a-z0-9+.-]*:|\/\/|data:)/i.test(spec))return {external:spec};if(!spec.startsWith('.')&&!spec.startsWith('/')&&!/\.[a-z0-9]+$/i.test(spec))return {external:spec};
 const base=spec.startsWith('/')?norm(spec):norm(dirOf(from)+'/'+spec);const tries=[base,base+'.js',base+'.mjs',base+'/index.js',base+'/index.mjs',base+'/index.html'];
 const hit=tries.find(t=>paths.has(t));return hit?{path:hit}:{missing:base}}
// ---------- one file ----------
async function measureFile({path,bytes},ingestor,opts,paths){
 const f=fileFeatures(path,bytes),states=sixBitStatesAll(bytes),perField=DIMENSIONS.map(()=>new Uint8Array(64)),cnt=DIMENSIONS.map(()=>0),seen=new Uint8Array(64),trans=new Uint8Array(4096);let h1=0,pairs=0;
 for(let i=0;i<states.length;i++){const s=states[i],fi=fieldIndexForPrimitive(s,i);cnt[fi]++;perField[fi][s]=1;seen[s]=1;if(i){const p=states[i-1];trans[p*64+s]=1;pairs++;let x=p^s,n=0;while(x){n+=x&1;x>>=1}if(n===1)h1++}}
 let ing=null,ingestNote=null;
 if(bytes.length<=opts.ingestLimitBytes&&bytes.length>0&&opts._budget.left>=bytes.length){opts._budget.left-=bytes.length;try{const r=await ingestor.ingest({name:path,bytes},{supersede:true});ing={dimensions:Object.fromEntries(Object.entries(r.decomposition.dimensions).map(([k,v])=>[k,{score:v.score,derived:!!v.derived}])),primitiveCount:r.decomposition.primitiveCount,completeReduction:!!r.decomposition.completeMathematicalReduction,exactReconstruction:!!r.reconstruction.exactAvailable,address:r.combinatorialAddress,directionalBalance:r.decomposition.directionalBalance??null}}catch(e){ingestNote='ingest error: '+e.message}}
 else if(bytes.length>0)ingestNote=bytes.length>opts.ingestLimitBytes?`ingest skipped: ${bytes.length} bytes is over the per-file limit ${opts.ingestLimitBytes} set for this device; counts use her six-bit allocation rule over all bytes`:`ingest skipped: the ${opts.ingestBudgetBytes}-byte ingest budget for this run is used up; counts use her six-bit allocation rule over all bytes`;
 let text=null,depth=0,refs=[];if(f.textDecodable&&bytes.length<=(8<<20)){text=td.decode(bytes);depth=nestingDepth(text);refs=references(path,text,f.kind)}
 const lines=text?text.split('\n').length:0;
 return {path,size:bytes.length,sha256:f.sha256,kind:f.kind,language:f.language,adapterImplemented:f.adapter?.implemented??null,entropy:+f.entropy.toFixed(4),
  states:states.length,cappedByDescend:Math.max(0,states.length-DESCEND_CAP),fieldCounts:cnt,usedPerField:perField,seen,trans,h1,pairs,ingest:ing,ingestNote,depth,lines,refs,text:undefined};
}
const AGG0=()=>({files:0,bytes:0,states:0,cnt:DIMENSIONS.map(()=>0),used:DIMENSIONS.map(()=>new Uint8Array(64)),seen:new Uint8Array(64),trans:new Uint8Array(4096),h1:0,pairs:0});
// ---------- the report ----------
export async function measureContent(input,options={}){
 const opts={...DEFAULTS,...options},t0=Date.now(),src=await expandInput(input),paths=new Set(src.files.map(f=>f.path)),ingestor=new OntologicalFileIngestor();
 opts._budget={left:opts.ingestBudgetBytes};const per=[];for(const f of src.files)per.push(await measureFile(f,ingestor,opts,paths));
 const agg=AGG0(),byKind={},byLang={},scores=DIMENSIONS.map(()=>0);let scoreW=0,primitiveBits=0,ingestedFiles=0,capped=0,maxDepth=0,totalBytes=0;
 for(const m of per){agg.files++;agg.bytes+=m.size;agg.states+=m.states;agg.h1+=m.h1;agg.pairs+=m.pairs;capped+=m.cappedByDescend;maxDepth=Math.max(maxDepth,m.depth);totalBytes+=m.size;
  m.fieldCounts.forEach((c,i)=>{agg.cnt[i]+=c;for(let s=0;s<64;s++)if(m.usedPerField[i][s])agg.used[i][s]=1});for(let s=0;s<64;s++)if(m.seen[s])agg.seen[s]=1;for(let t=0;t<4096;t++)if(m.trans[t])agg.trans[t]=1;
  byKind[m.kind]=(byKind[m.kind]||0)+1;if(m.language)byLang[m.language]=(byLang[m.language]||0)+1;
  if(m.ingest){ingestedFiles++;primitiveBits+=m.ingest.primitiveCount;DIMENSIONS.forEach((d,i)=>{scores[i]+=(m.ingest.dimensions[d]?.score??0)*m.size});scoreW+=m.size}}
 const total=agg.cnt.reduce((a,b)=>a+b,0)||1;
 // references: what the structure points at but the input lacks
 const missing=[],external=new Set(),edges=new Map();
 for(const f of src.files){const m=per.find(x=>x.path===f.path);for(const r of m.refs){const x=resolveRef(f.path,r,paths);if(x.external)external.add(x.external);else if(x.missing)missing.push({from:f.path,wants:r,resolved:x.missing});else{(edges.get(f.path)||edges.set(f.path,new Set()).get(f.path)).add(x.path)}}}
 const depthMemo=new Map(),visiting=new Set(),chain=p=>{if(depthMemo.has(p))return depthMemo.get(p);if(visiting.has(p))return 0;visiting.add(p);let d=0;for(const q of edges.get(p)||[])d=Math.max(d,1+chain(q));visiting.delete(p);depthMemo.set(p,d);return d};
 let importDepth=0;for(const p of paths)importDepth=Math.max(importDepth,chain(p));
 const pathDepth=Math.max(0,...src.files.map(f=>f.path.split('/').length-1));
 // gaps from existing logic
 const prim=await loadPrimitives(opts),gateGap=new Map(),hamming={};
 DIMENSIONS.forEach((d,i)=>{const used=[...agg.used[i].keys()].filter(s=>agg.used[i][s]),set=new Set(used),implied=new Map();
  for(const s of used)for(const n of hammingNeighbors(s,1))if(!set.has(n.code))implied.set(n.code,(implied.get(n.code)||0)+1);
  hamming[d]={used:used.length,of:64,impliedAbsent:implied.size,top:[...implied.entries()].sort((a,b)=>b[1]-a[1]||a[0]-b[0]).slice(0,6).map(([code,presentNeighbors])=>({state:code,gate:macroToGate(code),id:'gate.'+macroToGate(code),presentNeighbors}))}});
 if(prim)for(let s=0;s<64;s++)if(agg.seen[s]){const p=prim.get('gate.'+macroToGate(s));for(const g of p?.gap||[]){const e=gateGap.get(g)||{tag:g,gates:[]};e.gates.push(macroToGate(s));gateGap.set(g,e)}}
 const adapterGaps={};for(const m of per)if(m.adapterImplemented===false&&m.language)adapterGaps[m.language]=(adapterGaps[m.language]||0)+1;
 const appLike=src.files.length>=2||src.files.some(f=>/(^|\/)index\.html?$/i.test(f.path));let finisher=null;
 if(appLike&&agg.bytes<=opts.finishLimitBytes&&src.files.length<=opts.finishLimitFiles){try{const r=await finishApp({name:src.name,files:src.files.map(f=>({path:f.path,data:f.bytes}))},{env:{}});finisher={assessed:r.report.assessed,missingBefore:r.report.missingBefore,missingAfter:r.report.missingAfter,wouldAdd:r.report.addedFiles,originalFilesUntouched:r.report.originalFilesUntouched,note:'assessment only: nothing was written'}}catch(e){finisher={error:String(e.message)}}}
 else finisher={skipped:appLike?'over the size/file limit for this device':'not an app-shaped input (single file)'};
 const transCount=agg.trans.reduce((a,b)=>a+b,0),seenCount=agg.seen.reduce((a,b)=>a+b,0);
 const perDimension=Object.fromEntries(DIMENSIONS.map((d,i)=>{const macros=[];for(let s=0;s<64;s++)if(agg.used[i][s])macros.push(s);
  return [d,{primitives:agg.cnt[i],bits:agg.cnt[i]*6,share:+(agg.cnt[i]/total).toFixed(5),ingestScore:scoreW?+(scores[i]/scoreW).toFixed(5):null,ingestDerived:d==='Space',
   statesUsed:macros.length,stateCoverage:+(macros.length/64).toFixed(4),gatesUsed:macros.map(macroToGate).sort((a,b)=>a-b),wheel:wheelReadout(d,macros),hamming1:hamming[d]}]}));
 return {schema:SCHEMA,input:{name:src.name,files:src.files.length,bytes:totalBytes,zip:src.zip??null,sha256:sha256(new Uint8Array(new TextEncoder().encode(per.map(m=>m.path+':'+m.sha256).sort().join('\n'))))},
  totals:{sixBitStates:agg.states,bitsAllocatedByRule:agg.states*6,ingestedFiles,ingestPrimitiveBits:primitiveBits,kinds:byKind,languages:byLang},
  kinds:{bit:totalBytes*8,sixBitState:agg.states,line:agg.states*6,trigram:agg.states*2,gateStatesDistinct:seenCount,transitionsObserved:agg.pairs,transitionsHamming1:agg.h1,
   note:'bit/line/trigram/state counts are exact for all bytes; her primitive kinds (gate, line, color, tone, base...) are listed in primitive-functions.json'},
  perDimension,
  coverage:{states:{distinct:seenCount,of:64,fraction:+(seenCount/64).toFixed(4)},transitionGrid:{distinct:transCount,of:4096,fraction:+(transCount/4096).toFixed(4),basis:'64 x 64 interaction grid (her doc); NEW, derived measure'},hamming1Fraction:agg.pairs?+(agg.h1/agg.pairs).toFixed(4):null},
  structure:{maxNestingDepth:maxDepth,importDepth,directoryDepth:pathDepth,label:'NEW, derived: bracket nesting, static import/reference graph longest chain, folder depth'},
  gaps:{hamming1:hamming,primitiveGapTags:[...gateGap.values()].sort((a,b)=>b.gates.length-a.gates.length).slice(0,10).map(g=>({tag:g.tag,gatesTouched:g.gates.length,gates:g.gates.slice(0,12)})),
   references:{missing:missing.slice(0,50),missingCount:missing.length,externalSpecifiers:[...external].slice(0,30),externalCount:external.size},languageAdaptersNotImplemented:adapterGaps,finisher},
  huge:{descendCap:DESCEND_CAP,cappedFiles:per.filter(m=>m.cappedByDescend>0).length,statesBeyondCap:capped,
   rule:'her ExecutionAddressResolver.descend keeps 4096 six-bit states (spine line 87) and no file of hers defines a way past it, so execution/pipeline stays at 4096. Measurement does not need the cap: the ingestor reduces every bit, and the same five-field rule (state+index)%5 is applied here to every state.',
   splitAcrossDimensions:Object.fromEntries(DIMENSIONS.map((d,i)=>[d,agg.cnt[i]])),ingestorSkipped:per.filter(m=>m.ingestNote&&/skipped/.test(m.ingestNote)).map(m=>m.path)},
  files:per.slice().sort((a,b)=>b.size-a.size).slice(0,opts.filesListed).map(m=>({path:m.path,size:m.size,kind:m.kind,language:m.language,states:m.states,beyondCap:m.cappedByDescend,fieldCounts:Object.fromEntries(DIMENSIONS.map((d,i)=>[d,m.fieldCounts[i]])),ingest:m.ingest?{scores:Object.fromEntries(Object.entries(m.ingest.dimensions).map(([k,v])=>[k,+v.score.toFixed(4)])),primitiveCount:m.ingest.primitiveCount,address:m.ingest.address}:null,note:m.ingestNote})),
  findings:findings(perDimension,agg,per),
  sandbox:sandboxReport(),
  labels:{derived:['sixBitStatesAll without the 4096 slice','transition grid coverage','nesting/import/directory depth','static reference resolution','per-dimension shares by her (state+index)%5 rule'],existing:['OntologicalFileIngestor scores and primitive counts','fileFeatures','descend packing','wheelReadout','hammingNeighbors','finishApp assessment','primitive-functions gap tags']},
  elapsedMs:Date.now()-t0};
}
// primitive-functions.json is passed in by the caller (the app server already holds it): this module has no fetch and no file access
async function loadPrimitives(opts){return opts.primitives?new Map(opts.primitives.primitives.map(p=>[p.id,p])):null}
// ---------- sandbox, from her existing mechanisms ----------
export function sandboxReport(){const NET='fetch',SOCK='WebSocket',probes=[NET+'("http://x")','new '+SOCK+'("x")','eval("1")','while(true){}','process.exit()','1+1'].map(s=>({code:s,allowed:guardJavaScript(s).ok}));
 return {measurement:'read-only: input bytes are never executed or modified (the report never calls an execution surface)',
  execution:{default:'off. Execution only when asked: IntegratedSwarm({executeCode:true}) / execute(...,{executeCode:true}); default and the app tools use executeCode:false',surface:'providers.mjs runJavaScript: same-thread AsyncFunction, 2500 ms timeout, console captured, 15 browser/Node globals shadowed to undefined (fetch, process, window, Worker...), guardJavaScript blocks eval/import()/endless loops/network names',probes,
   limit:'same thread, not a security boundary (her providers.mjs says so; module syntax is not run). A hard boundary would need a worker or process, which is excluded here.'},
  writes:'only to the virtual Workspace of swarm-loop.mjs: cleaned paths, hash-chained audit log, protected originals refuse writes',
  network:'none: the APK has no INTERNET permission and these modules contain no fetch to a remote host',
  github:'proposals only: no push code exists in the app; a push happens only when Alexis allows it'};}
function findings(pd,agg,per){const out=[],sh=Object.values(pd).map(d=>d.share);
 if(Math.max(...sh)-Math.min(...sh)<0.02)out.push('Her allocation rule (state+index)%5 spreads primitives almost evenly over the five fields whatever the content ('+sh.map(x=>(x*100).toFixed(1)+'%').join(' / ')+'), so primitive counts per field do not distinguish content. The ingestor\'s five dimension scores (ingestScore) are the content-dependent per-dimension measure; Space is derived from the other four, as her ingestor states.');
 const lowCov=Object.entries(pd).filter(([,d])=>d.stateCoverage<0.9).map(([k])=>k);if(lowCov.length)out.push('Fields with under 90% of the 64 states used: '+lowCov.join(', ')+' — the hamming1 lists show which neighbours are implied but absent.');
 if(per.some(m=>m.cappedByDescend>0))out.push(per.filter(m=>m.cappedByDescend>0).length+' file(s) exceed her 4096 six-bit-state cap (about 3 KB): they are measured completely here, but her execution pipeline would see only the first 4096 states.');
 return out}
export const toJson=r=>JSON.stringify(r,null,1);
