// First swarm, pure JavaScript: ANY file goes through registration -> state-space placement -> swarm execution.
// Alexis's rules used: the swarm is the agent; PSO = genotype (background tooling, checks, mechanics); GA = phenotype (what shows);
// packs are 3, 2, 1, each pack two bits, three packs make a six-bit hexagram; uncertainty triggers investigation, not rejection;
// Space is not primitive; other languages come later (js, py, css, json adapters below are PLACEHOLDERS, marked as such).
// By default (this module alone) the uploaded file's own code is never run or modified: only its bytes are read and the NSNF swarm runtime runs on the file's event.
// integrated/integrated-swarm.mjs runs the file through Alexis's patches first and passes {executed:true, appCode} here; the bytes are still never modified.
// INVESTIGATING (not settled by Alexis or the spec; see also state-space.mjs and nsnf-runtime.mjs lists):
export const INVESTIGATING=Object.freeze([
 'placement below the address: level 1 uses the registered address (gate via King Wen->Fu Xi map, line as the sub-position) when present; levels 2.. use bytes of the file\'s SHA-256. Alexis has not said how a file maps to nested positions.',
 'with no usable gate, level 1 uses a content-hash byte. This is a minimal stand-in, not a rule from the spec.',
 'dimension comes only from the registered address. Without one the file is held (not placed) and still executes through the encoder; nothing guesses a dimension. A Space-led address is held too, because Space is emergent and cannot be written.',
 'file features (size, entropy, printable/high/zero-byte ratios, text flag, known-type flag) are an engineering choice to give the swarm numbers to read; Alexis has not specified features.',
 'language adapters for js, py, css and json are placeholders: only generic byte features are computed today.',
 'escalation: an uncertain registration/placement starts execution at tier 2 (Spec §17: ambiguity may recruit more components). Which uncertainty should recruit more is not specified.']);
import {SwarmComponent} from './swarm-component.mjs';
import {SwarmPair} from './swarm-nsnf.mjs';
import {sha256} from './sha256.mjs';
import {registerSimlish} from './synthia-registration.mjs';
import {finishApp} from './app-finisher.mjs';
import {missingPlaces} from './registration-swarm.mjs';
import {NsnfRuntime} from './nsnf-runtime.mjs';
import {PRIMITIVE_DIMENSIONS,gateToMacro,macroToGate} from './state-space.mjs';
const te=new TextEncoder();
export const FEATURE_NAMES=Object.freeze(['log-size','entropy','printable','newline-density','high-byte','zero-byte','is-text','known-type']);
const MAGIC=[['zip',[0x50,0x4b,3,4]],['png',[0x89,0x50,0x4e,0x47]],['jpeg',[0xff,0xd8,0xff]],['gif',[0x47,0x49,0x46,0x38]],['pdf',[0x25,0x50,0x44,0x46]],['gzip',[0x1f,0x8b]],['wasm',[0,0x61,0x73,0x6d]],['elf',[0x7f,0x45,0x4c,0x46]]];
// Placeholders: each language adapter is registered so it can be filled in later without changing the swarm. None parses anything yet.
export const LANGUAGE_ADAPTERS=Object.freeze({
 js:{ext:['js','mjs','cjs'],implemented:false,placeholder:true},py:{ext:['py'],implemented:false,placeholder:true},
 css:{ext:['css'],implemented:false,placeholder:true},json:{ext:['json'],implemented:false,placeholder:true}});
export const languageOf=name=>{const e=String(name||'').split('.').pop().toLowerCase();return Object.entries(LANGUAGE_ADAPTERS).find(([,a])=>a.ext.includes(e))?.[0]||null};
const toBytes=b=>b instanceof Uint8Array?b:typeof b==='string'?te.encode(b):b instanceof ArrayBuffer?new Uint8Array(b):Array.isArray(b)?Uint8Array.from(b):null;
export function fileFeatures(name,bytes){
 const b=toBytes(bytes);if(!b)throw new TypeError('bytes required (Uint8Array, ArrayBuffer, string or number array)');
 const n=b.length,hist=new Array(256).fill(0);for(const v of b)hist[v]++;
 let printable=0,newlines=hist[10],high=0;for(let i=0;i<256;i++){if(i===9||i===10||i===13||(i>=32&&i<127))printable+=hist[i];if(i>=128)high+=hist[i]}
 const entropy=n?-hist.reduce((s,c)=>c?s+(c/n)*Math.log2(c/n):s,0):0;
 let text=false;if(n&&hist[0]===0){try{new TextDecoder('utf-8',{fatal:true}).decode(b);text=true}catch{text=false}}
 const magic=MAGIC.find(([,m])=>m.every((v,i)=>b[i]===v))?.[0]||null,kind=magic||(text?'text':n?'binary':'empty'),language=languageOf(name);
 const vec=[Math.min(1,Math.log2(n+1)/32),entropy/8,n?printable/n:0,n?Math.min(1,newlines/n*20):0,n?high/n:0,n?hist[0]/n:0,text?1:0,magic?1:0];
 return {name:String(name||'upload'),size:n,sha256:sha256(b),kind,language,adapter:language?{...LANGUAGE_ADAPTERS[language],note:'placeholder: generic byte features only'}:null,entropy,textDecodable:text,vector:vec,featureNames:FEATURE_NAMES};
}
// ---- PSO genotype for a file: six members in packs of 3, 2, 1 (swarm-component.mjs). Count-of-passes -> pack value is INVESTIGATING there.
export const fileGenotype=new SwarmComponent({id:'file-intake',kind:'PSO',purpose:'Check that any file can be read, hashed, typed and given numbers for the swarm to execute on',
 openQuestions:['Pack-count-to-two-bits mapping and pack order are investigating, not rulings.','Language adapters are placeholders.'],
 members:[
  {id:'has-bytes',check:i=>i.features.size>0},
  {id:'hashed',check:i=>/^[0-9a-f]{64}$/.test(i.features.sha256)},
  {id:'features-finite',check:i=>i.features.vector.length===8&&i.features.vector.every(Number.isFinite)},
  {id:'type-recognised',check:i=>({ok:i.features.kind!=='binary'&&i.features.kind!=='empty',note:'unrecognised '+i.features.kind+' content is kept and investigated, not rejected'})},
  {id:'readable',check:i=>i.features.textDecodable||!!(i.features.kind!=='binary'&&i.features.kind!=='empty')},
  {id:'language-known',check:i=>i.features.language?{ok:true,note:'adapter for '+i.features.language+' is a placeholder'}:{ok:false,note:'no language adapter for this file; generic handling only'}}]});
const phenotypeComponent=new SwarmComponent({id:'file-view',kind:'GA',purpose:'Show what the swarm did with the file',members:[
 {id:'title',check:i=>!!i.visible.title},{id:'placement-label',check:i=>!!i.visible.placement},{id:'six-lines',check:i=>i.visible.lines?.length===6},
 {id:'status-matches',check:i=>i.visible.status===i.genotype.status},{id:'file-identity',check:i=>/^[0-9a-f]{64}$/.test(i.visible.sha256||'')},{id:'execution-reported',check:i=>typeof i.visible.userCodeExecuted==='boolean'}]});
const slug=s=>String(s||'file').toLowerCase().replace(/\.[a-z0-9]+$/,'').replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'')||'file';
const levelBytes=hex=>Uint8Array.from(hex.match(/../g).map(h=>parseInt(h,16)));
// ---- placement into the nested state space (see INVESTIGATING above). Returns provenance for every level.
export function placeFile({features,genotype,address,space}){
 const bytes=levelBytes(features.sha256),notes=[],dim=address?.dimension;
 let dimension=null,dimensionFrom=null;
 if(PRIMITIVE_DIMENSIONS.includes(dim)){dimension=dim;dimensionFrom='registered address'}
 else if(dim==='Space')notes.push('Space is emergent and cannot hold a file directly; file is held, not placed');
 else notes.push(dim?'address dimension "'+dim+'" is not one of the five dimensions; file is held':'no dimension in the address; nothing guesses one, file is held');
 const gate=Number.isInteger(address?.gate)&&address.gate>=1&&address.gate<=64?address.gate:null;
 const macro1=gate?gateToMacro(gate):bytes[1]&63,macro1From=gate?'address.gate (King Wen -> Fu Xi map)':'content hash';
 const line=Number.isInteger(address?.line)&&address.line>=1&&address.line<=space.split?address.line:null;
 const path=[{macro:macro1,sub:line?line-1:bytes[0]%space.split}],from=[{level:1,macro:macro1From,sub:line?'address.line':'content hash'}];
 for(let k=1;k<space.levels;k++){path.push({macro:bytes[2*k]&63,sub:bytes[2*k+1]%space.split});from.push({level:k+1,macro:'content hash',sub:'content hash'})}
 if(!gate)notes.push('no gate in the address: level 1 taken from the content hash');
 if(dimension===null)return {status:'held',dimension:null,path,provenance:from,nodeId:null,notes,label:'held (not placed): '+notes[0]};
 const node=space.ensure(dimension,path);
 const complete=!!gate&&!!line;
 return {status:complete?'placed':'placed-investigating',dimension,dimensionFrom,path,provenance:from,nodeId:node.id,level1_gate:macroToGate(path[0].macro),hexagram_code:node.hexagram_code,notes,
  label:dimension+' · level-1 gate '+macroToGate(path[0].macro)+(gate?'':' (from content)')+' · '+path.length+' levels deep ('+node.id+')'+(complete?'':' · investigating')};
}
const trait=(g,i)=>({title:(i.personalization?.name?i.personalization.name+' · ':'')+i.features.name,placement:i.placement.label,status:g.status,hexagram_code:g.hexagram_code,
 lines:[...g.hexagram_code].map(b=>b==='1'?'solid':'broken'),sha256:i.features.sha256,kind:i.features.kind,language:i.features.language||'unknown',
 execution:i.execution?{tier:i.execution.tier,active:i.execution.n_active,confidence:i.execution.confidence}:null,userCodeExecuted:i.executed===true,kept:true});
export const filePair=()=>new SwarmPair({id:'any-file',genotype:fileGenotype,phenotype:{component:phenotypeComponent,trait}});
const mem=()=>({d:{},getItem(k){return this.d[k]??null},setItem(k,v){this.d[k]=v}});
// input: {name, bytes}. options: personalization, address, source, kernel, storage, runtime, finish, target, env, client, fetchImpl, outbox, persist
export async function processFile(input,{personalization=null,address=null,source=null,kernel=globalThis.AgentomagotchiKernel,storage=mem(),runtime=new NsnfRuntime(),finish=true,target='reality',env={},client=null,fetchImpl,outbox,persist=true,appCode=null,executed=false}={}){
 const bytes=toBytes(input?.bytes);if(!bytes)throw new TypeError('input.bytes required');
 const name=input.name||'upload',fixed=address??personalization?.address??null,features=fileFeatures(name,bytes);
 const uploader=source||(personalization?.name?'uploader:'+personalization.name:'uploaded-file:'+name);
 const pair=filePair(),reasons=[];
 // 1 registration (swarm: PSO checks the full 13-place address; uncertain -> investigating, kept)
 const registration=registerSimlish(storage,{id:appCode||('file-'+slug(name)+'-'+features.sha256.slice(0,8)),version:'sha256:'+features.sha256.slice(0,12),source:uploader,address:fixed,capabilities:['uploaded-file'],kernel});
 if(registration.status!=='registered')reasons.push('registration investigating: '+(registration.investigation?[].concat(registration.investigation.missingPlaces).join(', ')||registration.investigation.failedChecks.join(', '):''));
 // 2 file genotype (PSO): packs 3,2,1 -> six-bit hexagram
 const g=pair.genotype.forward({features});
 if(g.status!=='complete')reasons.push('file genotype investigating: '+g.failed.join(', '));
 // 3 placement in the nested state space
 const placement=placeFile({features,genotype:g,address:fixed,space:runtime.space});
 if(placement.status!=='placed')reasons.push('placement '+placement.status+': '+(placement.notes.join('; ')||'sub-positions content-derived'));
 // 4 execution: the swarm runtime runs on the file's event; ambiguity starts at tier 2 (Spec §17)
 const uncertain=reasons.length>0,seeds=placement.nodeId?[{dimension:placement.dimension,path:placement.path,role:'placement'}]:[];
 const execution=runtime.execute({input:features.vector,seeds,tier:uncertain?2:null,reason:uncertain?'ambiguity: '+reasons.join(' | '):null});
 if(!execution.executed)reasons.push('execution blocked: '+execution.reason);
 // 5 finisher (additive files, translation, original to Supabase when unfinished). Never modifies the uploaded bytes.
 let finisher=null;
 if(finish){const r=await finishApp({name,bytes},{personalization,address:fixed,source:uploader,kernel,env,client,fetchImpl,outbox,target});
  finisher={status:r.report.status,added:r.report.addedFiles,missingBefore:r.report.missingBefore,supabase:r.report.supabase,translation:{target:r.report.translation?.target,status:r.report.translation?.status,note:r.report.translation?.note},zipBytes:r.zip.length}}
 // 6 phenotype (GA): what is shown, expressed from the genotype
 const ph=pair.phenotype.express(g,{features,placement,execution:execution.executed?execution:null,personalization,executed:executed===true});
 const status=reasons.length?'investigating':'complete';
 if(persist)runtime.checkpoint();
 return {status,reasons,file:{name:features.name,size:features.size,sha256:features.sha256,kind:features.kind,language:features.language,adapter:features.adapter},
  registration:{status:registration.status,investigation:registration.investigation},genotype_pso:{packs:g.packs,hexagram_code:g.hexagram_code,status:g.status,failed:g.failed,notes:g.notes},
  placement,execution:execution.executed?{tier:execution.tier,startTier:execution.startTier,n_active:execution.n_active,macs:execution.macs,confidence:execution.confidence,coherence:execution.coherence,
   cache_hit:execution.cache_hit,escalation:execution.escalation,reasons:execution.reasons,space:execution.space,trace:execution.trace,event_id:execution.event_id}:execution,
  phenotype_ga:{visible:ph.visible,status:ph.status,hexagram_code:ph.hexagram_code,failed:ph.failed},finisher,
  safety:{userCodeExecuted:executed===true,originalBytesModified:false},investigating:[...INVESTIGATING],swarm:pair.snapshot()};
}
