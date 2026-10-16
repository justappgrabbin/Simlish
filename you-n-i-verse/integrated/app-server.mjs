// The server that lives inside the app. Pure JavaScript, in-process: no socket, no backend, no worker, no network. In the Android shell it runs in the WebView page (assets are served by WebViewAssetLoader);
// in Node tests it runs the same code with file loaders injected. It gives the person (chat page) and the swarm (self-assembly loop) one set of tools and one progress feed.
import {AuditLog,Workspace,ProtectedOriginals,SwarmLoop,Heartbeat,CONSTITUTION,shaOf,ProtectedError} from './swarm-loop.mjs';
import {buildPlan} from './assembly-plan.mjs';
import {Trident,tokenize,detokenize,CONFIG as TRIDENT} from './trident-js.mjs';
import * as ARC from './arc-engine.mjs';
import {finishApp} from '../app-finisher.mjs';
const te=new TextEncoder();
const b64=u8=>{let s='';for(let i=0;i<u8.length;i+=0x8000)s+=String.fromCharCode(...u8.subarray(i,i+0x8000));return btoa(s)};
const unb64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
export const TOOLS=Object.freeze([
 {name:'primitives.get',route:'POST /api/tool/primitives.get',description:'one primitive (symbol, form, functions, open gaps) plus its qualities, by id (e.g. gate.1, color.3, dim.movement)'},
 {name:'primitives.search',route:'POST /api/tool/primitives.search',description:'search primitive forms/ids by text, optionally by kind'},
 {name:'primitives.sources',route:'POST /api/tool/primitives.sources',description:'the separate sources layer for one primitive id; shown only when asked'},
 {name:'address.fromLongitude',route:'POST /api/tool/address.fromLongitude',description:'her sentence-engine address (gate, line, color, tone, base, DMS, zodiac, house, lattice position 1..69120) from a longitude'},
 {name:'address.lattice',route:'POST /api/tool/address.lattice',description:'lattice position <-> gate/line/color/tone/base'},
 {name:'file.ingest',route:'POST /api/tool/file.ingest',description:'her integrated swarm: ingest a file, reduce to six-bit primitives, resolve its address; executes code only when execute=true'},
 {name:'build.finish',route:'POST /api/tool/build.finish',description:'her app finisher: assess files against her checklist and add the missing pieces beside them'},
 {name:'workspace.write',route:'POST /api/tool/workspace.write',description:'add a file under user/ in the swarm workspace (audited; protected originals cannot be written)'},
 {name:'chat',route:'POST /api/chat',description:'talk to Trident (her trident_syntia.pt, run in JavaScript on the device)'}]);
export async function createAppServer({loadJson,loadBytes,store=null,now=()=>new Date().toISOString(),schedule,cancel,cycleMs=1500,beatMs=1000,loadModel=true,verbose=false}={}){
 const clean=await loadJson('integrated/primitive-functions.json'),attrs=(await loadJson('integrated/primitive-attributes.json')).attributes,manifest=await loadJson('integrated/protected-originals.json');
 const P=new Map(clean.primitives.map(p=>[p.id,p]));
 const originals=new ProtectedOriginals();for(const [p,s] of Object.entries(manifest.files))originals.register(p,s);originals.register('constitution.md',shaOf(CONSTITUTION));originals.protectPattern('originals');
 let saved=null;try{saved=store?JSON.parse(store.get('synthia.loop.v1')||'null'):null}catch{saved=null}
 const audit=new AuditLog({now,entries:saved?.audit||[]}),ws=new Workspace({audit,originals});
 if(saved?.files)ws.restore(saved.files.map(f=>({path:f.path,data:unb64(f.b64)})));
 if(!saved)audit.record('boot','constitution installed (protected) and '+originals.size+' originals registered',{path:'constitution.md',afterSha:shaOf(CONSTITUTION),reversible:false});
 else audit.record('boot','restored '+ws.size+' workspace files and '+saved.audit.length+' audit entries',{reversible:false});
 let model=null,modelInfo={loaded:false,reason:'not loaded yet'};
 async function ensureModel(){if(model)return model;try{model=await Trident.fromBytes(await loadBytes('models/trident_syntia.pt'));modelInfo={loaded:true,params:3594755,tokenizer:'ord(c)%4096 on lowercased text (as train_syntia.py)',rag:'not ported'}}catch(e){modelInfo={loaded:false,reason:String(e.message||e)};throw e}return model}
 const think=async unit=>{if(!model)return {note:'model not loaded; plan order only'};const {router}=model.forward(tokenize('assemble '+unit.title));return {note:'Trident router code/math/research = '+router.map(v=>v.toFixed(2)).join('/')+' (advisory only; it does not choose the unit)',router}};
 const plan=buildPlan({data:{clean,attrs},tools:TOOLS});
 const loop=new SwarmLoop({plan,workspace:ws,ctx:{data:{clean,attrs}},think,now,schedule,cancel,cycleMs});
 if(saved){loop.restoreState(saved.state,saved.cycleNo);const r=loop.recheck();if(r)loop.emit('loop','restored; '+r+' unit(s) re-queued because their files were missing')}
 const persist=()=>{if(!store)return;store.set('synthia.loop.v1',JSON.stringify({state:loop.state,cycleNo:loop.cycleNo,audit:audit.entries(),files:ws.export().map(f=>({path:f.path,b64:b64(f.data)}))}))};
 loop.on(e=>{if(e.phase==='version')persist()});
 // heartbeat: beats are the pulse (they run even when the loop is asleep). Light checks every beat, protected-original verification in a rotating slice.
 let cursor=0;const orig=originals.list().filter(o=>o.path!=='constitution.md'&&!/trident_syntia\.pt$/.test(o.path)),verifyState={ok:null,checked:0,changed:[],missing:[],at:null};
 const hb=new Heartbeat({tickMs:beatMs,now:()=>Date.now(),schedule,cancel,onBeat:b=>loop.emit('heartbeat','beat '+b.beat+(b.ran.length?' ['+b.ran.join(', ')+']':''),{beat:b.beat,quiet:true})});
 hb.add('verify-protected',3,async()=>{const slice=orig.slice(cursor,cursor+4);cursor=(cursor+4)%Math.max(1,orig.length);const r=await new ProtectedOriginals_(slice).verify(loadBytes);verifyState.at=now();verifyState.checked+=r.checked;if(!r.ok){verifyState.changed.push(...r.changed);verifyState.missing.push(...r.missing);audit.record('tamper-detected','protected original differs from its recorded hash: '+[...r.changed,...r.missing].join(', '),{reversible:false,actor:'heartbeat'})}verifyState.ok=!verifyState.changed.length&&!verifyState.missing.length;return r});
 hb.add('audit-chain',5,async()=>{const v=audit.verify();if(!v.ok)audit.record('tamper-detected','audit chain broken at #'+v.brokenAt,{reversible:false,actor:'heartbeat'});return v});
 hb.add('wake',2,async()=>{if(loop.sleeping&&loop.next()){loop.sleeping=false;loop.emit('loop','woken by heartbeat: new work is available')}return {sleeping:loop.sleeping}});
 function ProtectedOriginals_(list){const o=new ProtectedOriginals();for(const x of list)o.register(x.path,x.sha);return o}
 const status=()=>({time:now(),loop:{running:loop.running,cycle:loop.cycleNo,sleeping:loop.sleeping,...loop.progress(),units:loop.plan.length},heartbeat:{running:hb.running,beats:hb.beats,lastBeatAt:hb.lastBeatAt,lastResults:hb.lastResults},
  protected:{originals:originals.size,...verifyState},audit:{length:audit.length,chain:audit.verify()},versions:ws.versions().length,model:modelInfo,constitution:'protected'});
 const need=(o,k)=>{if(o?.[k]===undefined||o[k]===null)throw new RangeError('missing '+k);return o[k]};
 const tool={
  'primitives.get':({id})=>{const p=P.get(need({id},'id'));if(!p)return {error:'no such primitive'};return {primitive:p,attributes:attrs[id]??null}},
  'primitives.search':({q,kind=null,limit=20})=>{const s=String(q||'').toLowerCase();return {results:clean.primitives.filter(p=>(!kind||p.kind===kind)&&(p.id.toLowerCase().includes(s)||String(p.form).toLowerCase().includes(s)||p.symbol.toLowerCase()===s)).slice(0,limit).map(p=>({id:p.id,symbol:p.symbol,kind:p.kind,form:p.form}))}},
  'primitives.sources':async({id})=>{const S=await loadJson('integrated/primitive-sources.json');return {id,sources:S.sources?.[id]??null,functions:(P.get(id)?.fn||[]).map(f=>({name:f,...S.functions?.[f]}))}},
  'address.fromLongitude':({longitude,planetary=null,dimension=null})=>ARC.addressFromLongitude(Number(need({longitude},'longitude')),{planetary,dimension}),
  'address.lattice':a=>a.position!==undefined?ARC.latticePositionToAddress(Number(a.position)):{position:ARC.calculateLatticePosition(+a.gate,+a.line,+a.color,+a.tone,+a.base)},
  'file.ingest':async({name,text=null,base64=null,execute=false})=>{const {IntegratedSwarm}=await import('./integrated-swarm.mjs');const bytes=text!==null?te.encode(text):unb64(need({base64},'base64'));
   const sw=new IntegratedSwarm({executeCode:!!execute}),r=await sw.execute({name:name||'upload',bytes},{executeCode:!!execute});
   return {status:r.status,reasons:r.reasons,file:r.file,execution:r.execution,address:r.address,app:r.app,fold:{foldHash:r.fold.foldHash,fields:r.fold.fields},stages:r.stages}},
  'build.finish':async({name,files})=>{const r=await finishApp({name:name||'app',files:need({files},'files').map(f=>({path:f.path,data:f.text??''}))},{env:{}});return {report:r.report,files:r.files.map(f=>({path:f.path,size:f.data.length}))}},
  'workspace.write':({path,text})=>{if(!String(path).startsWith('user/'))return {error:'only user/ paths can be written by tools'};try{return ws.write(path,need({text},'text'),{actor:'tool',reason:'tool workspace.write'})}catch(e){if(e instanceof ProtectedError)return {error:e.message};throw e}}
 };
 async function chat({text,head=null,maxNew=48,temp=0.8,topK=40,yieldMs=0}){
  const m=await ensureModel(),ids=tokenize(String(text||'')),n=ids.length;if(!n)return {error:'empty text'};
  const out=await m.generateAsync(ids,{maxNew,temp,topK,head,yieldMs}),{router}=m.forward(ids,{head});
  return {reply:detokenize(out.slice(n)),router:{code:router[0],math:router[1],research:router[2]},head,model:'trident_syntia.pt (her weights, JavaScript port of model.py)',
   note:'Trident is a ~3.6M-parameter character model trained on a few dozen sentences of her system text. It continues text in that style; it does not answer questions reliably. RAG is not ported.'};
 }
 async function handle(method,url,body=null){
  const u=new URL(url,'https://app.local'),path=u.pathname;
  try{
   if(method==='GET'&&path==='/api/status')return {ok:true,body:status()};
   if(method==='GET'&&path==='/api/feed'){const since=Number(u.searchParams.get('since')||0);return {ok:true,body:{total:loop.emitted||0,events:loop.feed.filter(e=>e.n>since),progress:loop.progress(),units:Object.fromEntries(Object.entries(loop.state).map(([k,v])=>[k,v.status]))}}}
   if(method==='GET'&&path==='/api/files')return {ok:true,body:{files:ws.list(),versions:ws.versions().slice(-20)}};
   if(method==='GET'&&path==='/api/file'){const p=u.searchParams.get('path'),t=ws.text(p);return t===null?{ok:false,status:404,body:{error:'no such file'}}:{ok:true,body:{path:p,text:t,sha:shaOf(t)}}}
   if(method==='GET'&&path==='/api/audit')return {ok:true,body:{entries:audit.tail(Number(u.searchParams.get('n')||30)),chain:audit.verify(),report:u.searchParams.get('report')?audit.report():undefined}};
   if(method==='GET'&&path==='/api/tools')return {ok:true,body:{tools:TOOLS}};
   if(method==='POST'&&path==='/api/chat')return {ok:true,body:await chat(body||{})};
   if(method==='POST'&&path.startsWith('/api/tool/')){const n=path.slice(10);if(!tool[n])return {ok:false,status:404,body:{error:'no such tool'}};return {ok:true,body:await tool[n](body||{})}};
   if(method==='POST'&&path==='/api/loop/cycle')return {ok:true,body:await loop.cycle()};
   if(method==='POST'&&path==='/api/loop/start'){loop.start();hb.start();return {ok:true,body:status()}}
   if(method==='POST'&&path==='/api/loop/stop'){loop.stop();hb.stop();return {ok:true,body:status()}}
   if(method==='POST'&&path==='/api/workspace/revert')return {ok:true,body:ws.revert(Number(body?.version),{actor:'creator'})};
   return {ok:false,status:404,body:{error:'no such route'}};
  }catch(e){return {ok:false,status:e instanceof RangeError?400:500,body:{error:String(e.message||e)}}}
 }
 if(loadModel)ensureModel().catch(()=>{});
 return {handle,loop,heartbeat:hb,workspace:ws,audit,originals,status,start(){loop.start();hb.start()},stop(){loop.stop();hb.stop()},persist,ensureModel,plan};
}
