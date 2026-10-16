// Self-assembly loop for the app. Pure JavaScript; no dependency, no backend, no web worker, no network, no payments.
// PATTERNS BORROWED from Conway-Research/automaton (MIT, github.com/Conway-Research/automaton @ d8f8168; read as reference, none of its code is copied):
//   - heartbeat daemon: recursive setTimeout (a beat finishes before the next is scheduled, never setInterval)     [src/heartbeat/daemon.ts]
//   - Think -> Act -> Observe loop with a sleep state                                                            [src/agent/loop.ts]
//   - loop detector: the same call with the same arguments three times in a row is blocked                        [src/agent/loop-detector.ts]
//   - append-only audit log of every self-modification (id, timestamp, type, description, path, diff, reversible)  [src/self-mod/audit-log.ts]
//   - protected files that cannot be modified, matched by path segment                                            [src/self-mod/code.ts PROTECTED_FILES]
//   - a protected constitution and a self-authored identity file (SOUL.md)                                        [constitution.md, src/soul/]
//   - every change versioned                                                                                      [src/git/state-versioning.ts]
// NOT taken (money / chain / replication): wallet, payments, on-chain identity, replication, Conway Cloud. Nothing in this file can spend, send, or call out.
import {sha256} from './../sha256.mjs';
const te=new TextEncoder(),td=new TextDecoder();
const bytesOf=d=>typeof d==='string'?te.encode(d):d instanceof Uint8Array?d:new Uint8Array(d);
export const shaOf=d=>sha256(bytesOf(d));
export class ProtectedError extends Error{constructor(path){super('BLOCKED: protected original, not modifiable: '+path);this.name='ProtectedError';this.path=path}}
// ---------- audit log (hash chained, append only) ----------
export class AuditLog{
 #entries=[];#now;
 constructor({now=()=>new Date().toISOString(),entries=[]}={}){this.#now=now;this.#entries=entries.map(e=>({...e}))}
 record(type,description,{path=null,beforeSha=null,afterSha=null,diff=null,reversible=true,actor='swarm'}={}){
  const prev=this.#entries.length?this.#entries[this.#entries.length-1].hash:'0'.repeat(64);
  const core={seq:this.#entries.length+1,ts:this.#now(),type,actor,description,path,beforeSha,afterSha,diff,reversible,prev};
  const e={...core,hash:shaOf(prev+JSON.stringify(core))};this.#entries.push(e);return {...e};
 }
 get length(){return this.#entries.length}
 entries(){return this.#entries.map(e=>({...e}))}
 tail(n=20){return this.#entries.slice(-n).map(e=>({...e}))}
 verify(){let prev='0'.repeat(64);for(const e of this.#entries){const {hash,...core}=e;if(core.prev!==prev||shaOf(prev+JSON.stringify(core))!==hash)return {ok:false,brokenAt:e.seq};prev=hash}return {ok:true,length:this.#entries.length,head:prev}}
 report(){return ['=== SELF-MODIFICATION AUDIT LOG ===','entries: '+this.#entries.length,...this.#entries.map(e=>'['+e.ts+'] #'+e.seq+' '+e.type+': '+e.description+(e.path?' ('+e.path+')':'')+' '+e.hash.slice(0,12)),'================================='].join('\n')}
}
// ---------- protected originals ----------
export class ProtectedOriginals{
 #by=new Map();#patterns=[];
 register(path,sha){this.#by.set(path,sha)}
 protectPattern(seg){this.#patterns.push(seg)}                      // path-segment match, like automaton PROTECTED_FILES
 has(path){if(this.#by.has(path))return true;const segs=path.split('/');return this.#patterns.some(p=>segs.includes(p))}
 get size(){return this.#by.size}
 list(){return [...this.#by].map(([path,sha])=>({path,sha}))}
 async verify(read){const changed=[],missing=[];for(const [path,sha] of this.#by){let b;try{b=await read(path)}catch{b=null}if(b==null)missing.push(path);else if(shaOf(b)!==sha)changed.push(path)}return {ok:!changed.length&&!missing.length,checked:this.#by.size,changed,missing}}
}
export const CONSTITUTION=`# Constitution (protected)
Alexis's rules for this build, written verbatim from her instructions in this task (not from the automaton repo):
1. Invent nothing that exists. Use what exists first, and show what is new.
2. Preserve every original. Nothing is deleted, merged or flattened; new work is added beside it.
3. A new thing is marked NEW and derived, until Alexis confirms it.
4. Sources are cited in a separate sources layer, not inside the primitives.
5. No money: nothing here pays, charges, sends, posts or publishes. No push to main, no merge, no deploy.
6. No placeholders: if something is missing it is listed as missing.
7. Pure JavaScript: no dependency, no backend, no web worker added silently.
`;
// ---------- workspace: where the swarm assembles the app (virtual files, versioned, audited) ----------
const cleanPath=p=>{p=String(p).replace(/\\/g,'/');if(!p||p.startsWith('/')||p.split('/').some(s=>s===''||s==='..'||s==='.'))throw new RangeError('bad workspace path: '+p);return p};
export class Workspace{
 #files=new Map();#versions=[];
 constructor({audit,originals=new ProtectedOriginals()}={}){this.audit=audit;this.originals=originals}
 write(path,data,{actor='swarm',reason='assemble'}={}){
  path=cleanPath(path);
  if(this.originals.has(path)){this.audit.record('blocked',reason+': attempted to modify a protected original',{path,actor,reversible:false});throw new ProtectedError(path)}
  const b=bytesOf(data),after=shaOf(b),prev=this.#files.get(path);
  if(prev&&prev.sha===after)return {path,sha:after,changed:false};
  this.#files.set(path,{data:b.slice(),sha:after});
  this.audit.record(prev?'modify':'create',reason,{path,beforeSha:prev?.sha??null,afterSha:after,diff:(prev?'size '+prev.data.length+' -> ':'size ')+b.length,actor});
  return {path,sha:after,changed:true};
 }
 read(path){return this.#files.get(cleanPath(path))?.data??null}
 text(path){const b=this.read(path);return b==null?null:td.decode(b)}
 has(path){return this.#files.has(path)}
 list(prefix=''){return [...this.#files].filter(([p])=>p.startsWith(prefix)).map(([path,f])=>({path,sha:f.sha,size:f.data.length})).sort((a,b)=>a.path<b.path?-1:1)}
 get size(){return this.#files.size}
 treeSha(){return shaOf(this.list().map(f=>f.path+':'+f.sha).join('\n'))}
 snapshot(label){const v={n:this.#versions.length+1,label,tree:this.treeSha(),files:new Map(this.#files)};this.#versions.push(v);this.audit.record('version',label,{afterSha:v.tree,reversible:false});return {n:v.n,tree:v.tree,label}}
 versions(){return this.#versions.map(v=>({n:v.n,label:v.label,tree:v.tree,files:v.files.size}))}
 revert(n,{actor='creator'}={}){const v=this.#versions.find(x=>x.n===n);if(!v)throw new RangeError('no version '+n);this.#files=new Map(v.files);this.audit.record('revert','reverted workspace to version '+n+' ('+v.label+')',{afterSha:v.tree,actor});return {n,tree:this.treeSha()}}
 export(){return this.list().map(f=>({path:f.path,data:this.#files.get(f.path).data.slice()}))}
 restore(files){this.#files=new Map(files.map(f=>[cleanPath(f.path),{data:bytesOf(f.data).slice(),sha:shaOf(f.data)}]))} // reload after a restart; the audit chain already records how each file came to be
}
// ---------- loop detector (borrowed pattern) ----------
export class LoopDetector{
 #h=[];constructor({maxIdentical=3}={}){this.max=maxIdentical}
 record(name,args){const k=name+'|'+shaOf(String(args)).slice(0,16);this.#h.push(k);if(this.#h.length>50)this.#h.shift();const r=this.#h.slice(-this.max);return r.length===this.max&&r.every(x=>x===k)?{blocked:true,reason:'"'+name+'" called with identical arguments '+this.max+' times in a row'}:{blocked:false}}
}
// ---------- heartbeat: recursive setTimeout, tasks due every N beats ----------
export class Heartbeat{
 #running=false;#timer=null;#tasks=[];
 constructor({tickMs=1000,now=()=>Date.now(),schedule=(fn,ms)=>setTimeout(fn,ms),cancel=t=>clearTimeout(t),onBeat=()=>{}}={}){Object.assign(this,{tickMs,now,schedule,cancel,onBeat});this.beats=0;this.lastBeatAt=null;this.lastResults={}}
 add(name,everyBeats,run){this.#tasks.push({name,everyBeats,run})}
 async beat(){this.beats++;this.lastBeatAt=this.now();const ran=[];
  for(const t of this.#tasks)if(this.beats%t.everyBeats===0){try{this.lastResults[t.name]=await t.run({beat:this.beats});ran.push(t.name)}catch(e){this.lastResults[t.name]={ok:false,error:String(e.message||e)}}}
  this.onBeat({beat:this.beats,at:this.lastBeatAt,ran});return {beat:this.beats,ran}}
 start(){if(this.#running)return;this.#running=true;const next=()=>{if(!this.#running)return;this.#timer=this.schedule(async()=>{try{await this.beat()}catch{}next()},this.tickMs)};this.beat().catch(()=>{});next()}
 stop(){this.#running=false;if(this.#timer!=null)this.cancel(this.#timer);this.#timer=null}
 get running(){return this.#running}
}
// ---------- the Think -> Act -> Observe loop ----------
// unit = {id,title,needs:[ids],act(ctx)->[{path,data}], observe(ctx)->{ok,detail}}
export class SwarmLoop{
 #running=false;#timer=null;#busy=false;
 constructor({plan,workspace,ctx={},think=null,now=()=>new Date().toISOString(),schedule=(fn,ms)=>setTimeout(fn,ms),cancel=t=>clearTimeout(t),cycleMs=1500,maxAttempts=2,feedLimit=600}){
  Object.assign(this,{plan,workspace,ctx,thinkFn:think,now,schedule,cancel,cycleMs,maxAttempts,feedLimit});
  this.state=Object.fromEntries(plan.map(u=>[u.id,{status:'pending',attempts:0,detail:null}]));
  this.feed=[];this.listeners=new Set();this.cycleNo=0;this.detector=new LoopDetector();this.sleeping=false;
 }
 on(fn){this.listeners.add(fn);return ()=>this.listeners.delete(fn)}
 emit(phase,msg,extra={}){this.emitted=(this.emitted||0)+1;const e={n:this.emitted,t:this.now(),cycle:this.cycleNo,phase,msg,...extra};if(!extra.quiet){this.feed.push(e);if(this.feed.length>this.feedLimit)this.feed.shift()}for(const f of this.listeners)try{f(e)}catch{}return e}
 progress(){const ids=Object.keys(this.state),done=ids.filter(i=>this.state[i].status==='done').length,failed=ids.filter(i=>this.state[i].status==='failed').length;return {done,failed,total:ids.length,files:this.workspace.size,tree:this.workspace.treeSha()}}
 next(){return this.plan.find(u=>this.state[u.id].status==='pending'&&u.needs.every(n=>this.state[n]?.status==='done'))||null}
 async cycle(){
  if(this.#busy)return {skipped:true};this.#busy=true;
  try{
   this.cycleNo++;const unit=this.next();
   if(!unit){const pend=this.plan.filter(u=>this.state[u.id].status==='pending').length;this.emit('think',pend?'waiting: '+pend+' unit(s) blocked by a failed dependency':'plan complete; nothing left to assemble. Sleeping until a heartbeat re-verifies.',{idle:true});this.sleeping=true;return {idle:true}}
   this.sleeping=false;const st=this.state[unit.id];st.attempts++;
   let advice=null;if(this.thinkFn){try{advice=await this.thinkFn(unit)}catch(e){advice={error:String(e.message||e)}}}
   this.emit('think','next: '+unit.title+(advice?.note?' ['+advice.note+']':''),{unit:unit.id,advice});
   const loop=this.detector.record(unit.id,st.attempts);if(loop.blocked){st.status='failed';st.detail=loop.reason;this.emit('observe','blocked by loop detector: '+loop.reason,{unit:unit.id});return {blocked:true}}
   let wrote=[];
   try{const files=await unit.act({...this.ctx,workspace:this.workspace,state:this.state});for(const f of files)wrote.push(this.workspace.write(f.path,f.data,{reason:'assemble '+unit.id}))}
   catch(e){st.detail=String(e.message||e);if(st.attempts>=this.maxAttempts)st.status='failed';this.emit('act','failed: '+st.detail,{unit:unit.id,error:true});return {error:st.detail}}
   const changed=wrote.filter(w=>w.changed);this.emit('act','assembled '+wrote.length+' file(s), '+changed.length+' new/changed',{unit:unit.id,files:wrote.map(w=>w.path)});
   let obs;try{obs=await unit.observe({...this.ctx,workspace:this.workspace,state:this.state,wrote})}catch(e){obs={ok:false,detail:String(e.message||e)}}
   st.detail=obs.detail??null;
   if(obs.ok){st.status='done';this.emit('observe','verified: '+(obs.detail||unit.id),{unit:unit.id,ok:true})}
   else{if(st.attempts>=this.maxAttempts)st.status='failed';this.emit('observe','NOT verified: '+(obs.detail||''),{unit:unit.id,ok:false})}
   const v=this.workspace.snapshot('cycle '+this.cycleNo+': '+unit.id+' '+st.status);this.emit('version','version '+v.n+' tree '+v.tree.slice(0,12),{version:v.n});
   return {unit:unit.id,status:st.status};
  }finally{this.#busy=false}
 }
 start(){if(this.#running)return;this.#running=true;this.emit('loop','started (cycle every '+this.cycleMs+' ms, no external controller)');const next=()=>{if(!this.#running)return;this.#timer=this.schedule(async()=>{try{await this.cycle()}catch(e){this.emit('loop','cycle error: '+e.message)}next()},this.cycleMs)};next()}
 stop(){this.#running=false;if(this.#timer!=null)this.cancel(this.#timer);this.#timer=null;this.emit('loop','stopped')}
 get running(){return this.#running}
 restoreState(state,cycleNo=0){for(const [id,st] of Object.entries(state||{}))if(this.state[id])this.state[id]={...this.state[id],...st};this.cycleNo=cycleNo}
 // a unit that was 'done' but whose files are gone from the workspace goes back to pending (the swarm re-assembles it)
 recheck(){let n=0;for(const u of this.plan){const st=this.state[u.id];if(st.status==='done'&&u.files&&!u.files().every(p=>this.workspace.has(p))){st.status='pending';st.attempts=0;n++}}return n}
}
