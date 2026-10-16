// NSNF runtime (Spec §5 execution contract, §7 component interface, §4 data model, §6 tiers, §17 activation on demand,
// §21 learning kept apart from events, §63 trace/replay). Pure JavaScript, no dependencies, deterministic.
// The swarm is the agent: PSO swarms are background tooling (here: PSO trains component parameters), GA swarms give what is seen.
// INVESTIGATING (not settled by Alexis or the spec; kept visible):
export const INVESTIGATING=Object.freeze([
 'tier values (max active, rounds, Hamming radius) are the v0.1 prototype defaults, an implementation hypothesis (Spec §6), not Alexis\'s numbers; all are configurable.',
 'topology rule "topo-js-v0": candidates are resolved from seeds, their ancestors and Hamming-neighbours at the same level. This is a recruiting rule only; Alexis\'s "Hamming distance 1 at every level" channel rule is not confirmed for NSNF (Spec §36 flag).',
 'edges emerge from co-activation (Spec §35). Rule "edge-coactivation-v0": nested, sibling and co-event relations among active nodes become connections with a starting weight. Not a source-confirmed rule.',
 'meso states are connected groups of the active graph, not nine centers; macro is a coherence-weighted blend. Nine emergent centers are NOT implemented (rule unspecified, Spec §13/§19).',
 'component output has no assigned meaning: it is a numeric state vector. Untrained components make no claims.',
 'cross-component transfer of learned behaviour is not implemented or claimed (Spec §21).',
 'time/compute budgets per tier are counted in multiply-accumulates (MACs) and active components, not wall-clock limits.']);
import {sha256} from './sha256.mjs';
import {fitParticles} from './swarm-fit.mjs';
import {StateSpace,PRIMITIVE_DIMENSIONS,DIMENSIONS,bitsOf,hammingNeighbors,nodeId,parseNodeId,MACRO_PER_LEVEL} from './state-space.mjs';
export const TOPOLOGY_VERSION='topo-js-v0',EDGE_RULE_VERSION='edge-coactivation-v0',RUNTIME_VERSION='0.1';
export const DEFAULT_TIERS=Object.freeze({
 1:Object.freeze({radius:0,maxActive:5,rounds:0,ancestors:false}),2:Object.freeze({radius:1,maxActive:15,rounds:1,ancestors:true}),
 3:Object.freeze({radius:1,maxActive:35,rounds:1,ancestors:true}),4:Object.freeze({radius:2,maxActive:70,rounds:2,ancestors:true}),
 5:Object.freeze({radius:2,maxActive:110,rounds:2,ancestors:true})});
export function stable(v){
 if(v===null||typeof v!=='object')return typeof v==='number'?(Number.isFinite(v)?String(+v.toFixed(12)):'null'):JSON.stringify(v);
 if(Array.isArray(v))return '['+v.map(stable).join(',')+']';
 return '{'+Object.keys(v).filter(k=>v[k]!==undefined).sort().map(k=>JSON.stringify(k)+':'+stable(v[k])).join(',')+'}';
}
export const hashOf=v=>sha256(stable(v));
const seedOf=s=>parseInt(sha256(String(s)).slice(0,8),16)>>>0;
export function rng(seed){let a=seed>>>0;return()=>{a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296}}
const gauss=r=>Math.sqrt(-2*Math.log(r()||1e-12))*Math.cos(2*Math.PI*r());
// ---------- Spec §7 component: tiny trainable MLP cell (initialize, forward, update, snapshot, restore) ----------
export class TinyMLP{
 constructor({dIn=8,hidden=6,dOut=4,seed=0}={}){this.dIn=dIn;this.hidden=hidden;this.dOut=dOut;this.version=0;this.initialize(seed)}
 get macs(){return this.dIn*this.hidden+this.hidden*this.dOut}
 initialize(seed=0){const r=rng(seed),{dIn,hidden,dOut}=this;
  this.W1=Array.from({length:hidden},()=>Array.from({length:dIn},()=>gauss(r)/Math.sqrt(dIn)));this.b1=new Array(hidden).fill(0);
  this.W2=Array.from({length:dOut},()=>Array.from({length:hidden},()=>gauss(r)/Math.sqrt(hidden)));this.b2=new Array(dOut).fill(0);this.version=0;return this}
 forward(x){
  if(!Array.isArray(x)||x.length!==this.dIn||x.some(v=>!Number.isFinite(v)))throw new TypeError('input must be '+this.dIn+' finite numbers');
  const h=this.W1.map((row,i)=>Math.tanh(row.reduce((s,w,j)=>s+w*x[j],this.b1[i])));
  return {out:this.W2.map((row,k)=>row.reduce((s,w,j)=>s+w*h[j],this.b2[k])),hidden:h};
 }
 loss(x,y){const {out}=this.forward(x);return out.reduce((s,o,k)=>s+(o-y[k])**2,0)/out.length}
 // one local SGD step on mean squared error; returns the loss before the step
 update({input,target,lr=0.05}){
  if(!Array.isArray(target)||target.length!==this.dOut||target.some(v=>!Number.isFinite(v)))throw new TypeError('target must be '+this.dOut+' finite numbers');
  const {out,hidden:h}=this.forward(input),before=out.reduce((s,o,k)=>s+(o-target[k])**2,0)/out.length;
  const go=out.map((o,k)=>2*(o-target[k])/out.length),gh=h.map((hv,j)=>(1-hv*hv)*go.reduce((s,g,k)=>s+g*this.W2[k][j],0));
  for(let k=0;k<this.dOut;k++){for(let j=0;j<this.hidden;j++)this.W2[k][j]-=lr*go[k]*h[j];this.b2[k]-=lr*go[k]}
  for(let j=0;j<this.hidden;j++){for(let i=0;i<this.dIn;i++)this.W1[j][i]-=lr*gh[j]*input[i];this.b1[j]-=lr*gh[j]}
  this.version++;return before;
 }
 params(){return [...this.W1.flat(),...this.b1,...this.W2.flat(),...this.b2]}
 setParams(p){let at=0;const take=n=>{const s=p.slice(at,at+n);at+=n;return s};
  if(p.length!==this.dIn*this.hidden+this.hidden+this.hidden*this.dOut+this.dOut)throw new RangeError('wrong parameter count');
  this.W1=Array.from({length:this.hidden},()=>take(this.dIn));this.b1=take(this.hidden);this.W2=Array.from({length:this.dOut},()=>take(this.hidden));this.b2=take(this.dOut)}
 snapshot(){return {kind:'tiny-mlp',dIn:this.dIn,hidden:this.hidden,dOut:this.dOut,version:this.version,params:this.params()}}
 restore(s){if(s.kind!=='tiny-mlp'||s.dIn!==this.dIn||s.hidden!==this.hidden||s.dOut!==this.dOut)throw new Error('snapshot shape differs');this.setParams(s.params);this.version=s.version;return this}
}
// PSO = background tooling: gradient-free training of one component's parameters with the bounded particle swarm already in swarm-fit.mjs.
// Honest scope: held-out loss is reported before and after; parameters are only replaced when held-out loss does not get worse.
export async function trainWithParticleSwarm(mlp,data,{heldOut=[],particles=20,iterations=30,seed=719,bound=3}={}){
 if(!data.length)throw new Error('training data required');
 const initial=mlp.params().map(v=>Math.max(-bound,Math.min(bound,v))),probe=new TinyMLP({dIn:mlp.dIn,hidden:mlp.hidden,dOut:mlp.dOut});
 const mse=(p,set)=>{probe.setParams(p);return set.reduce((s,d)=>s+probe.loss(d.input,d.target),0)/set.length};
 const fit=await fitParticles({bounds:initial.map(()=>[-bound,bound]),initial,score:p=>mse(p,data),particles,iterations,seed});
 const evalSet=heldOut.length?heldOut:data,before=mse(initial,evalSet),after=mse(fit.values,evalSet),accepted=after<=before;
 if(accepted){mlp.setParams(fit.values);mlp.version++}
 return {accepted,trainBefore:fit.initialScore,trainAfter:fit.score,heldOutBefore:before,heldOutAfter:accepted?after:before,evaluatedOn:heldOut.length?'held-out':'training data (no held-out given)',particles,iterations,backend:fit.backend};
}
// ---------- persistence: event_log is append-only; parameters live apart from events (Spec §21) ----------
export class MemoryStore{
 #t={};#cp=null;
 append(kind,rec){(this.#t[kind]??=[]).push(rec);return rec}
 list(kind){return [...(this.#t[kind]||[])]}
 saveCheckpoint(cp){this.#cp=JSON.stringify(cp)}
 loadCheckpoint(){return this.#cp?JSON.parse(this.#cp):null}
}
export function storageStore(storage,prefix='synthia.nsnf.'){ // browser localStorage; small state only
 const rd=k=>{const r=storage.getItem(prefix+k);return r?JSON.parse(r):[]};
 return {append(kind,rec){const a=rd(kind);a.push(rec);storage.setItem(prefix+kind,JSON.stringify(a));return rec},list:rd,
  saveCheckpoint(cp){storage.setItem(prefix+'checkpoint',JSON.stringify(cp))},loadCheckpoint(){const r=storage.getItem(prefix+'checkpoint');return r?JSON.parse(r):null}};
}
export async function fileStore(dir){ // Node: JSONL append-only logs + checkpoint.json
 const fs=await import('node:fs'),path=await import('node:path');fs.mkdirSync(dir,{recursive:true});
 return {append(kind,rec){fs.appendFileSync(path.join(dir,kind+'.jsonl'),JSON.stringify(rec)+'\n');return rec},
  list(kind){const f=path.join(dir,kind+'.jsonl');return fs.existsSync(f)?fs.readFileSync(f,'utf8').split('\n').filter(Boolean).map(l=>JSON.parse(l)):[]},
  saveCheckpoint(cp){fs.writeFileSync(path.join(dir,'checkpoint.json'),JSON.stringify(cp))},
  loadCheckpoint(){const f=path.join(dir,'checkpoint.json');return fs.existsSync(f)?JSON.parse(fs.readFileSync(f,'utf8')):null}};
}
const softmax=z=>{const m=Math.max(...z),e=z.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s)};
const argmax=a=>a.indexOf(Math.max(...a));
const mean=rows=>rows[0].map((_,k)=>rows.reduce((s,r)=>s+r[k],0)/rows.length);
const edgeKey=(a,b)=>a<b?a+'|'+b:b+'|'+a;
export class NsnfRuntime{
 #comps=new Map();#edges=new Map();#cache=new Map();#params=new Map();#proj;#events=0;#snaps=0;#trans=0;#lastSnap=null;
 constructor({space=new StateSpace(),store=new MemoryStore(),tiers=DEFAULT_TIERS,dIn=8,hidden=6,dOut=4,seed=7,baseTier=1,confidenceThreshold=0,alpha=0.5,checkpointEvery=10,actor='nsnf-runtime',now=()=>new Date().toISOString()}={}){
  Object.assign(this,{space,store,tiers,dIn,hidden,dOut,seed,baseTier,confidenceThreshold,alpha,checkpointEvery,actor,now});
  if(!tiers[baseTier])throw new RangeError('unknown base tier');
  this.globalVersion=0;this.topologyVersion=0;this.stats={hits:0,misses:0};
  const r=rng(seed);this.#proj=PRIMITIVE_DIMENSIONS.map(()=>Array.from({length:6},()=>Array.from({length:dIn},()=>gauss(r))));
 }
 get investigating(){return INVESTIGATING}
 get tierMax(){return Math.max(...Object.keys(this.tiers).map(Number))}
 // Spec §5 step 1: encode input into one six-bit code per primitive dimension. Space is not encoded: it is read.
 encode(x){
  if(!Array.isArray(x)||x.length!==this.dIn||x.some(v=>!Number.isFinite(v)))throw new TypeError('input must be '+this.dIn+' finite numbers');
  const c=Object.fromEntries(PRIMITIVE_DIMENSIONS.map((d,i)=>[d,this.#proj[i].reduce((code,row,b)=>code|((row.reduce((s,w,j)=>s+w*(x[j]-0.5),0)>0?1:0)<<b),0)]));return c;
 }
 #component(id){
  let c=this.#comps.get(id);
  if(!c){const mlp=new TinyMLP({dIn:this.dIn,hidden:this.hidden,dOut:this.dOut,seed:seedOf(this.seed+':'+id)});
   c={mlp,record:{id:'c:'+id,gate_id:id,architecture_type:'trainable-cell',artifact_hash:hashOf({kind:'tiny-mlp',dIn:this.dIn,hidden:this.hidden,dOut:this.dOut,init:id}),parameter_ref:null,version:RUNTIME_VERSION}};
   this.#setRef(c);this.#comps.set(id,c)}
  return c;
 }
 #setRef(c){const snap=c.mlp.snapshot(),ref=hashOf(snap);this.#params.set(ref,snap);c.record={...c.record,parameter_ref:ref}}
 componentRecords(){return [...this.#comps.values()].map(c=>({...c.record,param_version:c.mlp.version}))}
 parameterBlob(ref){return this.#params.get(ref)||null}
 // Spec §5 step 2: resolve candidates by the versioned topology rule; lists are nested across tiers.
 candidates(seeds,codes,tier){
  const t=this.tiers[tier],why=new Map(),order=[];
  const add=(node,reason)=>{if(!why.has(node.id)){why.set(node.id,reason);order.push(node)}};
  const ring0=[];
  for(const s of seeds){const n=this.space.ensure(s.dimension,s.path);add(n,s.role||'seed');ring0.push(n)}
  for(const d of PRIMITIVE_DIMENSIONS){const n=this.space.ensure(d,[{macro:codes[d],sub:0}]);add(n,'encoded');ring0.push(n)}
  if(t.ancestors)for(const n of ring0)for(const a of this.space.ancestors(n.id))add(a,'ancestor-of:'+n.id);
  for(let r=1;r<=t.radius;r++)for(const n of ring0)for(const h of hammingNeighbors(n.macro,r).filter(h=>h.distance===r))
   add(this.space.ensure(n.dimension,[...n.path.slice(0,-1),{macro:h.code,sub:n.sub}]),'hamming-'+r+'-of:'+n.id);
  return order.slice(0,t.maxActive).map(n=>({node:n,why:why.get(n.id)}));
 }
 #relations(active){ // permitted edges for this event; types are nested, sibling, co-event (rule edge-coactivation-v0)
  const ids=new Set(active.map(a=>a.node.id)),out=new Map(),seedIds=active.filter(a=>a.why==='encoded'||a.why==='seed'||a.why==='placement').map(a=>a.node.id);
  const put=(a,b,type)=>{if(a!==b&&ids.has(a)&&ids.has(b)){const k=edgeKey(a,b);if(!out.has(k))out.set(k,{src:a<b?a:b,dst:a<b?b:a,type})}};
  for(const {node:n} of active){
   if(n.parent)put(n.id,n.parent,'nested');
   for(const h of hammingNeighbors(n.macro,1))put(n.id,nodeId(n.dimension,[...n.path.slice(0,-1),{macro:h.code,sub:n.sub}]),'sibling');
  }
  for(let i=0;i<seedIds.length;i++)for(let j=i+1;j<seedIds.length;j++)put(seedIds[i],seedIds[j],'co-event');
  return [...out.values()].map(e=>{const s=this.#edges.get(edgeKey(e.src,e.dst));return {...e,weight:s?s.weight:0.1,existing:!!s}});
 }
 #key(input,seeds,tier,threshold){return hashOf({input,seeds,tier,threshold,alpha:this.alpha,gv:this.globalVersion,tv:this.topologyVersion,tiers:this.tiers,levels:this.space.levels,split:this.space.split,space:this.space.readSpace().status})}
 // seeds: [{dimension,path,role}] explicit placements (e.g. where a registered file lives). Returns the full trace.
 execute({input,seeds=[],tier=null,reason=null,confidenceThreshold=this.confidenceThreshold,log=true}={}){
  if(!Array.isArray(input)||input.length!==this.dIn||input.some(v=>!Number.isFinite(v)))return {status:'blocked',reason:'input must be '+this.dIn+' finite numbers; nothing was executed',executed:false};
  let cur=tier||this.baseTier;if(!this.tiers[cur])return {status:'blocked',reason:'unknown tier '+tier,executed:false};
  const forced=tier!==null,startTier=cur;
  const key=this.#key(input,seeds,startTier,forced?null:confidenceThreshold);
  if(this.#cache.has(key)){this.stats.hits++;const c=this.#cache.get(key);return {...c,cache_hit:true,macs:0,trace:{...c.trace,newEdges:0,cache:'reused: input, seeds, tier, version and relevant state all match'}}}
  this.stats.misses++;
  const codes=this.encode(input),rows=new Map(),escalation=[],reasons=[];let macs=0,res;
  if(reason)reasons.push(reason);
  for(;;){
   const active=this.candidates(seeds,codes,cur);
   if(!active.length)return {status:'blocked',reason:'no candidate components under tier '+cur+'; no neural execution happened',executed:false};
   for(const a of active)if(!rows.has(a.node.id)){const c=this.#component(a.node.id);rows.set(a.node.id,c.mlp.forward(input).out);macs+=c.mlp.macs}
   const ids=active.map(a=>a.node.id),z=new Map(ids.map(id=>[id,rows.get(id).slice()])),edges=this.#relations(active);
   const nb=new Map(ids.map(id=>[id,[]]));for(const e of edges){nb.get(e.src).push([e.dst,e.weight]);nb.get(e.dst).push([e.src,e.weight])}
   const messages=[];
   for(let r=0;r<this.tiers[cur].rounds;r++){
    const next=new Map();
    for(const id of ids){const n=nb.get(id);if(!n.length){next.set(id,z.get(id));continue}
     const w=n.reduce((s,[,x])=>s+x,0),m=z.get(id).map((_,k)=>n.reduce((s,[o,x])=>s+x*z.get(o)[k],0)/w);
     next.set(id,z.get(id).map((v,k)=>(1-this.alpha)*v+this.alpha*m[k]));
     for(const [o,x] of n)messages.push({round:r+1,type:'activation',from:o,to:id,weight:x});macs+=n.length*this.dOut}
    for(const [id,v] of next)z.set(id,v);
   }
   // meso: connected groups of the active graph (emergent, not nine hard-coded centers); macro: coherence-weighted blend
   const parent=new Map(ids.map(i=>[i,i])),find=i=>{while(parent.get(i)!==i){parent.set(i,parent.get(parent.get(i)));i=parent.get(i)}return i};
   for(const e of edges)parent.set(find(e.src),find(e.dst));
   const groups=new Map();for(const id of ids){const g=find(id);if(!groups.has(g))groups.set(g,[]);groups.get(g).push(id)}
   const meso=[...groups.entries()].map(([g,members])=>{const m=mean(members.map(i=>z.get(i)));return {group:g,members,state:m,confidence:Math.max(...softmax(m))}}).sort((a,b)=>a.group<b.group?-1:1);
   const wsum=meso.reduce((s,m)=>s+m.confidence*m.members.length,0),macro=meso[0].state.map((_,k)=>meso.reduce((s,m)=>s+m.confidence*m.members.length*m.state[k],0)/wsum);
   const probs=softmax(macro),conf=Math.max(...probs),coherence=meso.filter(m=>argmax(m.state)===argmax(macro)).length/meso.length;
   macs+=ids.length*this.dOut;
   res={tier:cur,active,edges,messages,meso,macro,probs,conf,coherence};
   if(forced||conf>=confidenceThreshold||cur>=this.tierMax)break;
   escalation.push('tier '+cur+'->'+(cur+1)+': confidence '+conf.toFixed(3)+' < '+confidenceThreshold);cur++;
  }
  // step 6: output + trace
  const out={status:'executed',executed:true,tier:res.tier,startTier,pred:argmax(res.probs),probs:res.probs,macro:res.macro,confidence:res.conf,coherence:res.coherence,
   n_active:res.active.length,macs,cache_hit:false,escalation,reasons,codes,space:this.space.readSpace(),
   trace:{topology_version:TOPOLOGY_VERSION,edge_rule_version:EDGE_RULE_VERSION,
    active:res.active.map(a=>({node:a.node.id,dimension:a.node.dimension,hexagram_code:a.node.hexagram_code,depth:a.node.depth,why:a.why,component:this.#component(a.node.id).record.id})),
    messages:{count:res.messages.length,sample:res.messages.slice(0,6)},meso:res.meso.map(m=>({group:m.group,size:m.members.length,confidence:m.confidence})),
    edges:res.edges.length,newEdges:res.edges.filter(e=>!e.existing).length}};
  // step 7: persist event, state transition and checkpoint; topology updates are versioned transitions
  this.space.clearActive();for(const a of res.active)this.space.activate(a.node.id);
  for(const d of PRIMITIVE_DIMENSIONS)this.space.setDimensionCode(d,codes[d]);
  let created=0;for(const e of res.edges){const k=edgeKey(e.src,e.dst),s=this.#edges.get(k);
   if(s)s.count++;else{this.#edges.set(k,{src:e.src,dst:e.dst,type:e.type,weight:e.weight,rule_version:EDGE_RULE_VERSION,count:1});created++}}
  if(created)this.topologyVersion++;
  if(log){
   const ev=this.#logEvent('execute',{input,seeds,tier:forced?tier:null,reason,confidenceThreshold,output:hashOf({tier:out.tier,macro:out.macro,active:out.trace.active.map(a=>a.node)})});
   out.event_id=ev.id;out.snapshot_id=this.#snapshot(ev.id,'execute','event '+ev.id+' at tier '+out.tier,res.meso).id;
   if(created)this.#snapshot(ev.id,'topology-update',created+' new connection(s) by rule '+EDGE_RULE_VERSION,res.meso);
   if(this.checkpointEvery&&this.#events%this.checkpointEvery===0)this.checkpoint();
  }
  this.#cache.set(key,out);this.#cache.set(this.#key(input,seeds,startTier,forced?null:confidenceThreshold),out);
  return out;
 }
 // Spec §7 update(loss/feedback): local learning on exactly the components the event activates.
 learn({input,target,seeds=[],tier=1,lr=0.05,log=true}){
  const codes=this.encode(input),active=this.candidates(seeds,codes,tier);if(!active.length)return {status:'blocked',reason:'no components to update',updated:0};
  const lossBefore=[],comps=active.map(a=>this.#component(a.node.id));
  for(const c of comps)lossBefore.push(c.mlp.update({input,target,lr}));
  for(const c of comps)this.#setRef(c);
  this.globalVersion++;this.#cache.clear();
  const lossAfter=comps.map(c=>c.mlp.loss(input,target)),m=a=>a.reduce((s,v)=>s+v,0)/a.length;
  const r={status:'updated',updated:comps.length,lossBefore:m(lossBefore),lossAfter:m(lossAfter),globalVersion:this.globalVersion,note:'local learning only; topology unchanged; no cross-component transfer claimed'};
  if(log){const ev=this.#logEvent('learn',{input,target,seeds,tier,lr});this.#snapshot(ev.id,'learn','feedback on '+comps.length+' component(s)');r.event_id=ev.id}
  return r;
 }
 #logEvent(kind,payload){
  const id=++this.#events,entry={id,actor:this.actor,payload_hash:hashOf({kind,payload}),timestamp:this.now()};
  this.store.append('event_log',entry);this.store.append('event_payload',{event_id:id,kind,payload});return entry;
 }
 #snapshot(eventId,type,why,meso=null){
  const id=++this.#snaps,row={id,timestamp:this.now(),event_id:eventId,dimension_codes:this.space.codes(),active_masks:this.space.maskHex(),
   center_states:meso?meso.map(m=>({group:m.group,size:m.members.length,state:m.state})):null,checkpoint_ref:null};
  this.store.append('state_snapshot',row);
  if(this.#lastSnap!==null)this.store.append('transition',{id:++this.#trans,before_id:this.#lastSnap,after_id:id,event_type:type,reason:why});
  this.#lastSnap=id;return row;
 }
 // Spec §4 data model rows, assembled from live state.
 dataModel(){
  return {dimension:DIMENSIONS.map((d,i)=>({id:i,name:d,primitive:d!=='Space',emergent:d==='Space',slots:MACRO_PER_LEVEL})),
   center:{emergence_rule_version:null,note:'nine centers are not implemented; rule unspecified'},
   gate:{count:5*MACRO_PER_LEVEL,center_binding:null,note:'one hexagram_code per (dimension,slot); slots exist even when no node is touched'},
   component:this.componentRecords(),connection:[...this.#edges.values()].map(e=>({src_id:e.src,dst_id:e.dst,type:e.type,weight:e.weight,rule_version:e.rule_version,co_activations:e.count})),
   state_snapshot:this.store.list('state_snapshot'),transition:this.store.list('transition'),event_log:this.store.list('event_log')};
 }
 // checkpoint / restore: topology, parameters and counters. Replay re-runs the event payloads and compares output hashes.
 checkpoint(){
  const cp={runtime_version:RUNTIME_VERSION,config:{dIn:this.dIn,hidden:this.hidden,dOut:this.dOut,seed:this.seed,alpha:this.alpha,tiers:this.tiers},
   space:this.space.snapshot(),components:Object.fromEntries([...this.#comps].map(([id,c])=>[id,c.mlp.snapshot()])),
   edges:[...this.#edges.values()],counters:{events:this.#events,snaps:this.#snaps,trans:this.#trans,globalVersion:this.globalVersion,topologyVersion:this.topologyVersion},last:this.#lastSnap};
  const ref=hashOf(cp);this.store.saveCheckpoint({ref,...cp});return ref;
 }
 restore(cp=this.store.loadCheckpoint()){
  if(!cp)throw new Error('No checkpoint to restore');
  if(cp.config.dIn!==this.dIn||cp.config.hidden!==this.hidden||cp.config.dOut!==this.dOut||cp.config.seed!==this.seed)throw new Error('Checkpoint shape/seed differs from this runtime');
  this.space.restore(cp.space);this.#comps.clear();this.#edges.clear();this.#cache.clear();
  for(const [id,s] of Object.entries(cp.components)){const c=this.#component(id);c.mlp.restore(s);this.#setRef(c)}
  for(const e of cp.edges)this.#edges.set(edgeKey(e.src,e.dst),{...e});
  Object.assign(this,{globalVersion:cp.counters.globalVersion,topologyVersion:cp.counters.topologyVersion});
  this.#events=cp.counters.events;this.#snaps=cp.counters.snaps;this.#trans=cp.counters.trans;this.#lastSnap=cp.last;return this;
 }
 // Replay: re-run logged payloads on a fresh runtime with the same config; every execute output hash must match.
 static replay(entries,payloads,config){
  const rt=new NsnfRuntime({...config,store:new MemoryStore(),now:()=>'replay'}),mismatches=[];
  const byId=new Map(payloads.map(p=>[p.event_id,p]));
  for(const e of entries){const p=byId.get(e.id);if(!p){mismatches.push({id:e.id,why:'payload missing'});continue}
   if(hashOf({kind:p.kind,payload:p.payload})!==e.payload_hash){mismatches.push({id:e.id,why:'payload hash differs (event log or payload altered)'});continue}
   if(p.kind==='execute'){const r=rt.execute({...p.payload,tier:p.payload.tier,reason:p.payload.reason});
    const h=hashOf({tier:r.tier,macro:r.macro,active:r.trace.active.map(a=>a.node)});if(h!==p.payload.output)mismatches.push({id:e.id,why:'output differs on replay'})}
   else if(p.kind==='learn')rt.learn(p.payload);}
  return {ok:mismatches.length===0,events:entries.length,mismatches,runtime:rt};
 }
 replayLog(){return NsnfRuntime.replay(this.store.list('event_log'),this.store.list('event_payload'),{space:new StateSpace({levels:this.space.levels,split:this.space.split}),dIn:this.dIn,hidden:this.hidden,dOut:this.dOut,seed:this.seed,tiers:this.tiers,baseTier:this.baseTier,confidenceThreshold:this.confidenceThreshold,alpha:this.alpha,checkpointEvery:0})}
}
