import {DIMENSIONS,PERSPECTIVES,SHADOWS,PRIMITIVE_LETTERS,normalizeAddress,traverse,CausalGraphError} from './causal-graph.mjs';

const clone=x=>structuredClone(x);
const finite=(n,k)=>{if(!Number.isFinite(n))throw new TypeError(`${k} must be finite`);return n;};
const EPS=1e-8;
export function legendre(order,x){ if(order===0)return 1;if(order===1)return x;let a=1,b=x;for(let n=2;n<=order;n++){const c=((2*n-1)*x*b-(n-1)*a)/n;a=b;b=c;}return b; }
export function moments(points,maxOrder=3){
 if(!Array.isArray(points)||!points.length)return [];
 const out=[]; for(let n=1;n<=maxOrder;n++)for(let p=0;p<=n;p++){
   const q=n-p;out.push(points.reduce((sum,pt)=>sum+legendre(p,finite(pt.x,'point.x'))*legendre(q,finite(pt.y,'point.y')),0)/points.length);
 }return out;
}
export function momentDistance(a,b){if(a.length!==b.length)throw new Error('moment length mismatch');return Math.hypot(...a.map((x,i)=>x-b[i]));}

// Mass-conserving push-sum: packets not delivered remain with sender; no fabricated reception.
// A supplied deterministic loss schedule allows exact reproducibility of recovery tests.
export function pushSumConsensus(initial,neighbors,{rounds=50,drop=()=>false}={}){
 const ids=Object.keys(initial).sort();if(!ids.length)return {};
 const width=initial[ids[0]].length;
 const mass=Object.fromEntries(ids.map(id=>[id,{w:1,values:[...initial[id]]}]));
 for(const id of ids)if(initial[id].length!==width)throw new Error('Vector width mismatch');
 for(let round=0;round<rounds;round++){
   const next=Object.fromEntries(ids.map(id=>[id,{w:0,values:Array(width).fill(0)}]));
   const receive=(id,w,vec)=>{next[id].w+=w;vec.forEach((n,i)=>next[id].values[i]+=n);};
   for(const id of ids){
     const adj=(neighbors[id]??[]).filter(n=>Object.hasOwn(mass,n));
     const dest=[id,...new Set(adj)];const unit=1/dest.length;
     const sender=mass[id];
     for(const target of dest){
       if(target!==id && drop(id,target,round))receive(id,sender.w*unit,sender.values.map(x=>x*unit));
       else receive(target,sender.w*unit,sender.values.map(x=>x*unit));
     }
   }
   Object.assign(mass,next);
 }
 const result=Object.fromEntries(ids.map(id=>[id,{estimate:mass[id].values.map(x=>x/mass[id].w),weight:mass[id].w}]));
 return result;
}

export class ConsciousnessSwarm {
 constructor({maxMomentOrder=3}={}){
   this.maxMomentOrder=maxMomentOrder;
   this.nodes=new Map();this.edges=new Map();this.events=[];this.nextId=1;this.nextEvent=1;
   this.operators=new Map();this.formation=new Map();this.links=new Map();this.tick=0;
   this.registerOperator('CONCAT',values=>values.map(v=>String(v)).join(''));
   this.registerOperator('JOIN_SPACE',values=>values.map(v=>String(v)).join(' '));
   this.registerOperator('SUM',values=>values.reduce((a,b)=>a+finite(Number(b),'SUM arg'),0));
 }
 registerOperator(name,fn){if(!/^[A-Z][A-Z0-9_]*$/.test(name)||typeof fn!=='function')throw new TypeError('operator must be a named function');this.operators.set(name,fn);return this;}
 _id(prefix){return `${prefix}:${this.nextId++}`;}
 _record(type,details){const event=Object.freeze({seq:this.nextEvent++,type,details:clone(details),at:new Date().toISOString()});this.events.push(event);return event;}
 _get(id){const item=this.nodes.get(id);if(!item)throw new Error(`Unknown node: ${id}`);return item;}
 createPrimitive({id,kind='symbol',value,scale=0,dimension=null,address={},properties={}}={}){
   if(value===undefined)throw new Error('Primitive value is required');
   if(!Number.isInteger(scale)||scale<0)throw new Error('Scale must be a nonnegative integer');
   const key=id??this._id('p');if(this.nodes.has(key))throw new Error(`Identity already exists: ${key}`);
   const dim=dimension===null && typeof value==='string' && value.length===1?PRIMITIVE_LETTERS[value.toUpperCase()]??null:dimension;
   if(dim!==null && !Object.hasOwn(DIMENSIONS,dim))throw new Error(`Unsupported dimension: ${dim}`);
   const addr=normalizeAddress({...address,...(dimension?{dimension:dim}:{})});
   // Fifth dimension is a valid quality of a primitive, never a fifth independent decision engine.
   const node=Object.freeze({id:key,kind,scale,value:clone(value),dimension:dim,address:addr.address,addressStatus:addr.status,
     unresolvedAddressFields:addr.unresolved,properties:clone(properties),parts:[],operator:null,createdSeq:this.nextEvent});
   this.nodes.set(key,node);this._record('primitive.created',{id:key,kind,scale,dimension:dim,addressStatus:node.addressStatus});
   return clone(node);
 }
 compose({id,parts,operator='CONCAT',kind='composition',scale=null,address={},dimension=null}={}){
   if(!Array.isArray(parts)||!parts.length)throw new Error('Composition requires existing parts');
   const child=parts.map(x=>this._get(x));const fn=this.operators.get(operator);
   if(!fn)throw new Error(`Operator not registered: ${operator}`);
   const resolvedScale=scale??Math.max(...child.map(x=>x.scale))+1;
   if(resolvedScale<=Math.max(...child.map(x=>x.scale)))throw new Error('Parent scale must exceed every child scale');
   const value=fn(child.map(x=>x.value)); const key=id??this._id('c');if(this.nodes.has(key))throw new Error(`Identity already exists: ${key}`);
   const addr=normalizeAddress({...address,...(dimension?{dimension}:{})});
   const node=Object.freeze({id:key,kind,scale:resolvedScale,value:clone(value),dimension:dimension??null,address:addr.address,
     addressStatus:addr.status,unresolvedAddressFields:addr.unresolved,properties:{},parts:[...parts],operator,createdSeq:this.nextEvent});
   this.nodes.set(key,node);
   const edges=parts.map((part,i)=>Object.freeze({id:`edge:${key}:${i}`,source:part,target:key,relation:'constituent-of',createdSeq:this.nextEvent}));
   edges.forEach(e=>this.edges.set(e.id,e));
   this._record('composition.created',{id:key,parts:[...parts],operator,value:clone(value),scale:resolvedScale,edgeIds:edges.map(e=>e.id)});
   return clone(node);
 }
 decompose(id,{recursive=true}={}){
   const node=this._get(id);return {node:clone(node),children:recursive?node.parts.map(p=>this.decompose(p)):node.parts.map(id=>clone(this._get(id)))};
 }
 ingestText(text){
   if(typeof text!=='string'||!text.length)throw new Error('Nonempty text required');
   const words=[];for(const raw of text.split(' ')){
     const letters=[...raw].map(value=>this.createPrimitive({value,kind:'letter'}));
     words.push(this.compose({parts:letters.map(c=>c.id),operator:'CONCAT',kind:'word'}));
   }
   return this.compose({parts:words.map(w=>w.id),operator:'JOIN_SPACE',kind:'text'});
 }
 connect(a,b,{relation,context={},provenance={}}={}){
   this._get(a);this._get(b);if(!relation||typeof relation!=='string')throw new Error('Mesh edge requires a named relation');
   const id=this._id('e');const edge=Object.freeze({id,source:a,target:b,relation,context:clone(context),provenance:clone(provenance),createdSeq:this.nextEvent});
   this.edges.set(id,edge);this._record('mesh.connected',{edge});return clone(edge);
 }
 perspectives(id){
   const n=this._get(id);
   const four=Object.fromEntries(PERSPECTIVES.map(d=>[d,{dimension:d,question:DIMENSIONS[d].question,nature:DIMENSIONS[d].nature,shadow:SHADOWS[d],identity:id,underlyingAddress:n.address}]));
   const incident=[...this.edges.values()].filter(e=>e.source===id||e.target===id);
   const space={dimension:'SPACE',question:'WHO',emergent:true,identity:id,underlyingAddress:n.address,
     relationshipCount:incident.length,relatedIds:[...new Set(incident.map(e=>e.source===id?e.target:e.source))],
     composedFrom:n.parts,form:n.value};
   return {identity:id,sourceNode:clone(n),perspectives:four,space};
 }
 observeMovement({id,from,to,sound=null,source='observation'}={}){
   this._get(id);for(const [k,p] of [['from',from],['to',to]])if(!p||!Number.isFinite(p.x)||!Number.isFinite(p.y))throw new Error(`movement.${k} needs numeric x and y`);
   if(sound!==null&&typeof sound!=='boolean'&&!(typeof sound==='object'&&sound!==null))throw new Error('sound must be boolean, a recorded observation object, or null (unknown)');
   const movement={id,from:clone(from),to:clone(to),delta:{x:to.x-from.x,y:to.y-from.y},sound:clone(sound),source};
   this._record('movement.observed',movement);return movement;
 }
 addFormationParticle({nodeId,x,y,motion='FREE'}={}){
   this._get(nodeId);finite(x,'x');finite(y,'y');if(!['FREE','UP','HORIZONTAL','DIAGONAL','STILL'].includes(motion))throw new Error(`Unknown motion ${motion}`);
   if(this.formation.has(nodeId))throw new Error(`Formation particle exists: ${nodeId}`);
   this.formation.set(nodeId,{id:nodeId,x,y,motion,active:true});this.links.set(nodeId,new Set());
   this._record('formation.particle.created',{nodeId,x,y,motion});return this.formation.get(nodeId);
 }
 linkTransport(a,b){
   if(!this.formation.get(a)?.active||!this.formation.get(b)?.active)throw new Error('Transport endpoints must be active particles');
   if(a===b)throw new Error('Self transport link');
   this.links.get(a).add(b);this.links.get(b).add(a);
   this._record('transport.linked',{a,b});
 }
 positions(){return [...this.formation.values()].filter(p=>p.active).map(p=>({id:p.id,x:p.x,y:p.y,motion:p.motion}));}
 shapeMoments(){return moments(this.positions(),this.maxMomentOrder);}
 formationError(targets){
   const current=this.positions();if(!current.length)throw new Error('No active formation particles');
   const target=current.map(p=>{const pt=targets[p.id];if(!pt)throw new Error(`Missing target for ${p.id}`);finite(pt.x,'target.x');finite(pt.y,'target.y');return pt;});
   return {momentError:momentDistance(moments(current,this.maxMomentOrder),moments(target,this.maxMomentOrder)),
     pointError:Math.sqrt(current.reduce((s,p,i)=>s+(p.x-target[i].x)**2+(p.y-target[i].y)**2,0)/current.length)};
 }
 stepFormation(targets,{rate=0.05,momentGain=0.1}={}){
   if(!(rate>0&&rate<=1)||momentGain<0)throw new Error('Invalid control gain');
   const current=this.positions();const target=current.map(p=>{const pt=targets[p.id];if(!pt)throw new Error(`Missing target for ${p.id}`);return pt;});
   const targetM=moments(target,this.maxMomentOrder);const errorVector=moments(current,this.maxMomentOrder).map((v,i)=>v-targetM[i]);
   const before=this.formationError(targets);
   for(const p of current){const live=this.formation.get(p.id);const desired=targets[p.id];
     let dx=rate*(desired.x-p.x),dy=rate*(desired.y-p.y);
     if(momentGain){const h=1e-4;const ix=current.findIndex(x=>x.id===p.id);
       for(const axis of ['x','y']){const plus=current.map((q,i)=>i===ix?{...q,[axis]:q[axis]+h}:q);
         const minus=current.map((q,i)=>i===ix?{...q,[axis]:q[axis]-h}:q);
         const pm=moments(plus,this.maxMomentOrder),mm=moments(minus,this.maxMomentOrder);
         const grad=pm.reduce((s,v,i)=>s+(v-mm[i])/(2*h)*errorVector[i],0);
         if(axis==='x')dx-=momentGain*grad;else dy-=momentGain*grad;
       }
     }
     // Explicit movement constraints operate on the projected formation, not the semantic ontology.
     if(live.motion==='STILL')dx=dy=0;
     else if(live.motion==='UP'){dx=0;dy=Math.max(0,dy);}
     else if(live.motion==='HORIZONTAL')dy=0;
     else if(live.motion==='DIAGONAL'){const d=(dx+dy)/2;dx=d;dy=d;}
     live.x=Math.max(-1,Math.min(1,live.x+dx));live.y=Math.max(-1,Math.min(1,live.y+dy));
   }
   this.tick++;const after=this.formationError(targets);
   this._record('formation.step',{tick:this.tick,before,after});return {tick:this.tick,before,after,space:this.emergentSpace()};
 }
 emergentSpace(){
   const p=this.positions();if(!p.length)return {kind:'empty-space',count:0,emergent:true};
   const x=p.reduce((s,v)=>s+v.x,0)/p.length,y=p.reduce((s,v)=>s+v.y,0)/p.length;
   return {kind:'collective-form',emergent:true,count:p.length,centroid:{x,y},moments:moments(p,this.maxMomentOrder),
     relatedParticleIds:p.map(t=>t.id)};
 }
 removeFromFormation(id){
   const p=this.formation.get(id);if(!p?.active)throw new Error(`No active particle ${id}`);
   p.active=false;const previous=[...(this.links.get(id)||[])];for(const x of previous)this.links.get(x)?.delete(id);
   this.links.get(id)?.clear();this._record('formation.particle.deactivated',{id,previousLinks:previous});
   // Transport recovery only: these links DO NOT imply semantic/ontological kinship.
   const active=this.positions().map(x=>x.id);if(active.length){
     const seen=new Set();const groups=[];
     for(const start of active){if(seen.has(start))continue;const q=[start],group=[];seen.add(start);
       while(q.length){const a=q.shift();group.push(a);for(const b of this.links.get(a)??[])if(!seen.has(b)&&this.formation.get(b)?.active){seen.add(b);q.push(b);}}
       groups.push(group);
     }
     for(let i=1;i<groups.length;i++){const a=groups[i-1][0],b=groups[i][0];this.links.get(a).add(b);this.links.get(b).add(a);
       this._record('transport.recovered',{a,b,reason:`deactivation:${id}`,relation:'transport-only'});
     }
   }
   return {deactivated:id,remaining:active,historyRetained:this._get(id)!=null};
 }
 estimateLocalMoments({rounds=80,drop=()=>false}={}){
   const ps=this.positions();const initial=Object.fromEntries(ps.map(p=>[p.id,moments([p],this.maxMomentOrder)]));
   const neighbors=Object.fromEntries(ps.map(p=>[p.id,[...(this.links.get(p.id)??[])].filter(k=>this.formation.get(k)?.active)]));
   return pushSumConsensus(initial,neighbors,{rounds,drop});
 }
 snapshot(){return {version:1,nextId:this.nextId,nextEvent:this.nextEvent,tick:this.tick,nodes:[...this.nodes.values()].map(clone),edges:[...this.edges.values()].map(clone),events:this.events.map(clone),formation:[...this.formation.values()].map(clone),links:[...this.links.entries()].map(([id,s])=>[id,[...s]])};}
 static restore(snapshot){if(snapshot.version!==1)throw new Error('Unknown snapshot version');const s=new ConsciousnessSwarm();s.nextId=snapshot.nextId;s.nextEvent=snapshot.nextEvent;s.tick=snapshot.tick;s.nodes=new Map(snapshot.nodes.map(n=>[n.id,Object.freeze(clone(n))]));s.edges=new Map(snapshot.edges.map(e=>[e.id,Object.freeze(clone(e))]));s.events=snapshot.events.map(e=>Object.freeze(clone(e)));s.formation=new Map(snapshot.formation.map(p=>[p.id,clone(p)]));s.links=new Map(snapshot.links.map(([id,l])=>[id,new Set(l)]));return s;}
 causalTrace(id){const seen=new Set();const walk=x=>{if(seen.has(x))return;seen.add(x);this._get(x).parts.forEach(walk);};walk(id);return {root:id,reachableNodes:[...seen],relations:[...this.edges.values()].filter(e=>seen.has(e.source)&&seen.has(e.target)).map(clone),events:this.events.filter(e=>{const t=JSON.stringify(e.details);return [...seen].some(k=>t.includes(`"${k}"`));}).map(clone)};}
 traverseAddress(address){const result=traverse(address);this._record('causal.traversal',{input:address,result});return result;}
}
