(function(global){
'use strict';
const K=global.AgentomagotchiKernel,A=global.AgentomagotchiAssembly,B=global.AgentomagotchiWorldBridge;
if(!K)throw new Error('kernel.js must load before sim-core.js');
if(!B)throw new Error('world-bridge.js must load before sim-core.js');
if(!A)throw new Error('mesh-assembly.js must load before sim-core.js');
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const dist=(a,b)=>Math.hypot((a.x||0)-(b.x||0),(a.z||0)-(b.z||0));
const deepClone=v=>JSON.parse(JSON.stringify(v));
function hash32(s){let h=0x811c9dc5;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,0x01000193)>>>0}return h>>>0}
const ACTION_LABEL=Object.freeze({walk:'Walk',inspect:'Inspect',eat:'Eat',sit:'Sit',talk:'Talk',tend:'Tend',morph:'Morph',idle:'Observe'});
const RELATION_SCALES=Object.freeze(['bit','binary','trinary','line','gate','filter','dimension']);
function relationScaleForCount(count){return RELATION_SCALES[Math.min(RELATION_SCALES.length-1,Math.max(0,(count||1)-1))]}
function positionLabel(p){const x=p.x||0,z=p.z||0;return `${x<0?'west':'east'}-${z<0?'north':'south'}`}
function actionWhy(agent,type){if(type==='eat')return 'hunger';if(type==='sit')return agent.needs.energy<agent.needs.comfort?'energy':'comfort';if(type==='talk')return 'connection';if(type==='tend')return 'care';if(type==='walk')return 'reach';if(type==='morph')return 'reshape';if(type==='inspect')return 'understand';return 'remain'}
function contextFor(agent,action,target,world){return {who:agent.id,what:target?(target.type||'agent'):'self',where:positionLabel(agent.position),when:'situation-'+Math.floor(world.tick/6),why:actionWhy(agent,action),perspective:'five-simultaneous',position:`${agent.position.x.toFixed(1)},${agent.position.z.toFixed(1)}`}}

// One event seed resolves to one complete coordinate in EACH of the five simultaneous dimensional views.
function completeAddressBundleFromKey(agent,action,target,tick){
  const key=`${agent.id}|${agent.seed}|${action}|${target?target.id:'none'}|${Math.floor(tick/2)}`;
  const h=hash32(key),h2=hash32(key+'|granular'),h3=hash32(key+'|arc');
  const common={
    planetary:K.PLANETARIES[(h>>>18)%K.PLANETARIES.length],gate:(h%64)+1,line:((h>>>6)%6)+1,color:((h>>>9)%6)+1,tone:((h>>>12)%6)+1,base:((h>>>15)%5)+1,
    degree:h2%30,minute:(h2>>>7)%60,second:(h2>>>13)%60,arc:h3%100,zodiac:K.ZODIACS[(h2>>>20)%12],house:((h3>>>11)%12)+1
  };
  return new A.FiveDimensionalState(K.DIMENSIONS.map(d=>new K.FullAddress(Object.assign({},common,{dimension:d}))));
}
function applyBundlePressure(agent,bundle,intensity,source){return bundle.addresses.map(addr=>agent.mesh.applyPressure(addr,intensity,source+':'+addr.dimension))}
function bundleResolution(agent,bundle){const views=bundle.addresses.map(addr=>agent.mesh.resolveAddress(addr));const signal=views.reduce((s,r)=>s+r.bit.charge+r.bit.tension*.5,0)/views.length;const specialty=views.reduce((s,r)=>s+r.specialty.microBias,0)/views.length;return {views,signal,specialty}}
function canonicalChannelFor(a,b){const id=[a.gate,b.gate].sort((x,y)=>x-y).join('-');return K.CHANNELS.find(c=>c.id===id)||null}
function connectBundles(agent,source,target,scale,metadata){
  const rels=[];
  for(const d of K.DIMENSIONS){
    const a=source.get(d),b=target.get(d);
    try{rels.push(agent.mesh.connect(a,b,scale,Object.assign({},metadata,{perspective:d})))}catch(_){/* invalid projection stays absent */}
    const channel=canonicalChannelFor(a,b);
    if(channel&&scale!=='channel'){
      try{rels.push(agent.mesh.connect(a,b,'channel',Object.assign({},metadata,{perspective:d,type:'channel:'+channel.id,channelName:channel.name,threshold:.7})))}catch(_){/* only canonical channels may form */}
    }
  }
  return rels;
}

class SimAgent{
  constructor({id,name,seed,x,z,player=false}){
    this.id=id;this.name=name;this.seed=seed>>>0;this.mesh=new K.AgentomagotchiField({seed:this.seed});this.assemblies=new A.CapabilityAssemblyRuntime({agentId:id,seed:this.seed});
    this.position={x,y:.8,z};this.target={x,z};this.player=player;this.speed=player?2.7:1.9;
    this.needs={energy:.82,hunger:.78,social:.62,comfort:.7};this.relationships={};this.interactionCounts={};this.memory=[];
    this.activeState=completeAddressBundleFromKey(this,'idle',null,0);this.currentAction={type:'idle',label:'Observe',until:0,targetId:null,state:this.activeState};
    this.lastDecisionAt=0;this.lastDirectPressureTick=-1;this.lastCapabilityExecutionAt=-999;this.lastCapabilityResolution=null;
  }
  remember(entry){this.memory.push(entry);if(this.memory.length>100)this.memory.shift()}
  capabilityBonus(action,targetKind='*'){return this.assemblies.bonus(action,{what:targetKind})}
  snapshot(){return {id:this.id,name:this.name,seed:this.seed,position:deepClone(this.position),needs:deepClone(this.needs),relationships:deepClone(this.relationships),interactionCounts:deepClone(this.interactionCounts),activeState:this.activeState.toObjects(),currentAction:{type:this.currentAction.type,label:this.currentAction.label,until:this.currentAction.until,targetId:this.currentAction.targetId,state:this.currentAction.state?this.currentAction.state.toObjects():null,origin:this.currentAction.origin||'world',capabilityId:this.currentAction.capabilityId||null},memory:deepClone(this.memory),assemblyRuntime:this.assemblies.snapshot(),lastCapabilityResolution:deepClone(this.lastCapabilityResolution)}}
}
class WorldObject{constructor(spec){Object.assign(this,{kind:'object',usable:true,removed:false,morphState:0},spec)}}

class AgentomagotchiWorld{
  constructor({seed=0xA7032026}={}){
    this.seed=seed>>>0;this.rng=new K.XorShift32(this.seed);this.time=0;this.tick=0;this.live=true;this.mode='LIVE';this.events=[];this.selectedId='agent-a';this.worldLedger=[];this.bridge=new B.WorldCapabilityBridge();
    this.agents=[new SimAgent({id:'agent-a',name:'Adaya Model',seed:this.seed^0xA11DA,x:-2.5,z:1.6,player:true}),new SimAgent({id:'agent-b',name:'Mirror Model',seed:this.seed^0xB00B5,x:2.6,z:-1.2,player:false})];
    this.objects=[new WorldObject({id:'chair',name:'Chair',type:'chair',x:-1.2,z:-2.2,sx:.9,sy:1,sz:.9}),new WorldObject({id:'table',name:'Table',type:'table',x:1,z:-2,sx:2.2,sy:1,sz:1.25}),new WorldObject({id:'food',name:'Food Bowl',type:'food',x:1.2,z:-1.6,sx:.6,sy:.3,sz:.6}),new WorldObject({id:'plant',name:'Plant',type:'plant',x:3.7,z:2.7,sx:.7,sy:1.4,sz:.7}),new WorldObject({id:'door',name:'Door',type:'door',x:-4.7,z:0,sx:.3,sy:2.2,sz:1.4}),new WorldObject({id:'beacon',name:'Goal Beacon',type:'beacon',x:4,z:-3.2,sx:.5,sy:1.6,sz:.5})];
    this.post('GENESIS','World entered post-deterministic history.',null,null);
  }
  agent(id){return this.agents.find(a=>a.id===id)} object(id){return this.objects.find(o=>o.id===id&&!o.removed)} entity(id){return this.agent(id)||this.object(id)} selected(){return this.entity(this.selectedId)}
  post(type,text,agent,state,extra){const bundle=state instanceof A.FiveDimensionalState?state:null;const e={id:'W'+(this.worldLedger.length+1),tick:this.tick,time:this.time,type,text,agentId:agent?agent.id:null,fiveDimensionalState:bundle?bundle.toObjects():null,fiveDimensionalStateStrings:bundle?bundle.toStrings():null,extra:extra||null};this.events.push(e);this.worldLedger.push(e);if(this.events.length>140)this.events.shift();return e}
  setMode(mode){this.mode=mode;this.live=mode==='LIVE';this.post('MODE','Mode → '+mode,null,null)} select(id){if(this.entity(id))this.selectedId=id}
  movePlayer(dx,dz,dt){const a=this.agents.find(x=>x.player);if(!a)return;const len=Math.hypot(dx,dz)||1,bonus=a.capabilityBonus('walk','self'),speed=a.speed*(1+bonus);a.position.x=clamp(a.position.x+dx/len*speed*dt,-4.25,4.25);a.position.z=clamp(a.position.z+dz/len*speed*dt,-4.25,4.25);a.target={x:a.position.x,z:a.position.z};a.currentAction={type:'walk',label:'Direct movement',until:this.time+.15,targetId:null,state:a.activeState,origin:'direct'};if(a.lastDirectPressureTick!==this.tick){const b=completeAddressBundleFromKey(a,'walk',null,this.tick);a.activeState=b;a.currentAction.state=b;applyBundlePressure(a,b,.12,'direct-control');a.lastDirectPressureTick=this.tick}}
  decayNeeds(a,dt){a.needs.energy=clamp(a.needs.energy-dt*.004,0,1);a.needs.hunger=clamp(a.needs.hunger-dt*.006,0,1);a.needs.social=clamp(a.needs.social-dt*.0025,0,1);a.needs.comfort=clamp(a.needs.comfort-dt*.002,0,1);a.assemblies.decay(dt*.003)}
  visibleObjects(){return this.objects.filter(o=>!o.removed)}

  // Ordinary world affordances remain legal possibilities. They are not the learned capability list.
  affordances(a){const arr=[];for(const o of this.visibleObjects()){const d=dist(a.position,o);if(d>7.5)continue;arr.push({type:'inspect',target:o,base:.18+1/(2+d)});if(o.type==='food')arr.push({type:'eat',target:o,base:(1-a.needs.hunger)*1.8+.15});if(o.type==='chair')arr.push({type:'sit',target:o,base:(1-a.needs.comfort)*1.2+(1-a.needs.energy)*.7+.1});if(o.type==='plant')arr.push({type:'tend',target:o,base:.18});if(d>1.25)arr.push({type:'walk',target:o,base:.22+1/(2+d)})}for(const other of this.agents){if(other===a)continue;const d=dist(a.position,other.position);arr.push({type:d>1.5?'walk':'talk',target:other,base:d>1.5?.25:(1-a.needs.social)*1.6+.2})}arr.push({type:'idle',target:null,base:.12+(a.needs.energy<.2?.8:0)});return arr.filter(x=>x.base>0)}
  evaluate(a,aff){const state=completeAddressBundleFromKey(a,aff.type,aff.target,this.tick),r=bundleResolution(a,state);let need=0;if(aff.type==='eat')need=1-a.needs.hunger;if(aff.type==='sit')need=(1-a.needs.comfort)+(1-a.needs.energy)*.5;if(aff.type==='talk')need=1-a.needs.social;if(aff.type==='walk')need=.12;if(aff.type==='inspect')need=.08;const kind=aff.target?(aff.target.type||'agent'):'self',cap=a.capabilityBonus(aff.type,kind);return {aff,state,resolved:r,weight:Math.max(.01,aff.base+need+r.signal*.35+(r.specialty-.5)*.12+cap)}}
  choose(cands){const total=cands.reduce((s,c)=>s+c.weight,0);let x=this.rng.next()*total;for(const c of cands){x-=c.weight;if(x<=0)return c}return cands[cands.length-1]}
  decide(a){const chosen=this.choose(this.affordances(a).map(x=>this.evaluate(a,x))),pressure=clamp(.18+(1-a.needs.energy)*.18+(1-a.needs.hunger)*.22+(1-a.needs.social)*.12,0,1),kernelEvents=applyBundlePressure(a,chosen.state,pressure,'sim-affordance:'+chosen.aff.type);a.activeState=chosen.state;a.lastDecisionAt=this.time;this.beginAction(a,chosen.aff,kernelEvents,{origin:'world'})}
  beginAction(a,aff,kernelEvents,meta={}){const t=aff.target,type=aff.type,kind=t?(t.type||'agent'):'self',bonus=a.capabilityBonus(type,kind),baseDur={walk:2,inspect:1.5,eat:1.2,sit:2.5,talk:2,tend:2.2,idle:1.3}[type]||1.5,duration=baseDur/(1+bonus);a.currentAction={type,label:ACTION_LABEL[type]||type,until:this.time+duration,targetId:t?t.id:null,state:a.activeState,kernelSelections:(kernelEvents||[]).map(e=>({dimension:e.fullAddress.dimension,selected:e.selected,probabilityRoll:e.probabilityRoll})),origin:meta.origin||'world',capabilityId:meta.capabilityId||null,composition:meta.composition||null};if(type==='walk'&&t)a.target={x:t.position?t.position.x:t.x,z:t.position?t.position.z:t.z};this.post(meta.origin==='capability'?'CAPABILITY_DECISION':'DECISION',`${a.name}: ${a.currentAction.label}${t?' → '+t.name:''}`,a,a.activeState,{kernelSelections:a.currentAction.kernelSelections,capabilityBonus:bonus,origin:a.currentAction.origin,capabilityId:a.currentAction.capabilityId,composition:a.currentAction.composition})}
  outcomeContext(a,action,target){return contextFor(a,action,target,this)}

  completeAction(a){const action=a.currentAction,target=this.entity(action.targetId);let success=true;
    switch(action.type){case'eat':if(target){a.needs.hunger=clamp(a.needs.hunger+.42,0,1);a.needs.comfort=clamp(a.needs.comfort+.05,0,1)}else success=false;break;case'sit':a.needs.comfort=clamp(a.needs.comfort+.28,0,1);a.needs.energy=clamp(a.needs.energy+.13,0,1);break;case'talk':if(target&&target.id){a.needs.social=clamp(a.needs.social+.31,0,1);if(target.needs)target.needs.social=clamp(target.needs.social+.16,0,1);a.relationships[target.id]=(a.relationships[target.id]||0)+1}else success=false;break;case'tend':if(target)target.morphState=(target.morphState+1)%3;a.needs.comfort=clamp(a.needs.comfort+.08,0,1);break;case'inspect':a.needs.comfort=clamp(a.needs.comfort+.025,0,1);break;case'idle':a.needs.energy=clamp(a.needs.energy+.04,0,1);break}
    const state=action.state instanceof A.FiveDimensionalState?action.state:a.activeState,ctx=this.outcomeContext(a,action.type,target),targetState=target instanceof SimAgent?target.activeState:(target?completeAddressBundleFromKey(a,'inspect',target,0):completeAddressBundleFromKey(a,'idle',null,0));let pointRelations=[];
    if(target){const key=`${action.type}|${target.id}`,count=a.interactionCounts[key]=(a.interactionCounts[key]||0)+1,scale=relationScaleForCount(count);pointRelations=connectBundles(a,state,targetState,scale,{type:action.type,intensity:clamp(.22+count*.035,0,1),verified:success,action:action.type,targetId:target.id,count,context:ctx})}
    const composition=action.composition||B.compositionForOutcome(action.type,target?(target.type||'agent'):'self');
    const observed=a.assemblies.observeOutcome({action:action.type,targetId:target?target.id:'self',targetKind:target?(target.type||'agent'):'self',context:ctx,sourceBundle:state,targetBundle:targetState,pointRelations,composition,success,utility:success?.82:.15});
    a.remember({tick:this.tick,action:action.type,composition:Array.from(composition),targetId:action.targetId,fiveDimensionalState:state.toObjects(),relationEdge:observed.edge.id,assembly:observed.assembly?observed.assembly.id:null,capability:observed.capability?observed.capability.id:null,origin:action.origin||'world'});
    this.post(action.origin==='capability'?'CAPABILITY_OUTCOME':'OUTCOME',`${a.name}: ${ACTION_LABEL[action.type]||action.type} became history.`,a,state,{composition:Array.from(composition),relationEdge:observed.edge.id,pointRelations:pointRelations.map(r=>r.id),assembly:observed.assembly?observed.assembly.id:null,capability:observed.capability?observed.capability.id:null,executedByCapability:action.capabilityId||null});
    if(observed.assembly&&observed.assembly.formedSeq===a.assemblies.seq)this.post('ASSEMBLY',`${a.name}: ${observed.assembly.id} recruited ${observed.assembly.memberAddresses.length} addressed members.`,a,state,{assembly:observed.assembly.id,composition:observed.assembly.composition,genome:observed.assembly.genome});
    if(observed.capability&&observed.capability.uses===0)this.post('CAPABILITY',`${a.name}: ${observed.capability.id} became reusable: ${observed.capability.composition.join(' → ')}.`,a,state,{capability:observed.capability});
    a.currentAction={type:'idle',label:'Observe',until:this.time+.2,targetId:null,state:a.activeState,origin:'world'};
  }

  updateAgentMotion(a,dt){if(a.currentAction.type!=='walk')return;const dx=a.target.x-a.position.x,dz=a.target.z-a.position.z,d=Math.hypot(dx,dz);if(d<.18){a.position.x=a.target.x;a.position.z=a.target.z;a.currentAction.until=Math.min(a.currentAction.until,this.time);return}const kind=a.currentAction.targetId?(this.entity(a.currentAction.targetId)?.type||'agent'):'self',sp=a.speed*(1+a.capabilityBonus('walk',kind)),step=Math.min(d,sp*dt);a.position.x+=dx/d*step;a.position.z+=dz/d*step}

  applyMorphOp(a,ent,op,{origin='direct',capabilityId=null}={}){
    if(!ent||ent instanceof SimAgent||ent.removed)return false;
    const state=completeAddressBundleFromKey(a,'morph',ent,this.tick),ke=applyBundlePressure(a,state,.55,origin+':morph:'+op);a.activeState=state;
    let success=true;
    if(op==='move'){const ox=ent.x,oz=ent.z;ent.x=clamp(ent.x+1.1,-4,4);ent.z=clamp(ent.z+.7,-4,4);success=ent.x!==ox||ent.z!==oz}
    else if(op==='transform'){ent.morphState=(ent.morphState+1)%3;ent.sy=[.72,1,1.35][ent.morphState]*(ent.type==='plant'?1.25:1)}
    else if(op==='remove'){if(ent.type!=='door')ent.removed=true;else success=false}
    else if(op==='duplicate'){const copy=new WorldObject(Object.assign({},deepClone(ent),{id:ent.id+'-'+(this.objects.length+1),name:ent.name+' copy',x:clamp(ent.x+.8,-4,4),z:clamp(ent.z+.8,-4,4),removed:false}));this.objects.push(copy);if(origin==='direct')this.selectedId=copy.id}
    else success=false;
    const key=`morph:${op}|${ent.id}`,count=a.interactionCounts[key]=(a.interactionCounts[key]||0)+1,targetState=completeAddressBundleFromKey(a,'inspect',ent,0),rels=connectBundles(a,state,targetState,relationScaleForCount(count),{type:'morph:'+op,intensity:clamp(.28+count*.04,0,1),verified:success,targetId:ent.id,count}),ctx=contextFor(a,'morph',ent,this),composition=B.compositionForOutcome('morph',ent.type,{op}),observed=a.assemblies.observeOutcome({action:'morph',targetId:ent.id,targetKind:ent.type,context:ctx,sourceBundle:state,targetBundle:targetState,pointRelations:rels,composition,success,utility:success?.88:.1});
    this.post(origin==='capability'?'EMERGENT_EXECUTION':'MORPH',`${origin==='capability'?'Emergent capability':'Morph'} ${op}: ${ent.name}`,a,state,{origin,capabilityId,kernelSelections:ke.map(e=>e.selected),composition:Array.from(composition),relationEdge:observed.edge.id,pointRelations:rels.map(r=>r.id),assembly:observed.assembly?observed.assembly.id:null,capability:observed.capability?observed.capability.id:null});
    if(observed.capability&&observed.capability.uses===0)this.post('CAPABILITY',`${a.name}: ${observed.capability.id} became reusable: ${observed.capability.composition.join(' → ')}.`,a,state,{capability:observed.capability});
    return success;
  }

  executeBestCapability(a,manual=false){
    const caps=[...a.assemblies.capabilities.values()];if(!caps.length){if(manual)this.post('INFO',`${a.name} has no promoted capability yet.`,a,a.activeState);return false}
    const resolved=this.bridge.resolve(a,this,caps,this.rng);a.lastCapabilityResolution=deepClone(this.bridge.lastResolution);if(!resolved){if(manual)this.post('INFO','No promoted capability has a legal affordance in this situation.',a,a.activeState);return false}
    const {capability:cap,possibility:p}=resolved,target=this.entity(p.targetId);a.lastCapabilityExecutionAt=this.time;cap.uses=(cap.uses||0)+1;
    this.post('CAPABILITY_RESOLUTION',`${a.name}: ${cap.id} resolved ${p.operation}${target?' → '+target.name:''}.`,a,a.activeState,{capabilityId:cap.id,composition:Array.from(cap.composition||[]),effect:p.effect,operation:p.operation,targetId:p.targetId,possibilityCount:this.bridge.lastResolution.possibilityCount,probabilityRoll:resolved.roll});
    if(p.operation==='move-object')return this.finishCapabilityExecution(a,cap,p,this.applyMorphOp(a,target,'move',{origin:'capability',capabilityId:cap.id}));
    if(p.operation==='transform-object')return this.finishCapabilityExecution(a,cap,p,this.applyMorphOp(a,target,'transform',{origin:'capability',capabilityId:cap.id}));
    if(p.operation==='duplicate-object')return this.finishCapabilityExecution(a,cap,p,this.applyMorphOp(a,target,'duplicate',{origin:'capability',capabilityId:cap.id}));
    if(p.operation==='remove-object')return this.finishCapabilityExecution(a,cap,p,this.applyMorphOp(a,target,'remove',{origin:'capability',capabilityId:cap.id}));
    const map={approach:'walk',consume:'eat',recover:'sit',exchange:'talk',tend:'tend',inspect:'inspect'},type=map[p.operation]||'inspect';
    const state=completeAddressBundleFromKey(a,'capability:'+cap.id,target,this.tick),ke=applyBundlePressure(a,state,.46,'capability-world:'+cap.id);a.activeState=state;this.beginAction(a,{type,target,base:1},ke,{origin:'capability',capabilityId:cap.id,composition:Array.from(cap.composition||[])});a.currentAction.until=Math.min(a.currentAction.until,this.time+.95);this.bridge.noteExecution({capabilityId:cap.id,operation:p.operation,targetId:p.targetId,queued:true});a.lastCapabilityResolution=deepClone(this.bridge.lastResolution);return true;
  }
  finishCapabilityExecution(a,cap,p,success){this.bridge.noteExecution({capabilityId:cap.id,operation:p.operation,targetId:p.targetId,success});a.lastCapabilityResolution=deepClone(this.bridge.lastResolution);return success}

  step(dt){dt=Math.min(.1,Math.max(0,dt||0));this.time+=dt;for(const a of this.agents){this.decayNeeds(a,dt);this.updateAgentMotion(a,dt);if(a.player&&this.mode==='PLAY')continue;if(this.mode==='MORPH')continue;if(this.time>=a.currentAction.until){if(a.currentAction.type!=='idle')this.completeAction(a);else if(this.time-a.lastDecisionAt>.6){const ready=a.assemblies.capabilities.size>0&&this.time-a.lastCapabilityExecutionAt>2.6;const emerged=ready&&this.rng.next()<.38&&this.executeBestCapability(a,false);if(!emerged)this.decide(a)}}}if(this.time>this.tick*.5)this.tick++}
  interactPlayer(){const a=this.agents.find(x=>x.player),targets=[...this.visibleObjects(),...this.agents.filter(x=>x!==a)].sort((x,y)=>dist(a.position,x.position||x)-dist(a.position,y.position||y)),t=targets[0];if(!t||dist(a.position,t.position||t)>1.8){this.post('INFO','Nothing close enough to interact with.',a,a.activeState);return}let type='inspect';if(t.type==='food')type='eat';else if(t.type==='chair')type='sit';else if(t.needs)type='talk';else if(t.type==='plant')type='tend';const aff={type,target:t,base:1},state=completeAddressBundleFromKey(a,type,t,this.tick),ke=applyBundlePressure(a,state,.34,'direct-interaction');a.activeState=state;this.beginAction(a,aff,ke,{origin:'direct'});a.currentAction.until=this.time+.7}
  morphSelected(op){const ent=this.selected();if(!ent||ent instanceof SimAgent)return false;const a=this.agents.find(x=>x.player);return this.applyMorphOp(a,ent,op,{origin:'direct'})}
  executeSelectedCapability(){const ent=this.selected(),a=ent instanceof SimAgent?ent:this.agents.find(x=>x.player);return this.executeBestCapability(a,true)}
  save(){const s={seed:this.seed,time:this.time,tick:this.tick,mode:this.mode,selectedId:this.selectedId,agents:this.agents.map(a=>a.snapshot()),objects:deepClone(this.objects),events:deepClone(this.worldLedger),bridge:deepClone(this.bridge.lastResolution)};localStorage.setItem('agentomagotchi-v07-save',JSON.stringify(s));return s}
  summary(){return {version:'0.7.0',tick:this.tick,time:this.time,mode:this.mode,addressedPoints:this.agents.reduce((s,a)=>s+a.mesh.addressedPoints,0),potentialExactAddressesPerAgent:this.agents[0]?this.agents[0].mesh.potentialExactAddressesString:'0',assemblies:this.agents.reduce((s,a)=>s+a.assemblies.assemblies.size,0),capabilities:this.agents.reduce((s,a)=>s+a.assemblies.capabilities.size,0),capabilityExecutions:this.worldLedger.filter(e=>e.type==='EMERGENT_EXECUTION'||e.type==='CAPABILITY_OUTCOME').length,events:this.worldLedger.length}}
}

global.AgentomagotchiSim=Object.freeze({VERSION:'0.7.0',AgentomagotchiWorld,SimAgent,WorldObject,completeAddressBundleFromKey,applyBundlePressure,bundleResolution,connectBundles,canonicalChannelFor,relationScaleForCount,ACTION_LABEL});
})(typeof window!=='undefined'?window:globalThis);
