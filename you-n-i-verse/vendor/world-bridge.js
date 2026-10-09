(function(global){
'use strict';
const K=global.AgentomagotchiKernel;
if(!K) throw new Error('kernel.js must load before world-bridge.js');
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,Number(v)||0));
const dist=(a,b)=>Math.hypot((a.x||0)-(b.x||0),(a.z||0)-(b.z||0));
const clone=v=>JSON.parse(JSON.stringify(v));

// These are world-effect contracts, not learned skills. A capability is the
// evidence-backed composition that recruits one or more contracts in context.
const EFFECT_CONTRACTS=Object.freeze({
  OBSERVE:Object.freeze({id:'observe',changesWorld:false}),
  MOVE_SELF:Object.freeze({id:'move:self',changesWorld:true}),
  TRANSFER_RESOURCE:Object.freeze({id:'transfer:resource',changesWorld:true}),
  RESTORE_SELF:Object.freeze({id:'restore:self',changesWorld:true}),
  EXCHANGE_RELATION:Object.freeze({id:'exchange:relation',changesWorld:true}),
  ALTER_TARGET:Object.freeze({id:'alter:target',changesWorld:true}),
  ALTER_MOVE:Object.freeze({id:'alter:move',changesWorld:true}),
  ALTER_TRANSFORM:Object.freeze({id:'alter:transform',changesWorld:true}),
  ALTER_DUPLICATE:Object.freeze({id:'alter:duplicate',changesWorld:true}),
  ALTER_REMOVE:Object.freeze({id:'alter:remove',changesWorld:true}),
  RETAIN:Object.freeze({id:'retain',changesWorld:false})
});

function normalizeComposition(xs){
  return Object.freeze((Array.isArray(xs)?xs:[]).map(String).filter(Boolean));
}

function compositionForOutcome(action,targetKind,meta={}){
  switch(action){
    case 'walk': return normalizeComposition(['observe','orient','move:self']);
    case 'inspect': return normalizeComposition(['observe','model','retain']);
    case 'eat': return normalizeComposition(['observe','contact','transfer:resource','integrate']);
    case 'sit': return normalizeComposition(['observe','contact','restore:self','retain']);
    case 'talk': return normalizeComposition(['observe','signal','exchange:relation','retain']);
    case 'tend': return normalizeComposition(['observe','contact','alter:target','retain']);
    case 'morph': return normalizeComposition(['observe','select:'+String(targetKind||'target'),'alter:'+String(meta.op||'transform'),'retain']);
    default: return normalizeComposition(['observe','retain']);
  }
}

function capabilityEffect(capability){
  const c=normalizeComposition(capability&&capability.composition);
  const priority=['alter:remove','alter:duplicate','alter:transform','alter:move','alter:target','transfer:resource','restore:self','exchange:relation','move:self','observe'];
  return priority.find(x=>c.includes(x))||'observe';
}

function targetKindOf(t){return t?(t.type||t.kind||'agent'):'self'}

class WorldCapabilityBridge{
  constructor(){this.lastResolution=null;this.executionCount=0}

  situationPressure(agent,world){
    const nearest=world.visibleObjects().reduce((best,o)=>Math.min(best,dist(agent.position,o)),99);
    const other=world.agents.find(a=>a!==agent);
    return Object.freeze({
      resource:clamp(1-agent.needs.hunger),
      recovery:clamp(((1-agent.needs.energy)+(1-agent.needs.comfort))/2),
      relation:clamp(1-agent.needs.social),
      proximity:clamp(1-nearest/8),
      other:other?clamp(1-dist(agent.position,other.position)/8):0
    });
  }

  isTargetCompatible(cap,target){
    const wanted=cap&&cap.context?cap.context.what:null;
    if(!wanted||wanted==='*'||wanted==='self') return true;
    return targetKindOf(target)===wanted;
  }

  legalPossibilities(agent,world,capability){
    if(!capability) return [];
    const effect=capabilityEffect(capability),p=this.situationPressure(agent,world),out=[];
    const objects=world.visibleObjects();
    const others=world.agents.filter(a=>a!==agent);
    const push=(target,operation,legal,why,pressure=0)=>{
      if(!legal) return;
      const d=target?dist(agent.position,target.position||target):0;
      const relationSignal=target&&agent.mesh&&agent.mesh.relationSummary?(()=>{
        const rs=agent.mesh.relationSummary().top||[];
        return rs.length?rs[0].signal:0;
      })():0;
      out.push({effect,operation,targetId:target?target.id:null,targetKind:targetKindOf(target),distance:d,
        score:clamp(.16+(capability.confidence||0)*.48+pressure*.24+relationSignal*.12+(d?1/(4+d):.08)),
        why,composition:Array.from(capability.composition||[]),capabilityId:capability.id});
    };
    if(effect==='move:self'){
      for(const t of [...objects,...others]) if(this.isTargetCompatible(capability,t)) push(t,'approach',true,'reachable target',p.proximity);
    }else if(effect==='transfer:resource'){
      for(const o of objects) push(o,'consume',o.type==='food'&&this.isTargetCompatible(capability,o),'resource is available',p.resource);
    }else if(effect==='restore:self'){
      for(const o of objects) push(o,'recover',o.type==='chair'&&this.isTargetCompatible(capability,o),'rest support is available',p.recovery);
    }else if(effect==='exchange:relation'){
      for(const o of others) push(o,'exchange',this.isTargetCompatible(capability,o),'another agent is reachable',p.relation);
    }else if(effect==='alter:target'){
      for(const o of objects) push(o,'tend',o.usable!==false&&this.isTargetCompatible(capability,o),'target permits alteration',.45);
    }else if(effect==='alter:move'){
      for(const o of objects) push(o,'move-object',o.usable!==false&&this.isTargetCompatible(capability,o),'target permits relocation',.5);
    }else if(effect==='alter:transform'){
      for(const o of objects) push(o,'transform-object',o.usable!==false&&this.isTargetCompatible(capability,o),'target permits transformation',.5);
    }else if(effect==='alter:duplicate'){
      for(const o of objects) push(o,'duplicate-object',o.usable!==false&&this.isTargetCompatible(capability,o),'target permits replication',.5);
    }else if(effect==='alter:remove'){
      for(const o of objects) push(o,'remove-object',o.usable!==false&&o.type!=='door'&&this.isTargetCompatible(capability,o),'target may be removed',.5);
    }else{
      for(const t of [...objects,...others]) if(this.isTargetCompatible(capability,t)) push(t,'inspect',true,'target can be observed',.2);
    }
    return out.sort((a,b)=>b.score-a.score||String(a.targetId).localeCompare(String(b.targetId)));
  }

  resolve(agent,world,capabilities,rng){
    const all=[];
    for(const cap of capabilities||[]) for(const p of this.legalPossibilities(agent,world,cap)) all.push({cap,p});
    if(!all.length){this.lastResolution={status:'none',reason:'no capability has a legal world affordance',possibilities:[]};return null}
    const total=all.reduce((s,x)=>s+Math.max(.001,x.p.score),0),roll=rng.next();let cursor=roll*total,chosen=all[all.length-1];
    for(const x of all){cursor-=Math.max(.001,x.p.score);if(cursor<=0){chosen=x;break}}
    this.lastResolution={status:'resolved',roll,total,capabilityId:chosen.cap.id,effect:chosen.p.effect,operation:chosen.p.operation,targetId:chosen.p.targetId,possibilityCount:all.length,possibilities:all.slice(0,12).map(x=>clone(x.p))};
    return {capability:chosen.cap,possibility:chosen.p,roll,total};
  }

  noteExecution(result){this.executionCount++;this.lastResolution=Object.assign({},this.lastResolution||{},{status:'executed',executionCount:this.executionCount,result:clone(result)})}
}

global.AgentomagotchiWorldBridge=Object.freeze({VERSION:'0.7.0',EFFECT_CONTRACTS,WorldCapabilityBridge,compositionForOutcome,capabilityEffect,normalizeComposition});
})(typeof window!=='undefined'?window:globalThis);
