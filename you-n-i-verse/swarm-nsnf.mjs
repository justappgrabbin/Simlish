// NSNF (Nested State-Space Neural Field v0.1) terms for swarm components, used as written in the spec:
//  component(id, gate_id, architecture_type, artifact_hash, parameter_ref, version)  (Spec §4)
//  event_log(id, actor, payload_hash, timestamp); transition(id, before_id, after_id, event_type, reason)
//  hexagram_code = six bits (Spec §3); component interface initialize/forward/update/snapshot/restore (Spec §7).
// Alexis's split: PSO = GENOTYPE (background tooling, the mechanical side, holds the hexagram_code and the
// checks); GA = PHENOTYPE (what is seen, the physical trait), expressed FROM a genotype and never without one.
import {SwarmComponent} from './swarm-component.mjs';
import {sha256} from './sha256.mjs';
export const INVESTIGATING=Object.freeze([
 'gate_id is null: the spec calls center/gate binding provisional (Spec §4/§19) and no gate has been assigned to these components.',
 'architecture_type "swarm-pso-genotype"/"swarm-ga-phenotype" are not among the spec\'s listed kinds (trainable cell, graph operator, model adapter).',
 'Pack count -> two-bit value and pack order in the six bits are not in Alexis\'s rules or the spec (Spec §3 says the 6-bit/64-bit role is a pending decision).',
 'Spec §7 update(loss/feedback) learning is NOT implemented: update() only logs feedback. No parameter_ref exists, so it is null.',
 'Whether PSO searches over the genotype (as GA normally evolves a genotype) is not settled; no PSO/GA search runs in these components yet.']);
export class EventLog{
 #entries=[];#n=0;
 append(actor,payload,now=()=>new Date().toISOString()){const e=Object.freeze({id:++this.#n,actor,payload_hash:sha256(JSON.stringify(payload)),timestamp:now()});this.#entries.push(e);return e}
 entries(){return [...this.#entries]}
}
export function componentRecord(c,architecture_type){
 return Object.freeze({id:c.id,gate_id:null,architecture_type,artifact_hash:sha256(JSON.stringify({id:c.id,kind:c.kind,members:c.packs.flat().map(m=>m.id)})),parameter_ref:null,version:'0.1'});
}
// PSO genotype unit: mechanics. Its output is the genotype: the hexagram_code plus the checks that made it.
export class GenotypeUnit{
 constructor(component){if(component.kind!=='PSO')throw new TypeError('A genotype unit is PSO');this.component=component;this.record=componentRecord(component,'swarm-pso-genotype')}
 initialize(){return this}
 forward(input){const r=this.component.runSync(input);return Object.freeze({genotype:true,component:r.component,hexagram_code:r.hexagram.bits,hexagram_number:r.hexagram.number,packs:r.packs.map(p=>p.value),status:r.status,failed:r.failed,notes:r.notes,record:this.record})}
}
// GA phenotype unit: what is seen. `trait(genotype,input)` builds the visible trait; the six GA members check it.
export class PhenotypeUnit{
 constructor({component,trait}){if(component.kind!=='GA')throw new TypeError('A phenotype unit is GA');if(typeof trait!=='function')throw new TypeError('trait function required');this.component=component;this.trait=trait;this.record=componentRecord(component,'swarm-ga-phenotype')}
 express(genotype,input){
  if(!genotype||genotype.genotype!==true)throw new TypeError('A phenotype is expressed from a genotype; none given');
  const visible=this.trait(genotype,input),check=this.component.runSync({visible,genotype,input});
  return Object.freeze({phenotype:true,component:this.component.id,genotype_ref:{hexagram_code:genotype.hexagram_code,artifact_hash:genotype.record.artifact_hash},visible,status:check.status,failed:check.failed,notes:check.notes,packs:check.packs.map(p=>p.value),hexagram_code:check.hexagram.bits,record:this.record});
 }
}
// A pair = one swarm building block: genotype (PSO) -> phenotype (GA), with an append-only event log.
export class SwarmPair{
 constructor({id,genotype,phenotype,log=new EventLog()}){this.id=id;this.genotype=new GenotypeUnit(genotype);this.phenotype=new PhenotypeUnit(phenotype);this.log=log;this.transitions=[];this.last=null}
 forward(input){
  const g=this.genotype.forward(input),p=this.phenotype.express(g,input);
  const e=this.log.append(this.id,{g:g.hexagram_code,p:p.hexagram_code,status:g.status});
  if(this.last&&this.last.status!==g.status)this.transitions.push(Object.freeze({id:this.transitions.length+1,before_id:this.last.event,after_id:e.id,event_type:'status-change',reason:this.last.status+' -> '+g.status}));
  this.last={status:g.status,event:e.id};
  return Object.freeze({genotype:g,phenotype:p,status:g.status,event:e});
 }
 update(feedback){return this.log.append(this.id,{feedback})}
 snapshot(){return Object.freeze({id:this.id,components:[this.genotype.record,this.phenotype.record],event_log:this.log.entries(),transitions:[...this.transitions],investigating:INVESTIGATING})}
 restore(){throw new Error('investigating: restore is not implemented')}
}
export {SwarmComponent};
