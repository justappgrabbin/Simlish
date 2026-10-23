/** Source-locked addition to the existing positioned state-space.
 * Authority: user's Google Doc 1pRXTJGn5gwZvYuX2NQwKinM72HM2pU8arUS7KKK-nh0,
 * final recap, sections 1-6 (WHO/WHAT/WHERE/WHEN/WHY, identity vs placement,
 * dominating frame, Being-led traversal, landed history, recursive primitives).
 * Only the rules ACTUALLY PROVIDED are encoded. In particular, the document
 * explicitly leaves the exact dimensional D/M/S <-> C/T/B arc equation open.
 */
import {resolvePosition} from './positioned-state-space.mjs';

const copy = x => structuredClone(x);
const FROZEN_KEY = Object.freeze({
  WHO:Object.freeze({dimension:'Space',nature:'Form',term:'Axon'}),
  WHAT:Object.freeze({dimension:'Evolution',nature:'Gravity',term:'Axion'}),
  WHERE:Object.freeze({dimension:'Being',nature:'Matter',term:'Arc-second'}),
  WHEN:Object.freeze({dimension:'Movement',nature:'Energy',term:'Axiom'}),
  WHY:Object.freeze({dimension:'Design',nature:'Structure',term:'Axis'})
});
const byDim = Object.fromEntries(Object.entries(FROZEN_KEY).map(([question,r])=>[r.dimension,{question,...r}]));
const byTerm = Object.fromEntries(Object.entries(FROZEN_KEY).map(([question,r])=>[r.term.toLowerCase(),{question,...r}]));
const BEING_ENTRY_ORDER=Object.freeze(['Being','Movement','Space','Design','Evolution']);
const key = id => {if(typeof id!=='string'||!id.trim())throw new TypeError('nonempty identity id required');return id;};
const dim = x => {if(!Object.hasOwn(byDim,x))throw new RangeError(`unknown intrinsic dimension: ${x}`);return x;};
const source = 'Google Doc 1pRXTJGn5gwZvYuX2NQwKinM72HM2pU8arUS7KKK-nh0, final dimensional/positional recap';

export function fixedDimensionalKey(){return copy(FROZEN_KEY);}
export function dimensionalIdentity(dimension){return copy(byDim[dim(dimension)]);}
export function beingLedTraversal(){return [...BEING_ENTRY_ORDER].map(d=>dimensionalIdentity(d));}
export function frameReading({subject,frame,slot}){
  if(!subject||!frame)throw new TypeError('subject and containing frame required');
  const slotRole=byTerm[String(slot||'').toLowerCase()];
  if(!slotRole)throw new RangeError(`unknown relational slot: ${slot}`);
  const identity=dimensionalIdentity(subject.dimension);
  const dominant=dimensionalIdentity(frame.dimension);
  return {
    subject:{id:key(subject.id),identity},
    placement:{frameId:key(frame.id),slot:slotRole.term,slotQuality:copy(slotRole)},
    dominatingFrame:{id:frame.id,quality:dominant},
    // The document specifies one FULL traversal order: when Being leads.
    traversal:frame.dimension==='Being'?beingLedTraversal():null,
    traversalStatus:frame.dimension==='Being'?'source-specified':'not specified in this document',
    preservation:'intrinsic identity is not replaced by the relational slot',
    source
  };
}

/** Companion of the already existing PositionedStateSpace, not a new address
 * authority. Nodes may have a verified 13-field address or null when no
 * coordinates were supplied; no zodiac/house/arc is synthesized here.
 * Every member may itself be entered as a dominating frame.
 */
export class PositionalRelationLedger {
  #nodes=new Map();
  #events=[];
  register({id,dimension,position=null,components=[]}){
    key(id);dim(dimension);
    if(this.#nodes.has(id))throw new Error(`duplicate node ${id}`);
    if(!Array.isArray(components))throw new TypeError('components array required');
    for(const child of components){
      if(!this.#nodes.has(child))throw new Error(`unknown constituent ${child}`);
      if(child===id)throw new Error('cannot contain itself');
    }
    const node={id,dimension,position:position==null?null:resolvePosition(position).position,components:[...components]};
    this.#nodes.set(id,copy(node));
    this.#events.push({index:this.#events.length,kind:'register',node:copy(node)});
    return this.get(id);
  }
  get(id){return this.#nodes.has(id)?copy(this.#nodes.get(id)):null;}
  place({subjectId,frameId,slot}){
    const subject=this.get(subjectId),frame=this.get(frameId);
    if(!subject||!frame)throw new Error('both the subject and dominating frame must already exist');
    if(subjectId===frameId)throw new Error('a node cannot occupy its own frame');
    // Explicit relational position, NEVER a rewrite of subject identity.
    const reading=frameReading({subject,frame,slot});
    this.#events.push({index:this.#events.length,kind:'placement',subjectId,frameId,slot:reading.placement.slot,reading:copy(reading)});
    return copy(reading);
  }
  enter(frameId){
    const frame=this.get(frameId);if(!frame)throw new Error('unknown frame');
    const readings=this.#events.filter(e=>e.kind==='placement'&&e.frameId===frameId).map(e=>copy(e.reading));
    return {frame,leadingQuality:dimensionalIdentity(frame.dimension),
      traversal:frame.dimension==='Being'?beingLedTraversal():null,
      readings,componentIds:[...frame.components]};
  }
  history(){return copy(this.#events);}
  checkpoint(){return {schema:'synthia-spec-locked-relations-v1',events:this.history()};}
  static restore(cp){
    if(cp?.schema!=='synthia-spec-locked-relations-v1'||!Array.isArray(cp.events))throw new Error('invalid ledger checkpoint');
    const l=new PositionalRelationLedger();
    for(const old of cp.events){
      if(old.index!==l.#events.length)throw new Error('history index mismatch');
      if(old.kind==='register')l.register(old.node);
      else if(old.kind==='placement')l.place({subjectId:old.subjectId,frameId:old.frameId,slot:old.slot});
      else throw new Error('unknown event kind');
      if(JSON.stringify(l.#events.at(-1))!==JSON.stringify(old))throw new Error('history proof mismatch');
    }
    return l;
  }
}
