// Swarm component: the building block for register, finisher and personalize.
// Alexis's rules used here: packs are 3, 2, 1; each pack is two bits; three packs make one
// six-bit hexagram; pieces build into 2-bit, 4-bit and 8-bit sizes; GA swarms make what is seen
// on screen; PSO swarms make background tooling; uncertainty triggers investigation, not rejection.
// INVESTIGATING (not stated by Alexis; chosen here so something can run, flagged in the report):
//  - a pack's two-bit value is the COUNT of its members that pass (3 members -> 0..3, 2 -> 0..2, 1 -> 0..1);
//  - pack order in the six bits is pack3, pack2, pack1 (first pack = first two bits).
import {hexagram} from './vendor/klein/iching.mjs';
export const PACK_SHAPE=Object.freeze([3,2,1]);
export const PACK_BITS=2;
export const PIECE_WIDTHS=Object.freeze([2,4,8]);
export const SWARM_KINDS=Object.freeze({GA:'visible-on-screen',PSO:'background-tooling'});
export function packValue(count){
 if(!Number.isInteger(count)||count<0||count>3)throw new RangeError('A pack value is two bits: 0..3');
 return count;
}
export function piece(width,value){
 if(!PIECE_WIDTHS.includes(width))throw new RangeError('Pieces are 2, 4 or 8 bits');
 if(!Number.isInteger(value)||value<0||value>=2**width)throw new RangeError('Value does not fit '+width+' bits');
 return Object.freeze({width,value,bits:value.toString(2).padStart(width,'0')});
}
export function hexagramFromPacks(values){
 if(!Array.isArray(values)||values.length!==3)throw new TypeError('Three packs make one hexagram');
 const bits=values.map(v=>packValue(v).toString(2).padStart(PACK_BITS,'0')).join('');
 return hexagram(bits);
}
// members: six {id,check(input)=>boolean|{ok,note}}; first 3 form pack A, next 2 pack B, last 1 pack C.
export class SwarmComponent{
 constructor({id,kind,purpose,members,openQuestions=[]}){
  if(!SWARM_KINDS[kind])throw new TypeError('kind must be GA or PSO');
  if(!Array.isArray(members)||members.length!==6)throw new TypeError('A swarm component has six members in packs of 3, 2 and 1');
  this.id=id;this.kind=kind;this.role=SWARM_KINDS[kind];this.purpose=purpose;this.openQuestions=Object.freeze([...openQuestions]);
  let at=0;this.packs=PACK_SHAPE.map(size=>{const m=members.slice(at,at+size);at+=size;return m});
 }
 runSync(input){
  const packs=[],notes=[],failed=[];
  for(const pack of this.packs){
   const results=pack.map(m=>{try{const r=m.check(input);if(r&&typeof r.then==='function')return {id:m.id,ok:false,note:'investigating: async check needs run()'};return typeof r==='object'&&r!==null?{id:m.id,ok:!!r.ok,note:r.note}:{id:m.id,ok:!!r}}catch(e){return {id:m.id,ok:false,note:'investigating: '+e.message}}});
   for(const r of results){if(!r.ok){failed.push(r.id);if(r.note)notes.push(r.id+': '+r.note)}}
   packs.push({members:results,value:packValue(results.filter(r=>r.ok).length)});
  }
  return this.#result(packs,failed,notes);
 }
 async run(input){
  const packs=[],notes=[],failed=[];
  for(const pack of this.packs){
   const results=await Promise.all(pack.map(async m=>{
    try{const r=await m.check(input);return typeof r==='object'&&r!==null?{id:m.id,ok:!!r.ok,note:r.note}:{id:m.id,ok:!!r}}
    catch(e){return {id:m.id,ok:false,note:'investigating: '+e.message}}}));
   for(const r of results){if(!r.ok){failed.push(r.id);if(r.note)notes.push(r.id+': '+r.note)}}
   packs.push({members:results,value:packValue(results.filter(r=>r.ok).length)});
  }
  return this.#result(packs,failed,notes);
 }
 #result(packs,failed,notes){
  const hx=hexagramFromPacks(packs.map(p=>p.value));
  return Object.freeze({component:this.id,kind:this.kind,role:this.role,packs,hexagram:{number:hx.number,bits:hx.bits},
   status:failed.length?'investigating':'complete',failed,notes,swarmBuilt:true,
   search:'none: members are deterministic checks; no GA/PSO search is run by this component'});
 }
}
