// The primitive population: a DIVERSE set of differentiated primitives sharing ONE ontology and ONE set of composition/resolution/execution rules.
// Alexis: "ONE ontology does NOT mean ONE primitive". Nothing here is an app, a file engine or a controller; nothing app-specific is in this file.
//
// SHARED (unity): every primitive is a node of the nested state space (state-space.mjs: five dimensions, 64 macro slots per level) holding a six-bit hexagram state
//   (line 1 = bit 0 = bottom, packs 3-2-1, two bits per pack); all are addressed by (dimension, path); all are composed by the same rule (a composite of <=64 cells
//   acts as ONE primitive of the next level, with an emergent dimension and six-bit code); all execute by the same tick rule (a signal read from a later position is
//   the previous tick's value: feedback/memory; no scheduler, no controller: each cell reacts to the signals it is wired to); all state is six-bit atoms.
// DIFFERENTIATED (diversity): the four writable dimensions are different kinds of primitive. Roles are from Alexis's ontological-ingest patch (ONTOLOGICAL-INGEST-PATCH.md):
//   "Movement is a directional/change field and records down/up polarity. Being is a persistence/position field and records left/right polarity.
//    Evolution measures change-of-change. Design measures internal organization. Space ... derived from the interaction of the other four."
// INVESTIGATING (her files give the roles above, not executable semantics; the mapping of each role to the rules below is an engineering reading, to be confirmed):
export const POPULATION_INVESTIGATING=Object.freeze([
 'executable semantics of the four dimension roles (Movement=transition table, Evolution=accumulated change, Being=held value, Design=routing and instantiation of organization) are an engineering reading of her one-line role descriptions.',
 'which hexagram lines carry which parameter is not specified by Alexis: lines 1-4 as a two-input table (Movement, Evolution), line 1 as invert flag (Being), line 6 + lines 1-3 as "instantiate organization #n" (Design), lines 5-6 polarity are recorded but do not change behaviour.',
 'Space has no executable semantics here by design (read-only, emergent): it is never a writable cell; the composite outputs are what is read. No emergence rule for Space exists in her spec (§70).',
 'the stream layout (header, definitions, cell records) is an engineering serialization of nodes + wiring; any byte stream decodes totally (missing atoms read as 0, references wrap) so arbitrary files execute, but only well-formed streams mean what their author intended.',
 'emergent dimension of a composite = the dimension with most cells (ties by Movement, Evolution, Being, Design); six-bit composite code = fold with exchangeTrigrams. Both are engineering choices.',
 'ABC (artificial bee colony): CORRECTION of an earlier wrong entry here ("no runnable ABC exists in her files"). It exists in swarm-os-v14/x/v9/swarm_os_full.py abc_phase() and v10/swarm_os_bantu.py, and in oct7src/computer/runtime/morph-search.mjs; ported rule for rule in abc.mjs. It is not part of reorganize() below (original kept as written); dynamics-experiment.mjs measures it.']);
import {ExecutionAddressResolver} from './spine/execution-address-resolver.mjs';
import {StateSpace,exchangeTrigrams,flipLines,hammingNeighbors} from '../state-space.mjs';
import {fitParticles} from '../swarm-fit.mjs';

export const WRITABLE=Object.freeze(['Movement','Evolution','Being','Design']);   // Space is emergent: never a cell
export const PRIMITIVE_KINDS=Object.freeze({
 Movement:Object.freeze({role:'change field: a transition of its inputs',stateful:false,inputs:2,properties:['directional','polarity down/up recorded (lines 5-6)'],constraints:['stateless: output depends only on current inputs and its hexagram'],transformation:'out = bit (a+2b) of lines 1-4'}),
 Evolution:Object.freeze({role:'change-of-change: state advances by the change applied to it',stateful:true,inputs:2,properties:['accumulates','persists between ticks'],constraints:['output depends on its own history'],transformation:'out = previous out XOR table(a,b), table = lines 1-4'}),
 Being:Object.freeze({role:'persistence/position: holds a value until told to take another',stateful:true,inputs:2,properties:['holds','left/right polarity recorded (lines 2-6)'],constraints:['changes only while its enable input is 1'],transformation:'out = (enable ? a : previous out) XOR line 1'}),
 Design:Object.freeze({role:'internal organization: routes signals and instantiates composed organization',stateful:false,inputs:3,properties:['selects','instantiates a definition as one primitive'],constraints:['instantiates only earlier definitions (no cycles)'],transformation:'line 6 = 0: out = sel ? b : a; line 6 = 1: instantiate definition (lines 1-3) with the wired signals'})});
export const GATE_TABLE=Object.freeze({AND:8,OR:14,XOR:6,NAND:7,NOR:1,XNOR:9,A:10,B:12,NOTA:5,NOTB:3,ANDNOTB:2,ONE:15,ZERO:0});

// ---- reading the six-bit stream (total: every stream decodes) ----
class Reader{constructor(a){this.a=a;this.i=0;this.overrun=0}next(){if(this.i<this.a.length)return this.a[this.i++]&63;this.overrun++;this.i++;return 0}}
export function decode(atoms){
 const r=new Reader(atoms),defs=[],nDefs=1+(r.next()%8),notes=[];
 for(let d=0;d<nDefs;d++){
  const nIn=r.next(),nCells=r.next(),nOut=r.next(),cells=[];let sig=nIn;
  for(let c=0;c<nCells;c++){
   const dim=WRITABLE[r.next()%4],H=r.next();let call=null,nref;
   if(dim==='Design'&&(H&32)&&d>0){call=(H&7)%d;nref=defs[call].nIn}else nref=dim==='Design'?3:2;
   const refs=[];for(let k=0;k<nref;k++)refs.push(r.next());
   const outs=call===null?1:defs[call].nOut;cells.push({dimension:dim,H,call,refs,slot:sig,outs});sig+=outs;
  }
  const outputs=[];for(let k=0;k<nOut;k++)outputs.push(r.next());
  defs.push({index:d,nIn,nOut,cells,outputs,nSig:sig,composite:emergent(cells,defs)});
 }
 const wellFormed=r.overrun===0&&defs.every(d=>d.nSig<=64);
 return {defs,top:defs[defs.length-1],wellFormed,atomsRead:Math.min(r.i,atoms.length),trailingAtoms:Math.max(0,atoms.length-r.i),overrun:r.overrun,notes};
}
function emergent(cells,defs){ // a composed structure acts as ONE primitive of the next level
 const n=Object.fromEntries(WRITABLE.map(k=>[k,0]));let code=0;
 for(const c of cells){n[c.call===null?c.dimension:defs[c.call].composite.dimension]++;code=exchangeTrigrams(code)^(c.H&63)}
 const dimension=WRITABLE.reduce((b,k)=>n[k]>n[b]?k:b,'Movement');
 return {dimension,code,cells:cells.length,counts:n};
}

// ---- execution: one tick rule for every kind ----
export class Instance{
 constructor(program,def=program.top){this.program=program;this.def=def;this.prev=new Uint8Array(Math.max(def.nSig,1));this.kids=new Map();
  for(const c of def.cells)if(c.call!==null)this.kids.set(c.slot,new Instance(program,program.defs[c.call]))}
 tick(inputs){
  const d=this.def,cur=new Uint8Array(Math.max(d.nSig,1)),prev=this.prev;
  for(let i=0;i<d.nIn;i++)cur[i]=(inputs[i]??0)&1;
  const rd=(r,slot)=>{if(d.nSig===0)return 0;const s=r%d.nSig;return s<slot?cur[s]:prev[s]}; // later position = previous tick
  for(const c of d.cells){
   if(c.call!==null){const o=this.kids.get(c.slot).tick(c.refs.map(r=>rd(r,c.slot)));for(let k=0;k<c.outs;k++)cur[c.slot+k]=o[k]??0;continue}
   const a=rd(c.refs[0],c.slot),b=rd(c.refs[1],c.slot),H=c.H;
   switch(c.dimension){
    case 'Movement':cur[c.slot]=(H>>(a+2*b))&1;break;
    case 'Evolution':cur[c.slot]=prev[c.slot]^((H>>(a+2*b))&1);break;
    case 'Being':cur[c.slot]=(b?a:prev[c.slot])^(H&1);break;
    case 'Design':cur[c.slot]=rd(c.refs[2],c.slot)?b:a;break;
   }
  }
  const out=d.outputs.map(r=>d.nSig?cur[r%d.nSig]:0);this.prev=cur;return out;
 }
 // state is six-bit atoms too (depth-first, six signals per atom, MSB first)
 stateBits(){const out=[...this.prev.slice(0,this.def.nSig)];for(const c of this.def.cells)if(c.call!==null)out.push(...this.kids.get(c.slot).stateBits());return out}
 stateAtoms(){const b=this.stateBits(),a=[];for(let i=0;i<b.length;i+=6){let v=0;for(let k=0;k<6;k++)v=(v<<1)|(b[i+k]??0);a.push(v)}return a}
 restoreAtoms(atoms){const bits=[];for(const v of atoms)for(let k=5;k>=0;k--)bits.push((v>>k)&1);let p=0;const rec=ins=>{for(let i=0;i<ins.def.nSig;i++)ins.prev[i]=bits[p++]??0;for(const c of ins.def.cells)if(c.call!==null)rec(ins.kids.get(c.slot))};rec(this);return this}
}
export const run=(program,inputRows)=>{const m=new Instance(program);return inputRows.map(r=>m.tick(r))};

// ---- packing six-bit atoms <-> bytes (same bit order as her ExecutionAddressResolver.descend: MSB first) ----
export function atomsToBytes(atoms){const out=new Uint8Array(Math.ceil(atoms.length*6/8));let buf=0,nb=0,o=0;for(const a of atoms){buf=(buf<<6)|(a&63);nb+=6;while(nb>=8){out[o++]=(buf>>>(nb-8))&255;nb-=8;buf&=(1<<nb)-1}}if(nb)out[o++]=(buf<<(8-nb))&255;return out}
export function bytesToAtoms(bytes,byteLength=bytes.length){const a=[];let buf=0,nb=0;for(let i=0;i<byteLength;i++){buf=(buf<<8)|bytes[i];nb+=8;while(nb>=6){a.push((buf>>>(nb-6))&63);nb-=6;buf&=(1<<nb)-1}}if(nb)a.push((buf<<(6-nb))&63);return a}

// ---- resolution: bytes -> six-bit primitives (her descend) -> organization -> nodes in the nested state space + resolved address ----
export class Population{
 constructor({space=new StateSpace(),resolver=new ExecutionAddressResolver()}={}){this.space=space;this.resolver=resolver;this.records=new Map()}
 resolve(bytes,{name='artifact'}={}){
  const artifact={name,bytes},dec=this.resolver.descend(artifact),atoms=[...dec.sixBitStates];
  const program=decode(atoms),nodes=this.#register(program);
  const probe=run(program,Array.from({length:4},()=>Array(program.top.nIn).fill(0)));
  const res=this.resolver.resolve({artifact,decomposition:dec,execution:{ok:true,path:'population-resolution',kind:'primitive-organization',result:{engine:'population-tick',stdout:[],returnValue:{probe,top:program.top.composite}}}});
  return {name,atoms,primitives:dec.sixBitStateCount,program,nodes,address:{key:res.key,address:res.address,basis:res.derivation.basis},top:program.top.composite,instance:()=>new Instance(program)};
 }
 #register(program){ // every cell becomes a node (dimension, path): nested by instantiation depth, macro = its hexagram, sub = position within the level
  const out=[],walk=(def,prefix)=>{def.cells.forEach((c,i)=>{
    const dim=c.call===null?c.dimension:program.defs[c.call].composite.dimension,macro=c.call===null?c.H&63:program.defs[c.call].composite.code,path=[...prefix,{macro,sub:i%this.space.split}];
    if(path.length<=this.space.levels){const n=this.space.ensure(dim,path);out.push({id:n.id,dimension:dim,hexagram_code:n.hexagram_code,instantiates:c.call})}
    if(c.call!==null&&prefix.length+1<this.space.levels)walk(program.defs[c.call],path)})};
  walk(program.top,[]);return out;
 }
}

// ---- reorganization by swarm dynamics (existing PSO: swarm-fit.mjs fitParticles; existing Hamming-1 neighbourhood: state-space.mjs) ----
// The genotype is the atom stream; fitness is how many output bits of the examples are right. Nothing app-specific: examples are plain input/expected-output bit rows.
export const fitness=(atoms,examples)=>{let bad=0;for(const ex of examples){const p=decode(atoms),out=run(p,ex.inputs);ex.expect.forEach((row,t)=>row.forEach((v,k)=>{if((out[t]?.[k]??-1)!==v)bad++}))}return bad};
export async function reorganize(atoms,examples,{particles=24,iterations=30,seed=719,refineRounds=400}={}){
 const start=atoms.slice(),startBad=fitness(start,examples),log=[];let cur=start.slice(),bad=startBad;
 const refine=()=>{for(let round=0;round<refineRounds&&bad>0;round++){ // Hamming-1 neighbours (her channel-emergence rule) of the organization, best single-line change first
  let best=null;
  for(let i=0;i<cur.length;i++)for(const h of hammingNeighbors(cur[i],1)){const t=cur.slice();t[i]=h.code;const b=fitness(t,examples);if(b<bad&&(!best||b<best.b))best={i,code:h.code,b}}
  if(!best)break;log.push({stage:'hamming-1',atom:best.i,from:cur[best.i],to:best.code,before:bad,after:best.b});cur[best.i]=best.code;bad=best.b}};
 refine();
 // when local change is stuck, the particle swarm (swarm-fit.mjs) explores globally, then local change continues
 if(bad>0&&particles>0){
  const r=await fitParticles({bounds:cur.map(()=>[0,63]),initial:cur.map(v=>v),score:x=>fitness(x.map(v=>Math.round(v)&63),examples),particles,iterations,seed});
  const cand=r.values.map(v=>Math.round(v)&63),cb=fitness(cand,examples);log.push({stage:'pso',before:bad,after:cb,changedAtoms:cand.filter((v,i)=>v!==cur[i]).length});
  if(cb<bad){cur=cand;bad=cb;refine()}
 }
 return {atoms:cur,startBad,bad,repaired:bad===0,log};
}
export {flipLines};
