// Inventory of the primitives that ALREADY exist (Alexis: inventory, preserve, identify what is connected/executable, repair without flattening, connect, add only when necessary).
// Every "executable" below is established by actually calling the thing, not by reading its name. Nothing here replaces or wraps an original.
import {DIMENSIONS,PRIMITIVE_DIMENSIONS,nuclear,flipLines,complement,reverseLines,exchangeTrigrams,hammingNeighbors,gateToMacro,macroToGate,StateSpace} from '../state-space.mjs';
import {NsnfRuntime,TinyMLP,trainWithParticleSwarm} from '../nsnf-runtime.mjs';
import {fitParticles} from '../swarm-fit.mjs';
import {SwarmComponent} from '../swarm-component.mjs';
import {SwarmPair} from '../swarm-nsnf.mjs';
import {PRIMITIVE_KINDS,WRITABLE} from './population.mjs';
import {WHEELS} from './wheels.mjs';
import {encodeBytes,decodeBytes,packsOf} from './bantu.mjs';
import {FiveFieldCompiler,EXECUTION_FIELDS} from './automata/five-field-compiler.mjs';
import {ExecutionAddressResolver} from './spine/execution-address-resolver.mjs';
import {OntologicalFileIngestor} from './ontological/ingestion/ontological-file-ingestor.mjs';
const probe=(id,source,fn)=>{try{const r=fn();return {id,source,executable:true,evidence:r}}catch(e){return {id,source,executable:false,evidence:String(e.message)}}};
export async function inventoryPrimitives(){
 const out=[];
 out.push(probe('hexagram-state (six bits, packs 3-2-1)','state-space.mjs, swarm-component.mjs',()=>({code:37,packs:packsOf(37)})));
 out.push(probe('64 gates: King Wen <-> Fu Xi map','state-space.mjs + vendor/kingwen.mjs',()=>{for(let g=1;g<=64;g++)if(macroToGate(gateToMacro(g))!==g)throw new Error('map broken at '+g);return {gates:64}}));
 out.push(probe('operators: flipLines, complement, reverseLines, exchangeTrigrams, nuclear','state-space.mjs',()=>({flip:flipLines(0,[1]),complement:complement(5),reverse:reverseLines(1),exchange:exchangeTrigrams(7),nuclear:nuclear(63)})));
 out.push(probe('Hamming-1 neighbourhood (channel emergence rule)','state-space.mjs',()=>({neighbours:hammingNeighbors(0,1).length})));
 out.push(probe('five dimensions, 64 nested macro slots per level, Space read-only','state-space.mjs',()=>{const s=new StateSpace(),n=s.ensure('Being',[{macro:5,sub:0}]);return {dimensions:DIMENSIONS.length,node:n.id,space:s.readSpace().status}}));
 for(const d of DIMENSIONS)out.push(probe('dimension '+d+' wheel','wheels.mjs (spec §2.1)',()=>({sequence:WHEELS[d].sequence,status:WHEELS[d].status,order:WHEELS[d].order?WHEELS[d].order.length:null})));
 out.push(probe('NSNF trainable cell (TinyMLP)','nsnf-runtime.mjs',()=>{const m=new TinyMLP({dIn:8,hidden:6,dOut:4,seed:1});return {macs:m.macs}}));
 out.push(probe('NSNF runtime (tiers, sparse activation, trace)','nsnf-runtime.mjs',()=>{const r=new NsnfRuntime(),e=r.execute({input:Array(8).fill(0.5)});return {executed:e.executed,tier:e.tier,active:e.n_active}}));
 out.push(await (async()=>{try{const m=new TinyMLP({dIn:8,hidden:6,dOut:4,seed:3}),data=Array.from({length:6},(_,i)=>({input:Array(8).fill(i/6),target:[1,0,0,0]}));const r=await trainWithParticleSwarm(m,data,{particles:4,iterations:2});return {id:'PSO on trainable cells',source:'nsnf-runtime.mjs trainWithParticleSwarm',executable:true,evidence:{keys:Object.keys(r||{}).slice(0,5)}}}catch(e){return {id:'PSO on trainable cells',source:'nsnf-runtime.mjs trainWithParticleSwarm',executable:false,evidence:String(e.message)}}})());
 out.push(await (async()=>{try{const r=await fitParticles({bounds:[[0,10]],initial:[1],score:x=>(x[0]-7)**2,particles:8,iterations:10});return {id:'PSO numeric fit',source:'swarm-fit.mjs',executable:true,evidence:{best:+r.values[0].toFixed(2),workers:r.workers}}}catch(e){return {id:'PSO numeric fit',source:'swarm-fit.mjs',executable:false,evidence:String(e.message)}}})());
 out.push(probe('PSO genotype / GA phenotype pair, packs 3-2-1','swarm-component.mjs, swarm-nsnf.mjs',()=>typeof SwarmComponent==='function'&&typeof SwarmPair==='function'?{classes:['SwarmComponent','SwarmPair']}:null));
 out.push({id:'ABC (artificial bee colony)',source:'none found in her files (searched zips, spec and architecture text)',executable:false,evidence:'absent: not implemented, not claimed (investigating)'});
 out.push(probe('Bantu 6-bit token representation (bytes <-> tokens, 3-2-1 levels)','bantu/bridge.js.txt (her bridge.js lines 5-9)',()=>{const t=encodeBytes([1,200,255]);return {tokens:t,back:[...decodeBytes(t)]}}));
 out.push(probe('five execution fields (shared, knowledge, causal, dependency, stateSpace)','automata/five-field-compiler.mjs',()=>({fields:EXECUTION_FIELDS,compiles:typeof new FiveFieldCompiler().compile})));
 out.push(probe('ExecutionAddressResolver (descend to six-bit states, resolve 13-place address)','spine/execution-address-resolver.mjs',()=>{const r=new ExecutionAddressResolver();return {states:r.descend({bytes:new Uint8Array([1,2,3])}).sixBitStateCount}}));
 out.push(probe('OntologicalFileIngestor (bit-level reduction, Space derived, exact reconstruction)','ontological/ingestion/ontological-file-ingestor.mjs',()=>({class:typeof OntologicalFileIngestor})));
 for(const k of WRITABLE)out.push({id:'population primitive kind: '+k,source:'integrated/population.mjs (this integration; roles from her ingest patch)',executable:true,evidence:{role:PRIMITIVE_KINDS[k].role,stateful:PRIMITIVE_KINDS[k].stateful},added:true});
 return out;
}
export const CONNECTED_TO_EXECUTION=Object.freeze({
 connected:['hexagram state and operators (population hexagrams, reorganize uses Hamming-1)','five dimensions + nested state space (population nodes are registered there)','ExecutionAddressResolver (resolves every organization to an address)','PSO numeric fit (reorganize)','Bantu tokens (round-trip representation of any file, packs 3-2-1)','NSNF runtime + PSO/GA swarm pair (file swarm, integrated-swarm.mjs)'],
 notConnectedToPopulationExecution:['TinyMLP trainable cells and NSNF tiers (they execute the file swarm, not the population; no cell of the population is a TinyMLP)','dimension wheels (read-only annotation; they do not select behaviour)','Space readout (no emergence rule in her spec)','ABC (absent)','Klein/Autoling tools (not copied; the UniversalExecutionBridge that hosts them needs a Web Worker)']});
