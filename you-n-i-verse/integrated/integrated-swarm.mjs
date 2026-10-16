// Integration: Alexis's patches + the swarm/state-space already built. Pure JavaScript, no external dependencies, no backend, no web worker.
// Per file: ingest -> normalize -> primitive-operator-graph -> five-field-allocation -> dependency-schedule-forwarding -> artifact-execution -> fold-recombine -> state-commit
//           -> observation -> ontological-address -> execution-cache (her Execution Pipeline Patch v0.5.1, copied as-is in ./spine)
// with her automata kernel (./automata) as the artifact-execution core, her ontological ingestor (./ontological) for exact-byte ingest and derived Space,
// then the swarm (any-file-swarm: registration, PSO packs 3-2-1, placement, NSNF runtime), a fold-up NSNF event, and history landed at the resolved address.
import {ExecutionPipeline,MemoryExecutionPipelineStore} from './spine/execution-pipeline.mjs';
import {ExecutionAddressResolver} from './spine/execution-address-resolver.mjs';
import {AutomataIngestionExecutionKernel} from './automata/ingestion-execution-kernel.mjs';
import {BrowserAutomataStore} from './automata/browser-automata-store.mjs';
import {OntologicalFileIngestor} from './ontological/ingestion/ontological-file-ingestor.mjs';
import {HistoryLedger} from './ledger.mjs';
import {createSurface,PROVIDER_INVESTIGATING} from './providers.mjs';
import {WHEELS,WHEELS_INVESTIGATING,TEN_WINGS,wheelReadout,wheelRank} from './wheels.mjs';
import {deriveAppCode,APP_CODE_INVESTIGATING} from './app-code.mjs';
import {processFile,fileFeatures,INVESTIGATING as FILE_INVESTIGATING} from '../any-file-swarm.mjs';
import {NsnfRuntime,hashOf} from '../nsnf-runtime.mjs';
import {PRIMITIVE_DIMENSIONS,gateToMacro} from '../state-space.mjs';

export const INTEGRATED_INVESTIGATING=Object.freeze([
 'her pipeline allocates primitives to all five fields including Space (her patch: "runtime placement rule only"); the NSNF state space keeps Space read-only. Nothing from the pipeline is written to Space slots, and the fold-up vector uses only the four primitive fields.',
 'fold-up vector (8 numbers: four primitive-field shares, execution ok, forwarding ratio, stdout size, unresolved count) is an engineering choice so the swarm has numbers to read; Alexis has not specified it.',
 'files over maxPipelineBytes go through the pipeline as a prefix (the SHA-256 and ingest still cover all bytes). Her resolver itself keeps at most 4096 six-bit states.',
 'when no address is registered, placement uses dimension/gate/line from her first-execution-derived address (her spine rule); the other places are not mapped to the repo 13-place schema, so registration stays "investigating".',
 'three address forms exist (repo 13-place, spine computational coordinate, ontological-ingest address); all three are reported, none is chosen as canonical.']);
const toBytes=b=>b instanceof Uint8Array?b:typeof b==='string'?new TextEncoder().encode(b):b instanceof ArrayBuffer?new Uint8Array(b):Array.isArray(b)?Uint8Array.from(b):null;
const clamp=x=>Math.max(0,Math.min(1,x));

// Her pipeline writes edges to a mesh when one is supplied (addEdge); this records them.
export class MeshLog{constructor(){this.edges=[]}addEdge(layer,from,to,relation,meta={}){this.edges.push({layer,from,to,relation,meta});return true}}

export {HistoryLedger};
export class IntegratedSwarm{
 constructor({runtime=new NsnfRuntime(),executeCode=true,maxPipelineBytes=8<<20,kernelStore=new BrowserAutomataStore(),pipelineStore=new MemoryExecutionPipelineStore()}={}){
  this.runtime=runtime;this.executeCode=executeCode;this.maxPipelineBytes=maxPipelineBytes;
  this.surface=createSurface({executeCode});
  this.kernel=new AutomataIngestionExecutionKernel({store:kernelStore,executionSurface:this.surface});
  this.resolver=new ExecutionAddressResolver();
  this.ingestor=new OntologicalFileIngestor();
  this.mesh=new MeshLog();
  this.history=new HistoryLedger(runtime.store);
  this.pipeline=new ExecutionPipeline({executeArtifactCore:(a,c)=>this.#core(a,c),addressResolver:this.resolver,mesh:this.mesh,store:pipelineStore});
 }
 // artifact-execution stage of her pipeline: her automata kernel reduces, compiles five fields, EXECUTES via the surface, signs; her resolver resolves the address.
 async #core(artifact,context){
  const rec=await this.kernel.process({name:artifact.name,bytes:artifact.bytes},{executeCode:context.executeCode,timeoutMs:context.timeoutMs,depth:0});
  const ex=rec.execution;
  const resolved=this.resolver.resolve({artifact,execution:{ok:ex.ok,path:ex.path,kind:ex.kind,runtimeAdapter:'automata-kernel-same-thread',result:ex.result,error:ex.error},decomposition:context.executionDecomposition});
  return {ok:ex.ok,kind:ex.kind,path:ex.path,runtimeAdapter:'automata-kernel-same-thread',result:ex.result,error:ex.error,unresolvedDependencies:ex.unresolvedDependencies,
   canonicalAddress:resolved.address,canonicalAddressKey:resolved.key,addressBasis:resolved.derivation.basis,executionDerivedAddress:resolved.address,executionDerivedAddressKey:resolved.key,
   recordId:'automata:'+rec.address.id,placementRecordId:rec.fingerprint,automata:{addressId:rec.address.id,fingerprint:rec.fingerprint,reused:rec.reusedCompiledForm,fieldCounts:rec.fields.counts,integritySignature:rec.address.integritySignature}};
 }
 foldVector(pipeline,core){
  const f=pipeline.fold.fields,byName=Object.fromEntries(f.map(x=>[x.field,x.primitiveCount])),total=PRIMITIVE_DIMENSIONS.reduce((s,d)=>s+byName[d],0)||1,n=pipeline.primitiveGraph.nodeCount||1;
  const out=(core.result?.stdout?.length)||0;
  return [...PRIMITIVE_DIMENSIONS.map(d=>byName[d]/total),core.ok?1:0,clamp(pipeline.fold.forwardingCount/n),clamp(Math.log2(1+out)/8),clamp((core.unresolvedDependencies?.length||0)/4)];
 }
 async execute(input,{address=null,personalization=null,source=null,kernel=globalThis.AgentomagotchiKernel,storage,finish=false,executeCode=this.executeCode,timeoutMs=2500,...rest}={}){
  const bytes=toBytes(input?.bytes);if(!bytes)throw new TypeError('input.bytes required');
  const name=input.name||'upload',features=fileFeatures(name,bytes),fixed=address??personalization?.address??null;
  // 1 ingest (her ontological ingest patch): original bytes kept for exact reconstruction, bit-level reduction, Space derived from the other four
  let ingest;try{const r=await this.ingestor.ingest({name,bytes},{supersede:true});
   ingest={status:'ingested',id:r.id,address:r.address,combinatorialAddress:r.combinatorialAddress,dimensions:r.decomposition.dimensions,spaceRule:r.decomposition.spaceRule,primitiveCount:r.decomposition.primitiveCount,exactReconstruction:r.reconstruction.exactAvailable}}
  catch(e){ingest={status:'investigating',error:String(e.message)}}
  // 2 her ExecutionPipeline: normalize ... artifact-execution ... fold-recombine ... state-commit ... ontological-address ... execution-cache
  const truncated=bytes.length>this.maxPipelineBytes,pb=truncated?bytes.subarray(0,this.maxPipelineBytes):bytes;
  const run=await this.pipeline.execute({name,bytes:pb},{executeCode,timeoutMs});
  const p=run.pipeline,derived=run.executionDerivedAddress;
  // 3 resolved code: a registered address is authoritative (her spine rule); otherwise the first-execution-derived address
  const resolved=fixed?{address:fixed,key:hashOf(fixed),basis:'creator-supplied'}:{address:derived,key:run.executionDerivedAddressKey,basis:run.addressBasis};
  const app=deriveAppCode(resolved);
  // 4 swarm: registration (id = derived app code), file genotype PSO (packs 3-2-1), placement, NSNF runtime, phenotype GA
  const placeAddress=fixed??{dimension:derived.dimension,gate:derived.gate,line:derived.line};
  const sw=await processFile({name,bytes},{personalization,address:placeAddress,source:source||(fixed?null:'first-execution-derived address'),kernel,storage,runtime:this.runtime,finish,appCode:app.code,executed:run.result?.engine==='same-thread-async-function',persist:false,...rest});
  // 5 fold up: the pipeline's fields become the input of a higher-scale NSNF event at the placed node (Space is read, never written)
  const seeds=sw.placement?.nodeId?[{dimension:sw.placement.dimension,path:sw.placement.path,role:'placement'}]:[];
  const vec=this.foldVector(p,run);
  const up=this.runtime.execute({input:vec,seeds,tier:null,reason:'fold-up of pipeline results'});
  // 6 wheels (spec §2.1), read only
  const byField={};for(const n of p.primitiveGraph.nodes)(byField[n.field]??=[]).push(n.sixBitState);
  const wheels=Object.fromEntries(Object.keys(WHEELS).map(d=>[d,wheelReadout(d,byField[d]||[])]));
  const placedWheel=sw.placement?.nodeId?{dimension:sw.placement.dimension,macro:sw.placement.path[0].macro,rank:wheelRank(sw.placement.dimension,sw.placement.path[0].macro)}:null;
  // 7 state commit + history at the resolved address
  const cp=this.runtime.checkpoint();
  const summary={ok:!!run.ok,kind:run.kind,path:run.path,engine:run.result?.engine??null,error:run.ok?null:run.error??null,unresolved:run.unresolvedDependencies??[],stdout:(run.result?.stdout||[]).slice(0,8),returnValue:summarize(run.result?.returnValue)};
  const hist=this.history.append(resolved.key,{at:this.runtime.now(),appCode:app.code,name,sha256:features.sha256,size:bytes.length,execution:summary,
   pipeline:{sequence:p.commit.sequence,fingerprint:p.fingerprint,foldHash:p.fold.foldHash,cacheHit:p.cache.analysisHit,primitives:p.primitiveGraph.nodeCount},
   addresses:{resolved:resolved.key,basis:resolved.basis,executionDerived:run.executionDerivedAddressKey,ingest:ingest.combinatorialAddress??null,automata:run.automata.addressId},
   nsnf:{placementEvent:sw.execution?.event_id??null,foldUpEvent:up.event_id??null,foldUpTier:up.tier??null}});
  const reasons=[...sw.reasons];if(!run.ok)reasons.push('artifact-execution: '+(run.error||'not ok'));if(ingest.status!=='ingested')reasons.push('ingest investigating: '+ingest.error);if(app.status!=='derived')reasons.push('app code investigating: '+app.reason);
  return {status:reasons.length?'investigating':'complete',reasons,file:sw.file,
   stages:p.stages.map(s=>({stage:s.stage,ok:s.ok})),
   ingest,execution:summary,executedUserCode:summary.engine==='same-thread-async-function',
   address:{resolved:{key:resolved.key,basis:resolved.basis},executionDerived:{key:run.executionDerivedAddressKey,address:derived},ingest:ingest.combinatorialAddress??null,automataId:run.automata.addressId},
   app,fold:{foldHash:p.fold.foldHash,vector:vec,fields:p.fold.fields.map(f=>({field:f.field,primitives:f.primitiveCount})),forwarding:p.fold.forwardingCount,truncatedPipelineInput:truncated},
   swarm:{status:sw.status,placement:sw.placement,genotype_pso:sw.genotype_pso,phenotype_ga:sw.phenotype_ga,execution:sw.execution&&sw.execution.tier!==undefined?{tier:sw.execution.tier,n_active:sw.execution.n_active,confidence:sw.execution.confidence,event_id:sw.execution.event_id}:sw.execution,
    foldUp:up.executed?{tier:up.tier,n_active:up.n_active,confidence:up.confidence,event_id:up.event_id,space:up.space}:up,registration:sw.registration},
   wheels:{perField:wheels,placed:placedWheel,tenWings:TEN_WINGS},history:{seq:hist.seq,addressKey:hist.addressKey,entriesAtAddress:this.history.recall(resolved.key).length},
   stateCommit:{pipelineSequence:p.commit.sequence,checkpoint:cp?.id??cp?.version??true,meshEdges:this.mesh.edges.length},
   safety:{userCodeExecuted:summary.engine==='same-thread-async-function',originalBytesModified:false,workerUsed:false,backendUsed:false}};
 }
 get investigating(){return [...INTEGRATED_INVESTIGATING,...PROVIDER_INVESTIGATING,...WHEELS_INVESTIGATING,...APP_CODE_INVESTIGATING,...FILE_INVESTIGATING]}
}
function summarize(v){if(v===null||v===undefined)return null;const s=JSON.stringify(v);return s&&s.length>2000?{truncated:true,preview:s.slice(0,400),length:s.length}:v}
export {deriveAppCode,wheelReadout,WHEELS};
