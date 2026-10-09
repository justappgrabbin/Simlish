(function(global){
'use strict';
const K=global.AgentomagotchiKernel,B=global.AgentomagotchiWorldBridge;
if(!K) throw new Error('kernel.js must load before mesh-assembly.js');
if(!B) throw new Error('world-bridge.js must load before mesh-assembly.js');
const clone=v=>JSON.parse(JSON.stringify(v));
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,Number(v)||0));
function hash32(s){let h=0x811c9dc5;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,0x01000193)>>>0}return h>>>0}
function stableContext(context={}){return Object.freeze({
  who:context.who??null,what:context.what??null,where:context.where??null,when:context.when??null,why:context.why??null,
  perspective:context.perspective??null,position:context.position??null
})}

// One event/state, simultaneously projected through all five canonical dimensions.
// Every projection remains a COMPLETE canonical address.
class FiveDimensionalState{
  constructor(addresses){
    if(!Array.isArray(addresses)||addresses.length!==K.DIMENSIONS.length)throw new Error('FiveDimensionalState requires exactly five complete addresses');
    const normalized=addresses.map(a=>a instanceof K.FullAddress?a:new K.FullAddress(a));
    const dims=normalized.map(a=>a.dimension);
    for(const d of K.DIMENSIONS)if(dims.filter(x=>x===d).length!==1)throw new Error('FiveDimensionalState requires one simultaneous '+d+' perspective');
    this.addresses=Object.freeze(K.DIMENSIONS.map(d=>normalized.find(a=>a.dimension===d)));
    Object.freeze(this);
  }
  get(dimension){return this.addresses.find(a=>a.dimension===dimension)||null}
  toObjects(){return this.addresses.map(a=>a.toObject())}
  toStrings(){return this.addresses.map(a=>a.toString())}
  signature(){return this.toStrings().join(' || ')}
}

// Entity-level relationship memory transplanted from the relational mesh idea:
// edges store context, evidence AND counterevidence. Exact addressed point-pairs are evidence,
// not replacements for the stable relationship object.
class RelationalEvidenceMesh{
  constructor(){this.edges=new Map();this.seq=0}
  key(from,relation,to,context){const c=stableContext(context);return [from,relation,to,c.what,c.where,c.why,c.perspective].map(x=>JSON.stringify(x)).join('|')}
  observe({from,relation,to,context={},evidence=null,confidence=.5,success=true}={}){
    if(!from||!to||!relation)throw new TypeError('relation requires from, relation, to');
    const ctx=stableContext(context),key=this.key(String(from),String(relation),String(to),ctx),prior=this.edges.get(key);
    const edge=prior||{id:'E'+(++this.seq),from:String(from),relation:String(relation),to:String(to),context:ctx,support:0,counter:0,confidence:clamp(confidence),charge:0,coherence:.05,active:false,threshold:.62,evidence:[],firstSeq:this.seq,lastSeq:this.seq};
    if(success){edge.support++;edge.charge=clamp(edge.charge+.12+clamp(confidence)*.08);edge.coherence=clamp(edge.coherence+.07)}
    else{edge.counter++;edge.charge=clamp(edge.charge-.13);edge.coherence=clamp(edge.coherence-.08)}
    edge.confidence=clamp((edge.confidence*.65)+clamp(confidence)*.35);
    edge.active=edge.charge>=edge.threshold&&edge.support>edge.counter;
    edge.lastSeq=++this.seq;
    if(evidence){edge.evidence.push(clone(evidence));if(edge.evidence.length>32)edge.evidence.shift()}
    this.edges.set(key,edge);return edge
  }
  forEntity(id){return [...this.edges.values()].filter(e=>e.from===id||e.to===id)}
  snapshot(){return [...this.edges.values()].map(clone)}
}

// These are NOT the ontology primitives. They are a numerical execution substrate
// transplanted from Primitive Neural Network Foundry for a capability that has already emerged.
const NEURAL_OPERATOR_KINDS=Object.freeze(['transformation','memory','relation','generation']);
function makeOperatorGenome(id,seed,memberCount){
  const width=Math.max(4,Math.min(24,memberCount||4));
  return Object.freeze({version:1,id:'genome-'+id,name:'Emergent capability genome '+id,seed:seed>>>0,
    nodes:Object.freeze([
      Object.freeze({id:'T',primitive:'transformation',label:'Transform',config:Object.freeze({inputSize:width,outputSize:width,activation:'tanh'})}),
      Object.freeze({id:'M',primitive:'memory',label:'Retain',config:Object.freeze({inputSize:width,outputSize:width,update:'integrate'})}),
      Object.freeze({id:'R',primitive:'relation',label:'Relate',config:Object.freeze({inputSize:width,outputSize:width,neighborhood:'assembly'})}),
      Object.freeze({id:'G',primitive:'generation',label:'Select',config:Object.freeze({inputSize:width,outputSize:4,temperature:1})})
    ]),
    connections:Object.freeze([
      Object.freeze({id:'T-R',from:'T',fromPort:'out',to:'R',toPort:'in',transform:'identity'}),
      Object.freeze({id:'M-R',from:'M',fromPort:'out',to:'R',toPort:'context',transform:'identity'}),
      Object.freeze({id:'R-G',from:'R',fromPort:'out',to:'G',toPort:'in',transform:'identity'}),
      Object.freeze({id:'G-M',from:'G',fromPort:'out',to:'M',toPort:'in',transform:'project'})
    ]),
    learning:Object.freeze({name:'reinforce',learningRate:.04}),
    metadata:Object.freeze({role:'capability-execution-substrate',ontologyPrimitives:['Being','Design','Movement','Evolution']})
  })
}

class CapabilityAssemblyRuntime{
  constructor({agentId,seed=1}={}){this.agentId=String(agentId||'agent');this.seed=seed>>>0;this.relational=new RelationalEvidenceMesh();this.motifs=new Map();this.assemblies=new Map();this.capabilities=new Map();this.seq=0}
  motifKey(action,context,composition=[]){const c=stableContext(context),comp=B.normalizeComposition(composition).join('>');return `${action}|${comp||'observe>retain'}|${c.what||'*'}|${c.where||'*'}|${c.why||'*'}`}
  observeOutcome({action,targetId='world',targetKind='world',context={},sourceBundle,targetBundle=null,pointRelations=[],composition=[],success=true,utility=.7}={}){
    if(!(sourceBundle instanceof FiveDimensionalState))throw new TypeError('sourceBundle must be FiveDimensionalState');
    const ctx=stableContext(context);
    const edge=this.relational.observe({from:this.agentId,relation:String(action||'acts'),to:String(targetId||'world'),context:ctx,confidence:utility,success,evidence:{
      seq:this.seq+1,sourceAddresses:sourceBundle.toObjects(),targetAddresses:targetBundle instanceof FiveDimensionalState?targetBundle.toObjects():null,pointRelationIds:pointRelations.map(r=>r.id),composition:Array.from(B.normalizeComposition(composition)),outcome:success?'success':'failure',utility
    }});
    const normalizedComposition=B.normalizeComposition(composition),key=this.motifKey(action,ctx,normalizedComposition),motif=this.motifs.get(key)||{key,action:String(action),composition:Array.from(normalizedComposition),context:ctx,support:0,counter:0,charge:0,coherence:.05,state:'forming',traces:[],assemblyId:null,capabilityId:null};
    if(success){motif.support++;motif.charge=clamp(motif.charge+.13+utility*.08);motif.coherence=clamp(motif.coherence+.075)}else{motif.counter++;motif.charge=clamp(motif.charge-.16);motif.coherence=clamp(motif.coherence-.09)}
    motif.traces.push({seq:++this.seq,targetId,targetKind,utility,success,composition:Array.from(normalizedComposition),sourceAddresses:sourceBundle.toObjects(),targetAddresses:targetBundle instanceof FiveDimensionalState?targetBundle.toObjects():null,pointRelations:pointRelations.map(r=>({id:r.id,scale:r.scale,charge:r.charge,coherence:r.coherence,active:r.active}))});
    if(motif.traces.length>20)motif.traces.shift();
    if(motif.support>=2&&motif.coherence>=.18)motif.state='primed';
    if(motif.support>=3&&motif.charge>=.48)this.formAssembly(motif,edge);
    if(motif.support>=5&&motif.coherence>=.38)this.promote(motif);
    this.motifs.set(key,motif);
    return {edge:clone(edge),motif:clone(motif),assembly:motif.assemblyId?clone(this.assemblies.get(motif.assemblyId)):null,capability:motif.capabilityId?clone(this.capabilities.get(motif.capabilityId)):null}
  }
  formAssembly(motif,edge){
    if(motif.assemblyId&&this.assemblies.has(motif.assemblyId)){const a=this.assemblies.get(motif.assemblyId);a.charge=clamp((a.charge+motif.charge)/2+.08);a.coherence=clamp((a.coherence+motif.coherence)/2+.06);a.state='active';a.lastSeq=this.seq;return a}
    const id='A'+String(this.assemblies.size+1).padStart(3,'0');
    const traces=motif.traces.slice(-3),members=[];for(const t of traces)for(const a of t.sourceAddresses)members.push(a);
    const unique=new Map(members.map(a=>[new K.FullAddress(a).toString(),a]));
    const assembly={id,state:'active',motifKey:motif.key,action:motif.action,composition:clone(motif.composition),context:clone(motif.context),charge:motif.charge,coherence:motif.coherence,support:motif.support,counter:motif.counter,memberAddresses:[...unique.values()],pointRelationIds:[...new Set(traces.flatMap(t=>t.pointRelations.map(r=>r.id)))],sourceRelationId:edge.id,formedSeq:this.seq,lastSeq:this.seq,genome:makeOperatorGenome(id,hash32(`${this.seed}|${id}|${motif.key}`),unique.size),provenance:{sourceAgent:this.agentId,motif:motif.key,interactionSeqs:traces.map(t=>t.seq),relationship:edge.id,canonicalAddressFields:Array.from(K.ADDRESS_FIELDS)}};
    this.assemblies.set(id,assembly);motif.assemblyId=id;motif.state='active';return assembly
  }
  promote(motif){
    if(motif.capabilityId&&this.capabilities.has(motif.capabilityId)){const c=this.capabilities.get(motif.capabilityId);c.uses++;c.confidence=clamp(c.confidence+.025);return c}
    if(!motif.assemblyId)return null;const a=this.assemblies.get(motif.assemblyId),id='CAP-'+String(this.capabilities.size+1).padStart(3,'0');
    const cap={id,name:`${id} · ${motif.composition.join(' → ')}`,action:motif.action,composition:clone(motif.composition),context:clone(motif.context),sourceAssembly:a.id,confidence:clamp((motif.coherence+motif.charge)/2),uses:0,genome:a.genome,provenance:clone(a.provenance)};
    this.capabilities.set(id,cap);motif.capabilityId=id;motif.state='promoted';a.state='promoted';return cap
  }
  bonus(action,context={}){const c=stableContext(context);let b=0;for(const cap of this.capabilities.values())if(cap.action===action&&(cap.context.what==null||cap.context.what===c.what))b=Math.max(b,.08+cap.confidence*.18);return b}
  decay(amount=.006){for(const a of this.assemblies.values())if(a.state==='active'){a.charge=clamp(a.charge-amount);if(a.charge<.22)a.state='decaying';}else if(a.state==='decaying'){a.charge=clamp(a.charge-amount*.4);}}
  summary(){const assemblies=[...this.assemblies.values()].sort((a,b)=>b.charge-a.charge);const capabilities=[...this.capabilities.values()];return {edges:this.relational.snapshot(),motifs:[...this.motifs.values()].map(clone),assemblies:assemblies.map(clone),capabilities:capabilities.map(clone)}}
  snapshot(){return this.summary()}
}

global.AgentomagotchiAssembly=Object.freeze({VERSION:'0.7.0',FiveDimensionalState,RelationalEvidenceMesh,CapabilityAssemblyRuntime,NEURAL_OPERATOR_KINDS,makeOperatorGenome,stableContext});
})(typeof window!=='undefined'?window:globalThis);
