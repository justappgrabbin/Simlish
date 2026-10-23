import {ConsciousnessSwarm} from '../vendor/consciousness/consciousness-swarm.mjs';
import {RecursivePrimitiveKernel} from '../vendor/pass05/recursive-primitive-kernel.mjs';
import {PositionedStateSpace,resolvePosition} from '../vendor/pass05/positioned-state-space.mjs';
import {sixForGate} from '../vendor/pass05/king-wen-fuxi-map.mjs';
import {expressPrimitive,toSVG} from '../vendor/pass05/tri-modal-expression.mjs';
import {composeDenseFormation,executeNeuralFormation} from '../vendor/neural/neural-formation-adapter.mjs';
import {packSpriteSheet} from './spritesheet.mjs';
import {cloudScene,cloudSVG} from './semantic-cloud.mjs';
import {projectThreeTwoOne,mountThreeTwoOne,transitionThreeTwoOne} from './three-two-one.mjs';
import {projectZhouyiSixLine,transitionZhouyiSixLine} from './zhouyi-six-line.mjs';
const clone=v=>structuredClone(v);
const boundedId=id=>typeof id==='string'&&/^[A-Za-z][\w:.-]{0,79}$/.test(id);
const PRIMITIVES=Object.freeze(['tool-factory','neural-net-maker','model-maker','foundry']);
/**
 * Existing genotype authority -> contextual formation + executable VM -> Simlish phenotype.
 * Does not assign missing semantic/astronomical coordinates or alter source implementations.
 */
export class UnifiedSwarm {
  constructor({source=null,simlishPlug=null,authorize=null,swarm=null,kernel=null}={}){
    this.source=source;this.simlishPlug=simlishPlug;this.authorize=authorize;
    this.swarm=swarm??new ConsciousnessSwarm();this.kernel=kernel??new RecursivePrimitiveKernel();
    this.capabilities=new Map();this.receipts=[];this.nextReceipt=1;
  }
  get primitives(){return [...PRIMITIVES];}
  /** Corrected 3–2–1: three, two, one SINGLE-BIT binary roles, low to high. */
  threeTwoOne(spec){return projectThreeTwoOne(spec);}
  /** Mount the 3+2+1 single-bit resident positions. */
  mount321(spec){return mountThreeTwoOne(this,spec);}
  /** Update one explicit binary position, not a Zhouyi moving line. */
  transition321(input,options){return transitionThreeTwoOne(input,options);}
  /** Source-backed Zhouyi remains a SEPARATE representation. */
  zhouyi(spec){return projectZhouyiSixLine(spec);}
  transitionZhouyi(spec,options){return transitionZhouyiSixLine(spec,options);}
  inspect(subject,context={}){
    if(typeof this.source?.structure!=='function')throw new Error('CONNECTED_GENOTYPE_SOURCE_REQUIRED');
    return this.source.structure(subject,{...context,addressMode:'structural'});
  }
  /** Caller-supplied IDs and evidence only; placement never changes semantic coordinates. */
  add({id,label,kind='capability',x=0,y=0,motion='FREE',address=null,provenance=null}={}){
    if(!boundedId(id)||typeof label!=='string'||!label.trim())throw new TypeError('stable id and nonempty label required');
    if(!Number.isFinite(x)||!Number.isFinite(y)||Math.abs(x)>1||Math.abs(y)>1)throw new RangeError('display x/y must be in [-1,1]');
    const node=this.swarm.createPrimitive({id,kind,value:label,address:address??{},properties:{provenance:provenance??'explicit-ui-node'}});
    this.swarm.addFormationParticle({nodeId:id,x,y,motion});
    return node;
  }
  /** A named semantic predicate requires provenance. Transport links are a separate layer. */
  relate({from,to,predicate,provenance,context={}}={}){
    if(typeof predicate!=='string'||!predicate||typeof provenance!=='string'||!provenance)throw new TypeError('semantic relation and its provenance required');
    const edge=this.swarm.connect(from,to,{relation:predicate,provenance:{source:provenance},context});
    this.swarm.linkTransport(from,to);
    return edge;
  }
  /** Existing consciousness-swarm moment-gradient step, bounded by its own motion constraints. */
  arrange(targets,options={}){return this.swarm.stepFormation(targets,options);}
  cloud(options={}){const scene=cloudScene(this.swarm,options);return {...scene,byId:undefined};}
  cloudSVG(options={}){return cloudSVG(cloudScene(this.swarm,options));}
  /** Existing Pass05 VM: hash in formation.address is ONLY a storage fingerprint, NOT semantic placement. */
  form(program){return this.kernel.form(program);}
  executeFormation(name,inputs={}){return this.kernel.execute(name,inputs);}
  neural({layers,address=null,id='neural-formation',input}){
    const formation=composeDenseFormation({layers,address,id});
    return executeNeuralFormation(this.kernel.vm,formation,input);
  }
  /** Gate->six is source lookup (King Wen/Fu Xi), not gate_number & 63. */
  expression({gate,value,provenance,action='form',generation=0}={}){
    if(!Number.isInteger(gate)||gate<1||gate>64||typeof provenance!=='string'||!provenance)throw new TypeError('source-supported gate and provenance required');
    return expressPrimitive({address:{hexagram6:sixForGate(gate),ontology:{gate,provenance}},value,action,generation});
  }
  expressionSVG(spec,options={}){return toSVG(this.expression(spec),options);}
  /** Pass05 positional projection requires explicit planet/axis; no missing-field defaults. */
  position(position){return resolvePosition(position);}
  positionLedger(initial){return new PositionedStateSpace(initial);}
  sprites(frames,options={}){return packSpriteSheet(frames,options);}
  /** Existing host abilities only. No fake built-in factory or arbitrary code evaluation. */
  register({name,execute,sideEffect=true,requires=[]}={}){
    if(!boundedId(name)||typeof execute!=='function'||typeof sideEffect!=='boolean'||!Array.isArray(requires)||requires.some(s=>typeof s!=='string'))throw new TypeError('invalid capability descriptor');
    if(this.capabilities.has(name)||this.capabilities.size>=26)throw new Error('duplicate capability or 26-slot action limit');
    this.capabilities.set(name,Object.freeze({execute,sideEffect,requires:[...requires]}));
  }
  choices(context={}){
    const features=new Set(context.features??[]);
    return [...this.capabilities].map(([name,c])=>({name,eligible:c.requires.every(key=>features.has(key)),required:[...c.requires]}));
  }
  async dispatch(name,input,{context={},ownerPolicy=null,autonomy='manual'}={}){
    const cap=this.capabilities.get(name);if(!cap)throw new Error('CAPABILITY_NOT_REGISTERED');
    if(!cap.requires.every(key=>(context.features??[]).includes(key)))throw new Error('CONTEXT_CONSTRAINT_UNSATISFIED');
    const checker=ownerPolicy??this.authorize;
    const ok=!cap.sideEffect||typeof checker==='function'&&await checker({op:name,input,context,autonomy})===true;
    if(!ok){const record={seq:this.nextReceipt++,name,status:'blocked',reason:'OWNER_AUTHORIZATION_REQUIRED'};this.receipts.push(record);return clone(record);}
    try{const result=await cap.execute(input,{context,autonomy});const record={seq:this.nextReceipt++,name,status:'executed',result:clone(result)};this.receipts.push(record);return clone(record);}
    catch(error){const record={seq:this.nextReceipt++,name,status:'failed',reason:String(error?.message??error)};this.receipts.push(record);return clone(record);}
  }
  /** Live Simlish plug owns canonical genotype validation, Klein witness and world mutation. */
  async morphSimlish({intent,genotypeAddress,genotypeSource,objectId,operation='transform',subject=null,context={}}={}){
    if(typeof this.simlishPlug?.morph!=='function')throw new Error('SIMLISH_PHENOTYPE_PLUG_NOT_CONNECTED');
    if(!['owner-explicit','externally-verified'].includes(genotypeSource))throw new Error('CANONICAL_GENOTYPE_PROVENANCE_REQUIRED');
    const result=await this.simlishPlug.morph({intent,genotypeAddress,genotypeSource,objectId,operation,subject,context});
    const rec={seq:this.nextReceipt++,name:'simlish.phenotype.morph',status:result.phenotype?.status??'unresolved',
      receipt:clone(result),note:'delegated to existing source-witnessed, owner-authorized Simlish plug'};
    this.receipts.push(rec);return clone(rec);
  }
  snapshot(){return {schema:'synthia.unified-swarm.v1',swarm:this.swarm.snapshot(),kernel:this.kernel.exportCheckpoint(),receipts:clone(this.receipts),nextReceipt:this.nextReceipt,
    policies:{originals:'immutable',semantic:'external-authority-only',primitives:PRIMITIVES}};}
  static restore(snapshot,options={}){
    if(snapshot?.schema!=='synthia.unified-swarm.v1'||!snapshot.swarm||!snapshot.kernel||!Array.isArray(snapshot.receipts)||!Number.isInteger(snapshot.nextReceipt))throw new Error('INVALID_SWARM_CHECKPOINT');
    const unified=new UnifiedSwarm({...options,swarm:ConsciousnessSwarm.restore(snapshot.swarm),kernel:RecursivePrimitiveKernel.fromCheckpoint(snapshot.kernel)});
    unified.receipts=clone(snapshot.receipts);unified.nextReceipt=snapshot.nextReceipt;return unified;
  }
}
export function createUnifiedSwarm(options={}){return new UnifiedSwarm(options);}
