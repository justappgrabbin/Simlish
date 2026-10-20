/** Additive recursive composition layer for the existing Synthia PrimitiveExecutionVM.
 * No external models, worker, network, or predefined phenotype templates.
 * A formation is an immutable graph of references to atoms or earlier formations.
 */
import {PrimitiveExecutionVM} from './universal-primitive-execution-bridge.mjs';
const copy = value => structuredClone(value);
const allowed = new Set(['identity','const','add','sub','mul','div','mod','pow','min','max','abs','sqrt','exp','log','sin','cos','tanh','relu','sigmoid','softmax','eq','neq','lt','lte','gt','gte','and','or','not','select','concat','slice','reshape','matmul','sum','mean']);
const arity = {identity:1,const:0,add:2,sub:2,mul:2,div:2,mod:2,pow:2,abs:1,sqrt:1,exp:1,log:1,sin:1,cos:1,tanh:1,relu:1,sigmoid:1,softmax:1,eq:2,neq:2,lt:2,lte:2,gt:2,gte:2,and:2,or:2,not:1,select:3,slice:1,reshape:1,sum:1,mean:1,matmul:2};
function assertName(name){if(typeof name!=='string'||!/^[A-Za-z][A-Za-z0-9_-]{0,63}$/.test(name))throw Error('invalid identifier');}
function fingerprint(data){const s=JSON.stringify(data);let h=2166136261;for(let i=0;i<s.length;i++){h=Math.imul(h^s.charCodeAt(i),16777619)>>>0;}return h.toString(16).padStart(8,'0');}
export class RecursivePrimitiveKernel {
  #formations=new Map(); #journal=[]; #counter=0;
  constructor({vm=new PrimitiveExecutionVM()}={}){this.vm=vm;}
  /** Atomic references are input names or literal values, not built-in functions. */
  form({name,inputs=[],steps=[],output,parents=[],ontology={}}){
    assertName(name);if(this.#formations.has(name))throw Error('formation already exists');
    if(!Array.isArray(inputs)||!Array.isArray(steps)||!Array.isArray(parents))throw Error('invalid formation');
    const inputNames=new Set();for(const input of inputs){assertName(input);if(inputNames.has(input))throw Error('duplicate input');inputNames.add(input);}
    const parentMap=new Map();for(const parent of parents){assertName(parent);if(parentMap.has(parent))throw Error('duplicate parent');const p=this.#formations.get(parent);if(!p)throw Error(`unknown parent ${parent}`);parentMap.set(parent,p);}
    const known=new Set(inputs);const normalized=[];
    for(const step of steps){
      assertName(step.id);if(known.has(step.id))throw Error('duplicate step ID');
      if(step.use){
        const parent=parentMap.get(step.use);if(!parent)throw Error('parent must be declared');
        if(!Array.isArray(step.args)||step.args.length!==parent.inputs.length)throw Error('parent argument count mismatch');
      }else{
        if(!allowed.has(step.op))throw Error(`unsupported operator ${step.op}`);
        if(!Array.isArray(step.args))throw Error('step arguments required');
        if(arity[step.op]!==undefined && step.args.length!==arity[step.op])throw Error('operator argument count mismatch');
      }
      for(const arg of step.args){if(typeof arg==='string'&&!known.has(arg))throw Error(`unknown reference ${arg}; literal strings use {value:...}`);if(arg&&typeof arg==='object'&&!Object.hasOwn(arg,'value'))throw Error('literal must use {value:...}');}
      normalized.push(copy(step));known.add(step.id);
    }
    if(!known.has(output))throw Error('output reference missing');
    const id=`f${++this.#counter}`;
    const genotype={name,inputs:[...inputs],steps:normalized,output,parents:[...parents],ontology:copy(ontology)};
    const address={hexagram6:parseInt(fingerprint(genotype).slice(-2),16)&63,structuralHash:fingerprint(genotype),ontology:copy(ontology)};
    const formation={id,...genotype,address};
    this.#formations.set(name,copy(formation));this.#journal.push(copy({event:'form',id,name,parents:[...parents],address,genotype}));
    return copy(formation);
  }
  get(name){const v=this.#formations.get(name);return v?copy(v):null;}
  history(){return copy(this.#journal);}
  /** Recursively lower child references into the existing VM instruction set. */
  lower(name){const formation=this.#formations.get(name);if(!formation)throw Error('unknown formation');
    let serial=0;const nodes=[];
    const inline=(form, bindings)=>{
      const env=new Map(form.inputs.map((input,i)=>[input,bindings[i]]));
      for(const step of form.steps){
        const refs=step.args.map(arg=>typeof arg==='string'?env.get(arg):copy(arg));
        if(step.use){const parent=this.#formations.get(step.use);env.set(step.id,inline(parent,refs));}
        else {const id=`r${serial++}`;const node={id,op:step.op,inputs:refs};
          for(const field of ['value','shape','start','end'])if(Object.hasOwn(step,field))node[field]=copy(step[field]);
          nodes.push(node);env.set(step.id,id);
        }
      }
      return env.get(form.output);
    };
    const output=inline(formation,formation.inputs);
    return {inputs:Object.fromEntries(formation.inputs.map(k=>[k,null])),nodes,outputs:[output],resultRef:output,address:copy(formation.address)};
  }
  async execute(name,inputs={}){
    const f=this.#formations.get(name);if(!f)throw Error('unknown formation');
    for(const key of f.inputs)if(!Object.hasOwn(inputs,key))throw Error(`missing input ${key}`);
    const compiled=this.lower(name);
    // Output may be a direct input; add an identity node to expose it consistently.
    if(!compiled.nodes.some(n=>n.id===compiled.resultRef)){
      compiled.nodes.push({id:`r${compiled.nodes.length}`,op:'identity',inputs:[compiled.resultRef]});compiled.outputs=[compiled.nodes.at(-1).id];
    }
    const result=await this.vm.run(compiled,{inputs});const value=result.returnValue[compiled.outputs[0]];
    this.#journal.push(copy({event:'execute',formationId:f.id,name,inputs,output:value,traceLength:result.trace.length}));
    return {value,trace:result.trace,address:copy(f.address),formationId:f.id};
  }
  exportCheckpoint(){return copy({schema:'synthia-recursive-primitives-v1',formations:[...this.#formations.values()],journal:this.#journal,counter:this.#counter});}
  static fromCheckpoint(checkpoint,options={}){
    if(checkpoint?.schema!=='synthia-recursive-primitives-v1')throw Error('invalid checkpoint');
    const kernel=new RecursivePrimitiveKernel(options);
    for(const saved of checkpoint.formations){
      const actual=kernel.form(saved);
      if(JSON.stringify(actual)!==JSON.stringify(saved))throw Error('checkpoint integrity mismatch');
    }
    if(!Array.isArray(checkpoint.journal)||checkpoint.counter!==checkpoint.formations.length)throw Error('invalid checkpoint history');
    kernel.#journal=copy(checkpoint.journal);return kernel;
  }
}
