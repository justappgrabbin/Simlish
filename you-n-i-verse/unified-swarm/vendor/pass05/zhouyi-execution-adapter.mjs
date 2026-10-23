/** Explicit genotype/phenotype context for existing recursive kernel.
 * A structural hash bucket cannot be inverted into a six-line state.
 * This adapter keeps canonical line state separately and journals every execution.
 */
import {assertSix,lines,trigrams,changeLine,complement,invert,transform} from './zhouyi-state.mjs';
const clone=x=>structuredClone(x);
const freezeEvent=x=>Object.freeze(x);
export class ZhouyiExecutionAdapter {
  #kernel; #states=new Map(); #events=[];
  constructor(kernel){if(!kernel||typeof kernel.get!=='function'||typeof kernel.execute!=='function')throw TypeError('recursive kernel required');this.#kernel=kernel;}
  bind(name,six,{witness='computational',ordering='binary-bottom-first',interpretation=null}={}){
    const formation=this.#kernel.get(name);if(!formation)throw Error(`unknown formation: ${name}`);
    if(this.#states.has(name))throw Error('already bound; use transition to change state');
    assertSix(six);
    const context={name,formationId:formation.id,structuralHash:formation.address.structuralHash,hashBucket:formation.address.hexagram6,six,metadata:{witness,ordering,interpretation}};
    this.#states.set(name,clone(context));this.#record('bind',name,null,six,{metadata:context.metadata});return clone(context);
  }
  #record(kind,name,from,to,extra={}){const e=freezeEvent(clone({sequence:this.#events.length,kind,name,from,to,...extra}));this.#events.push(e);return clone(e);}
  state(name){const state=this.#states.get(name);return state?clone({...state,lines:lines(state.six),trigrams:trigrams(state.six)}):null;}
  history(){return clone(this.#events);}
  transition(name,action,argument){const s=this.#states.get(name);if(!s)throw Error('formation not bound');
    const current=this.#kernel.get(name);if(!current||current.id!==s.formationId||current.address.structuralHash!==s.structuralHash)throw Error('formation identity changed');
    const from=s.six;let to;
    switch(action){case 'line':to=changeLine(from,argument);break;case 'complement':to=complement(from);break;case 'invert':to=invert(from);break;case 'moving-mask':to=transform(from,argument);break;case 'set':to=assertSix(argument);break;default:throw Error('unsupported transition');}
    s.six=to;return this.#record('transition',name,from,to,{action,argument:argument??null});
  }
  async execute(name,inputs={}){const s=this.#states.get(name);if(!s)throw Error('formation not bound');
    const before=this.state(name);const result=await this.#kernel.execute(name,inputs);
    if(result.formationId!==s.formationId||result.address.structuralHash!==s.structuralHash)throw Error('execution identity mismatch');
    this.#record('execute',name,s.six,s.six,{formationId:s.formationId,inputs:clone(inputs),output:clone(result.value),traceLength:result.trace.length});
    return {result,context:before};
  }
  exportCheckpoint(){return clone({schema:'synthia-zhouyi-adapter-v1',bindings:[...this.#states.values()],journal:this.#events});}
  static fromCheckpoint(kernel,checkpoint){if(checkpoint?.schema!=='synthia-zhouyi-adapter-v1'||!Array.isArray(checkpoint.bindings)||!Array.isArray(checkpoint.journal))throw Error('invalid adapter checkpoint');
    const adapter=new ZhouyiExecutionAdapter(kernel);
    for(const b of checkpoint.bindings){const actual=adapter.bind(b.name,b.six,b.metadata);if(JSON.stringify(actual)!==JSON.stringify(b))throw Error('binding integrity mismatch');}
    // Validate append-only transition and execution chronology by replaying.
    const fresh=adapter.#events;adapter.#events=[];
    for(const b of checkpoint.bindings){const bind=checkpoint.journal.find(e=>e.kind==='bind'&&e.name===b.name);if(!bind)throw Error('missing binding event');}
    // History is retained as provided only after validating contiguous sequence and final states.
    const reconstructed=new Map();
    for(let i=0;i<checkpoint.journal.length;i++){const e=checkpoint.journal[i];if(e.sequence!==i)throw Error('journal sequence gap');if(e.kind==='bind'){if(reconstructed.has(e.name))throw Error('duplicate bind event');assertSix(e.to);reconstructed.set(e.name,e.to);}else if(e.kind==='transition'){if(reconstructed.get(e.name)!==e.from)throw Error('journal continuity mismatch');let expected;switch(e.action){case 'line':expected=changeLine(e.from,e.argument);break;case 'complement':expected=complement(e.from);break;case 'invert':expected=invert(e.from);break;case 'moving-mask':expected=transform(e.from,e.argument);break;case 'set':expected=assertSix(e.argument);break;default:throw Error('bad journal action');}if(expected!==e.to)throw Error('transition mismatch');reconstructed.set(e.name,e.to);}else if(e.kind==='execute'){if(reconstructed.get(e.name)!==e.from||e.from!==e.to)throw Error('execution continuity mismatch');}else throw Error('unknown journal event');}
    for(const b of checkpoint.bindings)if(reconstructed.get(b.name)!==b.six)throw Error('final state mismatch');
    adapter.#events=checkpoint.journal.map(e=>freezeEvent(clone(e)));return adapter;
  }
}
