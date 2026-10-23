
function cloneable(value) {
  try { return structuredClone(value); } catch { return value; }
}

function flatten(value) {
  if (!Array.isArray(value)) return [value];
  return value.flat(Infinity);
}
function elementwise(a,b,fn) {
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) throw new Error('shape mismatch');
    return a.map((v,i)=>elementwise(v,b[i],fn));
  }
  if (Array.isArray(a)) return a.map(v=>elementwise(v,b,fn));
  if (Array.isArray(b)) return b.map(v=>elementwise(a,v,fn));
  return fn(Number(a),Number(b));
}
function mapDeep(a,fn) {
  return Array.isArray(a) ? a.map(v=>mapDeep(v,fn)) : fn(Number(a));
}
function matmul(a,b) {
  if (!Array.isArray(a)||!Array.isArray(b)||!Array.isArray(a[0])||!Array.isArray(b[0])) throw new Error('matmul expects rank-2 arrays');
  const rows=a.length, inner=a[0].length, cols=b[0].length;
  if (b.length!==inner) throw new Error('matmul dimension mismatch');
  const out=Array.from({length:rows},()=>Array(cols).fill(0));
  for(let i=0;i<rows;i++) for(let k=0;k<inner;k++) for(let j=0;j<cols;j++) out[i][j]+=Number(a[i][k])*Number(b[k][j]);
  return out;
}
function softmax(x) {
  const arr=flatten(x).map(Number);
  const m=Math.max(...arr);
  const e=arr.map(v=>Math.exp(v-m));
  const s=e.reduce((a,b)=>a+b,0)||1;
  return e.map(v=>v/s);
}
function reshape(flat,shape) {
  const data=flatten(flat);
  const total=shape.reduce((a,b)=>a*Number(b),1);
  if(total!==data.length) throw new Error(`reshape ${data.length} values into ${shape.join('x')}`);
  let cursor=0;
  function build(dim){
    const n=Number(shape[dim]);
    if(dim===shape.length-1) return Array.from({length:n},()=>data[cursor++]);
    return Array.from({length:n},()=>build(dim+1));
  }
  return shape.length?build(0):data[0];
}

const OPS = Object.freeze({
  identity: ([a])=>a,
  const: (_args,node)=>cloneable(node.value),
  add: ([a,b])=>elementwise(a,b,(x,y)=>x+y),
  sub: ([a,b])=>elementwise(a,b,(x,y)=>x-y),
  mul: ([a,b])=>elementwise(a,b,(x,y)=>x*y),
  div: ([a,b])=>elementwise(a,b,(x,y)=>x/y),
  mod: ([a,b])=>elementwise(a,b,(x,y)=>x%y),
  pow: ([a,b])=>elementwise(a,b,(x,y)=>x**y),
  min: (args)=>Math.min(...flatten(args)),
  max: (args)=>Math.max(...flatten(args)),
  abs: ([a])=>mapDeep(a,Math.abs),
  sqrt: ([a])=>mapDeep(a,Math.sqrt),
  exp: ([a])=>mapDeep(a,Math.exp),
  log: ([a])=>mapDeep(a,Math.log),
  sin: ([a])=>mapDeep(a,Math.sin),
  cos: ([a])=>mapDeep(a,Math.cos),
  tanh: ([a])=>mapDeep(a,Math.tanh),
  relu: ([a])=>mapDeep(a,x=>Math.max(0,x)),
  sigmoid: ([a])=>mapDeep(a,x=>1/(1+Math.exp(-x))),
  softmax: ([a])=>softmax(a),
  eq: ([a,b])=>a===b,
  neq: ([a,b])=>a!==b,
  lt: ([a,b])=>a<b,
  lte: ([a,b])=>a<=b,
  gt: ([a,b])=>a>b,
  gte: ([a,b])=>a>=b,
  and: ([a,b])=>Boolean(a&&b),
  or: ([a,b])=>Boolean(a||b),
  not: ([a])=>!a,
  select: ([cond,a,b])=>cond?a:b,
  concat: (args,node)=>[].concat(...args.map(v=>Array.isArray(v)?v:[v])),
  slice: ([a],node)=>a.slice(node.start??0,node.end??a.length),
  reshape: ([a],node)=>reshape(a,node.shape??[]),
  matmul: ([a,b])=>matmul(a,b),
  sum: ([a])=>flatten(a).reduce((x,y)=>Number(x)+Number(y),0),
  mean: ([a])=>{const x=flatten(a).map(Number);return x.length?x.reduce((m,n)=>m+n,0)/x.length:0;},
});

function normalizeProgram(program) {
  if (!program || typeof program !== 'object') throw new TypeError('primitive program must be an object');
  const nodes = Array.isArray(program.nodes) ? program.nodes
    : Array.isArray(program.ops) ? program.ops
      : null;
  if (!nodes) throw new TypeError('primitive program requires nodes[] or ops[]');
  return {
    inputs: program.inputs && typeof program.inputs === 'object' ? program.inputs : {},
    nodes,
    outputs: Array.isArray(program.outputs) ? program.outputs : null,
  };
}

export class PrimitiveExecutionVM {
  constructor({maxNodes=10000}={}) {
    this.maxNodes=maxNodes;
    this.operators=new Map(Object.entries(OPS));
  }
  registerOperator(name,fn) {
    if(!name || typeof fn!=='function') throw new TypeError('operator requires name and function');
    this.operators.set(String(name).toLowerCase(),fn);
    return this;
  }
  canRun(program) {
    try {
      const p=normalizeProgram(program);
      return p.nodes.length<=this.maxNodes && p.nodes.every(n=>this.operators.has(String(n.op??n.operator??'').toLowerCase()));
    } catch { return false; }
  }
  async run(program,{inputs={}}={}) {
    const p=normalizeProgram(program);
    if(p.nodes.length>this.maxNodes) throw new Error(`primitive program exceeds ${this.maxNodes} nodes`);
    const env=new Map(Object.entries({...p.inputs,...inputs}));
    const trace=[];
    for(let index=0;index<p.nodes.length;index++){
      const node=p.nodes[index]??{};
      const id=String(node.id??node.output??`n${index}`);
      const opName=String(node.op??node.operator??'').toLowerCase();
      const op=this.operators.get(opName);
      if(!op) throw new Error(`unsupported primitive operator: ${opName}`);
      const refs=Array.isArray(node.inputs)?node.inputs:Array.isArray(node.args)?node.args:[];
      const args=refs.map(ref=>{
        if(typeof ref==='string' && env.has(ref)) return env.get(ref);
        if(ref && typeof ref==='object' && Object.prototype.hasOwnProperty.call(ref,'value')) return ref.value;
        return ref;
      });
      const value=await op(args,node,{env,index});
      env.set(id,value);
      trace.push(Object.freeze({id,op:opName,inputRefs:Object.freeze([...refs]),value:cloneable(value)}));
    }
    const outputIds=p.outputs??(p.nodes.length?[String(p.nodes.at(-1).id??p.nodes.at(-1).output??`n${p.nodes.length-1}`)]:[]);
    const output=Object.fromEntries(outputIds.map(id=>[id,cloneable(env.get(id))]));
    return Object.freeze({ok:true,engine:'synthia-primitive-vm',stdout:Object.freeze([]),returnValue:Object.freeze(output),trace:Object.freeze(trace)});
  }
}

export class UniversalPrimitiveExecutionBridge {
  constructor({vm=new PrimitiveExecutionVM(),foreignLowerers=[]}={}) {
    this.vm=vm;
    this.lowerers=[...foreignLowerers];
  }
  registerLowerer(lowerer) {
    if(!lowerer || typeof lowerer.canLower!=='function' || typeof lowerer.lower!=='function') throw new TypeError('lowerer requires canLower() and lower()');
    this.lowerers.push(lowerer);
    return lowerer;
  }
  extractInlineProgram(normalized,context={}) {
    return context.primitiveProgram
      ?? normalized?.original?.primitiveProgram
      ?? normalized?.original?.program
      ?? (normalized?.kind==='primitive-ir' ? normalized?.original : null)
      ?? null;
  }
  async lower(normalized,context={}) {
    const inline=this.extractInlineProgram(normalized,context);
    if(inline) return {program:inline,lowerer:'inline-primitive-program'};
    for(const lowerer of this.lowerers){
      if(await lowerer.canLower(normalized,context)) {
        const program=await lowerer.lower(normalized,context);
        if(program) return {program,lowerer:lowerer.id??'anonymous-lowerer'};
      }
    }
    return null;
  }
  async execute(normalized,context={}) {
    const lowered=await this.lower(normalized,context);
    if(!lowered) return null;
    if(!this.vm.canRun(lowered.program)) {
      return {
        ok:false,kind:normalized.kind,path:'primitive-lowering-incomplete',
        error:'lowered program contains operators not mounted in the primitive VM',
        unresolvedDependencies:['primitive-operator'],
        backendUsed:false,workerUsed:false,
      };
    }
    try{
      const result=await this.vm.run(lowered.program,{inputs:context.inputs??{}});
      return {
        ok:true,kind:normalized.kind,path:'synthia-primitive-vm',
        lowering:lowered.lowerer,backendUsed:false,workerUsed:false,result,
      };
    }catch(error){
      return {
        ok:false,kind:normalized.kind,path:'synthia-primitive-vm',
        error:String(error?.message??error),backendUsed:false,workerUsed:false,
      };
    }
  }
}

export default UniversalPrimitiveExecutionBridge;
