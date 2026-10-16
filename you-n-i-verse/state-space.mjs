// NSNF state space (Nested State-Space Neural Field spec v0.1, Part IV.D) in pure JavaScript.
// What the spec and Alexis say, used as written:
//  - Five dimensions x 64 gate slots (Spec §3, §60, §63). Six-bit hexagram per slot; line 1 is the bottom (Spec §36).
//  - Alexis (2026-10-10): "64 macro nodes on each level, nested, broken down into 12 [she also said six]".
//  - Space is not a primitive (Alexis; Spec §70): its slots exist but are never written; it is READ from the other four.
//  - King Wen gate number, Fu Xi/binary order and HD gate numbering are separate named maps (Spec §36).
//  - Sparse and on demand: a node exists only once something touches it (Spec §5, §17).
// INVESTIGATING (not settled by Alexis or the spec; kept visible, nothing silently decided):
export const INVESTIGATING=Object.freeze([
 'split factor: Alexis said 12 and also six; both are accepted, 12 is the default only because she said it first (Spec §IV.D flag).',
 'number of nested levels: Spec §6 says "five levels" are tiers whose semantics are not fixed; depth 5 is configurable and is not tied to the five dimensions or centers.',
 'Space readout: no emergence rule is given (Spec §70 says the order of differentiation and integration MUST NOT be silently decided). Default is no value, reported as investigating; a rule can be injected.',
 'center_binding is null: nine centers must emerge (Spec §19); no gate-to-center table is stored and no modulo-nine mapping is used.',
 'role of the 64-bit mask (Spec §3 "pending design decision"): here it only marks which level-1 slots are active in the current event.',
 'gate numbering: the repo kernel treats address.gate as King Wen numbering; whether HD gate numbers are identical is not settled (Spec §36).']);
import {KING_WEN_TO_FUXI_DECIMAL,FUXI_DECIMAL_TO_KING_WEN} from './vendor/kingwen.mjs';
export const DIMENSIONS=Object.freeze(['Movement','Evolution','Being','Design','Space']);
export const PRIMITIVE_DIMENSIONS=Object.freeze(['Movement','Evolution','Being','Design']);
export const EMERGENT_DIMENSION='Space';
export const MACRO_PER_LEVEL=64,SPLIT_CHOICES=Object.freeze([12,6]),DEFAULT_SPLIT=12,DEFAULT_LEVELS=5;
// --- hexagram helpers: code 0..63, line 1 (bottom) = bit 0, bit string leftmost = line 6 (top). Same as vendor/kingwen.mjs.
export const bitsOf=code=>{checkCode(code);return code.toString(2).padStart(6,'0')};
export const codeOf=bits=>{if(!/^[01]{6}$/.test(bits))throw new RangeError('Six bits required');return parseInt(bits,2)};
export function checkCode(c){if(!Number.isInteger(c)||c<0||c>63)throw new RangeError('Hexagram code is 0..63');return c}
export const lineOf=(code,line)=>{checkCode(code);if(!Number.isInteger(line)||line<1||line>6)throw new RangeError('Line is 1..6');return (code>>(line-1))&1};
// Spec §36 candidate deterministic operators (different relations; never interchanged).
export const flipLines=(code,lines)=>lines.reduce((c,l)=>{lineOf(c,l);return c^(1<<(l-1))},checkCode(code));
export const complement=code=>checkCode(code)^63;
export const reverseLines=code=>{checkCode(code);let r=0;for(let i=0;i<6;i++)if((code>>i)&1)r|=1<<(5-i);return r};
export const exchangeTrigrams=code=>((checkCode(code)&7)<<3)|(code>>3);
export const nuclear=code=>{checkCode(code);return [2,3,4,3,4,5].reduce((r,l,i)=>r|(lineOf(code,l)<<i),0)}; // original lines [2,3,4,3,4,5] bottom to top
export const hammingNeighbors=(code,radius=1)=>{checkCode(code);const out=[];for(let m=1;m<64;m++){let n=0,x=m;while(x){n+=x&1;x>>=1}if(n<=radius)out.push({code:code^m,distance:n})}return out.sort((a,b)=>a.distance-b.distance||a.code-b.code)};
// Named maps, kept separate on purpose (Spec §36).
export const gateToMacro=gate=>{const m=KING_WEN_TO_FUXI_DECIMAL[gate];if(m===undefined)throw new RangeError('Gate is 1..64');return m};
export const macroToGate=macro=>{const g=FUXI_DECIMAL_TO_KING_WEN[macro];if(g===undefined)throw new RangeError('Macro is 0..63');return g};
// --- node addressing: a node is (dimension, path); path = [{macro,sub}, ...] one entry per level, depth 1..levels.
export const nodeId=(dimension,path)=>DIMENSIONS.indexOf(dimension)+':'+path.map(p=>p.macro+'.'+p.sub).join('/');
export function parseNodeId(id){
 const m=/^(\d):(\d+\.\d+(?:\/\d+\.\d+)*)$/.exec(String(id));if(!m||!DIMENSIONS[+m[1]])throw new TypeError('Not a node id: '+id);
 return {dimension:DIMENSIONS[+m[1]],path:m[2].split('/').map(s=>{const [a,b]=s.split('.').map(Number);return {macro:a,sub:b}})};
}
export function capacity({levels=DEFAULT_LEVELS,split=DEFAULT_SPLIT}={}){
 const per=BigInt(MACRO_PER_LEVEL*split),leaf=per**BigInt(levels);
 return {macroNodesPerLevel:MACRO_PER_LEVEL,split,levels,positionsPerLevel:Number(per),positionsPerDimensionAtFullDepth:leaf.toString(),
  dimensions:5,positionsAcrossFiveDimensions:(leaf*5n).toString(),primitiveDimensions:4,
  note:'Combinatorial capacity of the addressing scheme, not a claim of trained coverage. Nodes are created only when touched.'};
}
export const capacityTable=(levels=DEFAULT_LEVELS)=>SPLIT_CHOICES.map(split=>capacity({levels,split}));
export class StateSpace{
 #nodes=new Map();#active=new Set();#codes={};#emergence;
 constructor({levels=DEFAULT_LEVELS,split=DEFAULT_SPLIT,emergence=null}={}){
  if(!Number.isInteger(levels)||levels<1||levels>16)throw new RangeError('levels must be 1..16');
  if(!SPLIT_CHOICES.includes(split))throw new RangeError('split must be 12 or 6 (both are Alexis\'s words; anything else is not in the spec)');
  if(emergence!==null&&typeof emergence!=='function')throw new TypeError('emergence must be a function');
  this.levels=levels;this.split=split;this.#emergence=emergence;
 }
 get investigating(){return INVESTIGATING}
 capacity(){return capacity({levels:this.levels,split:this.split})}
 checkPath(path){
  if(!Array.isArray(path)||path.length<1||path.length>this.levels)throw new RangeError('path depth must be 1..'+this.levels);
  for(const p of path){if(!p||!Number.isInteger(p.macro)||p.macro<0||p.macro>=MACRO_PER_LEVEL)throw new RangeError('macro must be 0..63');if(!Number.isInteger(p.sub)||p.sub<0||p.sub>=this.split)throw new RangeError('sub must be 0..'+(this.split-1))}
  return path.map(p=>({macro:p.macro,sub:p.sub}));
 }
 // Create-or-get a node. Space cannot be written: it is emergent (Alexis; Spec §70).
 ensure(dimension,path){
  if(!DIMENSIONS.includes(dimension))throw new RangeError('Unknown dimension '+dimension);
  if(dimension===EMERGENT_DIMENSION)throw new Error('Space is emergent and non-primitive: it cannot be written directly; read it with readSpace()');
  const p=this.checkPath(path),id=nodeId(dimension,p);let n=this.#nodes.get(id);
  if(!n){n=Object.freeze({id,dimension,depth:p.length,path:Object.freeze(p),macro:p[p.length-1].macro,sub:p[p.length-1].sub,hexagram_code:bitsOf(p[p.length-1].macro),
    gate:macroToGate(p[p.length-1].macro),center_binding:null,parent:p.length>1?nodeId(dimension,p.slice(0,-1)):null});this.#nodes.set(id,n)}
  return n;
 }
 has(id){return this.#nodes.has(id)}
 get(id){return this.#nodes.get(id)||null}
 get size(){return this.#nodes.size}
 *children(id,{macros=null}={}){ // lazy: 64*split children, never materialised unless asked
  const n=this.get(id)||(()=>{const {dimension,path}=parseNodeId(id);return {dimension,path,depth:path.length}})();
  if(n.depth>=this.levels)return;
  for(let m=0;m<MACRO_PER_LEVEL;m++){if(macros&&!macros.includes(m))continue;for(let s=0;s<this.split;s++)yield {dimension:n.dimension,path:[...n.path,{macro:m,sub:s}]}}
 }
 ancestors(id){const out=[];let n=this.get(id);while(n&&n.parent){n=this.get(n.parent)||this.ensure(n.dimension,n.path.slice(0,-1));out.push(n)}return out}
 siblings(id,radius=1){const n=this.get(id);if(!n)throw new RangeError('Unknown node');
  return hammingNeighbors(n.macro,radius).map(h=>({node:this.ensure(n.dimension,[...n.path.slice(0,-1),{macro:h.code,sub:n.sub}]),distance:h.distance}))}
 // activation (current event only)
 activate(id){const n=this.get(id);if(!n)throw new RangeError('Unknown node '+id);this.#active.add(id);return n}
 clearActive(){this.#active.clear()}
 activeIds(){return [...this.#active]}
 // 64-bit slot masks per dimension: bit m set when a node under level-1 macro m is active. Space is derived-only (always 0 here).
 masks(){const out={};for(const d of DIMENSIONS)out[d]=0n;for(const id of this.#active){const n=this.#nodes.get(id);out[n.dimension]|=1n<<BigInt(n.path[0].macro)}return out}
 maskHex(){return Object.fromEntries(Object.entries(this.masks()).map(([d,m])=>[d,m.toString(16).padStart(16,'0')]))}
 setDimensionCode(dimension,code){if(dimension===EMERGENT_DIMENSION)throw new Error('Space code is read, not set');if(!PRIMITIVE_DIMENSIONS.includes(dimension))throw new RangeError('Unknown dimension');this.#codes[dimension]=checkCode(code)}
 codes(){return {...Object.fromEntries(PRIMITIVE_DIMENSIONS.map(d=>[d,this.#codes[d]??null])),Space:this.readSpace().code}}
 readSpace(){
  const four=Object.fromEntries(PRIMITIVE_DIMENSIONS.map(d=>[d,this.#codes[d]??null]));
  if(!this.#emergence)return {dimension:'Space',status:'investigating',code:null,slots:64,derivedOnly:true,note:'No emergence rule is specified (Spec §70); Space is not invented. Inject an emergence function to read it.'};
  if(PRIMITIVE_DIMENSIONS.some(d=>four[d]===null))return {dimension:'Space',status:'investigating',code:null,slots:64,derivedOnly:true,note:'Space is read from all four primitive dimensions; some have no state yet.'};
  const code=this.#emergence(Object.freeze({...four}));
  return {dimension:'Space',status:'derived',code:code===null?null:checkCode(code),slots:64,derivedOnly:true,from:four};
 }
 snapshot(){return {levels:this.levels,split:this.split,nodes:[...this.#nodes.keys()],codes:{...this.#codes}}}
 restore(s){
  if(s.levels!==this.levels||s.split!==this.split)throw new Error('Snapshot shape (levels/split) differs from this state space');
  this.#nodes.clear();this.#active.clear();for(const id of s.nodes){const {dimension,path}=parseNodeId(id);this.ensure(dimension,path)}
  for(const k of Object.keys(this.#codes))delete this.#codes[k];Object.assign(this.#codes,s.codes||{});
 }
}
