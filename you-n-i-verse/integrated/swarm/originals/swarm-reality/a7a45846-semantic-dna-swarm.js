/**
 * Semantic DNA Swarm Observer for Synthia / LCM
 *
 * ADDITIVE integration layer. It does not replace the LCM state space.
 * It observes exact resolved addresses produced by lcm-address-resolver.js,
 * preserves the five dimensional labels, projects them into an inspectable
 * feature vector, records transitions, and learns a sparse dictionary over
 * those computed frames.
 *
 * No microtubule physics, temperature analogue, invented Hamming chemistry,
 * or hard-coded disorder/swarm phase labels are introduced here.
 */
import { resolveAddress } from './lcm-address-resolver.js';
import { KW_TO_BITS, nuclearHexagram } from './hexagram_sequences.js';
import { GATE_TO_CHANNELS } from './lcm-chatbot.js';

export const DIMENSION_ORDER = Object.freeze(['Movement','Evolution','Being','Design','Space']);
const D = 5, G = 64, L = 6, C = 6, T = 6, B = 5;
const OFF = Object.freeze({ dim:0, gate:D, line:D+G, color:D+G+L, tone:D+G+L+C, base:D+G+L+C+T,
  gateBits:D+G+L+C+T+B, nuclearBits:D+G+L+C+T+B+6, dms:D+G+L+C+T+B+12 });
export const FEATURE_COUNT = OFF.dms + 4; // degree, minute, second, within-base

function clamp01(x){ return Math.max(0, Math.min(1, Number.isFinite(x) ? x : 0)); }
function oneHot(v, off, n, idx){ if(idx >= 0 && idx < n) v[off+idx] += 1; }
function bitFeatures(v, off, bits, n=6){ for(let i=0;i<n;i++) v[off+i] += (bits >> i) & 1; }
function norm(v){ let s=0; for(const x of v) s+=x*x; s=Math.sqrt(s)||1; return v.map(x=>x/s); }

export function resolveDimensionalFrame(input, {timestamp=Date.now(), id=null}={}) {
  if (!input || typeof input !== 'object') throw new TypeError('frame input must be an object keyed by dimension');
  const dimensions = {};
  for (const dim of DIMENSION_ORDER) {
    const spec = input[dim];
    if (spec == null) continue;
    const arc = typeof spec === 'number' ? spec : spec.arcSeconds;
    const wheelSource = typeof spec === 'object' && spec.wheelSource ? spec.wheelSource : 'bg';
    if (!Number.isFinite(arc)) throw new TypeError(`${dim}.arcSeconds must be finite`);
    const r = resolveAddress(arc, wheelSource);
    if (r?.error) throw new Error(`${dim} resolution failed: ${r.error}`);
    dimensions[dim] = r;
  }
  if (!Object.keys(dimensions).length) throw new Error('frame contains no dimensional positions');
  return Object.freeze({ id:id ?? `frame_${timestamp}`, timestamp, dimensions:Object.freeze(dimensions) });
}

export function encodeFrame(frame) {
  const v = Array(FEATURE_COUNT).fill(0);
  const gateOccupancy = Array(65).fill(0);
  const activeChannels = new Map();
  let n=0;
  for (const [dim, r] of Object.entries(frame.dimensions)) {
    n++;
    oneHot(v, OFF.dim, D, DIMENSION_ORDER.indexOf(dim));
    const gate=r.gate?.number, line=r.line?.number, color=r.color?.number, tone=r.tone?.number, base=r.base?.number;
    oneHot(v, OFF.gate, G, gate-1); oneHot(v, OFF.line, L, line-1); oneHot(v, OFF.color, C, color-1);
    oneHot(v, OFF.tone, T, tone-1); oneHot(v, OFF.base, B, base-1);
    if (gate>=1 && gate<=64) {
      gateOccupancy[gate]++;
      const bits=KW_TO_BITS[gate];
      if (bits != null) { bitFeatures(v,OFF.gateBits,bits); bitFeatures(v,OFF.nuclearBits,nuclearHexagram(bits)); }
      for (const ch of (GATE_TO_CHANNELS[gate] || [])) activeChannels.set(ch,(activeChannels.get(ch)||0)+1);
    }
    const p=r.dms?.position;
    if(p){ v[OFF.dms]+=p.degrees/359; v[OFF.dms+1]+=p.minutes/59; v[OFF.dms+2]+=p.seconds/60; v[OFF.dms+3]+=p.within_base_arc/18.75; }
  }
  for(let i=0;i<v.length;i++) v[i]/=n;
  return Object.freeze({ vector:Object.freeze(norm(v)), gateOccupancy:Object.freeze(gateOccupancy),
    activeChannels:Object.freeze([...activeChannels.entries()].map(([name,count])=>({name,count,complete:count>=2}))), dimensions:n });
}

export function transitionBetween(a,b){
  const events=[];
  for(const dim of DIMENSION_ORDER){
    const x=a.dimensions[dim], y=b.dimensions[dim];
    if(!x && y){events.push({dimension:dim,operator:'•',type:'ignition',to:addressKey(y)});continue;}
    if(x && !y){events.push({dimension:dim,operator:'/',type:'dormancy',from:addressKey(x)});continue;}
    if(!x||!y) continue;
    const levels=[['gate','number','°'],['line','number','.'],['color','number',','],['tone','number','.'],['base','number','.']];
    for(const [k,p,op] of levels) if(x[k]?.[p]!==y[k]?.[p]) events.push({dimension:dim,operator:op,type:'transform',level:k,from:x[k]?.[p]??null,to:y[k]?.[p]??null});
    const xp=x.dms?.position, yp=y.dms?.position;
    if(xp&&yp){ if(xp.degrees!==yp.degrees) events.push({dimension:dim,operator:'°',type:'transform',level:'degree',from:xp.degrees,to:yp.degrees});
      if(xp.minutes!==yp.minutes) events.push({dimension:dim,operator:'′',type:'transform',level:'minute',from:xp.minutes,to:yp.minutes});
      if(xp.seconds!==yp.seconds) events.push({dimension:dim,operator:'″',type:'transform',level:'second',from:xp.seconds,to:yp.seconds}); }
  }
  return Object.freeze(events);
}
function addressKey(r){return `${r.gate?.number??'?'}:${r.line?.number??'?'}:${r.color?.number??'?'}:${r.tone?.number??'?'}:${r.base?.number??'?'}`;}

function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;}
function soft(x,l){return x>l?x-l:x<-l?x+l:0;}
function sparseCode(x, atoms, lambda=.08, passes=12){
  const c=Array(atoms.length).fill(0);
  for(let p=0;p<passes;p++) for(let j=0;j<atoms.length;j++){
    let rho=dot(atoms[j],x); for(let k=0;k<atoms.length;k++) if(k!==j && c[k]!==0) rho-=c[k]*dot(atoms[j],atoms[k]);
    c[j]=soft(rho,lambda);
  }
  return c;
}
function normalizeAtom(a){let s=Math.sqrt(dot(a,a))||1;return a.map(x=>x/s);}

export class SparseSemanticDictionary {
  constructor({atoms=12,lambda=.08,iterations=20}={}){this.k=atoms;this.lambda=lambda;this.iterations=iterations;this.atoms=[];this.trained=false;}
  fit(frames){
    const X=frames.map(f=>Array.from(f.vector ?? encodeFrame(f).vector));
    if(X.length<this.k) throw new Error(`need at least ${this.k} frames to learn ${this.k} atoms`);
    this.atoms=Array.from({length:this.k},(_,i)=>normalizeAtom([...X[Math.floor(i*X.length/this.k)]]));
    let codes=[];
    for(let it=0;it<this.iterations;it++){
      codes=X.map(x=>sparseCode(x,this.atoms,this.lambda));
      for(let j=0;j<this.k;j++){
        const acc=Array(FEATURE_COUNT).fill(0); let w=0;
        for(let i=0;i<X.length;i++){const cj=codes[i][j]; if(Math.abs(cj)<1e-9)continue; w+=cj*cj; for(let q=0;q<acc.length;q++){let residual=X[i][q]; for(let k=0;k<this.k;k++)if(k!==j)residual-=codes[i][k]*this.atoms[k][q]; acc[q]+=cj*residual;}}
        if(w>1e-12)this.atoms[j]=normalizeAtom(acc.map(z=>z/w));
      }
    }
    this.trained=true; return {atoms:this.atoms.length,frames:X.length,lambda:this.lambda,iterations:this.iterations};
  }
  encode(frame){if(!this.trained)throw new Error('dictionary not trained'); const x=Array.from(frame.vector ?? encodeFrame(frame).vector); return sparseCode(x,this.atoms,this.lambda);}
  reconstruct(coeff){const out=Array(FEATURE_COUNT).fill(0);for(let j=0;j<this.atoms.length;j++)for(let q=0;q<out.length;q++)out[q]+=coeff[j]*this.atoms[j][q];return out;}
  mse(frame){const x=Array.from(frame.vector ?? encodeFrame(frame).vector), r=this.reconstruct(this.encode(frame));return x.reduce((s,z,i)=>s+(z-r[i])**2,0)/x.length;}
}

export class SynthiaStateObserver {
  constructor({dictionary={}}={}){this.frames=[];this.transitions=[];this.dictionary=new SparseSemanticDictionary(dictionary);}
  observe(input,meta={}){const frame=resolveDimensionalFrame(input,meta);const encoded=encodeFrame(frame);const record=Object.freeze({...frame,encoded});const prev=this.frames.at(-1);if(prev)this.transitions.push(...transitionBetween(prev,record));this.frames.push(record);return record;}
  learn(){return this.dictionary.fit(this.frames.map(f=>f.encoded));}
  activations(index=this.frames.length-1){const f=this.frames[index];if(!f)throw new Error('no frame');return this.dictionary.encode(f.encoded);}
  status(){return {frames:this.frames.length,transitions:this.transitions.length,dictionary:{trained:this.dictionary.trained,atoms:this.dictionary.atoms.length}};}
}

export default SynthiaStateObserver;
