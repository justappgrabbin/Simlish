// Probe: Hatcher's Scales (pp.444-448) as constraints on the axis order. Tests all 720 orders x 64 yang-pole choices.
import {FACES,AXES,adjacent,litFaces} from './dodeca-face-axes.mjs';
import {nuclear,gateToMacro} from './state-space.mjs';
export const SOVEREIGN_GATES=Object.freeze([24,19,11,34,43,1,44,33,12,20,23,2]); // Hatcher p.447 (Zhouyi 19.0); listed by moon at pp.453ff
export const SOVEREIGN=Object.freeze(SOVEREIGN_GATES.map(gateToMacro));
const orders=[];const pm=(a,r)=>{if(!a.length){orders.push(r);return}a.forEach((x,i)=>pm(a.filter((_,k)=>k!==i),[...r,x]))};pm([0,1,2,3,4,5],[]);
const isCap=fs=>fs.some(f=>fs.filter(g=>adjacent(f,g)).length===5);
// yang-pole choice y flips which face of each axis counts as yang: code c lights faces of (c XOR y) under litFaces convention
export function capCodes(order,y){const out=[];for(let c=0;c<64;c++)if(isCap(litFaces(c^y,order)))out.push(c);return out}
export function survey(){const sov=new Set(SOVEREIGN);let n=0,best=0;const hits=[];
  for(const o of orders)for(let y=0;y<64;y++){const caps=capCodes(o,y);const m=caps.filter(c=>sov.has(c)).length;best=Math.max(best,m);if(m===12){n++;hits.push({order:o,y})}}
  return {setups:46080,sovereignEqualsCaps:n,maxSovereignAmongCaps:best,hits}}
// nuclear (Hu Gua) images: 16 hexagrams
export const HU=Object.freeze([...new Set(Array.from({length:64},(_,c)=>nuclear(c)))].sort((a,b)=>a-b));
if(import.meta.url===`file://${process.argv[1]}`){const s=survey();console.log({...s,hits:s.hits.length});console.log('Hu Gua count',HU.length)}
