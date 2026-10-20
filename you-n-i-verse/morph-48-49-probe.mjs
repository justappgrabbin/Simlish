// Side-by-side probe, 48 vs 49 (Alexis: "It could be 49... science says 48, the book says 49; somebody's wrong"). Picks no winner.
// 49 = yarrow stalks used: Hatcher pp.46-47 ("Begin with 50 yarrow stalks... Set one aside"; "beginning with forty-nine stalks each time").
import {SLICES} from './morph-state-space.mjs';
const W={'69,120 slices':SLICES,'1,080 slices/gate':1080,'360°':360,'21,600′':21600,'64 gates':64,'12 caps':12,'120 dodeca symmetries':120,'2,304 colors':2304,'13,824 tones':13824,'384 lines':384};
export function divides(n){return Object.fromEntries(Object.entries(W).map(([k,v])=>[k,Number.isInteger(v/n)?v/n:'no']))}
// Exact yarrow procedure with `n` working stalks: 3 changes; each: split L+R (every split equally likely), take 1 from R, remove (L mod 4 || 4) and ((R-1) mod 4 || 4).
export function yarrow(n){let dist=new Map([[n,1]]);
  for(let ch=0;ch<3;ch++){const next=new Map();for(const [s,p] of dist){for(let L=1;L<s;L++){const R=s-L;if(R<1)continue;const rl=L%4||4,rr=(R-1)%4||4,left=s-1-rl-rr;next.set(left,(next.get(left)||0)+p/(s-1))}}dist=next}
  const out={};for(const [s,p] of dist){const k=s%4===0?String(s/4):`${s} (not ×4)`;out[k]=(out[k]||0)+p}return out}
export function probe48vs49(){return [48,49].map(n=>({n,...divides(n),yarrow:Object.entries(yarrow(n)).map(([k,p])=>`${k}:${(p*16).toFixed(3)}/16`).join(' ')}))}
if(import.meta.url===`file://${process.argv[1]}`)console.log(probe48vs49())
