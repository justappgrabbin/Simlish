// FLAGGED LAYER (additive): axis order from Hatcher's Scale of Twelve (p.447, Zhouyi 19.0: the Sovereign Gua of the Twelve Moons).
// Probe result (hatcher-scales-probe.mjs + this file's survey): the 12 caps can NEVER equal the 12 Sovereign Gua (0 of 46,080); at most 6 are caps.
// The setups reaching that maximum (6 caps) with Qian as a cap and the ring read linearly = exactly 240 = 2 classes: centre line 3 or centre line 4.
// In both, the 12 Sovereign Gua light only two shapes: 6 caps + 6 of the "222444" shape. Choice between 3 and 4 is INVESTIGATING (they mirror under line reversal).
import {buildOrder,facesForHexagram} from './axis-order.mjs';
import {litFaces,adjacent} from './dodeca-face-axes.mjs';
import {SOVEREIGN} from './hatcher-scales-probe.mjs';
export const SOVEREIGN_CENTRE_CANDIDATES=Object.freeze([3,4]);
export const sovereignOrders=()=>SOVEREIGN_CENTRE_CANDIDATES.map(c=>({centreLine:c,order:buildOrder(c)}));
const centreOf=fs=>fs.findIndex(f=>fs.filter(g=>adjacent(f,g)).length===5);
export function survey(){const orders=[];const pm=(a,r)=>{if(!a.length){orders.push(r);return}a.forEach((x,i)=>pm(a.filter((_,k)=>k!==i),[...r,x]))};pm([0,1,2,3,4,5],[]);
  let max=0,atMax=0,qcap=0,ring=0;const centre={};
  for(const o of orders)for(let y=0;y<64;y++){const k=SOVEREIGN.filter(c=>centreOf(litFaces(c^y,o))>=0).length;if(k>max){max=k;atMax=0;qcap=0;ring=0;for(const x in centre)delete centre[x]}if(k<max)continue;atMax++;
    const Q=litFaces(63^y,o),c=centreOf(Q);if(c<0)continue;qcap++;const r=[0,1,2,3,4,5].filter(l=>l!==c);if(r.every((l,i)=>i===0||adjacent(Q[r[i-1]],Q[l]))){ring++;centre[c+1]=(centre[c+1]||0)+1}}
  return {setups:46080,maxSovereignCaps:max,atMax,withQianCap:qcap,withLinearRing:ring,centreLines:centre}}
export const sovereignShapes=order=>SOVEREIGN.map(h=>{const fs=facesForHexagram(h,order);return fs.map(x=>fs.filter(y=>adjacent(x,y)).length).sort().join('')});
if(import.meta.url===`file://${process.argv[1]}`)console.log(survey())
