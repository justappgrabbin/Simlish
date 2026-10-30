// FLAGGED LAYER (additive): fixes which dodecahedron face-axis is line 1..6 and which pole is yang.
// Constraint set that pins it to ONE class up to the 120 symmetries (see axis-order-probe.mjs):
//  (a) all-yang (Qian) lights a connected cap: one face + its 5 neighbours; all-yin (Kun) is the opposite cap  [INFERRED from Kian/Kun diagonal]
//  (b) the 5 ring faces carry their lines in order, consecutive lines on neighbouring faces ("written linearly")  [INFERRED]
//  (c) the centre face carries line 5, "the most important… compared to the ruler" (Govinda p.70)  [SOURCED quote, INFERRED use]
// centreLine is a parameter (INVESTIGATING); each choice 1..6 is also one class.
import {FACES,AXES,adjacent} from './dodeca-face-axes.mjs';
export function buildOrder(centreLine=5,centreFace=0){
  if(!Number.isInteger(centreLine)||centreLine<1||centreLine>6)throw new RangeError('centreLine 1..6');
  const nb=FACES.map((_,i)=>i).filter(i=>adjacent(centreFace,i));
  const ring=[nb[0]];while(ring.length<5){const last=ring.at(-1);ring.push(nb.find(f=>!ring.includes(f)&&adjacent(last,f)))}
  const lines=[1,2,3,4,5,6].filter(l=>l!==centreLine);const yangFace={[centreLine]:centreFace};lines.forEach((l,i)=>yangFace[l]=ring[i]);
  const axisOf=f=>AXES.findIndex(a=>a.includes(f));
  return Object.freeze([1,2,3,4,5,6].map(l=>{const f=yangFace[l],ax=axisOf(f);return Object.freeze({line:l,axis:ax,yangFace:f,yinFace:AXES[ax].find(g=>g!==f)})}));
}
export const ORDER=buildOrder();
export const facesForHexagram=(code,order=ORDER)=>order.map((o,i)=>((code>>i)&1)?o.yangFace:o.yinFace);
