// FLAGGED HYPOTHESIS (additive): dodecahedron 12 faces = 6 opposite pairs = 6 axes; hexagram line i = axis i; yin/yang picks which face of the pair is lit.
// Face centres of the dodecahedron = vertices of the icosahedron (0,±1,±φ) cyclic. Axis order (which axis is line 1..6) is a parameter: INVESTIGATING.
import {complement,reverseLines,gateToMacro} from './state-space.mjs';
const P=(1+Math.sqrt(5))/2;
const raw=[];for(const s1 of[1,-1])for(const s2 of[1,-1]){raw.push([0,s1,s2*P],[s1,s2*P,0],[s2*P,0,s1])}
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
export const FACES=raw; // 12 face normals
export const AXES=(()=>{const used=new Set(),ax=[];FACES.forEach((f,i)=>{if(used.has(i))return;const j=FACES.findIndex(g=>g.every((x,k)=>Math.abs(x+f[k])<1e-9));used.add(i);used.add(j);ax.push([i,j])});return ax})();
const n2=dot(FACES[0],FACES[0]);
export const adjacent=(i,j)=>i!==j&&Math.abs(dot(FACES[i],FACES[j])/n2-1/Math.sqrt(5))<1e-9; // neighbouring faces at 63.43°
export function litFaces(code,order=[0,1,2,3,4,5]){return order.map((ax,line)=>AXES[ax][(code>>line)&1])}
export function connected(fs){const s=new Set(fs),seen=new Set([fs[0]]),q=[fs[0]];while(q.length){const x=q.pop();for(const y of s)if(!seen.has(y)&&adjacent(x,y)){seen.add(y);q.push(y)}}return seen.size===s.size}
const shape=fs=>{const deg=fs.map(x=>fs.filter(y=>adjacent(x,y)).length).sort();const edges=deg.reduce((a,b)=>a+b,0)/2;return `${edges} edges, degrees ${deg.join('')}`};
export function analyse(order=[0,1,2,3,4,5]){
  const out={sets:64,complementIsAntipode:0,connected:0,caps:0,shapes:{},complementSameShape:0};
  for(let c=0;c<64;c++){const L=litFaces(c,order),C=litFaces(complement(c),order);
    if(L.every((f,i)=>AXES[order[i]].includes(f)&&C[i]!==f))out.complementIsAntipode++;
    if(connected(L))out.connected++;
    if(L.some(f=>L.filter(g=>adjacent(f,g)).length===5))out.caps++;
    const sh=shape(L);out.shapes[sh]=(out.shapes[sh]||0)+1;if(shape(C)===sh)out.complementSameShape++;}
  return out;
}
// King Wen pairs (2k-1,2k): inversion (reverse lines) or complement (cuo)?
export function kingWenPairs(){let inv=0,cmp=0,both=0,none=0;const cmpPairs=[];for(let k=1;k<=63;k+=2){const a=gateToMacro(k),b=gateToMacro(k+1);const i=reverseLines(a)===b,c=complement(a)===b;if(i&&c)both++;else if(i)inv++;else if(c){cmp++;cmpPairs.push([k,k+1])}else none++}return {inversionOnly:inv,complementOnly:cmp,complementPairs:cmpPairs,both,none}}
// Any two axes meet at the same angle -> geometry does not prefer an order.
export function axisAngles(){const s=new Set();for(let i=0;i<6;i++)for(let j=i+1;j<6;j++){const a=FACES[AXES[i][0]],b=FACES[AXES[j][0]];s.add((Math.acos(Math.abs(dot(a,b))/n2)*180/Math.PI).toFixed(4))}return [...s]}
if(import.meta.url===`file://${process.argv[1]}`){console.log(AXES.length,'axes');console.log(analyse());console.log(kingWenPairs());console.log('axis-axis angles',axisAngles());
  const perms=[];const p=(a,r)=>{if(!a.length){perms.push(r);return}a.forEach((x,i)=>p(a.filter((_,k)=>k!==i),[...r,x]))};p([0,1,2,3,4,5],[]);
  const sig=new Set(perms.map(o=>{const a=analyse(o);a.shapes=Object.entries(a.shapes).sort();return JSON.stringify(a)}));console.log('distinct analyses over all 720 orders:',sig.size)}
