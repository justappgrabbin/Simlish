/** Structural equivalence pass 03: automorphisms of the six-dimensional Boolean cube.
 * Coordinates are bottom-first. Perspective is a reversible coordinate permutation
 * followed by an XOR polarity mask. This is a mathematical mapping, not a claim
 * that every qualitative or biological relation is preserved.
 */
import {assertSix,changeLine,complement,transform,lines,fromLines} from './zhouyi-state.mjs';
const VALID=[0,1,2,3,4,5];
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
export function perspective(permutation=VALID,polarity=0){
 if(!Array.isArray(permutation)||permutation.length!==6||!same([...permutation].sort((a,b)=>a-b),VALID))throw new TypeError('permutation must contain each coordinate 0..5 exactly once');
 assertSix(polarity);return Object.freeze({permutation:Object.freeze([...permutation]),polarity});
}
export const IDENTITY=perspective();
export function applyPerspective(p,state){assertSix(state);const bits=lines(state);return fromLines(p.permutation.map((source,out)=>bits[source]^((p.polarity>>out)&1)));}
export function inversePerspective(p){const inv=Array(6);for(let out=0;out<6;out++)inv[p.permutation[out]]=out;
 const polarity=fromLines(inv.map(out=>((p.polarity>>out)&1)));
 return perspective(inv,polarity);
}
/** compose(a,b) means a after b */
export function composePerspective(a,b){const permutation=a.permutation.map(i=>b.permutation[i]);const polarity=fromLines(a.permutation.map((i,out)=>((b.polarity>>i)&1)^((a.polarity>>out)&1)));return perspective(permutation,polarity);}
export function mappedLine(p,bottomBasedLine){if(!Number.isInteger(bottomBasedLine)||bottomBasedLine<1||bottomBasedLine>6)throw RangeError('line 1..6');return p.permutation.indexOf(bottomBasedLine-1)+1;}
export function mappedMask(p,mask){assertSix(mask);return fromLines(p.permutation.map(i=>(mask>>i)&1));}
export function hamming(a,b){assertSix(a);assertSix(b);let n=a^b,count=0;while(n){count+=n&1;n>>=1;}return count;}
export function edge(a,b){return hamming(a,b)===1;}
export function verifyPerspective(p){const issues=[];const inverse=inversePerspective(p);
 for(let x=0;x<64;x++){
  const y=applyPerspective(p,x);
  if(applyPerspective(inverse,y)!==x)issues.push({kind:'roundtrip',x});
  if(applyPerspective(p,complement(x))!==complement(y))issues.push({kind:'complement',x});
  for(let line=1;line<=6;line++){
   const mapped=applyPerspective(p,changeLine(x,line));
   if(mapped!==changeLine(y,mappedLine(p,line)))issues.push({kind:'line',x,line});
  }
  for(let mask=0;mask<64;mask++){
   if(applyPerspective(p,transform(x,mask))!==transform(y,mappedMask(p,mask)))issues.push({kind:'mask',x,mask});
  }
 }
 return {states:64,linesChecked:384,masksChecked:4096,bijective:true,complementPreserved:issues.every(e=>e.kind!=='complement'),hammingPreserved:!issues.length,issues};
}
/** Returns both perspectives, and the preserved relation witness. */
export function compareStates(p,a,b){const aa=applyPerspective(p,a),bb=applyPerspective(p,b);return {source:{a,b,hamming:hamming(a,b),edge:edge(a,b)},target:{a:aa,b:bb,hamming:hamming(aa,bb),edge:edge(aa,bb)},preserved:hamming(a,b)===hamming(aa,bb)};}
/** A useful counterexample: order-dependent interpretations are not automatically preserved. */
export function verifyTrigramIdentity(p){let preserved=0;for(let x=0;x<64;x++){
 const y=applyPerspective(p,x);
 if((x&7)===(y&7)&&((x>>3)&7)===((y>>3)&7))preserved++;
 }return {preserved,total:64,allPreserved:preserved===64};}
