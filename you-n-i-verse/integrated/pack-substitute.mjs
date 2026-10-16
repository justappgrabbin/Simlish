// NEW, DERIVED, PLEASE CONFIRM (gap fill; composed only from existing functions).
// Gap: Alexis said the DNA base substitution (her Drive doc 182wgMEPcQxhQbj6ObACelejqQ1Psr20a, method [3] "Substituting some of the DNA nucleotides with other nucleotides") and memory completion
// are forms of the same kind of transition inside Evolution's face space. A base is four-valued (A,C,G,T); a pack is four-valued (a bigram, 2 bits, her 3-2-1 rule). No function existed that substitutes ONE pack of a six-bit face.
// This does exactly that and nothing else: it uses packsOf (her bridge.js layout via bantu.mjs) and flipLines (state-space.mjs). Which bigram stands for which base is NOT decided here (the letter->bigram mapping is open in her files).
import {packsOf} from './bantu.mjs';
import {flipLines,checkCode} from '../state-space.mjs';
// pack 0 = the first two bits of her bits() string (lines 6,5), pack 1 = next two (lines 4,3), pack 2 = last two (lines 2,1)
const LINES=[[6,5],[4,3],[2,1]];
export function packSubstitute(code,pack,bigram){
 checkCode(code);if(![0,1,2].includes(pack))throw new RangeError('pack is 0..2');if(!Number.isInteger(bigram)||bigram<0||bigram>3)throw new RangeError('bigram is 0..3');
 const cur=packsOf(code)[pack],diff=cur^bigram,flip=[];
 if(diff&2)flip.push(LINES[pack][0]);if(diff&1)flip.push(LINES[pack][1]);
 return flipLines(code,flip);
}
export const packDistance=(a,b)=>packsOf(a).map((p,i)=>{let d=p^packsOf(b)[i],n=0;while(d){n+=d&1;d>>=1}return n});
