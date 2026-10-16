// Thin adapter: Alexis's wheels per nsnf-spec §2.1 ("Each of the five Base layers uses its own distinct classical I Ching sequence").
//   L1 Movement  = Fu Xi (binary/mathematical changes)       L2 Evolution = Mawangdui (matrix transitions)
//   L3 Being     = Eight Palaces / Taiji pivot (spec: not yet implemented)
//   L4 Design    = AST-style structural parsing (spec: not yet implemented)
//   L5 Space     = King Wen (received sequence, output)  -> read only here: Space is never written.
// Source for the data: hexagram_sequences.py in her zips (integrated/reference/hexagram_sequences.py). That file says: King Wen binary VERIFIED from a public dataset;
// Fu Xi and Mawangdui GENERATED from structural principles, not transcribed. The Mawangdui rule below is a line-for-line port of its mawangdui_order() (Python is not run).
import {KING_WEN_TO_FUXI_DECIMAL as KW2FX,FUXI_DECIMAL_TO_KING_WEN as FX2KW} from '../vendor/kingwen.mjs';
import {nuclear} from '../state-space.mjs';

export const WHEELS_INVESTIGATING=Object.freeze([
 'Mawangdui: her zips hold more than one version. hexagram_sequences.py (used here, the one spec §2.1 names as implemented) puts the doubled trigram first inside each upper-trigram group; hexagram_sequences.js orders the lower trigram by plain family order; vendor sequences.js uses yet another upper order (Qian, Kun, Zhen, Gen, Kan, Li, Dui, Xun). Which one is hers is not settled. Not checked against the silk manuscript.',
 'hexagram_sequences.js (her zip content-4) has King Wen bit values that differ from the py for 50 of 64 gates, and its trigram bit values look swapped; it is kept as a reference copy only and is not used. The repo King Wen map agrees with the py on all 64.',
 'Being (Eight Palaces / Taiji pivot) and Design (AST-style structural parsing): spec §2.1 says not yet implemented; no order is invented, readouts are null.',
 'Ten Wings: no Ten Wings text or order is in her files except assistant-written notes; nothing is encoded.',
 'Nuclear hexagram: lines [2,3,4,3,4,5] as in state-space.mjs (Spec §36 and her hexagram_sequences.js nuclearHexagram use the same inner lines).']);

const TRI={Qian:'111',Gen:'100',Kan:'010',Zhen:'001',Kun:'000',Dui:'011',Li:'101',Xun:'110'}; // leftmost = top line (py convention)
const FAMILY=['Qian','Gen','Kan','Zhen','Kun','Dui','Li','Xun'];
const gateOfBinary=bits=>FX2KW[parseInt(bits,2)];
// Port of mawangdui_order(): returns King Wen numbers, 64 long.
export function mawangduiKingWen(){
 const out=[];
 for(const upper of FAMILY){const lowers=[upper,...FAMILY.filter(t=>t!==upper)];for(const lower of lowers)out.push(gateOfBinary(TRI[upper]+TRI[lower]))}
 return out;
}
const fuxiOrder=Array.from({length:64},(_,n)=>n);                       // macro code = Fu Xi decimal; position n holds code n
const mawangduiOrder=mawangduiKingWen().map(g=>KW2FX[g]);               // positions hold macro codes
const kingWenOrder=Array.from({length:64},(_,i)=>KW2FX[i+1]);            // position i holds the macro code of gate i+1
export const WHEELS=Object.freeze({
 Movement:Object.freeze({layer:'L1',sequence:'Fu Xi',status:'implemented',order:Object.freeze(fuxiOrder),source:'hexagram_sequences.py fu_xi_order()'}),
 Evolution:Object.freeze({layer:'L2',sequence:'Mawangdui',status:'investigating',order:Object.freeze(mawangduiOrder),source:'hexagram_sequences.py mawangdui_order() (port)',note:WHEELS_INVESTIGATING[0]}),
 Being:Object.freeze({layer:'L3',sequence:'Eight Palaces / Taiji pivot',status:'investigating',order:null,source:null,note:'spec §2.1: not yet implemented'}),
 Design:Object.freeze({layer:'L4',sequence:'AST-style structural parsing',status:'investigating',order:null,source:null,note:'spec §2.1: not yet implemented'}),
 Space:Object.freeze({layer:'L5',sequence:'King Wen',status:'implemented',order:Object.freeze(kingWenOrder),readOnly:true,source:'repo vendor/kingwen.mjs (matches hexagram_sequences.py KING_WEN_BINARY on all 64)'})});
export const TEN_WINGS=Object.freeze({status:'investigating',note:WHEELS_INVESTIGATING[3]});
export const wheelRank=(dimension,macro)=>{const w=WHEELS[dimension];if(!w||!w.order)return null;const r=w.order.indexOf(macro);return r<0?null:r};
export const nuclearOf=macro=>nuclear(macro);
// Mean normalised wheel position (0..1) of a list of macro codes on one dimension's wheel; null when that wheel is not implemented. Read only: nothing is written to the state space.
export function wheelReadout(dimension,macros){
 const w=WHEELS[dimension];if(!w)throw new RangeError('unknown dimension '+dimension);
 const base={dimension,layer:w.layer,sequence:w.sequence,status:w.status,n:macros.length};
 if(!w.order||!macros.length)return {...base,meanPosition:null,nuclearChecksum:null};
 let s=0,nu=0;for(const m of macros){s+=w.order.indexOf(m)/63;nu=(nu*31+nuclear(m))>>>0}
 return {...base,meanPosition:+(s/macros.length).toFixed(4),nuclearChecksum:nu};
}
