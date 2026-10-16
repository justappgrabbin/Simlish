// ALL existing Mawangdui orderings in Alexis's files, side by side. NONE is chosen, NONE is King Wen, NONE is invented.
// Each version is a line-for-line port of a rule found in her sources (path + line cited). Output of every version: 64 positions, each a King Wen gate number
// and its Fu Xi decimal ("macro", line 1 = bit 0). Which one is hers is UNRESOLVED (see MAWANGDUI_UNRESOLVED). wheels.mjs (unchanged) reads version A only.
import {KING_WEN_TO_FUXI_DECIMAL as KW2FX,FUXI_DECIMAL_TO_KING_WEN as FX2KW} from '../vendor/kingwen.mjs';
// physical trigram values, line 1 = bit 0 (repo state-space.mjs, constants.js TRIGRAMS, hexagram_sequences.py strings parsed as ints, resonance-engine hexagrams.js)
const TV={Qian:7,Gen:4,Kan:2,Zhen:1,Kun:0,Dui:3,Li:5,Xun:6};
const FAMILY=['Qian','Gen','Kan','Zhen','Kun','Dui','Li','Xun'];          // upper-trigram family order, same in A, B(by its names), C, E
const macroOf=(upper,lower)=>(upper<<3)|lower;                              // upper = lines 4-6, lower = lines 1-3
const asVersion=macros=>({macros:Object.freeze([...macros]),kingWen:Object.freeze(macros.map(m=>FX2KW[m]))});

// A  hexagram_sequences.py mawangdui_order() (5 identical copies in zips; integrated/reference/hexagram_sequences.py lines 62-83): lower cycles through the family order with the pure doubled trigram FIRST.
const A=()=>{const o=[];for(const u of FAMILY){for(const l of [u,...FAMILY.filter(t=>t!==u)])o.push(macroOf(TV[u],TV[l]))}return o};
// C  mesh-state-space.js (zips/**/mesh-state-space.js, run-existing/.../vendor/spine/.../state-space/mesh-state-space.js lines 81-90, 108-) and state_space_core.mjs lines 40-70 mawangduiOrder():
//    sort by upper trigram family index, then by lower trigram family index (plain family order; no doubled-first). Spellings Kian/Jen/Sun are Qian/Zhen/Xun. Its own comment: intra-octet order is "the documented organizing PRINCIPLE, not a verbatim transcription" (H-MAWANGDUI-INTRA-OCTET, source: null).
const C=()=>{const o=[];for(const u of FAMILY)for(const l of FAMILY)o.push(macroOf(TV[u],TV[l]));return o};
// E  swarm-search/.../synthai-r21.22-self-cultivation-graph-integrated/suite/graphs/synthia_state_space_assembled.node.mjs lines 22 and 70-73: MW=[qian,gen,kan,zhen,kun,dui,li,xun], sort by upper rank then lower rank.
//    Same physical order as C IF its Hatcher dataset "binary" strings are line-1-first (the dataset was not located here: not verified).
const E=()=>C();
// D  resonance-engine data/sequences.js lines 16-45 (integrated/reference/sequences.js.txt) and SynthAi/computer/hover/src/state-space/mapping-providers.mjs lines 28-41:
//    MAWANGDUI_TRIGRAM_ORDER=[7,0,1,4,2,5,3,6] (its comment: Qian, Kun, Zhen, Gen, Kan, Li, Dui, Xun; "per Shaughnessy 1997 reconstruction"), rank = upperRank*8+lowerRank, upper = bits 3-5.
const D=()=>{const ord=[7,0,1,4,2,5,3,6],o=[];for(const u of ord)for(const l of ord)o.push(macroOf(u,l));return o};
// B  hexagram_sequences.js (content-4 zip; integrated/reference/hexagram_sequences.js.txt lines 114-141): executed from her file text. Its own trigram table has Gen=001, Zhen=100, Xun=011, Dui=110 (names swapped
//    against constants.js/py/resonance-engine), so read literally as line patterns its order differs from C although it uses the same family names. Both readings are kept: 'B-literal' and its King Wen labels via its own table.
// the caller passes the text of integrated/reference/hexagram_sequences.js.txt (no file access here: pure JavaScript, browser-safe)
const loadB=src=>import('data:text/javascript;base64,'+btoa(unescape(encodeURIComponent(src))));
export async function mawangduiVersions(bSource){
 if(typeof bSource!=='string')throw new TypeError('pass the text of integrated/reference/hexagram_sequences.js.txt');
 const B=await loadB(bSource),bMacros=Array.from({length:64},(_,i)=>B.MD_TO_BITS[i+1]);
 const bInternalKW=bMacros.map(b=>B.BITS_TO_KW[b]);
 const out={
  A:{id:'A',rule:'upper family order Qian,Gen,Kan,Zhen,Kun,Dui,Li,Xun; lower cycles the same family order with the doubled trigram first',source:'integrated/reference/hexagram_sequences.py mawangdui_order() lines 62-83 (5 identical copies in /workspace/zips)',usedBy:'integrated/wheels.mjs Evolution wheel',...asVersion(A())},
  C:{id:'C',rule:'upper family order Qian,Gen,Kan,Zhen,Kun,Dui,Li,Xun; lower sorted by the same family order (no doubled-first)',source:'mesh-state-space.js lines 81-90 + state_space_core.mjs lines 40-70 (many copies in /workspace/zips); claim H-MAWANGDUI-INTRA-OCTET has source:null',usedBy:'earlier Synthia state-space modules (not the integrated wheels)',...asVersion(C())},
  D:{id:'D',rule:'trigram order Qian,Kun,Zhen,Gen,Kan,Li,Dui,Xun for BOTH upper and lower (values [7,0,1,4,2,5,3,6])',source:'resonance-engine data/sequences.js lines 16-45; SynthAi computer/hover/src/state-space/mapping-providers.mjs lines 28-41',usedBy:'resonance-engine layers; SynthAi hover mapping provider',...asVersion(D())},
  E:{id:'E',rule:'same rule as C (MW=qian,gen,kan,zhen,kun,dui,li,xun)',source:'swarm-search/.../synthia_state_space_assembled.node.mjs lines 22, 70-73',usedBy:'r21.22 self-cultivation graph',note:'identical to C only if its Hatcher binary strings are line-1-first: not verified',...asVersion(E())},
  Bliteral:{id:'B-literal',rule:'her hexagram_sequences.js: family names Qian,Gen,Kan,Zhen,Kun,Dui,Li,Xun for both trigrams, but its trigram values differ (Gen=1,Zhen=4,Xun=3,Dui=6); bit values read as line patterns',source:'integrated/reference/hexagram_sequences.js.txt lines 114-141',usedBy:'nothing in integrated/',...asVersion(bMacros)},
  Binternal:{id:'B-internal-labels',rule:'same bit values as B-literal, labelled with that file\'s own King Wen table (KW_TO_BITS differs from the verified py table for 50 of 64 gates)',source:'integrated/reference/hexagram_sequences.js.txt',usedBy:'nothing in integrated/',macros:Object.freeze(bMacros),kingWen:Object.freeze(bInternalKW)}};
 return out;
}
export const differ=(x,y)=>x.macros.reduce((n,m,i)=>n+(m!==y.macros[i]?1:0),0);
export const MAWANGDUI_UNRESOLVED=Object.freeze({
 status:'UNRESOLVED',
 question:'Which Mawangdui ordering is hers (L2 Evolution wheel, spec §2.1)?',
 whatHerFilesSay:['nsnf-spec §2.1: "L2 — Evolution (Mind) | Mawangdui (matrix transitions)"; "Mawangdui generated from the verified upper-trigram-family-order principle rather than a separately transcribed table" (hexagram_sequences.py)',
  'the family order Qian,Gen,Kan,Zhen,Kun,Dui,Li,Xun for the UPPER trigram is common to A, B, C, E; D uses a different upper order',
  'the LOWER-trigram rule differs: A doubled-first, B/C/E plain family order, D its own order; C itself says its intra-octet rule is a principle with source:null'],
 evidenceNeeded:['a transcription of the silk manuscript table (the 64 hexagram names in manuscript order) from a source she accepts, to compare position by position with every version here',
  'her statement of which ordering (or which intra-octet rule) she means by "Mawangdui" in the spec, or the table she wants',
  'for E: the Hatcher dataset "binary" string orientation (line 1 first or top first)',
  'the "direct correction Aug 12 2026" cited in mesh-state-space.js (the text of that correction is not in the files found)'],
 notDone:['no version was chosen','King Wen was not substituted','no table was transcribed from memory or the web']});
