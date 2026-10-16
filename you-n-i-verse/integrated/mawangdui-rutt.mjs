// Mawangdui order as DOCUMENTED in Richard Rutt, Zhouyi: The Book of Changes (Routledge), a book in Alexis's Drive (id 1uiKe_YkJApeEo6s_hU65-0C_ep86IBYL).
// Source text (verbatim excerpt kept in integrated/reference/drive/rutt-excerpts.txt, "The Mawangdui order"):
//  upper trigram: "father, third son, second son, first son; mother, third daughter, second daughter, first daughter";
//  lower trigram: "father, mother; third son, third daughter; second son, second daughter; first son, first daughter; except that the first hexagram in each octet has the same trigram at the bottom as at the top."
// Rutt's Table 10 lists the whole order (Mawangdui number -> received/King Wen number); RUTT_TABLE_10 below is that table. This is evidence added NEXT TO versions A-E (mawangdui-versions.mjs, unchanged).
// It is NOT declared to be Alexis's choice: that is hers to confirm. Pure JavaScript.
import {FUXI_DECIMAL_TO_KING_WEN as FX2KW} from '../vendor/kingwen.mjs';
const TV={Qian:7,Gen:4,Kan:2,Zhen:1,Kun:0,Dui:3,Li:5,Xun:6};                        // same physical values as mawangdui-versions.mjs (line 1 = bit 0)
export const RUTT_UPPER=Object.freeze(['Qian','Gen','Kan','Zhen','Kun','Dui','Li','Xun']);  // father, third son, second son, first son; mother, third daughter, second daughter, first daughter
export const RUTT_LOWER=Object.freeze(['Qian','Kun','Gen','Dui','Kan','Li','Zhen','Xun']);  // father, mother; 3rd son, 3rd daughter; 2nd son, 2nd daughter; 1st son, 1st daughter
export function mawangduiRuttMacros(){const o=[];for(const u of RUTT_UPPER)for(const l of [u,...RUTT_LOWER.filter(t=>t!==u)])o.push((TV[u]<<3)|TV[l]);return o}
export const mawangduiRuttKingWen=()=>mawangduiRuttMacros().map(m=>FX2KW[m]);
// Rutt Table 10, column "Mawangdui - Received" (printed p.117; OCR in /workspace/zhouyi/rutt.txt lines 10658-10770), position 1..64 -> King Wen number
export const RUTT_TABLE_10=Object.freeze([1,12,33,10,6,13,25,44, 52,26,23,41,4,22,27,18, 29,5,8,39,60,63,3,48, 51,34,16,62,54,40,55,32, 2,11,15,19,7,36,24,46, 58,43,45,31,47,49,17,28, 30,14,35,56,38,64,21,50, 57,9,20,53,61,59,37,42]);
