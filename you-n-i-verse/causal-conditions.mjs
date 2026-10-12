// User-authored architecture supplied in chat, 2026-10-12. These are system
// semantics; center/biology labels are not measurements of biological function.
export const CAUSAL_SOURCE='user-consciousness-causal-spec/2026-10-12';
export const BASES=['Movement','Evolution','Being','Design','Space'];
const gateBases=[1,1,4,2,5,3,1,1,4,3,5,5,2,4,5,5,2,4,3,1,4,5,2,2,2,4,3,3,3,5,1,4,2,3,4,5,3,3,1,4,5,4,2,4,4,3,2,4,3,4,1,4,4,4,5,5,4,3,3,1,2,5,2,2];
const anchors='Creation|Landscape|Structure|Memory|Rhythm / Timing|Sex / Survival|Orientation|Seeing|Structure|Being|Illusion / Form|Hearing / Music|Memory|Progress|Rhythm|Music / Expression|Gravity|Structure / Decay|Touch|Environment|Structure|Music / Form|Light|Memory|Love|Life / Art|Survival|Survival|Matter|Fantasy / Form|Orientation|Continuity|Memory|Matter / Body|Progress|Illusion|Touch|Survival|Energy|Ego / Structure|Fantasy|Growth|Light|Smell / Structure|Manifestation|Biology / Body|Memory|Structure|Survival|Structure|Energy|Structure|Progress|Manifestation|Music / Freedom|Fantasy / Illusion|Smell|Biology|Sex|Limitation|Light|Form|Gravity|Memory'.split('|');
export const GATE_ANCHORS=Object.freeze(gateBases.map((base,i)=>Object.freeze({gate:i+1,base,anchor:anchors[i],source:CAUSAL_SOURCE})));
const resonanceTone=[3,2,6,1,6];
const allowedColors=[[1,2],[1,2],[3,4],[3,4],[4,5],[5,6]],forbiddenColor=[6,4,1,5,2,1];
export function compatibility({base,tone,color}){
 for(const [name,value,max] of [['base',base,5],['tone',tone,6],['color',color,6]])if(!Number.isInteger(value)||value<1||value>max)return {status:'invalid',reason:`${name} is outside its authored range`,source:CAUSAL_SOURCE};
 const status=allowedColors[tone-1].includes(color)?'valid':forbiddenColor[tone-1]===color?'invalid':'unresolved';
 return {status,reason:status==='unresolved'?'This Tone × Color cell was not specified':status==='invalid'?'Explicit Tone × Color conflict':null,resonant:resonanceTone[base-1]===tone,weight:resonanceTone[base-1]===tone?2:1,source:CAUSAL_SOURCE};
}
// Exact widths in arcseconds. Position is relative to an explicit gate origin;
// a zodiac longitude cannot yield a gate without the authored wheel order/origin.
export function subdivideGate(offsetArcseconds){
 if(!Number.isFinite(offsetArcseconds)||offsetArcseconds<0||offsetArcseconds>=20250)throw RangeError('Offset must lie inside one 5°37′30″ gate');
 let remainder=offsetArcseconds;const result={};
 for(const [field,width] of [['line',3375],['color',562.5],['tone',93.75],['base',18.75]]){result[field]=Math.floor(remainder/width)+1;remainder%=width;}
 return {...result,residualArcseconds:remainder,source:CAUSAL_SOURCE};
}
export function evaluateConditions(input,{chains={}}={}){
 const check=compatibility(input);if(check.status==='invalid')return {...check,admitted:false};
 if(!Number.isInteger(input.gate)||input.gate<1||input.gate>64||!Number.isInteger(input.line)||input.line<1||input.line>6)return {status:'invalid',admitted:false,reason:'Gate or line outside authored range',source:CAUSAL_SOURCE};
 const anchor=GATE_ANCHORS[input.gate-1],unresolved=[];
 if(check.status==='unresolved')unresolved.push('tone-color-cell');
 if(anchor.base!==input.base)unresolved.push('cross-base-anchor-edge');
 const chain=chains[input.base];if(!chain)unresolved.push('base-chain');
 // Compound labels are retained verbatim; no invented split or alias is used.
 else if(!chain.nodes?.includes(anchor.anchor))unresolved.push('anchor-node');
 if(!['Vertical','Horizontal','Diagonal'].includes(input.axis))unresolved.push('axis');
 if(!input.zodiac)unresolved.push('zodiac');if(!input.house)unresolved.push('house');
 if(input.line===3)unresolved.push('line-3-loop-policy');
 if(input.line===4)unresolved.push('external-node-selection');
 if(input.line===5)unresolved.push('universalization-map');
 const giftByResonance=check.resonant&&[5,6].includes(input.line);
 // Other gift activation needs the missing Color Binary × Axis lookup.
 const gift={status:giftByResonance?'active':'unresolved',reason:giftByResonance?'tone-resonance-with-line-5-or-6':'color-binary-axis-map-not-supplied'};
 return {status:unresolved.length?'unresolved':'conditions-valid',admitted:false,anchor:structuredClone(anchor),compatibility:check,gift,unresolved,source:CAUSAL_SOURCE};
}
