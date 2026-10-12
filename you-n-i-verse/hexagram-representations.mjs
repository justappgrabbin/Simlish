import {hexagramEncoding} from './hexagram-encodings.mjs';
import {evaluateConditions} from './causal-conditions.mjs';
// Both projections belong to the same King Wen gate. Neither representation
// substitutes for the other or resolves absent resident-address fields.
export function representHexagram(row,conditions,options={}){
 if(row.gate!==conditions.gate)throw Error('Embodied and action representations must share gate identity');
 return {gate:row.gate,embodied:hexagramEncoding(row),action:{representation:'action',...evaluateConditions(conditions,options)},admitted:false};
}
