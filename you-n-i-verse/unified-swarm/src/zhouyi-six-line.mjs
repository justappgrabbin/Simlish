/** Retained separately: source-backed Zhouyi lines and Fu Xi projection.
 * This is NOT the user's human 3–2–1 hierarchy. */
import {assertSix,lines,changeLine} from '../vendor/pass05/zhouyi-state.mjs';
import {sixForGate,gateForSix} from '../vendor/pass05/king-wen-fuxi-map.mjs';
export function projectZhouyiSixLine({gate=null,six=null,provenance}={}){
  if(typeof provenance!=='string'||!provenance.trim())throw new TypeError('source provenance required');
  if(gate===null&&six===null)throw new TypeError('gate or six-line state required');
  if(gate!==null&&(!Number.isInteger(gate)||gate<1||gate>64))throw new RangeError('invalid gate');
  if(six!==null)assertSix(six);
  const value=gate===null?six:sixForGate(gate);
  if(six!==null&&six!==value)throw new RangeError('gate and six-line state conflict');
  const bits=lines(value);
  return {schema:'synthia.zhouyi.six-line.v1',gate:gate??gateForSix(value),six:value,
    bits,provenance,threeBigrams:[bits.slice(0,2),bits.slice(2,4),bits.slice(4,6)],
    twoTrigrams:[bits.slice(0,3),bits.slice(3,6)],oneHexagram:bits,
    explicitlyNot:'synthia.321.binary.v2'};
}
export function transitionZhouyiSixLine(input,{line,provenance}={}){
  const before=projectZhouyiSixLine(input);
  const after=projectZhouyiSixLine({six:changeLine(before.six,line),provenance:provenance??before.provenance});
  return {before,after,line,policy:'moving-line change in Zhouyi representation only'};
}
