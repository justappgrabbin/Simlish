import { GATES } from './gate-matrix.mjs';

// These are the user's recorded correspondences, not validated physical laws.
export const DIMENSIONS = Object.freeze({
  MOVEMENT: Object.freeze({ question:'WHEN', human:'Individuality', nature:'Energy', axis:'VERTICAL', chain:['Movement','Energy','Creation','Seeing','Landscape','Environment'] }),
  EVOLUTION: Object.freeze({ question:'WHAT', human:'Mind', nature:'Gravity', axis:'HORIZONTAL', chain:['Evolution','Gravity','Memory','Taste','Love','Light'] }),
  BEING: Object.freeze({ question:'WHERE', human:'Body', nature:'Matter', axis:'VERTICAL', chain:['Being','Matter','Touch','Sex','Survival'] }),
  DESIGN: Object.freeze({ question:'WHY', human:'Ego', nature:'Structure', axis:'DIAGONAL', chain:['Design','Structure','Progress','Smell','Life','Art'] }),
  SPACE: Object.freeze({ question:'WHO', human:'Personality', nature:'Form', axis:'EMERGENT', chain:['Space','Form','Illusion','Hearing','Music','Freedom'] })
});
export const PERSPECTIVES = Object.freeze(['MOVEMENT','EVOLUTION','BEING','DESIGN']);
export const SHADOWS = Object.freeze({MOVEMENT:'STILLNESS',EVOLUTION:'DEVOLUTION',BEING:'NON-BEING',DESIGN:'CHAOS'});
export const PRIMITIVE_LETTERS = Object.freeze({I:'MOVEMENT',A:'EVOLUTION',E:'BEING',U:'DESIGN',O:'SPACE',M:'BEING'});
export const ADDRESS_FIELDS = Object.freeze(['planetary','dimension','gate','line','color','tone','base','degree','minute','second','arc','zodiac','house']);
export const TONE_COLORS = Object.freeze({1:[1,2],2:[1,2],3:[3,4],4:[3,4],5:[4,5],6:[5,6]});
export const INCOMPATIBLE = Object.freeze({1:[6],2:[4],3:[1],4:[5],5:[2],6:[1]});
export const RESONANT_TONES = Object.freeze({1:3,2:2,3:6,4:1,5:6});
export const COLORS = Object.freeze(['Fear','Hope','Desire','Need','Guilt','Innocence']);
export const TONES = Object.freeze(['Security','Uncertainty','Action','Meditation','Judgement','Acceptance']);
export const AXES = Object.freeze({VERTICAL:['MOVEMENT','BEING'],HORIZONTAL:['EVOLUTION','SPACE'],DIAGONAL:['DESIGN']});

export class CausalGraphError extends Error { constructor(code, detail={}) { super(`${code}: ${JSON.stringify(detail)}`); this.name='CausalGraphError'; this.code=code; this.detail=detail; } }
const validInt=(v,lo,hi,key,context)=>{
  if(!Number.isInteger(v)||v<lo||v>hi) throw new CausalGraphError('ADDRESS_RANGE_ERROR',{field:key,value:v,range:[lo,hi],context});
};
export function gateRow(gate){ validInt(gate,1,64,'gate','gateRow'); return GATES[gate-1]; }
export function validateAddress(a){
  if(!a||typeof a!=='object')throw new CausalGraphError('ADDRESS_REQUIRED',{received:a});
  const range={gate:[1,64],line:[1,6],color:[1,6],tone:[1,6],base:[1,5],degree:[0,29],minute:[0,59],second:[0,59]};
  for(const [name,[lo,hi]] of Object.entries(range)) if(a[name]!=null) validInt(a[name],lo,hi,name,a);
  if(a.arc!=null && (!Number.isFinite(a.arc)||a.arc<0))throw new CausalGraphError('ADDRESS_ARC_UNRESOLVED',{arc:a.arc});
  if(a.dimension!=null&&!Object.hasOwn(DIMENSIONS,String(a.dimension).toUpperCase()))throw new CausalGraphError('UNKNOWN_DIMENSION',{dimension:a.dimension});
  return Object.freeze(Object.fromEntries(ADDRESS_FIELDS.map(k=>[k,a[k]??null])));
}
export function normalizeAddress(input={}){
  const address=validateAddress(input);
  return {address,status:ADDRESS_FIELDS.every(k=>address[k]!==null)?'fully-provided':'partial',unresolved:ADDRESS_FIELDS.filter(k=>address[k]===null),provider:'caller-provided-not-inferred'};
}
function anchorPath(chain, anchor){
  const exact=chain.indexOf(anchor); if(exact>=0) return {path:chain.slice(exact), linkStatus:'existing-chain-edge'};
  const matches=anchor.split(/\s*\/\s*/).map(s=>s.trim());
  const first=matches.find(s=>chain.includes(s));
  if(first) return {path:[anchor,...chain.slice(chain.indexOf(first)+1)],linkStatus:'alias-anchor-expanded'};
  // Anchor remains the exact source node. An explicit interpretation bridge connects it to its base chain.
  return {path:[anchor,...chain],linkStatus:'bridge-proposed-not-source-attested'};
}
export function traverse({gate,line,color,tone,base,axis,zodiac,house}={}){
  const row=gateRow(gate);
  for(const [k,v,lo,hi] of [['line',line,1,6],['color',color,1,6],['tone',tone,1,6]]) validInt(v,lo,hi,k,{gate,line,color,tone,base});
  if(base!==undefined){validInt(base,1,5,'base',{gate,base});if(base!==row.base)throw new CausalGraphError('GATE_BASE_CONFLICT',{gate,expected:row.base,received:base});}
  if(!TONE_COLORS[tone].includes(color)) throw new CausalGraphError('COLOR_TONE_INCOMPATIBLE',{gate,line,color,tone,allowed:TONE_COLORS[tone]});
  if(axis!=null&&!Object.hasOwn(AXES,axis.toUpperCase()))throw new CausalGraphError('AXIS_UNKNOWN',{axis});
  if(zodiac!=null&&zodiac.toLowerCase()!==row.zodiac.toLowerCase()) throw new CausalGraphError('ZODIAC_CONFLICT',{given:zodiac,source:row.zodiac});
  if(house!=null&&house!==row.house)throw new CausalGraphError('HOUSE_CONFLICT',{given:house,source:row.house});
  const chain=DIMENSIONS[row.dimension].chain;
  const anchor=anchorPath(chain,row.anchor);let path;
  switch(line){
    case 1: path=anchor.path; break;
    case 2: path=[...chain]; break;
    case 3: path=[anchor.path[0],anchor.path[2]??anchor.path[0],anchor.path[1]??anchor.path[0],anchor.path[2]??anchor.path[0]]; break;
    case 4: path=[anchor.path[0],...anchor.path.slice(-2)].filter((n,i,a)=>i===0||n!==a[i-1]); break;
    case 5: path=anchor.path; break; // Universalizing interpretation is an overlay, not invented graph words.
    case 6: path=[anchor.path[0],anchor.path.at(-1)]; break;
  }
  const resonance=RESONANT_TONES[row.base]===tone;
  const result={gate,line,color,tone,base:row.base,axis:axis??DIMENSIONS[row.dimension].axis,zodiac:row.zodiac,house:row.house,
    dimension:row.dimension,name:row.name,archetype:row.archetype,anchor:row.anchor,linkStatus:anchor.linkStatus,
    path,modifiers:{color:COLORS[color-1],tone:TONES[tone-1],resonanceWeight:resonance?2:1,lineMode:['Foundation','Natural','Experimentation','Externalization','Projection','Transition'][line-1]},
    provenance:{source:'causal-graph-master-source.txt',type:'symbolic-interpretation',axisProof:axis===undefined?'source-default':'input-validated',siding:'undetermined: requires explicit axis/color-binary alignment rule'}
  };
  return Object.freeze(result);
}
export const GATE_COUNT=GATES.length;
