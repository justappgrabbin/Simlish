/** SWARM-OS additive positioned state-space assembly.
 * Joins three ALREADY EXISTING user components: verified finite address hierarchy,
 * Zhouyi six-line transformations + reversible perspectives, and the existing
 * trimodal expression renderer. Does NOT replace either source or invent a new ontology.
 */
import {encode,decode,bounds,UNITS,validateContext} from './verified-foundation-address.mjs';
import {assertSix,lines,trigrams,changeLine,complement,invert,transform} from './zhouyi-state.mjs';
import {perspective,applyPerspective,mappedLine,mappedMask} from './zhouyi-structural-isomorphism.mjs';
import {expressPrimitive} from './tri-modal-expression.mjs';
import {sixForGate,gateForSix,unicodeForGate} from './king-wen-fuxi-map.mjs';

const dimensions=Object.freeze(['Movement','Evolution','Being','Design','Space']);
const axes=Object.freeze(['Vertical','Horizontal','Diagonal']);
const dimensionOctave=Object.freeze({Movement:2,Evolution:3,Being:4,Design:5,Space:6});
const dimensionLayer=Object.freeze({Movement:'stroke',Evolution:'fill',Being:'glow',Design:'frame',Space:'ground'});
const SCALEGRAM=Object.freeze([0,2,5,7,9,10]);
const TIMBRES=Object.freeze(['sine','triangle','square','sawtooth','pulse','noise']);
const DURATIONS=Object.freeze([0.25,0.5,1,1.5,2,4]);
const VELOCITIES=Object.freeze([0.4,0.55,0.7,0.85,1]);
const clone=v=>structuredClone(v);
const fixed=v=>Object.freeze(clone(v));
const positiveInt=(v,min,max,k)=>{if(!Number.isInteger(v)||v<min||v>max)throw RangeError(`${k} must be ${min}..${max}`);return v;};

/** Coordinates that came from a real source/user request are retained as distinct
 * fields. We do NOT derive zodiac/house from sequential gate order; those use
 * different providers and conventions elsewhere in Synthia. */
export function validatePosition(a){
 if(!a||typeof a!=='object')throw TypeError('position object required');
 const {gate,line,color,tone,base,dimension,planet,zodiac,house,axis}=a;
 const index=encode({gate,line,color,tone,base});
 if(!dimensions.includes(dimension))throw RangeError('dimension must name one of the five dimensions');
 validateContext({dimension,planet,zodiac,house});
 positiveInt(planet,1,13,'planet');positiveInt(zodiac,1,12,'zodiac');positiveInt(house,1,12,'house');
 if(!axes.includes(axis))throw RangeError('axis must be Vertical, Horizontal or Diagonal');
 return index;
}
export function resolvePosition(a){
 const index=validatePosition(a);
 const minimum=bounds(index).startCentiArcsec;
 const maximum=bounds(index).endCentiArcsec;
 const six=sixForGate(a.gate);
 return fixed({position:clone(a),microIndex:index,angularBin:{startCentiArcsec:minimum,endCentiArcsec:maximum,convention:'sequential 360-degree address bins; NOT an ephemeris or validated mandala offset'},
  hexagram:{gate:a.gate,glyph:unicodeForGate(a.gate),six,bottomFirst:lines(six),trigrams:trigrams(six)},
  cardinality:{microPositions:UNITS,hexagramStates:64,singleLineTransitions:384,allMasksPerState:64},
  provenance:{micro:'swarm-state-space-foundation/state-space.mjs',six:'semantic-dna-swarm (1).html KING_WEN_TO_FUXI_DECIMAL',transition:'zhouyi-state.mjs'}});
}

/** Existing qualitative rules from user's semantic-dna-swarm (1).html.
 * This donor-defined expression layer has not been independently validated as
 * physically universal; it is intentionally exposed AS DONOR-DEFINED, not proof. */
export function expressPosition(a){
 const position=resolvePosition(a);
 const {gate,line,color,tone,base,dimension,house,zodiac}=a;
 const arcSecond=position.angularBin.startCentiArcsec/100;
 const hue=(arcSecond%1296000+1296000)%1296000/1296000*360;
 const saturation=30+line*10,lightness=25+tone*7;
 const pitchClass=zodiac-1,octave=dimensionOctave[dimension];
 const cents=((arcSecond%108000+108000)%108000)/108000*100;
 const scaleDegree=SCALEGRAM[line-1];
 const frequency=432*Math.pow(2,(octave*12+pitchClass+scaleDegree+cents/100-57)/12);
 // The original trimodal kernel already defines a reproducible computational
 // polygon. Reuse it; do not present this provisional shape as a discovered law.
 const existing=expressPrimitive({address:{hexagram6:position.hexagram.six,ontology:{dimension}},value:position.microIndex,action:'address',generation:0});
 return fixed({position,
  sound:{frequencyHz:frequency,cents,octave,pitchClass,scaleDegree,timbre:TIMBRES[color-1],durationSeconds:DURATIONS[tone-1],velocity:VELOCITIES[base-1],rhythmicSlot:house-1,source:'semantic-dna-swarm (1).html soundFor; rhythmicSlot is the source formula, NOT proof of a twelve-house rhythm'},
  color:{hue,saturation,lightness,layer:dimensionLayer[dimension],source:'semantic-dna-swarm (1).html colorFor'},
  shape:{...existing.shape,source:'existing tri-modal-expression.mjs expressPrimitive (computational shape projection)'},
  validation:'source-defined projections; cross-domain equivalence not yet established'});
}

export function changeAtPosition(a,kind,argument){
 const original=resolvePosition(a); const from=original.hexagram.six;
 let to;
 if(kind==='line')to=changeLine(from,argument);
 else if(kind==='mask')to=transform(from,argument);
 else if(kind==='complement')to=complement(from);
 else if(kind==='invert')to=invert(from);
 else throw RangeError('transition kind must be line, mask, complement or invert');
 // Preserve line/color/tone/base until a separate positional transition rule
 // explicitly moves them. That is a concrete, reversible operation, not a
 // silent reassignment of unrelated coordinates.
 const next={...a,gate:gateForSix(to)};
 return fixed({kind,argument:argument??null,from:original,to:resolvePosition(next),movingMask:from^to});
}
export function viewPosition(a,p){
 const original=resolvePosition(a);
 const projectedSix=applyPerspective(p,original.hexagram.six);
 const observed={...a,gate:gateForSix(projectedSix),line:mappedLine(p,a.line)};
 const output=resolvePosition(observed);
 return fixed({identity:original.position,view:output,perspective:{permutation:[...p.permutation],polarity:p.polarity},
  note:'reversible positional view, not change of original identity'});
}
export {perspective,mappedMask};
/** Local append-only positional ledger. State changes are never overwritten in history. */
export class PositionedStateSpace {
 #position;#journal=[];
 constructor(initial){this.#position=clone(resolvePosition(initial).position);this.#journal.push(fixed({sequence:0,action:'origin',from:null,to:clone(this.#position)}));}
 current(){return resolvePosition(this.#position);}
 history(){return clone(this.#journal);}
 change(kind,argument){const event=changeAtPosition(this.#position,kind,argument);const from=clone(this.#position);this.#position=clone(event.to.position);this.#journal.push(fixed({sequence:this.#journal.length,action:kind,argument:argument??null,from,to:clone(this.#position),mask:event.movingMask}));return event;}
 relocate(position){resolvePosition(position);const from=clone(this.#position);this.#position=clone(position);this.#journal.push(fixed({sequence:this.#journal.length,action:'relocate',from,to:clone(position)}));return this.current();}
 view(p){return viewPosition(this.#position,p);}
 express(){return expressPosition(this.#position);}
 checkpoint(){return fixed({schema:'synthia-positioned-space-v1',origin:this.#journal[0].to,current:this.#position,journal:this.history()});}
 static restore(cp){if(cp?.schema!=='synthia-positioned-space-v1'||!Array.isArray(cp.journal)||cp.journal.length===0)throw Error('invalid checkpoint');
  const x=new PositionedStateSpace(cp.origin);
  if(JSON.stringify(x.#journal[0])!==JSON.stringify(cp.journal[0]))throw Error('origin mismatch');
  for(const event of cp.journal.slice(1)){
   if(event.sequence!==x.#journal.length||JSON.stringify(event.from)!==JSON.stringify(x.#position))throw Error('history continuity mismatch');
   if(event.action==='relocate')x.relocate(event.to);
   else x.change(event.action,event.argument);
   if(JSON.stringify(x.#journal.at(-1))!==JSON.stringify(event))throw Error('history proof mismatch');
  }
  if(JSON.stringify(x.#position)!==JSON.stringify(cp.current))throw Error('checkpoint/current mismatch');
  return x;
 }
}
