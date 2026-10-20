// ONE morphological state space: joins (imports, does not replace) state-space.mjs (5 dimensions x 64, nesting),
// integrated/chart-engine.mjs + vendor humanDesignResolution (13 bodies, 12 signs, 302° wheel, gate/line/color/tone/base),
// and pass05/ (Pass-05 WH key + append-only relation ledger, copied unchanged from SWARM-OS-source-locked-relational-state-space-pass-05.zip, Drive 1dSi2UxxuveG76JESiNkamyReg5ygVziT).
// Pure JS. All arithmetic in integer hundredths of an arcsecond ("centi"): 360° = 129,600,000; one base slice = 1,875; 69,120 slices.
import {DIMENSIONS,PRIMITIVE_DIMENSIONS,EMERGENT_DIMENSION,StateSpace,gateToMacro,macroToGate} from './state-space.mjs';
import {SIGNS,BODIES} from './integrated/chart-engine.mjs';
import {humanDesignResolution} from './vendor/isohuman/GeonatalCalculator.browser.mjs';
import {fixedDimensionalKey,dimensionalIdentity,PositionalRelationLedger} from './pass05/spec-locked-dimensional-relations.mjs';

export const CIRCLE=129_600_000, SIGN=10_800_000, DEG=360_000, MIN=6_000, SEC=100;
export const GATE=CIRCLE/64, LINE=GATE/6, COLOR=LINE/6, TONE=COLOR/6, BASE=TONE/5; // 2,025,000 / 337,500 / 56,250 / 9,375 / 1,875
export const SLICES=CIRCLE/BASE; // 69,120
export const FIELDS=Object.freeze(['planet','dimension','gate','line','color','tone','base','degree','minute','second','arc','zodiac','house']);

// INVESTIGATING parameters: each is explicit, none is decided here. Defaults keep what existing code already does, flagged.
export const INVESTIGATING=Object.freeze({
  wheelStartDegrees:{value:302,candidates:[302,0,272],note:'existing chart-engine/vendor value (gate 41 at 2° Aquarius, matched Free Human Design); not decided by Alexis'},
  split:{value:12,candidates:[12,6,13],note:'state-space.mjs allows 12 or 6; 12+1 is her phrase, rule unknown'},
  arcReading:{value:Object.fromEntries(DIMENSIONS.map(d=>[d,'centi-second'])),candidates:['centi-second','gate-percent'],note:'arc 0–99 read as two linear digits (hundredths of a second) in every dimension by default; she said it reads that way "in at least one and not all" states — which ones is undecided'},
  houseRule:{value:'given',candidates:['given','whole-sign','equal-from-ascendant'],note:'house is accepted as given and range-checked only'},
  houseCount:{value:12,candidates:[8,9,10,11,12],note:'"8 to 12 houses" unclear'},
});

const int=(v,lo,hi,n)=>{if(!Number.isInteger(v)||v<lo||v>hi)throw new RangeError(`${n} must be an integer ${lo}..${hi}, got ${v}`);return v};
const mod=(a,m)=>((a%m)+m)%m;

export function makeWheel(startDegrees=INVESTIGATING.wheelStartDegrees.value){
  const OFF=Math.round(startDegrees*DEG);
  // gate order read from her vendored engine at gate centres (not retyped)
  // gate sequence read from her vendored engine (starts at 302°, gate 41 first); a candidate start moves the whole sequence so gate 41 begins at startDegrees.
  const ORDER=Array.from({length:64},(_,i)=>humanDesignResolution((302*DEG+i*GATE+GATE/2)/DEG).gate);
  const IDX=Object.fromEntries(ORDER.map((g,i)=>[g,i]));
  const sliceOf=P=>Math.floor(mod(P-OFF,CIRCLE)/BASE);
  const hd=s=>{const g=Math.floor(s/1080),r=s%1080;return {gate:ORDER[g],line:Math.floor(r/180)+1,color:Math.floor(r%180/30)+1,tone:Math.floor(r%30/5)+1,base:r%5+1}};
  const sliceIndex=({gate,line,color,tone,base})=>{const g=IDX[gate];if(g===undefined)throw new RangeError('gate 1..64');
    int(line,1,6,'line');int(color,1,6,'color');int(tone,1,6,'tone');int(base,1,5,'base');return g*1080+(line-1)*180+(color-1)*30+(tone-1)*5+(base-1)};
  const sliceStart=s=>mod(OFF+s*BASE,CIRCLE);
  return {startDegrees,OFF,ORDER,sliceOf,hd,sliceIndex,sliceStart};
}

// arc readings. centi-second: arc = hundredths of the second (two linear digits, 9 and 9 = 99). gate-percent: arc = floor(100*offsetInGate/GATE).
export const ARC_READERS={
  'centi-second':{fromP:(P)=>P%SEC},
  'gate-percent':{fromP:(P,w)=>Math.floor(100*mod(P-w.OFF,GATE)/GATE)},
};

export function encode(P,{planet,dimension,house},{wheel=makeWheel(),arcReading=INVESTIGATING.arcReading.value,houseCount=INVESTIGATING.houseCount.value}={}){
  int(P,0,CIRCLE-1,'position');
  const r=P%SIGN, a={planet,dimension,...wheel.hd(wheel.sliceOf(P)),degree:Math.floor(r/DEG),minute:Math.floor(r%DEG/MIN),second:Math.floor(r%MIN/SEC),
    arc:ARC_READERS[arcReading[dimension]].fromP(P,wheel),zodiac:Math.floor(P/SIGN)+1,house};
  validate(a,{houseCount});return Object.freeze(a);
}
export function validate(a,{houseCount=INVESTIGATING.houseCount.value}={}){
  if(!a||typeof a!=='object')throw new TypeError('address required');
  for(const f of FIELDS)if(!(f in a))throw new RangeError('missing '+f);
  int(a.planet,1,13,'planet');if(!DIMENSIONS.includes(a.dimension))throw new RangeError('dimension');
  int(a.gate,1,64,'gate');int(a.line,1,6,'line');int(a.color,1,6,'color');int(a.tone,1,6,'tone');int(a.base,1,5,'base');
  int(a.degree,0,29,'degree');int(a.minute,0,59,'minute');int(a.second,0,59,'second');int(a.arc,0,99,'arc');int(a.zodiac,1,12,'zodiac');int(a.house,1,houseCount,'house');
  return true;
}
// decode: address -> exact wheel position. Only centi-second arcs carry exact position; gate-percent cannot (reported by the probe).
export function decode(a,{wheel=makeWheel(),arcReading=INVESTIGATING.arcReading.value,houseCount}={}){
  validate(a,{houseCount});
  const mode=arcReading[a.dimension];
  let P=(a.zodiac-1)*SIGN+a.degree*DEG+a.minute*MIN+a.second*SEC+(mode==='centi-second'?a.arc:0);
  const s=wheel.sliceOf(P),h=wheel.hd(s);
  for(const k of ['gate','line','color','tone','base'])if(h[k]!==a[k])throw new RangeError(`contradiction: ${k} ${a[k]} but degree/minute/second places it at ${k} ${h[k]}`);
  if(mode!=='centi-second'&&ARC_READERS[mode].fromP(P,wheel)!==a.arc)throw new RangeError('contradiction: arc does not match position');
  return {position:P,slice:s,exact:mode==='centi-second'};
}
// downward nesting: span of any prefix gate→line→color→tone→base
export function span(prefix,{wheel=makeWheel()}={}){
  const lv=['gate','line','color','tone','base'],size=[GATE,LINE,COLOR,TONE,BASE];let depth=0;while(depth<5&&prefix[lv[depth]]!=null)depth++;
  if(!depth)throw new RangeError('gate required');
  const full={gate:prefix.gate,line:prefix.line??1,color:prefix.color??1,tone:prefix.tone??1,base:prefix.base??1};
  const start=wheel.sliceStart(wheel.sliceIndex(full));return {level:lv[depth-1],start,end:start+size[depth-1],children:depth<5?(depth===4?5:6):0};
}
// upward: the chain of parents of an address
export const ancestors=a=>[{gate:a.gate},{gate:a.gate,line:a.line},{gate:a.gate,line:a.line,color:a.color},{gate:a.gate,line:a.line,color:a.color,tone:a.tone}];

// REFUSE-TO-START: whole space must prove valid.
export function proveValid({wheel=makeWheel(),arcReading=INVESTIGATING.arcReading.value,houseCount=INVESTIGATING.houseCount.value,split=INVESTIGATING.split.value}={}){
  const fail=[],ok=c=>c;
  if(SLICES!==69120||BASE!==1875)fail.push('slice arithmetic');
  if(new Set(wheel.ORDER).size!==64)fail.push('gate order not 64 distinct');
  let mism=0,rt=0;
  for(let s=0;s<SLICES;s++){const P=wheel.sliceStart(s);
    const v=humanDesignResolution((P+BASE/2)/DEG);const h=wheel.hd(s);
    if((v.gate!==h.gate||v.line!==h.line||v.color!==h.color||v.tone!==h.tone||v.base!==h.base))mism++;
    if(wheel.sliceIndex(h)!==s||wheel.sliceOf(P)!==s||wheel.sliceOf(P+BASE-1)!==s)rt++;}
  if(mism)fail.push(mism+' slices disagree with her vendored engine');if(rt)fail.push(rt+' slices fail round-trip');
  const ss=new StateSpace({split:[6,12].includes(split)?split:12});
  for(const d of PRIMITIVE_DIMENSIONS){const g=new Set();for(let m=0;m<64;m++)g.add(ss.ensure(d,[{macro:m,sub:0}]).gate);if(g.size!==64)fail.push(d+' lacks 64');}
  {const g=new Set();for(let m=0;m<64;m++)g.add(macroToGate(m));for(let x=1;x<=64;x++)if(macroToGate(gateToMacro(x))!==x)fail.push('gate<->macro '+x);if(g.size!==64)fail.push('Space lacks 64 slots');}
  let addrFail=0;
  for(let s=0;s<SLICES;s+=37)for(const d of DIMENSIONS){const P=wheel.sliceStart(s)+((s*7919)%BASE);
    const a=encode(P,{planet:(s%13)+1,dimension:d,house:(s%houseCount)+1},{wheel,arcReading,houseCount});
    try{const r=decode(a,{wheel,arcReading,houseCount});if(r.exact&&r.position!==P)addrFail++;if(!r.exact&&r.slice!==s)addrFail++;}catch{addrFail++}}
  if(addrFail)fail.push(addrFail+' addresses fail round-trip');
  return {valid:fail.length===0,fail};
}

export class MorphStateSpace{
  #ledger=new PositionalRelationLedger();#addr=new Map();#events=[];
  constructor(opts={}){const proof=proveValid(opts);if(!proof.valid)throw new Error('REFUSE TO START: '+proof.fail.join('; '));
    this.opts={wheel:opts.wheel??makeWheel(),arcReading:opts.arcReading??INVESTIGATING.arcReading.value,houseCount:opts.houseCount??INVESTIGATING.houseCount.value};
    this.space=new StateSpace({split:opts.split??INVESTIGATING.split.value});this.proof=proof;Object.freeze(this.opts);}
  get key(){return fixedDimensionalKey()}
  identity(d){return dimensionalIdentity(d)}
  encode(P,ctx){return encode(P,ctx,this.opts)}
  decode(a){return decode(a,this.opts)}
  span(p){return span(p,this.opts)}
  register(id,address,components=[]){const r=decode(address,this.opts);
    this.#ledger.register({id,dimension:address.dimension,position:null,components});
    this.#addr.set(id,Object.freeze({...address}));this.#events.push(Object.freeze({index:this.#events.length,kind:'address',id,address:{...address},position:r.position}));return r}
  place(subjectId,frameId,slot){return this.#ledger.place({subjectId,frameId,slot})}
  enter(id){return {...this.#ledger.enter(id),address:this.#addr.get(id)??null}}
  history(){return {relations:this.#ledger.history(),addresses:structuredClone(this.#events)}}
}
export {SIGNS,BODIES,DIMENSIONS,EMERGENT_DIMENSION};
