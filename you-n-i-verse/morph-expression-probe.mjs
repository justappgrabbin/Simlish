// Expression probe: run the finished space with each INVESTIGATING candidate and report what it expresses. Decides nothing.
import {INVESTIGATING,makeWheel,proveValid,encode,decode,SLICES,BASE,DEG,CIRCLE,GATE,DIMENSIONS} from './morph-state-space.mjs';
import {humanDesignResolution} from './vendor/isohuman/GeonatalCalculator.browser.mjs';
const mod=(a,m)=>((a%m)+m)%m;
export function probe(){
  const rows=[];
  for(const start of INVESTIGATING.wheelStartDegrees.candidates){const w=makeWheel(start);let dis=0;
    for(let s=0;s<SLICES;s++){const P=w.sliceStart(s),v=humanDesignResolution((P+BASE/2)/DEG),h=w.hd(s);if(v.gate!==h.gate||v.line!==h.line||v.color!==h.color||v.tone!==h.tone||v.base!==h.base)dis++;}
    const g41=w.sliceStart(w.sliceIndex({gate:41,line:1,color:1,tone:1,base:1}))/DEG;
    rows.push({parameter:'wheel start',candidate:start+'°',roundTrip:proveValid({wheel:w}).fail.filter(f=>!f.includes('vendored')).length===0,
      expresses:`gate 41 line 1 begins at ${g41}°; 2° Aquarius = 302°: ${g41===302?'yes':'no'}; slices disagreeing with her vendored engine: ${dis}/69120`,
      rejected:dis>0});}
  for(const split of INVESTIGATING.split.candidates){const per=1080/split,even=Number.isInteger(per);
    rows.push({parameter:'split',candidate:split===13?'12+1 (13)':String(split),roundTrip:true,expresses:`64×${split}=${64*split} per level; 1080 slices per gate / ${split} = ${per.toFixed(3)}; 360°/${64*split} = ${(360/(64*split)).toFixed(6)}°`,rejected:!even});}
  for(const mode of INVESTIGATING.arcReading.candidates){const ar=Object.fromEntries(DIMENSIONS.map(d=>[d,mode]));let exact=0,n=0,lost=0,contra=0;
    for(let P=0;P<CIRCLE;P+=104729){n++;const a=encode(P,{planet:1,dimension:'Being',house:1},{arcReading:ar});try{const r=decode(a,{arcReading:ar});if(r.exact&&r.position===P)exact++;else lost++;}catch{lost++;contra++}}
    rows.push({parameter:'arc reading',candidate:mode,roundTrip:lost===0,expresses:mode==='centi-second'?`two linear digits 00–99 = 1/100″; position exact ${exact}/${n}; 99 = 0.99″`:`one step = ${GATE/100/100}″ (${(GATE/100/100/60).toFixed(3)}′) of the gate; position recoverable only to the second, exact ${exact}/${n}, contradictions ${contra}/${n}`,rejected:lost>0});}
  for(const hc of INVESTIGATING.houseCount.candidates){const deg=360/hc;rows.push({parameter:'house count',candidate:String(hc),roundTrip:proveValid({houseCount:hc}).valid,expresses:`360/${hc} = ${Number.isInteger(deg)?deg+'°':deg.toFixed(4)+'° (not whole)'}; 30° signs ${30%deg===0||deg%30===0?'nest evenly':'do not nest evenly'}`,rejected:!Number.isInteger(deg)});}
  return rows;
}
if(import.meta.url===`file://${process.argv[1]}`)console.table(probe());
