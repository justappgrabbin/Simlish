// Address from arc position. PORT to plain JavaScript of her sentence-engine.ts (Dual-engine- / Hdgnn repo; reference copy integrated/reference/her/sentence-engine.ts.txt): same constants, same HD wheel
// gate start degrees, same longitudeTo* functions, same calculateLatticePosition (69,120 positions). Types removed; nothing else changed, nothing invented.
// Other longitude->gate variants exist in her files and DISAGREE (field_core.js: sequential gates, no wheel order; OntologicalAddress.js: same wheel order starting at 0 deg). They are not chosen here.
export const DEGREES_PER_GATE=360/64,DEGREES_PER_LINE=DEGREES_PER_GATE/6,DEGREES_PER_COLOR=DEGREES_PER_LINE/6,DEGREES_PER_TONE=DEGREES_PER_COLOR/6,DEGREES_PER_BASE=DEGREES_PER_TONE/5; // base = tone/5 (her doc3 §M, spec, Colors book, field_core.js)
export const LATTICE_SIZE=5*6*6*6*64; // 69,120
export const HD_WHEEL_GATES=Object.freeze([41,19,13,49,30,55,37,63,22,36,25,17,21,51,42,3,27,24,2,23,8,20,16,35,45,12,15,52,39,53,62,56,31,33,7,4,29,59,40,64,47,6,46,18,48,57,32,50,28,44,1,43,14,34,9,5,26,11,10,58,38,54,61,60]);
// tropical start degree of each gate: gate 41 at 272.0 and +5.625 per step along the wheel (her table lists the same numbers explicitly)
export const GATE_START_DEGREES=Object.freeze(Object.fromEntries(HD_WHEEL_GATES.map((g,i)=>[g,(272+i*DEGREES_PER_GATE)%360])));
const norm=l=>((l%360)+360)%360;
export function longitudeToGate(longitude){
 const n=norm(longitude);
 for(const gate of HD_WHEEL_GATES){const start=GATE_START_DEGREES[gate],end=(start+DEGREES_PER_GATE)%360;
  if(start<end){if(n>=start&&n<end)return gate}else if(n>=start||n<end)return gate}
 return 41;
}
function within(longitude,gate){let off=norm(longitude)-GATE_START_DEGREES[gate];if(off<0)off+=360;if(off>=DEGREES_PER_GATE)off-=360;return Math.max(0,Math.min(DEGREES_PER_GATE-0.0001,off))}
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
export const longitudeToLine=l=>{const g=longitudeToGate(l);return clamp(Math.floor(within(l,g)/DEGREES_PER_LINE)+1,1,6)};
export const longitudeToColor=l=>{const g=longitudeToGate(l);return clamp(Math.floor((within(l,g)%DEGREES_PER_LINE)/DEGREES_PER_COLOR)+1,1,6)};
export const longitudeToTone=l=>{const g=longitudeToGate(l);return clamp(Math.floor((within(l,g)%DEGREES_PER_COLOR)/DEGREES_PER_TONE)+1,1,6)};
export const longitudeToBase=l=>{const g=longitudeToGate(l);return clamp(Math.floor((within(l,g)%DEGREES_PER_TONE)/DEGREES_PER_BASE)+1,1,5)};
export function longitudeToDMS(longitude){const n=norm(longitude),degree=Math.floor(n),rm=(n-degree)*60,minute=Math.floor(rm),second=Math.floor((rm-minute)*60);return {degree,minute,second}}
export const ZODIAC_SIGNS=Object.freeze(['Aries','Taurus','Gemini','Cancer','Leo','Virgo','Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces']);
export const longitudeToZodiac=l=>ZODIAC_SIGNS[Math.floor(norm(l)/30)]||'Aries';
export const gateToHouse=gate=>((gate-1)%12)+1; // her sentence-engine.ts: "simplified: gate mod 12"
export function calculateLatticePosition(gate,line,color,tone,base){
 const g=clamp(gate,1,64),l=clamp(line,1,6),c=clamp(color,1,6),t=clamp(tone,1,6),b=clamp(base,1,5);
 return (b-1)*13824+(t-1)*2304+(c-1)*384+(l-1)*64+(g-1)+1;
}
// inverse of calculateLatticePosition (NEW, derived: plain mixed-radix inverse of her formula; please confirm)
export function latticePositionToAddress(pos){
 if(!Number.isInteger(pos)||pos<1||pos>LATTICE_SIZE)throw new RangeError('lattice position is 1..'+LATTICE_SIZE);
 let n=pos-1;const gate=n%64+1;n=Math.floor(n/64);const line=n%6+1;n=Math.floor(n/6);const color=n%6+1;n=Math.floor(n/6);const tone=n%6+1;n=Math.floor(n/6);return {gate,line,color,tone,base:n+1};
}
export const gateLineToLongitude=(gate,line)=>(GATE_START_DEGREES[gate]+(line-1)*DEGREES_PER_LINE)%360;
// the whole address that can be read from a longitude alone (planetary and dimension are inputs, not derivable; house here is her simplified gate rule)
export function addressFromLongitude(longitude,{planetary=null,dimension=null}={}){
 const gate=longitudeToGate(longitude),line=longitudeToLine(longitude),color=longitudeToColor(longitude),tone=longitudeToTone(longitude),base=longitudeToBase(longitude),{degree,minute,second}=longitudeToDMS(longitude);
 return {planetary,dimension,gate,line,color,tone,base,degree,minute,second,arcSecond:null,zodiac:longitudeToZodiac(longitude),house:gateToHouse(gate),longitude:norm(longitude),latticePosition:calculateLatticePosition(gate,line,color,tone,base)};
}
