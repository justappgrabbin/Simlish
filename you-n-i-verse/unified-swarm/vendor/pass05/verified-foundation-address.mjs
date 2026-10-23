// Additive, dependency-free deterministic address kernel. Does not replace existing Synthia components.
export const GATES=64, LINES=6, COLORS=6, TONES=6, BASES=5;
export const UNITS=GATES*LINES*COLORS*TONES*BASES; // 69120
const assertInt=(v,lo,hi,name)=>{if(!Number.isInteger(v)||v<lo||v>hi)throw new RangeError(`${name}: ${lo}..${hi}`);};
export function encode({gate,line,color,tone,base}){
  for(const [n,v,max] of [['gate',gate,64],['line',line,6],['color',color,6],['tone',tone,6],['base',base,5]])assertInt(v,1,max,n);
  return (((((gate-1)*6+(line-1))*6+(color-1))*6+(tone-1))*5+(base-1));
}
export function decode(index){
  assertInt(index,0,UNITS-1,'index');
  let n=index;const base=n%5+1;n=Math.floor(n/5);const tone=n%6+1;n=Math.floor(n/6);const color=n%6+1;n=Math.floor(n/6);const line=n%6+1;n=Math.floor(n/6);return {gate:n+1,line,color,tone,base};
}
// Integer hundredths of an arcsecond avoid float boundary ambiguity.
export const CIRCLE_CENTIARCSEC=360*3600*100;
export function fromAngleCentiArcsec(angle){assertInt(angle,0,CIRCLE_CENTIARCSEC-1,'angle');return decode(Math.floor(angle*UNITS/CIRCLE_CENTIARCSEC));}
export function bounds(index){assertInt(index,0,UNITS-1,'index');return {startCentiArcsec:index*CIRCLE_CENTIARCSEC/UNITS,endCentiArcsec:(index+1)*CIRCLE_CENTIARCSEC/UNITS};}
export function changeLines(hexagramBits,mask){assertInt(hexagramBits,0,63,'hexagramBits');assertInt(mask,0,63,'mask');return hexagramBits^mask;}
export function validateContext({dimension,planet,zodiac,house}={}){
  if(dimension!==undefined&&!['Movement','Evolution','Being','Design','Space'].includes(dimension))throw new RangeError('dimension');
  if(planet!==undefined)assertInt(planet,1,13,'planet');
  if(zodiac!==undefined)assertInt(zodiac,1,12,'zodiac');
  if(house!==undefined)assertInt(house,1,12,'house');
  return true;
}
