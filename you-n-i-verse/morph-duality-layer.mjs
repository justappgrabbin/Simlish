// FLAGGED LAYER (additive): Movement, Evolution, Being, Design = dualities (a whole split into two UNEQUAL halves); Space = the one.
// Alexis: "the fifth one is one while the others are dualities"; "even the Chinese say yin and yang are not equal".
// Each scheme is a candidate ratio a:b. The probe checks whether it closes exactly on the space's own wholes. Nothing here is set as her rule.
import {SLICES,CIRCLE,DIMENSIONS} from './morph-state-space.mjs';
const PHI=(1+Math.sqrt(5))/2;
export const SCHEMES=Object.freeze({
  'phi (golden)':{a:PHI,b:1,src:'dodecahedron geometry (φ); irrational'},
  'dodeca:icosa 20:12 = 5:3':{a:5,b:3,src:'dual pair vertex counts (geometry)'},
  'old yang:old yin 9:6 = 3:2':{a:9,b:6,src:'Hatcher p.42 line values'},
  'yarrow old yang:old yin 3:1':{a:3,b:1,src:'Hatcher p.46: bu shi odds 6=1/16, 9=3/16'},
  'yarrow young yin:young yang 7:5':{a:7,b:5,src:'Hatcher p.46: 8=7/16, 7=5/16'},
  'yarrow all yin:all yang 8:8':{a:8,b:8,src:'Hatcher p.46: 1/16+7/16 = 5/16+3/16'},
  'line: yin 2 halves : yang 1 whole':{a:2,b:1,src:'broken vs unbroken line (Hatcher p.42 drawing)'},
  'Lo Shu 9:1':{a:9,b:1,src:'Govinda p.51 magic square, pairs through centre 5 sum to 10'},
  'Lo Shu 8:2':{a:8,b:2,src:'Govinda p.51'},
  'Lo Shu 7:3':{a:7,b:3,src:'Govinda p.51'},
  'Lo Shu 6:4':{a:6,b:4,src:'Govinda p.51'},
});
export const WHOLES=Object.freeze({'64 gates':64,'69,120 slices':SLICES,'360°':360,'centi-arcsec circle':CIRCLE});
export function splitWhole(whole,{a,b}){const big=whole*a/(a+b);return {big,small:whole-big,exact:Number.isInteger(big)&&Number.isInteger(whole-big)}}
export function probeDualities(){
  return Object.entries(SCHEMES).map(([name,s])=>{const r=Object.fromEntries(Object.entries(WHOLES).map(([w,n])=>[w,splitWhole(n,s)]));
    return {scheme:name,equal:s.a===s.b,...Object.fromEntries(Object.entries(r).map(([w,x])=>[w,x.exact?`${x.big}:${x.small}`:'no'])),
      closesAll:Object.values(r).every(x=>x.exact),rejected:Object.values(r).some(x=>!x.exact),src:s.src}});
}
// INVESTIGATING: which dimension takes which ratio is not decided. Default: unassigned (null). Space is always the one (no split).
export const ASSIGNMENT=Object.freeze(Object.fromEntries(DIMENSIONS.map(d=>[d,d==='Space'?'one':null])));
export function dualityOf(dimension,scheme,whole=SLICES){
  if(!DIMENSIONS.includes(dimension))throw new RangeError('dimension');
  if(dimension==='Space')return {dimension,kind:'one',whole};
  const s=SCHEMES[scheme];if(!s)throw new RangeError('unknown scheme');const r=splitWhole(whole,s);
  if(!r.exact)throw new RangeError(`scheme ${scheme} does not close on ${whole}; the space rejects it`);
  return {dimension,kind:'duality',whole,halves:[r.big,r.small],scheme,flag:'INVESTIGATING: assignment not decided by Alexis'};
}
if(import.meta.url===`file://${process.argv[1]}`)console.table(probeDualities().map(({src,...x})=>x));
