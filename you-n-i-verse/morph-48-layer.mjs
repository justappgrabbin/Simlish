// FLAGGED LAYER (additive): Alexis: "It's 48" — each of the 4 duality dimensions' cubes carries 48 symmetries (full cube group).
// Her photon remark ("how many dimensions can be in a single photon of light") is recorded as her words, NOT as physics.
// Only exact closures are kept; failures are reported, never rounded.
import {SLICES} from './morph-state-space.mjs';
export const N=48, DUALITY_DIMS=Object.freeze(['Movement','Evolution','Being','Design']), ONE='Space';
const COUNTS={'wheel slices (base)':SLICES,'tones':64*216,'colors':64*36,'lines':64*6,'gates':64,'slices per gate':1080,'degrees':360,'dodecahedron symmetries (full)':120,'dodecahedron rotations':60};
export function closures(){return Object.entries(COUNTS).map(([k,v])=>({count:k,value:v,per48:v/N,exact:Number.isInteger(v/N)}))}
// 48-sector wheel: 1,440 slices each = 1 gate + 2 lines = 7.5°
export const SECTOR=SLICES/N;
export function sectorOf(slice){if(!Number.isInteger(slice)||slice<0||slice>=SLICES)throw new RangeError('slice');return Math.floor(slice/SECTOR)}
export function sectorSpan(k){if(!Number.isInteger(k)||k<0||k>=N)throw new RangeError('sector 0..47');return {startSlice:k*SECTOR,endSlice:(k+1)*SECTOR,degrees:[k*7.5,(k+1)*7.5]}}
export const FACTS=Object.freeze({cubeFull:48,cubeKeptInsideDodecahedron:24,cubes:5,check:5*24===120,note:'48 does not divide 120: each inscribed cube keeps only 24 of its 48 symmetries (pyritohedral) inside the dodecahedron; 5×24 = 120'});
if(import.meta.url===`file://${process.argv[1]}`){console.table(closures());console.log(FACTS,SECTOR)}
