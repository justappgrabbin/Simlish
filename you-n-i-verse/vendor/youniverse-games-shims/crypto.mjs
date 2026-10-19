// FILLED IN, no source: createHash('sha256') via the repo sha256.mjs; randomUUID is a deterministic counter.
import {sha256} from '../../sha256.mjs';
export function createHash(alg){if(alg!=='sha256')throw new Error('only sha256 in browser shim');let s='';const h={update(d){s+=typeof d==='string'?d:new TextDecoder().decode(d);return h},digest(enc){if(enc!=='hex')throw new Error('only hex digest in browser shim');return sha256(s)}};return h}
let n=0;export function randomUUID(){n++;return '00000000-0000-4000-8000-'+n.toString(16).padStart(12,'0')}
export default {createHash,randomUUID};
