// Thin adapter: an app's code is DERIVED from the resolved code. There is no per-app code table and nothing is stored per app.
// Alexis (2026-10-16): "apps have codes but they should derive from a resolved code".
// What her files say: nsnf-spec §18 (execution idea) ends "... higher-scale state -> resolve address -> land history/bookkeeping"; her Execution Pipeline Patch names that
// stage "ontological-address" and Synthia's ExecutionAddressResolver.resolve() is where the address is resolved ("Resolve the complete address AFTER the first execution attempt").
// The exact phrase "resolved code" is not defined in her files. So here: resolved code = the resolved address (+ its six-bit hexagram part). INVESTIGATING (below).
// Also her rules used: values are "derived, never stored" (spec §1.5/§2.x), and six-bit code = Fu Xi macro of the gate (state-space.mjs gateToMacro).
import {gateToMacro,bitsOf,DIMENSIONS} from '../state-space.mjs';
import {stable} from '../nsnf-runtime.mjs';
import {sha256} from '../sha256.mjs';
export const APP_CODE_INVESTIGATING=Object.freeze([
 'The phrase "resolved code" is not defined in her files; it is taken to mean the resolved address from the "resolve address" step (spec §18) / ontological-address stage. Confirm or correct.',
 'Which part of the resolved address is the "code" is not specified: here the six-bit hexagram code comes from the gate (King Wen -> Fu Xi macro), the structural places are dimension, gate, line, color, tone, base, and a 48-bit fingerprint of the whole resolved address key keeps codes unique. The format is an engineering choice.',
 'A registered (creator-supplied) address is treated as authoritative and preserved, as her spine does; otherwise the first-execution-derived address from ExecutionAddressResolver is used. The repo 13-place address and the spine/ingest address schemas differ and no mapping between them was chosen.']);
const REQUIRED=['dimension','gate','line','color','tone','base'];
const INITIAL={Movement:'M',Evolution:'E',Being:'B',Design:'D',Space:'S'};
export const resolvedKeyOf=resolved=>resolved?.key??(resolved?.address?sha256(stable(resolved.address)):null);
export function deriveAppCode(resolved){
 const a=resolved?.address;
 if(!a||typeof a!=='object')return {code:null,status:'investigating',reason:'no resolved address yet; nothing is guessed',derived:true,stored:false};
 const missing=REQUIRED.filter(k=>a[k]===undefined||a[k]===null);
 if(missing.length)return {code:null,status:'investigating',reason:'resolved address lacks '+missing.join(', ')+'; nothing is guessed',missing,derived:true,stored:false};
 if(!DIMENSIONS.includes(a.dimension)||!Number.isInteger(a.gate)||a.gate<1||a.gate>64)return {code:null,status:'investigating',reason:'resolved address has an unusable dimension or gate',derived:true,stored:false};
 const key=resolvedKeyOf(resolved),fp=sha256(String(key)).slice(0,12),hex=bitsOf(gateToMacro(a.gate));
 const code='app-'+hex+'-'+INITIAL[a.dimension]+a.gate+'.'+a.line+'.'+a.color+'.'+a.tone+'.'+a.base+'-'+fp;
 return {code,status:'derived',hexagram_code:hex,dimension:a.dimension,gate:a.gate,line:a.line,
  derivedFrom:{basis:resolved.basis??'resolved-address',resolvedKey:key,fingerprint:fp},derived:true,stored:false,spaceWritten:false,investigating:[...APP_CODE_INVESTIGATING]};
}
// Verification is re-derivation: the code is accepted only if it equals what the resolved address derives. Nothing is looked up.
export const verifyAppCode=(code,resolved)=>{const d=deriveAppCode(resolved);return d.status==='derived'&&d.code===code};
