// Address gate: every app launch and every file coming into the Computer passes through
// Alexis's state-space resolver (address-engine/resolver.mjs, built from her State Space doc)
// BEFORE anything else happens. No address -> nothing gets in. Rules live in the resolver, untouched here.
import { state, encode } from './address-engine/resolver.mjs';
export const LEDGER_KEY='computerAddressLedger';
export function ledger(){try{return JSON.parse(localStorage[LEDGER_KEY]||'[]')}catch{return []}}
export function admit({ kind, id, name, address }) {
  if (!address) return { ok:false, reason:'no address given — every app or file needs a gate.line.color.tone.base address first' };
  let st; try { st = state(address); } catch (e) { return { ok:false, reason:e.message }; }
  if (st.unresolved) return { ok:false, reason:st.reason };
  const layers = encode(address);
  const rec = { at:new Date().toISOString(), kind, id, name, address:st.address, index:st.index, gate:st.gateState?.names?.[0]||null, layers };
  const l=ledger(); l.unshift({...rec,layers:undefined}); localStorage[LEDGER_KEY]=JSON.stringify(l.slice(0,300));
  return { ok:true, record:rec, state:st };
}
