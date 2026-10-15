// Registration as a swarm component (PSO: background tooling). Six members in packs of 3, 2, 1.
// Incomplete or uncertain input is never rejected: it is kept and marked 'investigating'.
import {SwarmComponent} from './swarm-component.mjs';
import {ADDRESS_ORDER,normalizeAddress,resolveAddress} from './address-resolution.mjs';
export function missingPlaces(address){
 let n;try{n=normalizeAddress(address)}catch{return {conflict:true,missing:[]}}
 return {conflict:false,missing:ADDRESS_ORDER.filter(k=>n[k]===undefined||n[k]===null||n[k]==='')};
}
export const registrationSwarm=new SwarmComponent({id:'register',kind:'PSO',
 purpose:'Check that a registration carries the full 13-place address before it is marked registered',
 openQuestions:['Pack-count-to-two-bits mapping and pack order are investigating, not rulings.',
  'Whether arc-second range differs by dimension is Alexis\'s open theory; the kernel uses one range for all.'],
 members:[
  {id:'identity',check:i=>!!(i.id&&i.version)},
  {id:'address-present',check:i=>!!(i.address&&typeof i.address==='object')},
  {id:'all-13-places',check:i=>{if(!i.address||typeof i.address!=='object')return {ok:false,note:'no address given'};const m=missingPlaces(i.address);return m.conflict?{ok:false,note:'arcSecond and legacy arc conflict'}:{ok:m.missing.length===0,note:m.missing.length?'missing '+m.missing.join(', '):undefined}}},
  {id:'kernel-valid',check:i=>{
    if(!i.address||typeof i.address!=='object')return {ok:false,note:'no address to validate'};
    if(!i.kernel)return {ok:false,note:'address kernel unavailable; not validated'};
    const m=missingPlaces(i.address);if(m.conflict||m.missing.length)return {ok:false,note:'needs the full address first'};
    const r=resolveAddress({kernel:i.kernel,leadingDimension:i.address.dimension,coordinates:i.address,evidence:{source:i.source||'unspecified'}});
    return {ok:r.ok,note:r.ok?undefined:r.reason}}},
  {id:'arc-consistent',check:i=>{try{normalizeAddress(i.address);return true}catch(e){return {ok:false,note:e.message}}}},
  {id:'provenance',check:i=>!!i.source}]});
// GA phenotype for registration: the visible trait (what the person sees), expressed from the PSO genotype above.
import {SwarmPair} from './swarm-nsnf.mjs';
const phenotypeComponent=new SwarmComponent({id:'register-view',kind:'GA',purpose:'Show registration state on screen',members:[
 {id:'has-label',check:i=>!!i.visible.label},{id:'has-lines',check:i=>i.visible.lines?.length===6},{id:'has-code',check:i=>/^[01]{6}$/.test(i.visible.hexagram_code)},
 {id:'names-missing',check:i=>i.genotype.status==='complete'||i.visible.missing!==undefined},{id:'status-matches-genotype',check:i=>i.visible.status===i.genotype.status},{id:'keeps-input',check:i=>i.visible.kept===true}]});
export function registrationTrait(g,input){
 const m=input.address&&typeof input.address==='object'?missingPlaces(input.address).missing:ADDRESS_ORDER.slice();
 return {label:g.status==='complete'?'Registered with the full address':'Investigating: address not complete yet',status:g.status,hexagram_code:g.hexagram_code,
  lines:[...g.hexagram_code].map(b=>b==='1'?'solid':'broken'),missing:g.status==='complete'?[]:m,kept:true};
}
export const registrationPair=()=>new SwarmPair({id:'register',genotype:registrationSwarm,phenotype:{component:phenotypeComponent,trait:registrationTrait}});
