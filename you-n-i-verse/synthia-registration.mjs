// Preservation-safe local registry with update receipts and optional shared-computer relay.
import {registrationPair,missingPlaces} from './registration-swarm.mjs';
const unit=registrationPair(); // swarm building block: PSO genotype -> GA phenotype
export const registrationUnit=unit;
export const REGISTRY_KEY = 'synthia.simlish.registry.v1';
export const OUTBOX_KEY = 'synthia.simlish.registration.outbox.v1';
function load(storage,key) {
  const raw=storage.getItem(key);
  if(raw===null)return {};
  let data;
  try {data=JSON.parse(raw)} catch {throw new Error('Unreadable '+key+'; original preserved')}
  if(!data||typeof data!=='object'||Array.isArray(data))throw new Error('Invalid '+key+'; original preserved');
  return data;
}
function requireStorage(storage){
  if(!storage||typeof storage.getItem!=='function'||typeof storage.setItem!=='function')throw new TypeError('Persistent storage required');
}
export function readSimlishRegistrations(storage){requireStorage(storage);return load(storage,REGISTRY_KEY)}
export function registerSimlish(storage,input){
  requireStorage(storage);
  if(!input||typeof input!=='object'||typeof input.id!=='string'||!input.id.trim()||!input.version)throw new TypeError('id and version required');
  const records=load(storage,REGISTRY_KEY),id=input.id.trim(),old=records[id]||{};
  const record={...old,id,version:String(input.version),computer:'gaming',
    source:input.source??old.source??null,address:input.address??old.address??null,
    capabilities:input.capabilities??old.capabilities??[],
    updatedAt:new Date().toISOString()};
  if(!Array.isArray(record.capabilities))throw new TypeError('capabilities must be an array');
  // Swarm: PSO genotype checks the full 13-place address; GA phenotype is what is shown. Uncertain -> 'investigating', kept.
  const out=unit.forward({id:record.id,version:record.version,address:record.address,source:record.source,kernel:input.kernel??globalThis.AgentomagotchiKernel});
  record.status=out.status==='complete'?'registered':'investigating';
  record.investigation=record.status==='registered'?null:{missingPlaces:record.address&&typeof record.address==='object'?missingPlaces(record.address).missing:'address not given',failedChecks:out.genotype.failed,notes:out.genotype.notes};
  record.swarm={genotype:{component:out.genotype.component,kind:'PSO',packs:out.genotype.packs,hexagram_code:out.genotype.hexagram_code},phenotype:{component:out.phenotype.component,kind:'GA',visible:out.phenotype.visible,status:out.phenotype.status},event:out.event};
  records[id]=record;
  storage.setItem(REGISTRY_KEY,JSON.stringify(records));
  const outbox=load(storage,OUTBOX_KEY);
  outbox[id]={record,delivery:'pending'};
  storage.setItem(OUTBOX_KEY,JSON.stringify(outbox));
  return record;
}
export function pendingRegistrations(storage){requireStorage(storage);return load(storage,OUTBOX_KEY)}
export async function syncRegistrations(storage,{endpoint,token,fetchImpl=fetch}={}){
  requireStorage(storage);
  if(!endpoint)throw new Error('Explicit trusted registration endpoint required');
  const url=new URL(endpoint);
  if(url.protocol!=='https:'&&url.hostname!=='localhost'&&url.hostname!=='127.0.0.1')throw new Error('HTTPS required');
  const outbox=load(storage,OUTBOX_KEY),results=[];
  for(const [id,item] of Object.entries(outbox)){
    const response=await fetchImpl(url.toString(),{method:'POST',headers:{
      'content-type':'application/json',...(token?{'authorization':'Bearer '+token}:{})
    },body:JSON.stringify({schema:'synthia.registration.v1',registration:item.record})});
    if(!response.ok){results.push({id,ok:false,status:response.status});continue}
    delete outbox[id];storage.setItem(OUTBOX_KEY,JSON.stringify(outbox));
    results.push({id,ok:true,status:response.status});
  }
  return results;
}
