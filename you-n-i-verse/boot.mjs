// App startup: register Simlish itself (full 13-place address or investigating), sync registrations when an endpoint is
// configured, and retry queued Supabase originals when env is configured. Nothing here is configured in the repo.
import {registerSimlish,syncRegistrations,pendingRegistrations} from './synthia-registration.mjs';
import {readPersonalization} from './personalize.mjs';
import {flushOutbox,readSupabaseEnv} from './supabase-original.mjs';
export const APP_ID='simlish',APP_VERSION='0.1-debug';
export const CAPABILITIES=Object.freeze(['scene-events','reality-readiness','app-finisher','supabase-original-outbox','personalize','swarm-pso-genotype','swarm-ga-phenotype']);
export function currentAddressContext(storage){
 const p=readPersonalization(storage);if(!p?.address)return null;
 return {leadingDimension:p.address.dimension,coordinates:p.address,evidence:{source:'uploader-personalization:'+(p.name||'unnamed')}};
}
export async function bootSimlish({storage,env={},endpoint=env.SIMLISH_REGISTRY_ENDPOINT,token=env.SIMLISH_REGISTRY_TOKEN,fetchImpl=globalThis.fetch,client=null,outbox=null,kernel=globalThis.AgentomagotchiKernel}={}){
 const p=readPersonalization(storage);
 const registration=registerSimlish(storage,{id:APP_ID,version:APP_VERSION,source:p?.name?'uploader-personalization:'+p.name:null,address:p?.address??undefined,capabilities:[...CAPABILITIES],kernel});
 let sync={status:'skipped',note:'No registration endpoint configured; registration stays pending on this device'};
 if(endpoint){try{sync={status:'attempted',results:await syncRegistrations(storage,{endpoint,token,fetchImpl})}}catch(e){sync={status:'pending',note:e.message}}}
 let originals={status:'skipped',note:'No outbox or Supabase env'};
 if(outbox&&(client||readSupabaseEnv(env)))originals={status:'attempted',results:await flushOutbox({env,client,fetchImpl,outbox})};
 return {registration,sync,pending:Object.keys(pendingRegistrations(storage)),originals};
}
