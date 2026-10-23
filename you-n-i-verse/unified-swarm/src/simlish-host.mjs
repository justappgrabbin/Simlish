/** This host does not clone a state-space. It composes the existing Simlish plug. */
import {createUnifiedSwarm,UnifiedSwarm} from './unified-swarm.mjs';
/** Caller must supply the actual connected plug exported by Simlish, plus source assembly. */
export function attachUnifiedSwarmToSimlish({assembly,simlishPlug,authorize=null,snapshot=null}={}){
  if(typeof assembly?.structure!=='function'||typeof simlishPlug?.morph!=='function')throw new Error('CONNECTED_SOURCE_AND_SIMLISH_REQUIRED');
  if(snapshot){
    // Snapshot restore is intentionally explicit; no silent reset.
    return UnifiedSwarm.restore(snapshot,{source:assembly,simlishPlug,authorize});
  }
  return createUnifiedSwarm({source:assembly,simlishPlug,authorize});
}
