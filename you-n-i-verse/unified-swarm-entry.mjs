// Place beside Simlish generative-state-space-runtime.mjs in you-n-i-verse/.
// This boots the ACTUAL source assembly + ACTUAL Simlish world once, then mounts the original sourced swarm.
import {bootSimlishStateSpace} from './generative-state-space-runtime.mjs';
import {attachUnifiedSwarmToSimlish} from './unified-swarm/src/simlish-host.mjs';
export async function bootUnifiedSimlishStateSpace({snapshot=null,...hostOptions}={}){
  const {assembly,world,plug}=await bootSimlishStateSpace(hostOptions);
  const swarm=attachUnifiedSwarmToSimlish({assembly,simlishPlug:plug,authorize:hostOptions.authorize,snapshot});
  return {assembly,world,plug,swarm};
}
