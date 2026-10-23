// Direct local import of the existing state-space assembly; no vendor archive copies.
import {resolve, join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {access} from 'node:fs/promises';
import './vendor/kernel.js';
import './vendor/world-bridge.js';
import './vendor/mesh-assembly.js';
import './vendor/sim-core.js';
import {connectGenerativeStateSpace} from './generative-state-space-plug.mjs';
export async function bootSimlishStateSpace({stateSpaceRoot=process.env.SYNTHIA_STATE_SPACE_ROOT,
  seed=777, authorize=null, renderer=null, actionRegistry=null}={}) {
  if (!stateSpaceRoot) throw new Error('SYNTHIA_STATE_SPACE_ROOT must point to the Synthia runnable root');
  const entry=resolve(join(stateSpaceRoot,'src/index-generative.mjs'));
  await access(entry);
  const {SynthiaGenerativeAssembly}=await import(pathToFileURL(entry).href);
  if (typeof SynthiaGenerativeAssembly!=='function') throw new Error('Source assembly unavailable');
  const assembly=new SynthiaGenerativeAssembly({morphology:{weightsFile:null}});
  const world=new globalThis.AgentomagotchiSim.AgentomagotchiWorld({seed});
  const plug=connectGenerativeStateSpace({assembly,world,authorize,renderer,actionRegistry});
  return {assembly,world,plug};
}
