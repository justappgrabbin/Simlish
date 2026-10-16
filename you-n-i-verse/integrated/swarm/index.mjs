// THE swarm area: one place that holds the registry of every swarm she has, loads the connected ones through her own modules, and grows ONE emergent mesh from any content.
// A continuation of the existing morphing swarm, not a new system: members are her modules imported as-is; the mesh sits on her mesh-state-space.js layers and state-space.mjs relations.
// Not decided here: the dimension-to-model mapping for the four LLM positions (her swarmdoc.md line 1342 says it is still open), and the Mawangdui sequence version.
import {EmergentMesh} from './mesh.mjs';
import {summarize} from './registry.mjs';
import {measureContent,expandInput} from '../measure.mjs';
// Loaders: literal dynamic imports, so the repo's import graph (registry.mjs computeConnections) sees them and a member counts as connected only if it is really loadable here.
export const MEMBER_LOADERS=Object.freeze({
 'integrated-swarm':[()=>import('../integrated-swarm.mjs'),['IntegratedSwarm']],
 'swarm-loop':[()=>import('../swarm-loop.mjs'),['SwarmLoop','Workspace','AuditLog']],
 'any-file-swarm':[()=>import('../../any-file-swarm.mjs'),['fileFeatures','placeFile']],
 'app-finisher':[()=>import('../../app-finisher.mjs'),['CHECKLIST']],
 'swarm-fit':[()=>import('../../swarm-fit.mjs'),['fitParticles']],
 'abc':[()=>import('../abc.mjs'),['abcFit','AbcV9']],
 'population':[()=>import('../population.mjs'),['Population']],
 'trident-js':[()=>import('../trident-js.mjs'),['Trident']],
 // byte-identical copy of her oct7src/computer/runtime/morph-search.mjs (sha256 in registry.json); the original is untouched
 'morph-search':[()=>import('./originals/morph-search/5cf53ad3-morph-search.mjs'),['createMorphSearch']]});
export async function createSwarmArea({loadJson,primitives=null,registry=null}={}){
 const reg=registry||await loadJson('integrated/swarm/registry.json'),mesh=new EmergentMesh({primitives});
 const area={registry:reg,mesh,
  inventory:()=>summarize(reg),
  members:()=>reg.entries.map(e=>({id:e.id,family:e.family,name:e.name,origin:e.origin,sha256:e.sha256,bytes:e.bytes,copy:e.copy?.path??null,...reg.connections.byId[e.id],loadable:e.id in MEMBER_LOADERS})),
  async loadMember(id){const l=MEMBER_LOADERS[id];if(!l)throw new Error('not a connected member: '+id);const m=await l[0]();for(const n of l[1])if(!(n in m))throw new Error(id+' lacks export '+n);return m},
  // any content (file, folder, zip) -> each file becomes units in the same mesh; the measurement is returned beside it
  async weave(input,opts={}){const src=await expandInput(input),files=[];for(const f of src.files){if(f.bytes.length)files.push({path:f.path,...mesh.weave(f.bytes,{name:f.path,...opts})})}
   return {input:{name:src.name,files:src.files.length},woven:files.length,units:files.reduce((a,f)=>a+f.units,0),newNodes:files.reduce((a,f)=>a+f.newNodes,0),newEdges:files.reduce((a,f)=>a+f.newEdges,0),mesh:mesh.summary()}},
  measure:(input,o)=>measureContent(input,{...o,primitives}),
  status:()=>({members:Object.keys(MEMBER_LOADERS),mesh:mesh.summary({clusterLimit:3}),inventory:summarize(reg)})};
 return area}
