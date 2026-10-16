// The one swarm area's registry: reads registry.json and computes, from the repo's own import graph, which swarms are CONNECTED.
// connected-copy = a byte-identical copy under swarm/originals is imported by the app (the original stays where it was)
// connected   = an in-repo module reachable by static/dynamic import from the app's entry points (so it actually runs in the app)
// sharesRules = that module's import closure includes the shared primitive rule modules (state-space.mjs, population.mjs, bantu.mjs, the ontological ingestor)
// referenced  = not connected, but a connected module's text names the file (a documented link, no code link)
// Pure JS; no dependency. `read(path)` returns the text of a repo file (path relative to you-n-i-verse/) or null.
import {references} from '../measure.mjs';
export const ENTRY_POINTS=Object.freeze(['integrated/app-server.mjs','integrated/integrated-swarm.mjs','integrated/swarm-loop.mjs','synthia-app.mjs','measure.mjs','integrated/swarm/index.mjs']);
export const PRIMITIVE_RULE_MODULES=Object.freeze(['state-space.mjs','integrated/population.mjs','integrated/bantu.mjs','integrated/ontological/ingestion/ontological-file-ingestor.mjs']);
const isCode=p=>/\.(mjs|js)$/.test(p);
const rel=(from,spec)=>{if(!spec.startsWith('.'))return null;const out=from.split('/').slice(0,-1);for(const s of spec.split('/')){if(s==='.'||!s)continue;if(s==='..')out.pop();else out.push(s)}return out.join('/')};
async function closure(start,read,cache){const seen=new Map(),order=[],q=[[start,null]];while(q.length){const [p,from]=q.shift();if(seen.has(p))continue;seen.set(p,from);order.push(p);let t=cache.get(p);if(t===undefined){t=await read(p);cache.set(p,t)}if(t==null)continue;for(const r of references(p,t,'text')){const base=rel(p,r);if(base&&isCode(base)&&!seen.has(base))q.push([base,p])}}return {seen,order}}
export async function computeConnections(registry,read,{entries=ENTRY_POINTS}={}){
 const cache=new Map(),reach=new Map(),chain=p=>{const out=[];for(let c=p;c;c=reach.get(c))out.unshift(c);return out};
 for(const e of entries){const c=await closure(e,read,cache);for(const [p,from] of c.seen)if(!reach.has(p)&&cache.get(p)!=null)reach.set(p,from??null)}
 const rules=new Set(PRIMITIVE_RULE_MODULES),texts=[...reach.keys()].map(p=>[p,cache.get(p)]),out={};
 for(const e of registry.entries){const relPath=e.origin.kind==='repo'?e.origin.path.replace(/^you-n-i-verse\//,''):null;let status,via=null,sharesRules=null,referencedBy=null;
  if(relPath&&isCode(relPath)){if(reach.has(relPath)){status='connected';via=chain(relPath);const c=await closure(relPath,read,cache);sharesRules=[...c.seen.keys()].some(p=>rules.has(p))}else{status='in-repo-not-imported'}}
  else if(relPath){status='in-repo-reference'}else if(e.copy&&isCode(e.copy.path)&&reach.has(e.copy.path)){status='connected-copy';via=chain(e.copy.path);const c=await closure(e.copy.path,read,cache);sharesRules=[...c.seen.keys()].some(p=>rules.has(p))}else status='not-connected';
  const bn=(e.origin.path.split('/').pop()||'');if(status!=='connected'&&status!=='connected-copy'&&bn){const hit=texts.filter(([p,t])=>p!==relPath&&t.includes(bn)).map(([p])=>p);if(hit.length){referencedBy=hit.slice(0,4);if(status==='not-connected')status='referenced'}}
  out[e.id]={status,via,sharesRules,referencedBy}}
 return {entryPoints:[...entries],reachableModules:reach.size,byId:out};
}
export function summarize(registry,conn=registry.connections){const by={},fam={};for(const e of registry.entries){const c=conn?.byId?.[e.id]?.status||'unknown';by[c]=(by[c]||0)+1;const f=fam[e.family]||(fam[e.family]={total:0,connected:0,items:[]});f.total++;if(c==='connected'||c==='connected-copy')f.connected++;f.items.push({id:e.id,name:e.name,status:c,sharesRules:conn?.byId?.[e.id]?.sharesRules??null,sha256:e.sha256.slice(0,12),bytes:e.bytes})}
 return {entries:registry.entries.length,byStatus:by,families:fam,uncurated:registry.uncurated.length,reachableModules:conn?.reachableModules??null}}
export async function loadRegistry(loadJson,path='integrated/swarm/registry.json'){return loadJson(path)}
