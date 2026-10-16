// Build tool (Node built-ins only; not shipped logic). Writes registry.json: ONE inventory of every swarm found in her repos, zips and HF downloads.
// Originals are never touched: out-of-repo files are listed BY REFERENCE (path, sha256, size). Small text originals are also copied byte-for-byte into swarm/originals/ with the hash re-verified.
// Run on the box that holds her files:  node integrated/swarm/registry.build.mjs [/workspace]
import {readFileSync,writeFileSync,readdirSync,statSync,mkdirSync,existsSync,copyFileSync} from 'node:fs';
import {join,dirname,basename,extname,relative} from 'node:path';import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {readZip,isZip} from '../../zip-lite.mjs';
const HERE=dirname(fileURLToPath(import.meta.url)),REPO=join(HERE,'..','..'),W=process.argv[2]||'/workspace';
const sha=b=>createHash('sha256').update(b).digest('hex'),rd=p=>new Uint8Array(readFileSync(p));
const TEXT=new Set(['.mjs','.js','.py','.md','.json','.txt','.ts','.tsx','.jsx','.html','.cjs','.sh']),COPY_MAX=70000;
// ---- families (names are hers: the file names, the zip names, her doc headings) ----
// each: id, family, name, why (quoted/derived from the file itself), paths (workspace-relative or repo-relative with 'repo:' prefix)
const M=[
 ['swarm-os-v9','swarm-os','SWARM-OS v9 (PSO+GA+ABC hybrid, Python)','swarm-os-v14/x/v9/swarm_os_full.py','Header of integrated/abc.mjs cites V9 swarm_os_full.py lines 3, 15 ("PSO+GA+ABC Hybrid"), 89, 132-155'],
 ['swarm-os-v6','swarm-os','SWARM-OS v6 (Python)','swarm-os-v14/x/v6/swarm_os.py','member of swarm_os_v14_complete-1.zip'],
 ['swarm-os-v10','swarm-os','SWARM-OS v10 Bantu (Python)','swarm-os-v14/x/v10/swarm_os_bantu.py','member of swarm_os_v14_complete-1.zip; also abc.mjs cites v10 lines 13, 114, 178'],
 ['swarm-os-v12','swarm-os','SWARM-OS v12 cosmic engine (Python)','swarm-os-v14/x/v12/cosmic_engine.py','member of swarm_os_v14_complete-1.zip'],
 ['swarm-os-v14-zip','swarm-os','SWARM-OS v14 complete (zip: v6, v9, v10, v12, v13, v14, pictures)','swarm-os-v14/swarm_os_v14_complete-1.zip','the zip itself'],
 ['swarm-os-v14-run','swarm-os','SWARM-OS v14 run output','swarm-os-v14/runs/v6out/swarm_result.json','result of running her v6'],
 ['swarm-os-pipe-v6','swarm-os','swarm_os.py (pipe copy)','pipe/all/swarm_os.py',''],
 ['swarm-os-pipe-v9','swarm-os','swarm_os_full.py (pipe copy)','pipe/all/swarm_os_full.py',''],
 ['swarm-os-pipe-v10','swarm-os','swarm_os_bantu.py (pipe copy)','pipe/all/swarm_os_bantu.py',''],
 ['swarm-os-doc','swarm-os','SWARM-OS foundational architecture (doc)','repo:docs/SWARM-OS-FOUNDATIONAL-ARCHITECTURE.md','in this repo'],
 ['swarm-state-space-doc','swarm-os','Swarm state space (doc)','repo:docs/SWARM-STATE-SPACE.md','in this repo'],
 ['swarm-finisher-doc','swarm-os','Swarm finisher (doc)','repo:docs/SWARM-FINISHER.md','in this repo'],
 ['swarm-pipeline-report','swarm-os','swarm pipeline from her files (report)','bundle/reports/swarm-pipeline-from-her-files.md',''],
 ['integrated-swarm','integrated','Integrated swarm (ingest, pipeline, swarm, fold-up, wheels, history)','repo:integrated/integrated-swarm.mjs','entry of the morphing swarm on this branch'],
 ['swarm-loop','integrated','Self-assembly loop (heartbeat, workspace, audit log, protected originals)','repo:integrated/swarm-loop.mjs',''],
 ['any-file-swarm','file-swarm','First swarm: any file -> registration -> state-space placement -> execution','repo:any-file-swarm.mjs',''],
 ['registration-swarm','file-swarm','Registration swarm (packs 3,2,1)','repo:registration-swarm.mjs',''],
 ['swarm-component','file-swarm','Swarm component (register, finisher, personalize building block)','repo:swarm-component.mjs',''],
 ['swarm-nsnf','file-swarm','NSNF swarm pair (genotype/phenotype units, event log)','repo:swarm-nsnf.mjs',''],
 ['app-finisher','file-swarm','App finisher (assess against checklist, fill gaps additively)','repo:app-finisher.mjs',''],
 ['swarm-fit','pso','PSO numeric search (fitParticles)','repo:swarm-fit.mjs',''],
 ['abc','abc','ABC artificial bee colony (V9 role split 0.4/0.4/0.2)','repo:integrated/abc.mjs',''],
 ['population','ga','Primitive population (GA: diverse primitives, one ontology)','repo:integrated/population.mjs',''],
 ['trident-js','trident','Trident in pure JavaScript (reads trident_syntia.pt)','repo:integrated/trident-js.mjs',''],
 ['trident-model','trident','trident_syntia.pt (her weights)','repo:models/trident_syntia.pt',''],
 ['trinity-engine','trident','Trinity engine (JS reference copy)','repo:integrated/reference/hf-synthia/trinity-engine.js.txt','reference copy of HF stellarproximology/Synthia'],
 ['trinity-system','trident','Trinity system (JS reference copy)','repo:integrated/reference/hf-synthia/trinity-system.js.txt',''],
 ['trinity-knowledge-loader','trident','Trinity knowledge loader (JS reference copy)','repo:integrated/reference/hf-synthia/trinity-knowledge-loader.js.txt',''],
 ['trident-onnx','trident','trident_single.onnx (HF download)','hf-all/trident_single.onnx',''],
 ['hdgnn','trident','hdgnn (HF; address/trust resolver, does not talk)','hf-all/hdgnn/README.md',''],
 ['trinity-orchestrator','trident','trinity-orchestrator.js (The-seed repo)','swarm-search/all/The-seed/trinity-orchestrator.js',''],
 ['trinity-field-py','trident','trinity_field.py (Dual-engine- repo)','swarm-search/all/Dual-engine-/trinity_field.py',''],
 ['trinity-resonance-engine','trident','trinity-resonance-engine.ts (Synthia- repo)','swarm-search/all/Synthia-/trinity-resonance-engine.ts',''],
 ['morph-search','morph-search','morph-search.mjs: local bounded ABC / PSO / GA search (JS)','oct7src/computer/runtime/morph-search.mjs','its own header: "Local bounded ABC / PSO / GA search. Lower finite fitness is better."'],
 ['morph-search-test','morph-search','morph-search test','oct7src/computer/tests/morph-search.test.mjs',''],
 ['realm-swarm','morph-search','realm-swarm.mjs: five-field projections per unit, local mesh (JS)','oct7src/computer/runtime/realm-swarm.mjs','uses StateSpaceEngine + DIMENSION_QUALITIES; not in this repo'],
 ['realm-swarm-test','morph-search','realm-swarm test','oct7src/computer/tests/realm-swarm.test.mjs',''],
 ['emergent-edge-resolver','morph-search','EmergentEdgeResolver (36 edge types)','oct7src/computer/donors/recovered/you-n-i-verse-corrected/emergent-edge-resolver.ported.mjs','her emergent-edge code; not in this repo'],
 ['state-space-engine-v2','morph-search','state-space-engine-v2 (ported)','oct7src/computer/donors/recovered/you-n-i-verse-corrected/state-space-engine-v2.ported.mjs',''],
 ['dimension-qualities','morph-search','DIMENSION_QUALITIES (Black Book p126 statements per dimension)','oct7src/computer/donors/dream-habitat/dimension-qualities.mjs','vendored byte-for-byte into vendor/mesh/'],
 ['morphing-particle-swarm','morph-search','morphing-particle-swarm-main (zip)','zips/_drive/morphing-particle-swarm-main.zip',''],
 ['morphing-substrate','morph-search','MorphingSubstrate.js (Resonance-morph-os repo)','swarm-search/all/Resonance-morph-os/src/lib/MorphingSubstrate.js',''],
 ['morphing-net-py','morph-search','morphing_net.py','swarm-search/all/We-as-in-me-and-mCP-with-kimi-kinda-chat-nearly-enough-Joe-did-it/morphing_net.py',''],
 ['morphing-net-exec','morph-search','MorphingNet_Execution_v1.ts (-SynthAi repo)','swarm-search/all/-SynthAi/Src/Engine/MorphingNet_Execution_v1.ts',''],
 ['neural-morphing-engine','morph-search','neuralMorphingEngine.ts (SynthAi-Proto-mobile)','swarm-search/all/SynthAi-Proto-mobile/neuralMorphingEngine.ts',''],
 ['four-llm-swarmdoc','four-llm','Swarm architecture doc: four active LLM positions (Movement, Being, Evolution, Design), Space emerges as the swarm; clustering and mesh formation (sections 4-7)','drive/swarmdoc.md','her design doc; sections 4-7 are the spec for the mesh. Exact dimension-to-model mapping is stated OPEN in the doc itself (line 1342)'],
 ['modelmaker-app','four-llm','Model Maker repair app (Stellarproximology-lab)','chassis-work/src/Stellarproximology-lab/Modelmaker-repair/app.py','builds small local models'],
 ['modelmaker-readme','four-llm','Model Maker repair README','chassis-work/src/Stellarproximology-lab/Modelmaker-repair/README.md',''],
 ['modelmaker-movement-run','four-llm','Model Maker run: movement model','chassis-work/modelmaker-run/run_movement.py','only the movement model was run; the other three dimensions were not'],
 ['modelmaker-movement-build','four-llm','Model Maker build record (movement)','chassis-work/modelmaker-run/movement-model/modelmaker_build.json',''],
 ['pure-synthia-swarm-body','pure-synthia','swarmBody.mjs (Pure Synthia swarm body)','swarm-search/all/Back-up-/vendor/pure-synthia-v0.4.0/src/synthia/swarm/swarmBody.mjs',''],
 ['pure-synthia-architecture','pure-synthia','SWARM-ARCHITECTURE.json','swarm-search/all/Back-up-/vendor/pure-synthia-v0.4.0/SWARM-ARCHITECTURE.json',''],
 ['pure-synthia-start','pure-synthia','START-SWARM.mjs','swarm-search/all/Back-up-/vendor/pure-synthia-v0.4.0/START-SWARM.mjs',''],
 ['pure-synthia-verify','pure-synthia','VERIFY-SWARM.mjs','swarm-search/all/Back-up-/vendor/pure-synthia-v0.4.0/VERIFY-SWARM.mjs',''],
 ['pure-synthia-selfhosted-hand','pure-synthia','selfhostedSwarmHand.mjs','swarm-search/all/Back-up-/vendor/pure-synthia-v0.4.0/host/selfhostedSwarmHand.mjs',''],
 ['pure-synthia-yni-hands','pure-synthia','yniSwarmHands.mjs','swarm-search/all/Back-up-/vendor/pure-synthia-v0.4.0/host/yniSwarmHands.mjs',''],
 ['pure-synthia-body-zip','pure-synthia','Pure-Synthia-Swarm-Body-v0.3.0-PHYSIOLOGY.zip','zips/_drive/Pure-Synthia-Swarm-Body-v0.3.0-PHYSIOLOGY.zip',''],
 ['dream-swarm-zip','dream-swarm','Dream-Swarm-System-v0.1.0.zip','zips/_drive/Dream-Swarm-System-v0.1.0.zip',''],
 ['swarm-reality-trial-zip','swarm-reality','Swarm-Reality-Trial-v0.1.0.zip','zips/_drive/Swarm-Reality-Trial-v0.1.0.zip',''],
 ['resonance-swarm-reality-zip','swarm-reality','Resonance-Network-Swarm-Reality-INTEGRATED-TRIAL-2026-10-02.zip','chassis-work/candidates/Resonance-Network-Swarm-Reality-INTEGRATED-TRIAL-2026-10-02.zip',''],
 ['bantu-321-swarm-bridge-zip','swarm-reality','BANTU-321-Swarm-Bridge-v0.2.zip','extra/BANTU-321-Swarm-Bridge-v0.2.zip',''],
 ['contact-automata-swarm-body','swarm-reality','contact-automata swarmBody.mjs','contact-automata/src/swarmBody.mjs',''],
 ['semantic-dna-swarm','swarm-reality','semantic-dna-swarm.js','drive-pieces/content-4/synthia-semantic-state-integration/semantic-dna-swarm.js',''],
 ['grok-swarm-directive','swarm-reality','GROK_SWARM_DIRECTIVE.md (Resonance North Star)','chassis-work/candidates/north-star/Resonance-North-Star-Production/GROK_SWARM_DIRECTIVE.md',''],
 ['alphabet-swarm-control','swarm-reality','11-alphabet-swarm-control.md (open question)','research/open-questions/11-alphabet-swarm-control.md',''],
 ['swarm-repo-readme','swarm-reality','Swarm repo (README + Index.html)','swarm-search/Swarm/README.md',''],
 ['the-horde','swarm-reality','The-horde repo (Index.html)','swarm-search/The-horde/Index.html',''],
 ['morphgnn-server','morph-search','morphgnn_server.py','repo-root:morphgnn_server.py','in the Simlish repo root, outside you-n-i-verse']
];
const SCAN_ROOTS=['drive','drive-pieces','extra','contact-automata','zips','SynthAi','chassis-work','pipe','oct7src','swarm-os-v14','research','swarm-search','hf-all','Synthia-State-Space-Foundation'],SKIP=/(^|\/)(node_modules|\.git|mm-venv\d*|venv|__pycache__|scratch)(\/|$)/,NAME=/swarm|morph|trinity|particle|boids|colony|modelmaker|emergent-edge|genetic|horde/i,MEDIA=/\.(png|jpe?g|gif|webp|mp3|wav|mp4|h|pyc|so|dylib)$/i;
const walk=(d,out=[])=>{let ents;try{ents=readdirSync(d,{withFileTypes:true})}catch{return out}for(const e of ents){const p=join(d,e.name);if(SKIP.test(p))continue;if(e.isDirectory())walk(p,out);else if(e.isFile())out.push(p)}return out};
const resolve=ref=>ref.startsWith('repo:')?join(REPO,ref.slice(5)):ref.startsWith('repo-root:')?join(REPO,'..',ref.slice(10)):join(W,ref);
const entries=[],bySha=new Map(),missing=[];
for(const [id,family,name,ref,why] of M){const p=resolve(ref);if(!existsSync(p)){missing.push({id,ref});continue}const b=rd(p),h=sha(b),inRepo=ref.startsWith('repo');
 entries.push({id,family,name,why:why||null,origin:{kind:inRepo?'repo':'workspace',path:inRepo?'you-n-i-verse/'+(ref.startsWith('repo:')?ref.slice(5):'../'+ref.slice(10)).replace('../','')  :ref,repo:inRepo?'https://github.com/justappgrabbin/Simlish':null},sha256:h,bytes:b.length,ext:extname(p).slice(1)||null});}
// fix repo path form
for(const e of entries)if(e.origin.kind==='repo'){e.origin.path=e.origin.path.replace(/^you-n-i-verse\/morphgnn/,'morphgnn')}
// zip members of the swarm zips (names + hashes, by reference)
const ZIPMEM=/swarm|morph|pso|abc|boid|trinity|particle|genetic|\.py$|SWARM/i;
for(const e of entries){if(e.ext!=='zip')continue;const p=e.origin.kind==='repo'?join(REPO,e.origin.path.replace(/^you-n-i-verse\//,'')):join(W,e.origin.path);let list=[];try{const b=rd(p);if(isZip(b)){const z=await readZip(b);e.zipEntries=z.length;list=z.filter(m=>ZIPMEM.test(m.path)&&m.data.length>0&&!MEDIA.test(m.path)).slice(0,40).map(m=>({path:m.path,bytes:m.data.length,sha256:sha(m.data)}))}}catch(x){e.zipError=String(x.message)}if(list.length)e.members=list}
// non-destructive copies of small text originals (byte-identical, hash verified)
const OUT=join(HERE,'originals');mkdirSync(OUT,{recursive:true});let copied=0;
for(const e of entries){if(e.origin.kind==='repo'||!TEXT.has('.'+e.ext)||e.bytes>COPY_MAX)continue;const src=join(W,e.origin.path),dst=join('originals',e.family,e.sha256.slice(0,8)+'-'+basename(src)+(/\.test\.m?js$/.test(src)?'.txt':''));mkdirSync(join(HERE,dirname(dst)),{recursive:true});copyFileSync(src,join(HERE,dst));if(sha(rd(join(HERE,dst)))!==e.sha256)throw Error('copy hash mismatch '+e.id);e.copy={path:'integrated/swarm/'+dst,sha256:e.sha256};copied++}
// everything else with a swarm-ish name that was not curated: listed so nothing found is dropped
const known=new Set(entries.map(e=>e.sha256)),extra=new Map();
for(const r of SCAN_ROOTS){for(const p of walk(join(W,r))){const n=basename(p);if(!NAME.test(n)||MEDIA.test(n))continue;let st;try{st=statSync(p)}catch{continue}if(st.size>400e6)continue;const b=rd(p),h=sha(b);if(known.has(h))continue;const rec=extra.get(h)||{sha256:h,bytes:b.length,name:n,paths:[]};if(rec.paths.length<6)rec.paths.push(relative(W,p));rec.count=(rec.count||0)+1;extra.set(h,rec)}}
const uncurated=[...extra.values()].sort((a,b)=>a.name.localeCompare(b.name)||a.sha256.localeCompare(b.sha256));
const reg={schema:'synthia-swarm-registry/1',note:'ONE area for every swarm found. Out-of-repo items are references with sha256; copies under swarm/originals are byte-identical. Nothing merged, edited or deleted. connected/shares-rules are computed by registry.mjs from the repo import graph, not stored here.',builtFrom:{root:W,node:process.version},entries,uncurated,missingAtBuild:missing};
writeFileSync(join(HERE,'registry.json'),JSON.stringify(reg,null,1));
console.log({entries:entries.length,copied,uncurated:uncurated.length,missing:missing.length,missingList:missing.map(m=>m.id)});
// connections: computed from the repo's own import graph (registry.mjs) and stored as a snapshot; the test recomputes it and must match
const {computeConnections}=await import('./registry.mjs');
const read=async p=>{try{return readFileSync(join(REPO,p),'utf8')}catch{return null}};
reg.connections=await computeConnections(reg,read);
writeFileSync(join(HERE,'registry.json'),JSON.stringify(reg,null,1));
const st={};for(const v of Object.values(reg.connections.byId))st[v.status]=(st[v.status]||0)+1;console.log('connections',st,'reachable modules',reg.connections.reachableModules);
