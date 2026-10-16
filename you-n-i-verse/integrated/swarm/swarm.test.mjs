import test from 'node:test';import assert from 'node:assert/strict';
import '../../vendor/kernel.js';
import {readFileSync,existsSync,readdirSync,statSync} from 'node:fs';import {join,dirname} from 'node:path';import {fileURLToPath} from 'node:url';import {createHash} from 'node:crypto';
import {EmergentMesh,potentialEdges,verifyEdges,RELATIONS,REL} from './mesh.mjs';
import {computeConnections,summarize} from './registry.mjs';
import {createSwarmArea,MEMBER_LOADERS} from './index.mjs';
import {createAppServer,TOOLS} from '../app-server.mjs';
import {StateSpace as MeshSpace} from '../../vendor/mesh/state-space/mesh-state-space.js';
import {CHANNELS} from '../chart-engine.mjs';import {DIMENSIONS,macroToGate,gateToMacro} from '../../state-space.mjs';
const ROOT=join(dirname(fileURLToPath(import.meta.url)),'..','..'),REG=JSON.parse(readFileSync(join(ROOT,'integrated/swarm/registry.json'),'utf8')),sha=b=>createHash('sha256').update(b).digest('hex');
const loadBytes=async p=>new Uint8Array(readFileSync(join(ROOT,p))),loadJson=async p=>JSON.parse(readFileSync(join(ROOT,p),'utf8')),te=new TextEncoder();
const file=p=>new Uint8Array(readFileSync(join(ROOT,p)));
const PRIM=JSON.parse(readFileSync(join(ROOT,'integrated/primitive-functions.json'),'utf8'));
// ---------- registry ----------
test('registry: every swarm she named is present, ids unique, nothing dropped',()=>{const ids=REG.entries.map(e=>e.id);assert.equal(new Set(ids).size,ids.length);
 for(const must of ['swarm-os-v9','swarm-os-v14-zip','integrated-swarm','four-llm-swarmdoc','modelmaker-app','any-file-swarm','swarm-fit','abc','population','trident-js','trinity-engine','morph-search','swarm-loop','app-finisher','pure-synthia-body-zip','dream-swarm-zip'])assert.ok(ids.includes(must),must);
 for(const f of ['swarm-os','integrated','file-swarm','pso','abc','ga','trident','morph-search','four-llm'])assert.ok(REG.entries.some(e=>e.family===f),f);assert.equal(REG.missingAtBuild.length,0);assert.ok(REG.uncurated.length>100)});
test('registry: hashes are real. In-repo files match, copies are byte-identical to the origin, out-of-repo originals still match when present',()=>{let repo=0,copies=0,ext=0;
 for(const e of REG.entries){if(e.origin.kind==='repo'){const p=e.origin.path==='morphgnn_server.py'?join(ROOT,'..',e.origin.path):join(ROOT,e.origin.path.replace(/^you-n-i-verse\//,''));if(existsSync(p)){assert.equal(sha(readFileSync(p)),e.sha256,e.id);repo++}}
  else{const p=join('/workspace',e.origin.path);if(existsSync(p)){assert.equal(sha(readFileSync(p)),e.sha256,'origin untouched: '+e.id);ext++}}
  if(e.copy){const c=readFileSync(join(ROOT,e.copy.path.replace(/^integrated/,'integrated')));assert.equal(sha(c),e.sha256,'copy '+e.id);assert.equal(e.copy.sha256,e.sha256);copies++}}
 assert.ok(repo>=15&&copies>=30,`repo ${repo} copies ${copies} external ${ext}`)});
test('registry: connected flags are computed from the import graph (snapshot equals recomputation) and the known gaps stay visible',async()=>{
 const read=async p=>{try{return readFileSync(join(ROOT,p),'utf8')}catch{return null}};const now=await computeConnections(REG,read);assert.deepEqual(now,REG.connections);
 const c=REG.connections.byId;for(const id of ['integrated-swarm','swarm-loop','any-file-swarm','app-finisher','swarm-fit','abc','population','trident-js','swarm-nsnf'])assert.equal(c[id].status,'connected',id);
 assert.equal(c['morph-search'].status,'connected-copy');for(const id of ['swarm-os-v9','swarm-os-v14-zip','four-llm-swarmdoc','modelmaker-app','emergent-edge-resolver','realm-swarm','pure-synthia-body-zip','dream-swarm-zip'])assert.ok(!['connected','connected-copy'].includes(c[id].status),id+' is not connected: '+c[id].status);
 const s=summarize(REG);assert.equal(Object.values(s.byStatus).reduce((a,b)=>a+b,0),REG.entries.length);assert.equal(c.population.sharesRules,true);assert.equal(c['swarm-fit'].sharesRules,false,'PSO search is numeric only: it does not use the shared primitive rules')});
test('area: every connected member really loads and exposes its named exports',async()=>{const area=await createSwarmArea({loadJson,primitives:PRIM});for(const id of Object.keys(MEMBER_LOADERS)){const m=await area.loadMember(id);assert.ok(m);assert.ok(area.members().find(x=>x.id===id).loadable)}
 await assert.rejects(()=>area.loadMember('swarm-os-v9'));
 const ms=await area.loadMember('morph-search');let calls=0;const s=ms.createMorphSearch({bounds:[[0,1],[0,1]],evaluate:v=>{calls++;return (v[0]-.3)**2+(v[1]-.6)**2},population:8,random:(()=>{let x=7;return()=>(x=(x*16807)%2147483647)/2147483647})()});assert.ok(s&&calls>=0)});
// ---------- mesh: analytic and independent verification ----------
const popc=x=>x.toString(2).split('1').length-1;
test('mesh: the possible edge count equals an independent brute-force count over all 320 nodes (analytic: 960 + 3840 Hamming-1, 640 identity, 36 channels x 25...)',()=>{const pot=potentialEdges();
 const rev=x=>{let r=0;for(let i=0;i<6;i++)if(x>>i&1)r|=1<<(5-i);return r},ex=x=>((x&7)<<3)|(x>>3),nuc=x=>{let r=0;[2,3,4,3,4,5].forEach((l,i)=>{r|=((x>>(l-1))&1)<<i});return r};
 const chan=new Set();for(const [a,b] of CHANNELS){chan.add(gateToMacro(a)+','+gateToMacro(b));chan.add(gateToMacro(b)+','+gateToMacro(a))}
 const cnt=Object.fromEntries(RELATIONS.map(r=>[r,0]));
 for(let i=0;i<320;i++)for(let j=0;j<320;j++){const a=i&63,b=j&63,sameDim=(i>>6)===(j>>6);if(i===j)continue;
  if(i<j){if(a===b&&!sameDim)cnt.identity++;if(popc(a^b)===1)cnt['hamming-1']++;if((a^b)===63)cnt.complement++;if(a!==b&&rev(a)===b)cnt.reverse++;if(a!==b&&ex(a)===b)cnt['exchange-trigrams']++;if(chan.has(a+','+b))cnt.channel++}
  if(a!==b&&nuc(a)===b)cnt.nuclear++}
 for(const r of RELATIONS)assert.equal(pot.byRelation[r].total,cnt[r],r);assert.equal(pot.byRelation['hamming-1'].withinDimension,960);assert.equal(pot.byRelation['hamming-1'].crossDimension,3840);assert.equal(pot.byRelation.identity.total,640);assert.equal(pot.byRelation.channel.total,36*5+36*2*10);assert.equal(pot.byRelation.complement.withinDimension,5*32);
 assert.equal(pot.total,Object.values(cnt).reduce((a,b)=>a+b,0))});
test('mesh: her own mesh-state-space.js layers give 192 Hamming-1 pairs per dimension and the same gate in all five dimensions',()=>{const sp=new MeshSpace();let total=0;
 for(const d of DIMENSIONS){const ns=sp.dimensions[d].nodes;assert.equal(ns.length,64);let n=0;for(let i=0;i<64;i++)for(let j=i+1;j<64;j++){let diff=0;for(let k=0;k<6;k++)diff+=ns[i].binary[k]!==ns[j].binary[k]?1:0;if(diff===1)n++}assert.equal(n,192,d);total+=n}
 assert.equal(total,960);const a=sp.address(13);assert.deepEqual(Object.keys(a),DIMENSIONS);assert.equal(new Set(DIMENSIONS.map(d=>a[d].gate)).size,1);
 const orders=DIMENSIONS.map(d=>sp.dimensions[d].nodes.map(n=>n.binary.join('')).join());assert.equal(new Set(orders).size,5,'five layers, five different arrangements of the 64 hexagrams (binary order)')});
test('mesh: her EmergentEdgeResolver (original copy) and her field_core channel list DISAGREE; the mesh uses field_core CHANNELS and the difference is recorded, not hidden',async()=>{
 const m=await import('./originals/morph-search/13dca3de-emergent-edge-resolver.ported.mjs'),k=a=>[...a].sort((x,y)=>x-y).join('-');
 const hers=new Set(m.EDGE_FUNCTIONS.map(f=>k([f.gateA+1,f.gateB+1]))),mine=new Set(CHANNELS.map(k));assert.equal(m.EDGE_FUNCTIONS.length,37);assert.equal(hers.size,33,'37 entries, 33 distinct pairs (TRUTH and IMAGINATION repeat LOGIC and ABSTRACTION reversed; DISCOVERY 46-29 and RECOGNITION 29-30 are not standard channel pairs)');
 assert.deepEqual([...hers].filter(x=>!mine.has(x)),['29-30']);assert.deepEqual([...mine].filter(x=>!hers.has(x)).sort(),['10-34','10-57','20-34','20-57'])});
// ---------- mesh: emergence ----------
const corpus=()=>{const out=[];for(const f of readdirSync(join(ROOT,'integrated')))if(f.endsWith('.mjs')&&!f.includes('test'))out.push({path:'integrated/'+f,bytes:file('integrated/'+f)});return out};
test('mesh: emerges only from interaction. Nothing in, or no interaction window, means no edges',()=>{const m=new EmergentMesh();assert.equal(m.edges.size,0);m.weave(new Uint8Array(0));assert.equal(m.edges.size,0);m.weave(new Uint8Array([200]),{name:'one'});assert.equal(m.edges.size,0);
 const z=new EmergentMesh({window:0});z.weave(file('integrated/chart-engine.mjs'));assert.equal(z.edges.size,0,'window 0: units never meet');assert.ok(z.nodes.size>=60);
 const w=new EmergentMesh();w.weave(file('integrated/chart-engine.mjs'));assert.ok(w.edges.size>1000)});
test('mesh: every edge it grew is a true instance of its relation (independent bit-operation re-check), with source -> relation -> target and the five record fields',async()=>{
 const m=new EmergentMesh({primitives:PRIM}),a=file('integrated/chart-engine.mjs');assert.ok(a.length>3072);m.weave(a,{name:'chart-engine.mjs'});const v=verifyEdges(m);assert.equal(v.bad.length,0);assert.equal(v.checked,m.edges.size);
 const rels=new Set();for(const e of m.edges.values()){const r=m.edge(e);rels.add(r.relation);for(const k of ['source','relation','target','address','dimension','transformation','context','provenance'])assert.ok(k in r,k);assert.ok(r.context.contents.length>=1&&r.provenance.rule&&r.weight>0)}
 assert.deepEqual([...rels].sort(),[...RELATIONS].sort(),'all seven relations emerged from one 12 KB file');
 const ham=[...m.edges.values()].find(e=>RELATIONS[e.r]==='hamming-1'),h=m.edge(ham);assert.equal(h.transformation.flippedLines.length,1);const idn=m.edge([...m.edges.values()].find(e=>RELATIONS[e.r]==='identity'));assert.notEqual(idn.dimension[0],idn.dimension[1]);assert.equal(idn.address.gates[0],idn.address.gates[1])});
test('mesh: nodes carry differentiated dimensional qualities (the same gate is a different node in each dimension)',()=>{const m=new EmergentMesh({primitives:PRIM});m.weave(file('integrated/chart-engine.mjs'));
 for(const g of [1,13,57]){const per=DIMENSIONS.map(d=>m.node(DIMENSIONS.indexOf(d)*64+gateToMacro(g)));assert.equal(new Set(per.map(n=>n.id)).size,5);assert.equal(new Set(per.map(n=>n.layerRole)).size,5);assert.equal(new Set(per.map(n=>n.macroChain[0])).size,5);assert.equal(new Set(per.map(n=>n.micro.name)).size,5);assert.ok(new Set(per.map(n=>n.layerPosition)).size>=3);assert.ok(per.every(n=>n.gate===g&&n.trigrams.upper&&n.gateForm))}
 const q=m.node(2*64+gateToMacro(13));const slots=DIMENSIONS.map((d,i)=>m.node(i*64+gateToMacro(13)).layerSlot);assert.deepEqual(slots,[14,13,13,13,7],'her Movement and Space layers seat gate 13 under other slot labels');assert.equal(q.micro.name,'The Body');assert.equal(q.layerRole,'state');assert.equal(q.sequence,'fuxi')});
test('mesh: different content grows a different mesh, repetition strengthens, adapt() forgets what is not re-made',()=>{
 const A=file('integrated/chart-engine.mjs'),B=file('integrated/wheels.mjs');assert.ok(B.length>3072);
 const ma=new EmergentMesh(),mb=new EmergentMesh();ma.weave(A);mb.weave(B);const wa=new Map([...ma.edges].map(([k,e])=>[k,e.w])),wb=new Map([...mb.edges].map(([k,e])=>[k,e.w]));let diff=0;for(const k of new Set([...wa.keys(),...wb.keys()]))if(wa.get(k)!==wb.get(k))diff++;assert.ok(diff>200,'weights differ on '+diff+' edges');
 ma.weave(A);for(const [k,e] of ma.edges)assert.equal(e.w,wa.get(k)*2,'second pass doubles every weight');
 const m=new EmergentMesh();m.weave(A);const before=m.edges.size;const r=m.adapt({decay:1,floor:0.5});assert.equal(m.edges.size,0);assert.equal(r.dropped,before);m.weave(B);assert.deepEqual([...m.edges.keys()].sort(),[...mb.edges.keys()].sort());
 const c=new EmergentMesh();c.weave(A);c.weave(B);assert.equal(c.contents.size,2);assert.ok(c.edges.size>=Math.max(ma.edges.size,mb.edges.size)&&c.edges.size<=ma.edges.size+mb.edges.size)});
test('mesh: three unrelated inputs (the repo\'s own modules, a binary blob, a zip-shaped document) all grow valid meshes; coverage and clusters reported',async()=>{
 const area=await createSwarmArea({loadJson,primitives:PRIM}),res={};
 const blob=new Uint8Array(40000).map((_,i)=>(i*i+(i>>3)+(i*31>>5))&255),doc=te.encode(JSON.stringify(PRIM.primitives.slice(0,60)));
 for(const [name,input] of [['repo-modules',{name:'repo',files:corpus()}],['blob',{name:'blob.bin',bytes:blob}],['json-doc',{name:'doc.json',bytes:doc}]]){const m=new (await import('./mesh.mjs')).EmergentMesh({primitives:PRIM});const fresh=await createSwarmArea({loadJson,primitives:PRIM});const r=await fresh.weave(input);assert.equal(verifyEdges(fresh.mesh).bad.length,0,name);assert.ok(r.mesh.edges>0&&r.mesh.nodes>=60,name);res[name]={nodes:r.mesh.nodes,edges:r.mesh.edges,saturation:r.mesh.saturation,clusters:r.mesh.clusters.count,mixed:r.mesh.clusters.mixedDimension,cross:r.mesh.crossDimensionEdges,within:r.mesh.withinDimensionEdges}}
 assert.ok(res['repo-modules'].saturation>0.4,JSON.stringify(res));assert.ok(res['repo-modules'].mixed>=1);assert.notEqual(res.blob.edges,res['json-doc'].edges);
 for(const r of Object.values(res))assert.ok(r.cross>r.within,'edges mostly connect different dimensions: '+JSON.stringify(r));console.log('MESH_RESULTS '+JSON.stringify(res))});
// ---------- through the app ----------
test('app server: swarm.registry, swarm.weave and content.measure are tools; the registry answer states what is and is not connected',async()=>{
 const srv=await createAppServer({loadJson,loadBytes,loadModel:false,schedule:()=>0,cancel:()=>{}});for(const n of ['swarm.registry','swarm.weave','content.measure'])assert.ok(TOOLS.some(t=>t.name===n),n);
 const r=await srv.handle('POST','/api/tool/swarm.registry',{});assert.equal(r.ok,true);assert.ok(r.body.inventory.byStatus.connected>=10);assert.ok(r.body.members.find(m=>m.id==='abc').status==='connected');assert.ok(r.body.members.find(m=>m.id==='swarm-os-v9').status!=='connected');
 const w=await srv.handle('POST','/api/tool/swarm.weave',{name:'x.js',text:'const a=1;\n'.repeat(900)});assert.equal(w.ok,true);assert.ok(w.body.mesh.edges>0);
 const z=await srv.handle('POST','/api/tool/swarm.weave',{reset:true,files:[{path:'a.txt',text:'hello world '.repeat(400)},{path:'b.txt',text:'second file '.repeat(400)}]});assert.equal(z.body.mesh.contents.length,2);
 const m=await srv.handle('POST','/api/tool/content.measure',{name:'t.txt',text:'hello '.repeat(900)});assert.equal(m.ok,true);assert.equal(m.body.totals.sixBitStates,Math.ceil(5400*8/6));
 const b64=Buffer.from('zip-ish bytes '.repeat(300)).toString('base64'),mb=await srv.handle('POST','/api/tool/content.measure',{name:'b.bin',base64:b64});assert.equal(mb.ok,true,'base64 input is accepted (regression: a wrong key name in need() rejected it)');assert.equal(mb.body.input.bytes,4200);
 const e=await srv.handle('POST','/api/tool/content.measure',{name:'none'});assert.equal(e.ok,false)});
