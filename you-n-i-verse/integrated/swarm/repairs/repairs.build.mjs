// Build tool (box only; her trees are not in the GitHub repo). Repairs her failing material WITHOUT touching it:
// each repair is a working copy under /workspace/scratch/repairs/<id>; only the missing piece is added, taken from HER OWN files; tests are her own scripts.
// Writes repairs.json (sources, sha256 of every added file, exact commands, results).  Run:  node integrated/swarm/repairs/repairs.build.mjs [/workspace]
import {execFileSync,spawnSync} from 'node:child_process';import {readFileSync,writeFileSync,existsSync,readdirSync,statSync,rmSync,mkdirSync} from 'node:fs';import {join,dirname,relative} from 'node:path';import {fileURLToPath} from 'node:url';import {createHash} from 'node:crypto';
const HERE=dirname(fileURLToPath(import.meta.url)),W=process.argv[2]||'/workspace',OUT=join(W,'scratch','repairs'),NODE22=process.env.NODE22||join(W,'node22/bin/node');
const sha=p=>createHash('sha256').update(readFileSync(p)).digest('hex');
const walk=d=>readdirSync(d,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(join(d,e.name)):[join(d,e.name)]);
const treeHash=d=>{const h=createHash('sha256');for(const f of walk(d).sort()){h.update(relative(d,f));h.update(sha(f))}return h.digest('hex')};
const run=(cmd,args,cwd)=>{const t0=Date.now(),r=spawnSync(cmd,args,{cwd,encoding:'utf8',timeout:600000,maxBuffer:64<<20});const o=(r.stdout||'')+(r.stderr||''),g=k=>Number((new RegExp('^# '+k+' (\\d+)','m').exec(o)||[])[1]??NaN);return {cmd:[cmd.replace(W,'<W>'),...args].join(' '),cwd:cwd.replace(W,'<W>'),exit:r.status,tests:g('tests'),pass:g('pass'),fail:g('fail'),ms:Date.now()-t0}};
const fresh=d=>{rmSync(d,{recursive:true,force:true});mkdirSync(dirname(d),{recursive:true})};
const H=join(W,'SynthAi/computer/donors/synthia58/Synthia-Solo-Hover-v0.5.8'),out=[];
{ // 1. Hover: vendor/klein-mesh-game-engine-v5.1/kmge_v5_fixed has her .ts sources but no dist/; registered-app-runtime.mjs requires dist/*.js.
  // Her own donor zip of the same engine (identical .ts files, checked below) carries the dist built by her package.json "build": "tsc --noEmitOnError".
  const id='hover-klein-dist',dst=join(OUT,id,'Synthia-Solo-Hover-v0.5.8'),K='vendor/klein-mesh-game-engine-v5.1/kmge_v5_fixed',Z=join(H,'vendor/trainable-assembly-v0.4.2/donors/klein-mesh-game-engine-v5.1-FIXED(2).zip');
  const before=treeHash(join(H,K));fresh(dst);execFileSync('cp',['-a',H,dst]);const tmp=join(OUT,id,'zip');fresh(tmp);mkdirSync(tmp);execFileSync('unzip',['-q',Z,'-d',tmp]);
  const zk=join(tmp,'kmge_v5_fixed'),same=readdirSync(join(H,K)).filter(f=>f.endsWith('.ts')).map(f=>({file:f,identical:sha(join(H,K,f))===sha(join(zk,f))}));
  if(!same.every(s=>s.identical))throw new Error('zip sources differ from the Hover copy: not her build of this source');
  execFileSync('cp',['-a',join(zk,'dist'),join(dst,K,'dist')]);
  const added=walk(join(dst,K,'dist')).map(f=>({path:relative(dst,f),bytes:statSync(f).size,sha256:sha(f)}));
  const node20=run('node',['--test','--test-concurrency=1',...readdirSync(join(dst,'test')).filter(f=>f.endsWith('.test.mjs')).map(f=>'test/'+f)],dst);
  const node22=existsSync(NODE22)?run(NODE22,['--test','--test-concurrency=1',...readdirSync(join(dst,'test')).filter(f=>f.endsWith('.test.mjs')).map(f=>'test/'+f)],dst):null;
  const unrepaired=run('node',['--test','--test-concurrency=1',...readdirSync(join(H,'test')).filter(f=>f.endsWith('.test.mjs')).map(f=>'test/'+f)],H);
  out.push({id,problem:'Hover test files fail: '+K+'/dist missing (registered-app-runtime.mjs line 5-6 require it)',original:relative(W,H),originalUntouched:treeHash(join(H,K))===before,
   fix:'working copy of the whole Hover tree + dist/ from her own donor zip',source:{zip:relative(W,Z),zipSha256:sha(Z),sourcesIdentical:same},added,
   results:{originalUnrepaired:unrepaired,repairedNode20:node20,repairedNode22:node22},
   note:'test/19-solo-hover-shell uses the global WebSocket (Chrome DevTools), which Node 20 does not define; Node >= 22 has it. Her package.json says "node": ">=20", so on Node 20 that one file still fails. Not changed.'});
}
{ // 2. integrated-tool-factory v1.6.0: test/ato-native-bridge.test.mjs imports ../../ato-core-native-test/src/index.mjs (a sibling the zip does not include).
  // Her own Synthia browser continuation v0.6.8 vendors the same factory (src identical) next to ato-core and imports ../../ato-core/src/index.mjs; that ato-core is used as the sibling.
  const id='tool-factory-ato-sibling',T=join(W,'zips/_drive_x/integrated-tool-factory-v1_6_0'),U=join(H,'vendor/synthia-browser-continuation-v0.6.8/src/UPGRADES/vendor'),base=join(OUT,id);
  const srcSame=readdirSync(join(T,'src')).map(f=>({file:'src/'+f,identical:existsSync(join(U,'integrated-tool-factory/src',f))&&sha(join(T,'src',f))===sha(join(U,'integrated-tool-factory/src',f))}));
  const before=treeHash(T);fresh(base);mkdirSync(base);execFileSync('cp',['-a',T,join(base,'integrated-tool-factory-v1_6_0')]);execFileSync('cp',['-a',join(U,'ato-core'),join(base,'ato-core-native-test')]);
  const unrepaired=run('node',['--test'],T),repaired=run('node',['--test'],join(base,'integrated-tool-factory-v1_6_0')),hers=run('node',['--test'],join(U,'integrated-tool-factory'));
  out.push({id,problem:'1 of 9 test files fails standalone: missing sibling ato-core-native-test/ (her DATASET_PROVENANCE.md line 62 says so)',original:relative(W,T),originalUntouched:treeHash(T)===before,
   fix:'working copy with her ato-core (from synthia-browser-continuation-v0.6.8/src/UPGRADES/vendor/ato-core) placed as the sibling',source:{atoCore:relative(W,join(U,'ato-core')),atoCoreTreeSha256:treeHash(join(U,'ato-core')),factorySrcIdentical:srcSame},
   added:[{path:'../ato-core-native-test/',treeSha256:treeHash(join(base,'ato-core-native-test')),files:walk(join(base,'ato-core-native-test')).length}],
   results:{originalUnrepaired:unrepaired,repaired,hersInV068:hers},
   note:'her provenance doc says the 11/11 run used ato-core from ato-mcp-v0_1-fixed.zip; that zip is not on this box (searched /), so the ato-core she vendored in v0.6.8 is used. Whether the two are identical is not known.'});
}
writeFileSync(join(HERE,'repairs.json'),JSON.stringify({schema:'synthia-repairs/1',repairs:out},null,1));
for(const r of out)console.log(r.id,'untouched',r.originalUntouched,JSON.stringify(Object.fromEntries(Object.entries(r.results).map(([k,v])=>[k,v&&`${v.pass}/${v.tests}`]))));
