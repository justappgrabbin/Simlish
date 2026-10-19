// Build tool (Node built-ins only). Produces INVENTORY.json and INVENTORY.md from inventory.data.mjs after checking every claim it can:
// files exist (sha256 recorded), named functions appear in the file, the tests named were run (results in inventory-probes.json) and the status is consistent with them.
// Run on the box that holds her files:  node integrated/swarm/inventory.build.mjs [/workspace]
import {readFileSync,writeFileSync,existsSync,statSync,readdirSync} from 'node:fs';
import {join,dirname} from 'node:path';import {fileURLToPath} from 'node:url';import {createHash} from 'node:crypto';import {spawn} from 'node:child_process';
import {ROWS,SPEC,ACCEPTANCE,BASE_RECONCILIATION,REFERENCES_ONLY} from './inventory.data.mjs';
const HERE=dirname(fileURLToPath(import.meta.url)),REPO=join(HERE,'..','..'),W=process.argv[2]||'/workspace';
const sha=p=>createHash('sha256').update(readFileSync(p)).digest('hex');
const pr=JSON.parse(readFileSync(join(HERE,'inventory-probes.json'),'utf8'));
// repo tests: run now (this branch), 4 at a time
const repoIds=[...new Set(ROWS.flatMap(r=>r.tests).filter(t=>t.startsWith('repo-test:')))];
const have=new Map(pr.probes.map(p=>[p.id,p]));
const runOne=id=>new Promise(res=>{const file=id.slice(10),t0=Date.now();let buf='';const c=spawn('node',['--test',file],{cwd:REPO,env:{...process.env,CI:'1'}});const k=setTimeout(()=>{c.kill('SIGKILL')},300000);c.stdout.on('data',d=>buf=(buf+d).slice(-6000));c.stderr.on('data',d=>buf=(buf+d).slice(-6000));c.on('close',code=>{clearTimeout(k);const m=/# pass (\d+)/.exec(buf),n=/# fail (\d+)/.exec(buf);res({id,cwd:'you-n-i-verse',cmd:'node --test '+file,status:code===0?'pass':'fail',code,ms:Date.now()-t0,tail:`pass ${m?m[1]:'?'} fail ${n?n[1]:'?'}`})})});
let q=[...repoIds],done=[];await Promise.all([0,1,2,3].map(async()=>{while(q.length)done.push(await runOne(q.shift()))}));
pr.probes=pr.probes.filter(p=>!p.id.startsWith('repo-test:')).concat(done);writeFileSync(join(HERE,'inventory-probes.json'),JSON.stringify(pr,null,1));
const probe=new Map(pr.probes.map(p=>[p.id,p]));
if(existsSync(join(HERE,'repairs','repairs.json')))for(const r of JSON.parse(readFileSync(join(HERE,'repairs','repairs.json'),'utf8')).repairs){const v=r.results.repairedNode22||r.results.repaired;probe.set('repair:'+r.id,{id:'repair:'+r.id,cmd:v.cmd,cwd:v.cwd,status:v.exit===0&&v.fail===0?'pass':'fail',tail:`pass ${v.pass} fail ${v.fail} (repaired working copy; original unrepaired: ${r.results.originalUnrepaired.pass}/${r.results.originalUnrepaired.tests})`})}
const abs=p=>p.startsWith('repo:')?join(REPO,p.slice(5)):join(W,p);
const rows=[],problems=[];
for(const r of ROWS){const files=[];for(const f of r.files){const p=abs(f.path);if(!existsSync(p)){problems.push(r.id+': file missing '+f.path);continue}const st=statSync(p),text=st.size<3e6&&/\.(m?js|py|json|md|ts)$/.test(p)?readFileSync(p,'utf8'):null;
  files.push({path:f.path,bytes:st.size,sha256:sha(p),functions:f.fns.map(n=>({name:n,found:text?new RegExp('\\b'+n.replace(/^(class|def) /,'$1\\s+')+'\\b').test(text):null}))})}
 const dirs=(r.dirs||[]).map(d=>{const p=abs(d);if(!existsSync(p)){problems.push(r.id+': dir missing '+d);return {path:d,exists:false}}return {path:d,exists:true,entries:readdirSync(p).length}});
 const tests=r.tests.map(id=>{const p=probe.get(id);if(!p){problems.push(r.id+': no probe result for '+id);return {id,status:'no-result'}}return {id,status:p.status,cmd:p.cmd,cwd:p.cwd,result:p.tail}});
 const passed=tests.filter(t=>t.status==='pass').length,ev=r.runEvidence&&existsSync(abs(r.runEvidence));
 if(r.status==='EXISTS-RUNS'&&!passed&&!ev)problems.push(r.id+': EXISTS-RUNS without a passing test');
 if(r.status==='MISSING'&&(files.length||dirs.length))problems.push(r.id+': MISSING but has files');
 if(r.status==='EXISTS-UNTESTED'&&!files.length&&!dirs.filter(d=>d.exists).length)problems.push(r.id+': EXISTS-UNTESTED without files');
 for(const f of files)for(const fn of f.functions)if(fn.found===false)problems.push(r.id+': function '+fn.name+' not in '+f.path);
 rows.push({id:r.id,component:r.name,specItems:r.spec,status:r.status,files,dirs,tests,testsPassed:passed,testsTotal:tests.length,runEvidence:r.runEvidence||null,dependencies:r.dependencies,missingConnections:r.missing,note:r.note||null})}
if(problems.length){console.error('INVENTORY PROBLEMS\n'+problems.join('\n'));process.exit(1)}
// reference-only research: confirm no code of theirs is in this repo
const srcFiles=[];const walk=d=>{for(const e of readdirSync(d,{withFileTypes:true})){if(['node_modules','.git','originals','models'].includes(e.name))continue;const p=join(d,e.name);e.isDirectory()?walk(p):/\.(m?js|json|py|html)$/.test(e.name)&&statSync(p).size<2e6&&srcFiles.push(p)}};walk(REPO);
const refs=REFERENCES_ONLY.map(n=>{const re=new RegExp(n.replace(/[-\s]/g,'[-\\s]?'),'i');const hits=srcFiles.filter(p=>!/inventory\.|INVENTORY|registry\.json|SCORECARD/.test(p)&&re.test(readFileSync(p,'utf8'))).map(p=>p.slice(REPO.length+1)).slice(0,3);return {name:n,status:'PROPOSED',packagedHere:false,mentionsInRepoSources:hits}});
const count=s=>rows.filter(r=>r.status===s).length,reg=JSON.parse(readFileSync(join(HERE,'registry.json'),'utf8')),regBy={};for(const v of Object.values(reg.connections.byId))regBy[v.status]=(regBy[v.status]||0)+1;
const out={schema:'synthia-swarm-inventory/1',builtOn:'2026-10-16 (PT)',statusCounts:Object.fromEntries(['EXISTS-RUNS','EXISTS-UNTESTED','PROPOSED','UNVERIFIED','MISSING'].map(s=>[s,count(s)+(s==='PROPOSED'?refs.length:0)])),spec:Object.fromEntries(SPEC),rows,acceptance:ACCEPTANCE,baseReconciliation:BASE_RECONCILIATION,referencesOnly:refs,swarmRegistry:{entries:reg.entries.length,connections:regBy,uncurated:reg.uncurated.length,file:'integrated/swarm/registry.json'},probesFile:'integrated/swarm/inventory-probes.json',
 rules:'EXISTS-RUNS only when a command listed in the row passed on this box (or a recorded run output exists). Nothing is called integrated because it belongs in the design: a component is connected only if registry.json says connected / connected-copy.'};
writeFileSync(join(HERE,'INVENTORY.json'),JSON.stringify(out,null,1));
// ---- markdown ----
const sh=h=>h.slice(0,10),cell=s=>String(s??'').replace(/\|/g,'\\|').replace(/\n/g,' ');
let md=`# SWARM-OS inventory: every component in the ecosystem table mapped to the specification\n\nBuilt 2026-10-16 (PT) by \`integrated/swarm/inventory.build.mjs\` from \`inventory.data.mjs\`; the builder refuses a row whose files are missing, whose named functions are not in the file, or whose status the listed tests do not back. Raw command results: \`inventory-probes.json\`. Machine-readable copy: \`INVENTORY.json\`. All of her files are untouched.\n\n**Status words:** EXISTS-RUNS (a listed command passed here) · EXISTS-UNTESTED (files exist, nothing passed here) · PROPOSED (in the spec, no connection built) · UNVERIFIED (no file confirms the name or capability; candidates listed, none claimed) · MISSING (searched, nothing found).\n\n`;
md+=`**Counts:** ${Object.entries(out.statusCounts).map(([k,v])=>k+' '+v).join(' · ')} (PROPOSED includes the ${refs.length} research references, which are not packaged).\n\n## Specification items\n\n${SPEC.map(([k,v])=>`- **${k}** ${v}`).join('\n')}\n\n## Components\n\n| Component | Spec | Status | Tests (passed/listed) | Exact files | Missing connections |\n|---|---|---|---|---|---|\n`;
for(const r of rows)md+=`| ${cell(r.component)} | ${r.specItems.join(' ')||'–'} | **${r.status}** | ${r.testsTotal?r.testsPassed+'/'+r.testsTotal:(r.runEvidence?'recorded run':'–')} | ${cell(r.files.map(f=>f.path+(f.functions.length?' ('+f.functions.map(x=>x.name).join(', ')+')':'')).concat(r.dirs.map(d=>d.path+'/')).join('; '))||'–'} | ${cell(r.missingConnections.join(' · '))} |\n`;
md+=`\n### Test evidence per component\n\n`;for(const r of rows){if(!r.tests.length)continue;md+=`- **${r.component}**: ${r.tests.map(t=>`${t.id} → ${t.status}`).join('; ')}\n`}
md+=`\n## The 15-row acceptance table, judged on this branch\n\n| # | Row | Status | Evidence | Missing |\n|---|---|---|---|---|\n${ACCEPTANCE.map(a=>`| ${a.n} | ${a.row} | **${a.status}${a.verdict?' · '+a.verdict:''}** | ${cell(a.evidence.join(' · '))||'–'} | ${cell(a.missing.join(' · '))||'–'} |`).join('\n')}\n`;
md+=`\n## Six bases vs five bases per tone: reconciled against the source files\n\n${BASE_RECONCILIATION.reading}\n\n**Five:**\n${BASE_RECONCILIATION.fiveBases.map(x=>`- \`${x.file}\`${x.line?':'+x.line:''}: ${x.quote}${x.also?' ('+x.also+')':''}`).join('\n')}\n\n**Six:**\n${BASE_RECONCILIATION.sixBases.map(x=>`- \`${x.file}\`${x.line?':'+x.line:''}: ${x.quote}${x.arithmetic?' — '+x.arithmetic:''}`).join('\n')}\n`;
md+=`\n## Research references (not packaged: no donor code without a purpose, a connection and a test)\n\n${refs.map(r=>`- ${r.name}: PROPOSED reference only; no local copy used${r.mentionsInRepoSources.length?'; the name is mentioned in: '+r.mentionsInRepoSources.join(', '):''}`).join('\n')}\n\n## The 68-entry swarm registry\n\n\`integrated/swarm/registry.json\`: ${reg.entries.length} swarms by reference with sha256 (${Object.entries(regBy).map(([k,v])=>k+' '+v).join(', ')}), plus ${reg.uncurated.length} further swarm-named files found and listed so none is dropped.\n`;
writeFileSync(join(HERE,'INVENTORY.md'),md);
console.log(out.statusCounts,'rows',rows.length,'problems 0');
