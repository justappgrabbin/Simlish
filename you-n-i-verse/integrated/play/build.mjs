// Build mode: the user builds a real project/page/tool WITH the agents, through the EXISTING pipeline:
//  personalize.mjs (her world theme, set once; PSO checks -> GA view) + app-finisher.mjs finishApp (PSO genotype assesses the
//  files against her checklist, GA phenotype is the report that is shown; adds manifest/registration/scene/reality beside the
//  files, never over them) -> zip-lite writeZip. Output is written as real files through the audited workspace.write tool
//  (user/builds/<slug>/...) when a writer is given, and returned as a zip.
// Agents: each agent takes one utility turn per build step (agents.mjs); their lines are kept in the build log (FILLED: the
// agents' role here is to run and log the pipeline steps; they do not write content).
import {personalize} from '../../personalize.mjs';
import {finishApp} from '../../app-finisher.mjs';
import {agentTurn} from './agents.mjs';
export const BUILD_KINDS=Object.freeze({page:'index.html',tool:'tool.mjs',project:'README.md'});
const slug=s=>String(s||'build').toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'')||'build';
const td=new TextDecoder();
export async function buildWithAgents({kind,title,text,files,chart,program,agents=[],storage=null,worldTheme,write=null}){
 if(!BUILD_KINDS[kind])throw new Error('kind must be page, tool or project');
 const own=files?.length?files:[{path:BUILD_KINDS[kind],data:String(text??'')}];
 const p=personalize({name:chart?.name||title,worldTheme,target:'reality'},storage),log=[];let crew=agents;
 const step=label=>{crew=crew.map(agentTurn);for(const a of crew)log.push({step:label,agent:a.id,line:a.last.line})};
 step('personalize');
 const r=await finishApp({name:title,files:own},{personalization:{...p,address:null},env:{}});
 step('finish');
 const base='user/builds/'+slug(title)+'/',written=[];
 const meta={kind,title,program:program?{fingerprint:program.fingerprint,level:program.level,address:program.address,lessons:program.lessons.map(l=>l.title)}:null,agents:crew.map(a=>({id:a.id,hexagram_code:a.hexagram_code})),log,status:r.report.status,originalSha256:r.report.originalSha256};
 const outFiles=[...r.files.map(f=>({path:f.path,text:td.decode(f.data)})),{path:'simlish/build-with-agents.json',text:JSON.stringify(meta,null,2)}];
 if(write)for(const f of outFiles)written.push(await write(base+f.path,f.text));
 return {report:r.report,files:outFiles,zip:r.zip,base,written,log,agents:crew,personalization:p}}
