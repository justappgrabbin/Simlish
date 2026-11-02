import {PROGRAMS,createGamingComputerHost,resolveProgram} from './gaming-computer-core.mjs';

const $=id=>document.getElementById(id);
const host=createGamingComputerHost();
const EVENTS_KEY='simlish.gaming-computer.launch-trace.v1';
let traces=[];
try {const stored=JSON.parse(localStorage.getItem(EVENTS_KEY)||'[]');if(Array.isArray(stored))traces=stored.slice(-20);}catch{}
const programList=$('program-list'),traceList=$('trace-list');
function addTrace(kind,detail){
  const event={at:new Date().toISOString(),kind,detail};
  traces=[event,...traces].slice(0,20);
  try{localStorage.setItem(EVENTS_KEY,JSON.stringify(traces));}catch{}
  drawTraces();
}
function drawTraces(){
  traceList.replaceChildren();
  if(!traces.length){const li=document.createElement('li');li.textContent='No resolver events yet. Try ENGINE CHECK.';traceList.append(li);return;}
  for(const item of traces.slice(0,6)){
    const li=document.createElement('li');const name=document.createElement('b');name.textContent=item.kind;
    const meta=document.createElement('small');meta.textContent=' · '+item.at.slice(11,19);
    const detail=document.createElement('span');detail.textContent=typeof item.detail==='string'?item.detail:JSON.stringify(item.detail);
    li.append(name,meta,detail);traceList.append(li);
  }
}
function setStatus(value){$('resolver-status').textContent=value;}
async function openProgram(id){
  try {
    const faculty=resolveProgram(host,id);
    const receipt=await host.execute(['local','program',...PROGRAMS.find(p=>p.id===id).caps],{action:'launch'});
    if(receipt.facultyId!==faculty.id)throw new Error('Resolver selected a different program.');
    const entry=receipt.result.entry;
    if(!PROGRAMS.some(p=>p.id===id&&p.entry===entry))throw new Error('Unexpected program route');
    addTrace('ROUTED', {faculty:faculty.id,entry});
    location.assign(entry);
  }catch(error){setStatus('Route failed: '+error.message);addTrace('ERROR',error.message);}
}
function drawPrograms(){
  for(const program of PROGRAMS){
    const card=document.createElement('button');
    card.className='program';
    card.style.setProperty('--accent',program.color);
    card.setAttribute('aria-label','Open '+program.label);
    const type=document.createElement('span');type.className='type';type.textContent=program.type+' / '+program.id.toUpperCase();
    const name=document.createElement('strong');name.textContent=program.label;
    const desc=document.createElement('span');desc.className='desc';desc.textContent=program.description;
    const arrow=document.createElement('span');arrow.className='go';arrow.textContent='OPEN PROGRAM  ↗';
    card.append(type,name,desc,arrow);
    card.addEventListener('click',()=>openProgram(program.id));
    programList.append(card);
  }
}
async function inspectEngine(){
  const button=$('engine-check');button.disabled=true;setStatus('Resolving local programs and game state…');
  try{
    const receipt=await host.execute(['local','game','state','inspect'],{action:'inspect'});
    const state=receipt.result;
    const found=PROGRAMS.map(p=>({program:p.id,resolver:resolveProgram(host,p.id).id}));
    $('module-count').textContent=String(found.length);
    $('gate-number').textContent=String(state.gate);
    $('line-number').textContent=String(state.choiceCount);
    $('capability-result').textContent=JSON.stringify({resolvedBy:receipt.facultyId,gameState:state,programs:found},null,2);
    setStatus('LOCAL PASS · '+found.length+' executable entry points · '+state.choiceCount+' gate/line choices');
    addTrace('ENGINE CHECK',{resolver:receipt.facultyId,gate:state.gate,choices:state.choiceCount,routes:found.length});
  }catch(error){setStatus('Engine check failed: '+error.message);$('capability-result').textContent=String(error.stack||error);addTrace('ERROR',error.message);}
  finally{button.disabled=false;}
}
drawPrograms();drawTraces();
$('engine-check').addEventListener('click',inspectEngine);
$('play-now').addEventListener('click',()=>openProgram('five-lights'));
$('clear-traces').addEventListener('click',()=>{traces=[];try{localStorage.removeItem(EVENTS_KEY);}catch{}drawTraces();setStatus('Local trace cleared.');});
inspectEngine();
