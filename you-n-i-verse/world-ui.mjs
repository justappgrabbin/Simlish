// Same existing world page. Only evidence-backed mutations change the rendered world.
import { WorldSession } from './world-session.mjs';
import { drawWorld } from './world-canvas.mjs';

const $=selector=>document.querySelector(selector);
const STORE='yni-executable-world-session-v1';
let session=new WorldSession(),mode='Play',voice=false,last=null;
const copy={
  Watch:['Your recorded world','Review the last locally executed operations without changing the world.','Read world history ↗'],
  Play:['What will you change?','Choose an object and a supported operation. The original Simlish world mutates and the canvas shows its actual state.','Execute transformation ↗'],
  Create:['Compose a new object','Duplicate the chosen world object and form a composite in the original swarm world.','Duplicate and compose ↗'],
  Mission:['Prove a transformation','Measured mission: transform the chair and produce an executable formation receipt. Other objects do not complete this mission.','Execute measured mission ↗']
};
function setMode(next) {
  mode=next;
  document.querySelectorAll('[data-mode]').forEach(button=>button.classList.toggle('active',button.dataset.mode===mode));
  const [title,description,action]=copy[mode];
  $('#scene-title').textContent=title;
  $('#scene-copy').textContent=description;
  $('#act').textContent=action;
  $('#worldOperation').disabled=mode==='Watch'||mode==='Create'||mode==='Mission';
  $('#worldTarget').disabled=mode==='Watch';
  if(mode==='Create')$('#worldOperation').value='duplicate';
  else if(mode==='Mission')$('#worldOperation').value='transform';
}
function refresh(options={}) {
  const snapshot=session.snapshot();
  drawWorld($('#liveWorld'),snapshot,{selected:$('#worldTarget').value});
  const picker=$('#worldTarget'),chosen=picker.value;
  picker.replaceChildren();
  for(const object of snapshot.objects.filter(o=>!o.removed)) {
    const option=document.createElement('option');
    option.value=object.id;option.textContent=object.name;
    picker.appendChild(option);
  }
  if([...picker.options].some(o=>o.value===chosen))picker.value=chosen;
  else if(picker.options.length)picker.selectedIndex=0;
  $('#worldStatus').textContent=
    snapshot.objects.filter(o=>!o.removed).length+' visible objects · '+snapshot.composites.length+
    ' real swarm composites · '+snapshot.interactions+' local actions · ATO formation revision '+snapshot.activeRevision;
  $('#history-title').textContent=snapshot.interactions?
    'Your chapter '+snapshot.interactions:'Your first chapter';
  $('#history').textContent=snapshot.interactions?
    'Each operation is recorded with the Simlish world event, executable formation revision and local swarm state.':
    'No world action has run. Objects are the initial state of the packaged Simlish simulation.';
  $('#export').disabled=!snapshot.interactions;
}
async function save() {
  try {localStorage.setItem(STORE,JSON.stringify(session.export()));return true}
  catch {return false}
}
async function boot() {
  let restored=false;
  try {
    const raw=localStorage.getItem(STORE);
    if(raw) {
      session=await WorldSession.fromCommands(JSON.parse(raw));
      restored=true;
    }
  } catch(error) {
    $('#receipt').textContent='Saved world could not be replayed: '+error.message+'. Fresh world is available; saved data has not been erased.';
    session=new WorldSession();
  }
  refresh();
  if(restored)$('#receipt').textContent='Replayed '+session.commands.length+' local world operations and rebuilt their executable state.';
}
for(const button of document.querySelectorAll('[data-mode]'))button.onclick=()=>setMode(button.dataset.mode);
$('#worldTarget').addEventListener('change',()=>refresh());
$('#voice').onclick=()=>{
  voice=!voice;
  $('#voice').textContent=voice?'Voice on':'Voice off';
  $('#voice').setAttribute('aria-pressed',String(voice));
  if(!voice){globalThis.speechSynthesis?.cancel();$('.waves').classList.remove('speaking')}
};
function speak(text) {
  if(!voice||!globalThis.speechSynthesis||!globalThis.SpeechSynthesisUtterance)return;
  const utterance=new SpeechSynthesisUtterance(text);
  utterance.onstart=()=>$('.waves').classList.add('speaking');
  utterance.onend=utterance.onerror=()=>$('.waves').classList.remove('speaking');
  speechSynthesis.cancel();speechSynthesis.speak(utterance);
}
$('#act').onclick=async()=>{
  const button=$('#act');button.disabled=true;
  try {
    if(mode==='Watch') {
      if(!session.commands.length) {
        $('#receipt').textContent='No recorded operations yet. Play, Create or Mission changes the local world.';
        return;
      }
      refresh();
      $('#receipt').textContent='World history contains '+session.commands.length+' replayable actions. Export it below.';
      return;
    }
    const receipt=await session.perform({
      mode,targetId:$('#worldTarget').value,op:$('#worldOperation').value
    });
    last=receipt;
    const saved=await save();
    refresh();
    $('#receipt').textContent='Executed '+receipt.command.op+' on '+receipt.command.targetId+
      ' · actual world mutation: '+receipt.changed+
      ' · ATO: '+receipt.formationRevision+
      ' · '+(receipt.compositeId?'composite '+receipt.compositeId+' · ':'')+
      (receipt.completed===null?'':receipt.completed?'Mission verified · ':'Mission not completed · ')+
      (saved?'local replay saved.':'local storage unavailable; export to retain the record.');
    speak(receipt.completed===false?'Mission not complete.':'Your world state has changed and the operation was recorded.');
  }catch(error){$('#receipt').textContent='Execution failed: '+error.message}
  finally{button.disabled=false}
};
$('#reference').onclick=()=>{$('#refs').open=true;$('#refs').scrollIntoView({behavior:'smooth'})};
$('#export').onclick=()=>{
  if(!session.commands.length)return;
  const data=session.export();
  const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
  const anchor=document.createElement('a');
  anchor.href=url;anchor.download='you-n-i-verse-executable-world-session.json';
  anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
setMode('Play');
boot().catch(error=>{$('#receipt').textContent='World could not boot: '+error.message;$('#act').disabled=true});
