import {runSlice} from './bridge.mjs';
import {readProject} from './project-store.mjs';
import {createSimlishMorphProvider} from './morph-provider.mjs';
let mode='Play',voice=false,result;const $=s=>document.querySelector(s);
const localCharacter = createSimlishMorphProvider();
let savedCharacterShown = false;
// Player-facing world only consumes a previously prepared morph. It never
// opens the editor, collects a photo, or blocks gameplay to reconstruct one.
async function reusePreparedCharacter() {
  if (savedCharacterShown) return true;
  const project = await readProject();
  if (!project?.morphedImage || !project?.photo) return false;
  const prepared = await localCharacter.prepare({intent:'portrait',reuseSavedPortrait:true});
  if (prepared.status !== 'ready') return false;
  const picture = $('.guide');
  picture.src = prepared.output.dataUrl;
  picture.alt = 'Your prepared character in YOU-N-I-VERSE';
  picture.classList.add('personalized');
  savedCharacterShown = true;
  return true;
}
const copy={Watch:['A moment in the residence','Read the recorded interaction and its consequences.','Observe the scene ↗'],Play:['What will you change?','Try the learned chair transformation in the simulated world.','Transform the chair ↗'],Create:['Create a new possibility','Run the authored scene recipe, then export its model and event for your project.','Build the example scene ↗'],Mission:['Make an action count','Complete one legal world interaction and retain its execution evidence.','Complete the first mission ↗']};
for(const b of document.querySelectorAll('[data-mode]'))b.onclick=()=>{mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.classList.toggle('active',x===b));const [t,c,a]=copy[mode];$('#scene-title').textContent=t;$('#scene-copy').textContent=c;$('#act').textContent=a;};
$('#voice').onclick=()=>{voice=!voice;$('#voice').textContent=voice?'Voice on':'Voice off';$('#voice').setAttribute('aria-pressed',String(voice));if(!voice){globalThis.speechSynthesis?.cancel();$('.waves').classList.remove('speaking')}};
function speak(text){if(!voice||!globalThis.speechSynthesis||!globalThis.SpeechSynthesisUtterance)return;const u=new SpeechSynthesisUtterance(text);u.onstart=()=>$('.waves').classList.add('speaking');u.onend=u.onerror=()=>$('.waves').classList.remove('speaking');speechSynthesis.cancel();speechSynthesis.speak(u)}
$('#act').onclick=async()=>{const b=$('#act');b.disabled=true;try{if(mode==='Watch'){const saved=localStorage.getItem('synthia-world-episode');if(!saved){$('#receipt').textContent='Play, Create or Mission once to record a scene to watch.';return}result=JSON.parse(saved)}else{result=await runSlice();try{localStorage.setItem('synthia-world-episode',JSON.stringify(result))}catch{}}
let characterReady = false;
if (mode !== 'Watch') {
  try { characterReady = await reusePreparedCharacter(); } catch { /* Existing gameplay remains available without a saved portrait. */ }
}
$('#receipt').textContent=`${mode==='Watch'?'Recorded':'Executed'}: ${result.event.structure.join(' ')} · ${result.event.projections.length} addressed projections. ${characterReady?'Saved character displayed.':'Default artwork displayed.'} This is still a world prototype.`;$('#history-title').textContent='An interaction remembered';$('#history').textContent='The learned transformation changed the simulated chair. Klein rule and AutoNovel lineage are recorded with the model and world receipt.';$('#export').disabled=false;speak('Your interaction is recorded. Another possibility has become part of your journey.');}catch(e){$('#receipt').textContent=e.message}finally{b.disabled=false}};
$('#reference').onclick=()=>{$('#refs').open=true;$('#refs').scrollIntoView({behavior:'smooth'})};
$('#export').onclick=()=>{if(!result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='you-n-i-verse-episode.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
