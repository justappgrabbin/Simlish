// Page script for the gaming + build area. Wires: chart-engine -> adaptive programme -> her youniverse worlds (wrapped,
// integrated/play/games-host.mjs) + the repo's line-walk -> swarm agents -> build mode. No worker, no network.
import './vendor/youniverse-games-shims/buffer.mjs';
import {calculateBirthChart} from './integrated/chart-engine.mjs';
import {createPlaySession,WORLDS} from './integrated/play/session.mjs';
import {avatarIdentity,rgbaToBmp} from './integrated/play/avatar.mjs';
import {newMemory,verifyMemory,program,INVESTIGATING,FILLED_IN} from './integrated/play/adaptive.mjs';
import {AGENT_FILLED_IN} from './integrated/play/agents.mjs';
import {HOOK_CONTRACT} from './integrated/play/hooks.mjs';
import {readPersonalization} from './personalize.mjs';
import {readProject} from './project-store.mjs';
import {createAppServer} from './integrated/app-server.mjs';
import {readFileSync,existsSync} from './vendor/youniverse-games-shims/fs.mjs';
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const loadBytes=async p=>{const r=await fetch(p);if(!r.ok)throw new Error(p+' '+r.status);return new Uint8Array(await r.arrayBuffer())};
const loadJson=async p=>{const r=await fetch(p);if(!r.ok)throw new Error(p+' '+r.status);return r.json()};
const store={get:k=>{try{return localStorage.getItem(k)}catch{return null}},set:(k,v)=>{try{localStorage.setItem(k,v)}catch{}}};
const srv=await createAppServer({loadJson,loadBytes,store,loadModel:false});window.synthiaPlay=srv;
const call=(n,b)=>srv.handle('POST','/api/tool/'+n,b||{}).then(r=>{if(!r.ok)throw new Error(r.body.error||'failed');return r.body});
const WORLD_COLORS={cosmic:{top:'#251038',bottom:'#090916',color:'#ed55ca'},forest:{top:'#183c29',bottom:'#06160f',color:'#54d27b'},ocean:{top:'#073f62',bottom:'#061324',color:'#44cdeb'},desert:{top:'#764a32',bottom:'#241b25',color:'#efb46c'}}; // values from studio.mjs
const applyTheme=t=>{const w=WORLD_COLORS[t]||WORLD_COLORS.cosmic,s=document.documentElement.style;s.setProperty('--top',w.top);s.setProperty('--bottom',w.bottom);s.setProperty('--accent',w.color)};
const pers=readPersonalization(localStorage);if(pers?.worldTheme){$('theme').value=pers.worldTheme;$('theme').disabled=true}applyTheme($('theme').value);$('theme').onchange=()=>applyTheme($('theme').value);
$('inv').innerHTML=INVESTIGATING.map(esc).join(' · ')+'<br><span class="note">Filled in (no source): '+esc([...FILLED_IN,...AGENT_FILLED_IN].join('; '))+'</span>';
// --- avatars: Studio morph, a photo, or a state from her supplied board (crop boxes FILLED: measured by eye on assets/avatar-states.jpg)
const BOARD=[{label:'Embodied agent',x:.205,y:.105,w:.255,h:.84},...['idle','walk','turn','reach','scan','navigate'].map((l,i)=>({label:'0'+(i+1)+' '+l,x:.467+i*.0877,y:.172,w:.083,h:.19})),...['activate','sit','stand on chair','lift chair','throw chair','neutral return'].map((l,i)=>({label:String(i+7).padStart(2,'0')+' '+l,x:.467+i*.0877,y:.397,w:.083,h:.23}))];
const N=48;let chosen=null;
const loadImg=src=>new Promise((ok,no)=>{const i=new Image();i.onload=()=>ok(i);i.onerror=()=>no(new Error('image failed: '+String(src).slice(0,40)));i.src=src});
function tile(img,box,label,ref){const c=document.createElement('canvas');c.width=c.height=N;c.title=label;const g=c.getContext('2d');const sx=box?box.x*img.naturalWidth:0,sy=box?box.y*img.naturalHeight:0,sw=box?box.w*img.naturalWidth:img.naturalWidth,sh=box?box.h*img.naturalHeight:img.naturalHeight;
 g.drawImage(img,sx,sy,sw,sh,0,0,N,N);c.onclick=()=>{for(const o of $('avatars').children)o.classList.remove('on');c.classList.add('on');chosen={canvas:c,ref}};$('avatars').append(c);return c}
try{const p=await Promise.race([readProject(),new Promise(r=>setTimeout(()=>r(null),3000))]);/* IndexedDB can stall in headless runs */const src=p?.morphedImage||p?.photo;if(src)tile(await loadImg(src),null,'My Studio morph','studio:project').click()}catch{}
try{const b=await loadImg('assets/avatar-states.jpg');BOARD.forEach((bx,i)=>{const c=tile(b,bx,bx.label,'board:avatar-states.jpg#'+bx.label);if(!chosen&&i===0)c.click()})}catch(e){$('err').textContent=String(e.message)}
$('photo').onchange=async()=>{const f=$('photo').files[0];if(!f)return;const u=URL.createObjectURL(f);tile(await loadImg(u),null,f.name,'photo:'+f.name).click()};
const identityOf=c=>avatarIdentity(rgbaToBmp(N,N,c.getContext('2d').getImageData(0,0,N,N).data),chosen.ref);
// --- saved people (guide's people.save)
try{const l=await call('people.list');for(const p of l.people){const o=document.createElement('option');o.value=p.id;o.textContent=(p.name||p.id)+' · '+p.hd.type+' '+p.hd.profile;$('saved').append(o)}}catch{}
const chartFromForm=async()=>{if($('saved').value)return (await call('people.get',{id:$('saved').value})).chart;return calculateBirthChart({name:$('name').value||'Traveler',date:$('date').value,time:$('time').value,timeZone:$('tz').value,latitude:Number($('lat').value),longitude:Number($('lon').value)})};
const memKey=chart=>'synthia.play.memory.v1:'+chart.utc+':'+chart.input.latitude+':'+chart.input.longitude;
const loadMem=chart=>{try{const m=JSON.parse(localStorage.getItem(memKey(chart))||'null');return m&&verifyMemory(m)?m:newMemory()}catch{return newMemory()}};
let S=null,CHART=null;
function showProgram(){const p=S.program;$('progCard').hidden=false;
 $('prog').innerHTML=`<p>Level <b>${p.level}</b> · address <b>gate ${p.address.gate}.${p.address.line}</b> (${esc(p.address.source)}) · ${S.memory.transitions.length} transitions remembered · programme <code>${p.fingerprint}</code></p>`+
 p.lessons.map(l=>`<div class="lesson"><b>${esc(l.title)}</b><br>${esc(l.text)}<br><span class="note">${esc(l.why.join(' · '))}</span></div>`).join('')+
 `<p class="note">Tools for you: ${p.tools.map(t=>esc(t.label)+(t.weight===0?' (not significant for your chart, §1.7)':'')).join(' · ')}<br>Views in your order: ${p.views.map(esc).join(' → ')}</p>`+(p.relevance.investigating?`<p class="flag">${esc(p.relevance.investigating)}</p>`:'')}
function showAgents(list){$('agents').innerHTML=S.agents.map(a=>`<div><b>${esc(a.name)}</b> <span class="note">swarm hexagram ${a.hexagram_code} · authority ${esc(a.authority)}</span></div>`).join('')+(list||[]).map(x=>`<div>${esc(x.line)}</div>`).join('')}
function showMorph(r){const id=r?.morph?.embodimentId||r?.entry?.morph?.embodimentId;const p='/youniverse/renders/'+S.player.playerId+'-'+id+'.svg';$('morph').innerHTML=id&&existsSync(p)?readFileSync(p):''}
function choicesFor(world,entry){if(world==='hd-rpg'){const m=entry.mechanics||{};return [...new Map((m.abilities||[]).map(a=>[a.canonicalGate,a])).values()].map(a=>({label:'Use '+a.name,c:{type:'use-ability',abilityGate:a.canonicalGate}})).concat((m.pathways||[]).map(p=>({label:'Traverse '+p.name,c:{type:'traverse',channel:p.channel}})),[{label:'Confront the shadow',c:{type:'confront',target:'shadow'}}])}
 if(world==='iching-story')return (entry.choices||[]).map(x=>({label:x.label,c:x.id}));
 if(world==='oracles-thread')return [{label:'Wait in stillness',c:{mode:'wait'}},...['sky-arc','lake-crescent','flame-flick','thunder-stomp','wind-sweep','water-trickle','mountain-press','earth-open'].map(g=>({label:'Gesture '+g,c:{mode:'gesture',symbol:g}})),...((entry.availableEntities||entry.journey?.session_state?.available_entities)||[]).slice(0,6).map(e=>({label:'Touch '+e,c:{mode:'click',entity:e}}))];
 if(world==='line-walk')return entry.moves.map(m=>({label:`Line ${m.line}: ${m.label} → gate ${m.toGate}`,c:{line:m.line}}));return []}
let lastEntry=null;
function renderScene(world,entry,lesson,extra=''){lastEntry=entry;const txt=[entry.scene?.situation,entry.scene,entry.situation,entry.prose,entry.situation?.prose].find(x=>typeof x==='string')||'';const sug=JSON.stringify(lesson.suggested);const ch=choicesFor(world,entry);
 $('scene').innerHTML=`<p><b>${esc(world)}</b> — ${esc(txt)}</p><div class="lesson"><b>Lesson: ${esc(lesson.chart?.title)}</b><br>${esc(lesson.chart?.keynote)}<br><span class="note">${esc(lesson.game||'')} ${esc(lesson.practise)}</span><br><span class="note">${esc(lesson.filled)}</span></div>${extra}<div id="choices"></div>`;
 ch.forEach(x=>{const b=document.createElement('button');const s2=JSON.stringify(x.c)===sug;b.textContent=(s2?'★ ':'')+x.label;if(s2)b.className='primary';b.onclick=()=>choose(world,x.c);$('choices').append(b)})}
async function choose(world,c){try{const r=await S.choose(c);localStorage.setItem(memKey(CHART),JSON.stringify(S.memory));const e=world==='line-walk'?{moves:r.lesson&&S.program.moves}:{...lastEntry,...r.result,mechanics:r.result.mechanics||lastEntry.mechanics,choices:r.result.scene?.choices||lastEntry.choices};
  if(world==='line-walk')e.moves=S.program.moves;
  renderScene(world,e,r.lesson,`<p class="${r.transition.outcome==='success'?'ok':r.transition.outcome==='miss'?'bad':'note'}">Outcome: ${r.transition.outcome}${r.programChanged?' · your programme changed':''}</p>`);showAgents(r.agents);showProgram();showMorph(r.result)}catch(e){$('scene').insertAdjacentHTML('beforeend','<p class="bad">'+esc(e.message)+'</p>')}}
async function enter(world){const e=await S.enter(world);renderScene(world,e.entry,e.lesson);showAgents(e.agents);showMorph(e)}
$('enterWorld').onclick=async()=>{$('err').textContent='';try{if(!chosen)throw new Error('Pick an avatar first');CHART=await chartFromForm();
 S=createPlaySession({chart:CHART,dataDir:'/youniverse',visualIdentity:identityOf(chosen.canvas),memory:loadMem(CHART)});
 $('playCard').hidden=$('buildCard').hidden=$('bodyCard').hidden=false;showProgram();
 $('worlds').innerHTML='';for(const w of WORLDS){const b=document.createElement('button');b.textContent={'hd-rpg':'HD RPG','iching-story':'I Ching Story',"oracles-thread":"Oracle's Thread",'line-walk':'Line Walk'}[w];b.onclick=()=>enter(w).catch(er=>$('scene').textContent=er.message);$('worlds').append(b)}
 const body=S.body();$('body').innerHTML=body.length?body.map(m=>`<p><b>${esc(m.title)}</b></p>`+(m.sections||[]).map(s=>`<p>${esc(s.heading)}: ${esc(s.text)} <span class="note">${esc(s.source)}</span></p>`).join('')).join(''):`<p class="note">No body &amp; science module is installed on this branch. Hook ready: ${esc(HOOK_CONTRACT.register)}.</p>`;
 await enter(S.program.views.includes('worlds')?'hd-rpg':'line-walk')}catch(e){$('err').textContent=e.message}};
$('build').onclick=async()=>{try{const r=await S.build({kind:$('bkind').value,title:$('btitle').value,text:$('btext').value,worldTheme:$('theme').value,storage:localStorage,write:(path,text)=>call('workspace.write',{path,text})});
 localStorage.setItem(memKey(CHART),JSON.stringify(S.memory));$('theme').disabled=true;const url=URL.createObjectURL(new Blob([r.zip],{type:'application/zip'}));
 $('bout').innerHTML=`<p class="ok">${esc(r.report.status)} · ${r.written.length} files saved under <b>${esc(r.base)}</b> (open them in Swarm → files)</p><p><a download="${esc(r.base.split('/')[2])}.zip" href="${url}">Download the build (.zip)</a></p><div class="log">${r.log.map(l=>`<div>${esc(l.step)} · ${esc(l.line)}</div>`).join('')}</div>`;showProgram();showAgents()}catch(e){$('bout').innerHTML='<p class="bad">'+esc(e.message)+'</p>'}};
// Self-test (?selftest=1): two charts, a generated avatar, every world, a build. Prints PLAY_SELFTEST PASS|FAIL.
if(location.search.includes('selftest')){let ok=false,d={};try{
 const A=calculateBirthChart({name:'A',date:'1990-05-17',time:'08:30',timeZone:'America/Los_Angeles',latitude:34.05,longitude:-118.24}),B=calculateBirthChart({name:'B',date:'1984-11-02',time:'22:10',timeZone:'Europe/London',latitude:51.5,longitude:-0.12});
 const px=new Uint8ClampedArray(N*N*4).map((_,i)=>(i*7)%256),vi=avatarIdentity(rgbaToBmp(N,N,px),'avatar:selftest');
 const sa=createPlaySession({chart:A,dataDir:'/selftest-a',visualIdentity:vi}),sb=createPlaySession({chart:B,dataDir:'/selftest-b',visualIdentity:vi});
 const f0=sa.program.fingerprint,outs=[];for(const w of WORLDS){const e=await sa.enter(w);const r=await sa.choose(e.lesson.suggested);outs.push(w+':'+r.transition.outcome)}await sa.exit();
 const eb=await sb.enter('hd-rpg'),mem={d:{},getItem(k){return this.d[k]??null},setItem(k,v){this.d[k]=v}};
 const b=await sa.build({kind:'page',title:'selftest page',text:'<!doctype html><h1>t</h1>',worldTheme:'cosmic',storage:mem,write:(path,text)=>call('workspace.write',{path,text})});
 d={programA:f0,programB:sb.program.fingerprint,after:sa.program.fingerprint,outs,typeA:A.hd.type,rpgB:eb.entry.mechanics.type,transitions:sa.memory.transitions.length,chain:verifyMemory(sa.memory),built:b.written.length,status:b.report.status,agents:sa.agents.length,board:BOARD.length};
 ok=d.programA!==d.programB&&d.after!==d.programA&&d.chain&&d.transitions===WORLDS.length+1&&d.rpgB===B.hd.type&&d.built>=4&&b.written.every(w=>w.path?.startsWith('user/builds/'))}catch(e){d={error:String(e&&e.stack||e)}}
 console.log('PLAY_SELFTEST '+(ok?'PASS':'FAIL')+' '+JSON.stringify(d))}
