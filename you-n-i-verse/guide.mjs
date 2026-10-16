// Guide page: everything goes through the in-process app server (integrated/app-server.mjs). No network, no worker.
import {createAppServer} from './integrated/app-server.mjs';
import {CENTERS,CENTER_OF_GATE} from './integrated/chart-engine.mjs';
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const loadBytes=async p=>{const r=await fetch(p);if(!r.ok)throw new Error(p+' '+r.status);return new Uint8Array(await r.arrayBuffer())};
const loadJson=async p=>{const r=await fetch(p);if(!r.ok)throw new Error(p+' '+r.status);return r.json()};
const store={get:k=>{try{return localStorage.getItem(k)}catch{return null}},set:(k,v)=>{try{localStorage.setItem(k,v)}catch{}}};
const srv=await createAppServer({loadJson,loadBytes,store,loadModel:false});window.synthia=srv;
const call=(n,b)=>srv.handle('POST','/api/tool/'+n,b||{}).then(r=>{if(!r.ok)throw new Error(r.body.error||'failed');return r.body});
let chart=null;
const FLAGS=[
 ['Which zodiac wheel is yours','Charts use the SynthAi isohuman wheel (gate 41 starts at 2°00\' Aquarius = 302°). It matches the published Human Design table, doc3 (gate 1 in Scorpio) and NatalEngine. Your sentence-engine.ts wheel starts gate 41 at 272° (30° earlier); both are kept, none chosen for her.'],
 ['2.5-minute boundary','Free Human Design places every gate/line boundary 2′30″ later than your donor and NatalEngine. This moves a line (rarely a gate) in about 4% of placements. Using the donor\'s exact table.'],
 ['House numbers in the donor','The donor\'s ascendant is actually the descendant (180° off), which shifts its houses by 6. The app uses the corrected ascendant (checked against Free Human Design to 0.01°); the donor file is unchanged. Houses are equal houses, not Placidus.'],
 ['Gate 22 center','Your attribute field centerSG says Throat; your Black Book text and the standard map say Solar Plexus. Defined centers use Solar Plexus.'],
 ['North Node','Your donor interpolates the true node between node events; Free Human Design differs by up to 0.31°. Node gates can differ near a boundary.'],
 ['Standard rules (not from her sources)','The 36-channel table, Type, Authority, Profile, Definition, aspect angles/orbs (8° for conjunction/square/trine/opposition, 6° sextile) and the relationship terms (companionship, electromagnetic, dominance, compromise) are standard Human Design/astrology conventions, labelled in every output.'],
 ['Missing','Incarnation cross, circuitry, variables/arrows, Placidus houses, a place search, planetary transits, and prose descriptions of Type/Authority (none are in her data here). The Trident chat is a toy and is not used by the guide.']];
$('flaglist').innerHTML=FLAGS.map(([t,x])=>`<div class="sec"><b>${esc(t)}</b><br><span class="flag">${esc(x)}</span></div>`).join('');
// tabs
const show=t=>{for(const b of document.querySelectorAll('#tabs button'))b.classList.toggle('on',b.dataset.t===t);for(const s of document.querySelectorAll('section.tab'))s.classList.toggle('on',s.id==='t-'+t);if(t==='saved'||t==='compare')refreshSaved()};
$('tabs').onclick=e=>{if(e.target.dataset.t)show(e.target.dataset.t)};
// tz
try{$('tz').value=Intl.DateTimeFormat().resolvedOptions().timeZone||'';$('tzs').innerHTML=(Intl.supportedValuesOf?Intl.supportedValuesOf('timeZone'):[]).map(z=>`<option value="${z}">`).join('')}catch{}
const input=()=>({name:$('name').value.trim(),date:$('date').value,time:$('time').value,timeZone:$('tz').value.trim(),latitude:parseFloat($('lat').value),longitude:parseFloat($('lon').value)});
const rowsHtml=list=>list.map(a=>`<tr class="act" data-o="${a.orientation}" data-b="${esc(a.body)}"><td>${esc(a.body)}</td><td>${esc(a.text)}</td><td>${a.house}</td><td><b>${a.gate}.${a.line}</b></td><td>${a.color}.${a.tone}.${a.base}</td></tr>`).join('');
function bodygraph(h){const pos={Head:[100,20],Ajna:[100,58],Throat:[100,106],G:[100,158],Heart:[145,150],Spleen:[40,200],'Solar Plexus':[160,200],Sacral:[100,214],Root:[100,262]};
  return `<svg viewBox="0 0 200 290" role="img" aria-label="centers, defined ones filled">${h.channels.map(([a,b])=>{const ca=h.centerOf[a],cb=h.centerOf[b];return `<line x1="${pos[ca][0]}" y1="${pos[ca][1]}" x2="${pos[cb][0]}" y2="${pos[cb][1]}" stroke="#9dffb0" stroke-width="3"><title>${a}-${b}</title></line>`}).join('')}
 ${CENTERS.map(c=>{const d=h.definedCenters.includes(c);return `<g><circle cx="${pos[c][0]}" cy="${pos[c][1]}" r="15" fill="${d?'#7c6cff':'#13132f'}" stroke="#a9a3ff"/><text x="${pos[c][0]}" y="${pos[c][1]+3}" font-size="7" text-anchor="middle" fill="#fff">${c==='Solar Plexus'?'SP':c}</text></g>`}).join('')}</svg><p class="note">Generic layout of the nine centers (positions are only for drawing). Filled = defined.</p>`}
function render(c){chart=c;const h=c.hd,A=c.astrology;h.centerOf=Object.fromEntries(c.hd.gates.map(g=>[g,centerOf(g)]));
 const dm=x=>x.slice(0,16).replace('T',' ')+' UTC';
 $('result').innerHTML=`<div class="card"><h3>${esc(c.name||'Chart')} · design</h3><div class="kv"><b>Type</b><span>${esc(h.type)}</span><b>Authority</b><span>${esc(h.authority)}</span><b>Profile</b><span>${esc(h.profile)}</span><b>Definition</b><span>${esc(h.definition)}</span><b>Defined</b><span>${esc(h.definedCenters.join(', ')||'none')}</span><b>Open</b><span>${esc(h.openCenters.join(', ')||'none')}</span><b>Channels</b><span>${esc(h.channels.map(x=>x.join('-')).join(', ')||'none')}</span></div><p class="flag">Type, Authority, Profile, Definition, channels: standard rules, not from her sources — confirm (see “To confirm”).</p>${bodygraph(h)}</div>
 <div class="card"><h3>Moments</h3><div class="kv"><b>Birth (UTC)</b><span>${dm(c.utc)}</span><b>Design (88° of sun before)</b><span>${dm(c.designUtc)}</span><b>Ascendant</b><span>${esc(A.ascendant.text)}</span></div><p class="note">${esc(c.methods.ephemeris)}. Houses: ${esc(A.houseMethod)}.</p></div>
 <div class="card"><h3>Personality (conscious) — tap a row</h3><div class="scroll"><table><tr><th>Body</th><th>Position</th><th>H</th><th>G.L</th><th>C.T.B</th></tr>${rowsHtml(c.activations.filter(a=>a.orientation==='personality'))}</table></div></div>
 <div class="card"><h3>Design (unconscious) — tap a row</h3><div class="scroll"><table><tr><th>Body</th><th>Position</th><th>H</th><th>G.L</th><th>C.T.B</th></tr>${rowsHtml(c.activations.filter(a=>a.orientation==='design'))}</table></div></div>
 <div class="card"><h3>Aspects (birth moment, tropical)</h3><p class="flag">Angles and orbs are conventional, not from her sources.</p><div class="scroll"><table>${A.aspects.map(x=>`<tr><td>${esc(x.a)}</td><td>${x.aspect}</td><td>${esc(x.b)}</td><td>${x.orb}°</td></tr>`).join('')}</table></div></div>
 <div class="card"><h3>Other zodiac frames (birth moment)</h3><div class="scroll"><table><tr><th>Body</th><th>Sidereal (Fagan-Bradley)</th><th>Draconic</th></tr>${A.sidereal.map((s,i)=>`<tr><td>${esc(s.body)}</td><td>${esc(s.text)}</td><td>${esc(A.draconic[i].text)}</td></tr>`).join('')}</table></div></div>`;
 srv.persist();}
const centerOf=g=>CENTER_OF_GATE[g];
$('calc').onclick=async()=>{$('err').textContent='';try{const c=await call('chart.calculate',input());render(c)}catch(e){$('err').textContent=e.message}};
$('save').onclick=async()=>{if(!chart){$('err').textContent='Calculate a chart first';return}await call('people.save',{chart});$('err').textContent='';$('err').className='ok';$('err').textContent='Saved on this device.';setTimeout(()=>{$('err').className='bad';$('err').textContent=''},2000)};
$('result').onclick=async e=>{const tr=e.target.closest('tr.act');if(!tr||!chart)return;const a=chart.activations.find(x=>x.orientation===tr.dataset.o&&x.body===tr.dataset.b);const r=await call('guide.ask',{q:`${a.gate}.${a.line}.${a.color}.${a.tone}.${a.base}`,chart});
 $('detail').innerHTML=`<h3>${esc(a.orientation)} ${esc(a.body)} · ${esc(a.text)} · house ${a.house}</h3><p class="note">primitive ids: ${Object.values(a.ids).map(esc).join(' · ')}</p>`+answerHtml(r);show('detail')};
const answerHtml=r=>`<div class="card"><h3>${esc(r.title)}</h3>${r.sections.map(s=>s.text?`<div class="sec"><b>${esc(s.label)}</b><br>${esc(s.text)} ${s.from&&s.from!=='—'?`<small>[${esc(s.from)}]</small>`:''}</div>`:`<div class="sec"><b>${esc(s.label)}</b></div>`).join('')}</div>`;
$('ask').onclick=async()=>{const r=await call('guide.ask',{q:$('q').value,chart});$('answer').innerHTML=answerHtml(r)};$('q').onkeydown=e=>{if(e.key==='Enter')$('ask').click()};
async function refreshSaved(){const {people}=await call('people.list');const opts=people.map(p=>`<option value="${p.id}">${esc(p.name||p.input.date)} · ${esc(p.hd.type)} ${esc(p.hd.profile)}</option>`).join('');$('ca').innerHTML=opts;$('cb').innerHTML=opts;if(people.length>1)$('cb').selectedIndex=1;
 $('savedlist').innerHTML=people.length?people.map(p=>`<div class="sec"><b>${esc(p.name||'(no name)')}</b> · ${esc(p.input.date)} ${esc(p.input.time)} ${esc(p.input.timeZone||'')} · ${esc(p.hd.type)} ${esc(p.hd.profile)}<div class="row" style="margin-top:6px"><button data-l="${p.id}">Open</button><button data-d="${p.id}">Delete</button></div></div>`).join(''):'<p class="note">Nothing saved yet.</p>'}
$('savedlist').onclick=async e=>{if(e.target.dataset.l){const {chart:c}=await srv.handle('POST','/api/tool/people.get',{id:e.target.dataset.l}).then(r=>r.body);const i=c.input;$('name').value=c.name;$('date').value=i.date;$('time').value=i.time;$('tz').value=i.timeZone||'';$('lat').value=i.latitude;$('lon').value=i.longitude;render(c);show('chart')}
 if(e.target.dataset.d){await call('people.delete',{id:e.target.dataset.d});refreshSaved()}};
$('cmp').onclick=async()=>{const get=async id=>(await srv.handle('POST','/api/tool/people.get',{id})).body.chart;const a=await get($('ca').value),b=await get($('cb').value);if(!a||!b){$('cmpout').innerHTML='<p class="bad">Pick two saved charts.</p>';return}
 const r=await call('chart.compare',{a,b});const L=r.relationship;$('cmpout').innerHTML=`<div class="card"><h3>${esc(a.name||'A')} ↔ ${esc(b.name||'B')} · channels</h3><p class="flag">${esc(r.labels.relationship)}</p>${L.length?L.map(x=>`<div class="sec"><b>${x.channel}</b> ${x.kind}${x.holder?' (held by '+x.holder+')':''}</div>`).join(''):'<p class="note">No channel relationships.</p>'}</div>
 <div class="card"><h3>Centers</h3><table><tr><th></th><th>A</th><th>B</th></tr>${r.centers.map(c=>`<tr><td>${c.center}</td><td>${c.A}</td><td>${c.B}</td></tr>`).join('')}</table></div>
 <div class="card"><h3>Shared gates</h3><p>${r.sharedGates.join(', ')||'none'}</p></div>
 <div class="card"><h3>Her emergence rule (Hamming-1, ${r.emergence.pairs.length} pairs)</h3><p class="note">${esc(r.emergence.rule)}</p><div class="scroll"><table>${r.emergence.pairs.slice(0,40).map(p=>`<tr><td>${esc(p.A)}</td><td>${esc(p.B)}</td><td>${p.gateLine?'gate/line ':''}${p.colorToneBase?'color/tone/base':''}</td></tr>`).join('')}</table></div></div>
 <div class="card"><h3>Synastry aspects</h3><p class="flag">${esc(r.labels.synastry)}</p><div class="scroll"><table>${r.synastry.slice(0,60).map(x=>`<tr><td>A ${esc(x.A)}</td><td>${x.aspect}</td><td>B ${esc(x.B)}</td><td>${x.orb}°</td></tr>`).join('')}</table></div></div>`};
// progress feed (the swarm keeps assembling by itself)
let seen=0;async function feed(){const f=await srv.handle('GET','/api/feed?since='+seen).then(r=>r.body);seen=f.total;const el=$('feed');for(const e of f.events){const d=document.createElement('div');d.className=e.phase;d.textContent=(e.t||'').slice(11,19)+' '+e.phase+' · '+e.msg;el.prepend(d)}while(el.childElementCount>150)el.lastChild.remove();const p=f.progress;$('pbar').style.width=(100*p.done/Math.max(1,p.total))+'%';$('ptext').textContent=p.done+' of '+p.total+' units assembled'+(p.failed?' · '+p.failed+' failed':'')}
setInterval(feed,1000);await feed();srv.start();
// self-test for CI / emulator (?selftest=1)
if(location.search.includes('selftest')){
 let ok=false,detail={};try{
  const c=await call('chart.calculate',{name:'selftest',date:'1965-09-19',time:'02:52',utcOffsetMinutes:0,latitude:46.77,longitude:-105.3});
  const g=await call('guide.ask',{q:'gate 13.2'}),d=await call('guide.ask',{q:'my profile',chart:c}),s=await call('people.save',{chart:c}),c2=await call('chart.calculate',{name:'b',date:'1984-02-03',time:'03:15',timeZone:'Europe/London',latitude:51.5,longitude:-0.12}),cmp=await call('chart.compare',{a:c,b:c2});
  detail={type:c.hd.type,authority:c.hd.authority,profile:c.hd.profile,definition:c.hd.definition,channels:c.hd.channels.map(x=>x.join('-')).join(','),asc:Math.round(c.astrology.ascendant.longitude*1000)/1000,guideOk:g.ok,saved:s.count,cmpCenters:cmp.centers.length,activations:c.activations.length};
  ok=detail.type==='Manifesting Generator'&&detail.authority==='Emotional'&&detail.profile==='4/6'&&detail.definition==='triple split'&&detail.channels==='5-15,9-52,12-22,35-36,47-64'&&detail.guideOk&&detail.saved>=1&&detail.cmpCenters===9&&detail.activations===26&&d.text.includes('4/6');
 }catch(e){detail={error:String(e)}}
 console.log('GUIDE_SELFTEST '+(ok?'PASS':'FAIL')+' '+JSON.stringify(detail));
}
