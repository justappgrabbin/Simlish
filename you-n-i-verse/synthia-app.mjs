// Page script: starts the in-process server (integrated/app-server.mjs) and shows what the swarm is doing. No worker, no network.
import {createAppServer} from './integrated/app-server.mjs';
const $=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const loadBytes=async p=>{const r=await fetch(p);if(!r.ok)throw new Error(p+' '+r.status);return new Uint8Array(await r.arrayBuffer())};
const loadJson=async p=>{const r=await fetch(p);if(!r.ok)throw new Error(p+' '+r.status);return r.json()};
const store={get:k=>{try{return localStorage.getItem(k)}catch{return null}},set:(k,v)=>{try{localStorage.setItem(k,v)}catch{}}};
const srv=await createAppServer({loadJson,loadBytes,store});
window.synthia=srv; // so the person (or a test) can poke the same server
const api=(m,u,b)=>srv.handle(m,u,b).then(r=>r.body);
let seen=0;
async function refresh(){
 const f=await api('GET','/api/feed?since='+seen);seen=f.total;const feed=$('feed');
 for(const e of f.events){const d=document.createElement('div');d.className=e.phase;d.textContent=(e.t||'').slice(11,19)+' '+e.phase+' · '+e.msg;feed.prepend(d)}
 while(feed.childElementCount>150)feed.lastChild.remove();
 const p=f.progress;$('pbar').style.width=(100*p.done/Math.max(1,p.total))+'%';$('ptext').textContent=p.done+' of '+p.total+' units assembled'+(p.failed?' · '+p.failed+' failed':'');
 $('units').innerHTML=Object.entries(f.units).map(([k,v])=>'<span class="'+v+'" title="'+esc(k)+': '+v+'"></span>').join('');
 const s=await api('GET','/api/status');$('pulse').textContent='heartbeat '+s.heartbeat.beats+' · cycle '+s.loop.cycle+(s.loop.sleeping?' · asleep (plan complete)':s.loop.running?' · running':' · paused');
 $('chain').innerHTML='hash chain: <b class="'+(s.audit.chain.ok?'ok':'bad')+'">'+(s.audit.chain.ok?'intact':'BROKEN at #'+s.audit.chain.brokenAt)+'</b> · '+s.audit.length+' entries · '+s.versions+' versions';
 $('prot').innerHTML=s.protected.originals+' originals registered · checked '+s.protected.checked+' so far · '+(s.protected.ok===null?'first check pending':s.protected.ok?'<b class="ok">none changed</b>':'<b class="bad">CHANGED: '+esc([...s.protected.changed,...s.protected.missing].join(', '))+'</b>');
 $('modelnote').textContent=s.model.loaded?'Trident loaded: '+s.model.params.toLocaleString()+' parameters, character model, no RAG. It continues text in the style of her training sentences; it does not answer questions reliably.':'model: '+s.model.reason;
 const au=(await api('GET','/api/audit?n=25')).entries.slice().reverse();$('audit').innerHTML=au.map(e=>'<div class="'+e.type+'">#'+e.seq+' '+e.type+' '+esc(e.path||'')+' <span class="note">'+e.hash.slice(0,10)+'</span><br><span class="note">'+esc(e.description)+'</span></div>').join('');
 const fl=(await api('GET','/api/files')).files.filter(x=>x.path.startsWith('app/')||x.path.startsWith('user/'));$('files').innerHTML=fl.map(x=>'<div><a href="#" data-p="'+esc(x.path)+'">'+esc(x.path)+'</a> <span class="note">'+x.size+' B</span></div>').join('')||'<span class="note">nothing yet</span>';
}
$('files').onclick=async e=>{const p=e.target.dataset?.p;if(!p)return;e.preventDefault();const r=await api('GET','/api/file?path='+encodeURIComponent(p));if(/\.html$/.test(p))$('viewer').srcdoc=r.text;else $('viewer').srcdoc='<pre>'+esc(r.text)+'</pre>'};
$('start').onclick=()=>api('POST','/api/loop/start');$('stop').onclick=()=>api('POST','/api/loop/stop');$('step').onclick=async()=>{await api('POST','/api/loop/cycle');refresh()};
const log=(c,t)=>{const p=document.createElement('p');p.className=c;p.textContent=t;$('chatlog').append(p);$('chatlog').scrollTop=1e9};
$('send').onclick=async()=>{const t=$('say').value.trim();if(!t)return;$('say').value='';log('me',t);$('send').disabled=true;
 const h=$('head').value;const r=await api('POST','/api/chat',{text:t,maxNew:40,head:h===''?null:Number(h),yieldMs:0});$('send').disabled=false;
 log('bot',r.error?('(error) '+r.error):r.reply+'   [router code '+r.router.code.toFixed(2)+' · math '+r.router.math.toFixed(2)+' · research '+r.router.research.toFixed(2)+']')};
$('say').onkeydown=e=>{if(e.key==='Enter')$('send').click()};
const tl=(await api('GET','/api/tools')).tools;$('tools').innerHTML=tl.map(t=>'<div><b>'+esc(t.name)+'</b> — '+esc(t.description)+'</div>').join('');
$('runtool').onclick=async()=>{let a={};try{a=JSON.parse($('toolargs').value||'{}')}catch{$('toolout').textContent='args must be JSON';return}
 const n=$('toolname').value.trim(),r=n==='chat'?await api('POST','/api/chat',a):await api('POST','/api/tool/'+n,a);$('toolout').textContent=JSON.stringify(r,null,1).slice(0,4000)};
srv.loop.on(()=>{});setInterval(refresh,1000);await refresh();srv.start(); // starts by itself: no external controller
// Self-test marker for CI/emulator runs (page opened with ?selftest=1): proves on the real WebView that the swarm assembled everything, the tools answer, and her model runs.
if(location.search.includes('selftest')){
 const t0=Date.now(),wait=async()=>{while((await api('GET','/api/feed?since=0')).progress.done<(await api('GET','/api/feed?since=0')).progress.total&&Date.now()-t0<240000)await new Promise(r=>setTimeout(r,500))};
 await wait();const f=await api('GET','/api/feed?since=999999'),st=await api('GET','/api/status'),g=await api('POST','/api/tool/primitives.get',{id:'gate.1'});
 let chat;try{chat=await api('POST','/api/chat',{text:'hello',maxNew:8,temp:0})}catch(e){chat={error:String(e)}}
 const ok=st.loop.done===st.loop.units&&st.loop.failed===0&&st.audit.chain.ok&&g.primitive?.form==='The Creative'&&st.model.loaded&&typeof chat.reply==='string';
 console.log('SYNTHIA_SELFTEST '+(ok?'PASS':'FAIL')+' '+JSON.stringify({done:st.loop.done,units:st.loop.units,failed:st.loop.failed,chain:st.audit.chain.ok,auditEntries:st.audit.length,protectedOk:st.protected.ok,model:st.model.loaded,params:st.model.params,gate1:g.primitive?.form,chatChars:chat.reply?.length,chatError:chat.error||null,ms:Date.now()-t0}));
}
