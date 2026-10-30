import { admit, ledger } from './address-gate.mjs';
const $=s=>document.querySelector(s), esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ADDR='computerAppAddresses', EDITS='computerAppEdits';
const store=k=>{try{return JSON.parse(localStorage[k]||'{}')}catch{return {}}}, save=(k,v)=>localStorage[k]=JSON.stringify(v);
let apps=[];
async function load(){ apps=await (await fetch('./apps/apps.json',{cache:'no-store'})).json(); const extra=store('computerAppsExtra'); apps=[...apps,...Object.values(extra)]; render(); }
function addrFor(a){ return store(ADDR)[a.id]||a.address||''; }
function gateApp(a){
  let address=addrFor(a);
  if(!address){ address=prompt(`Address for ${a.name}? (gate.line.color.tone.base, e.g. 18.5.3.4.2). Every app must pass the address engine before it opens.`); if(!address) return null; }
  const r=admit({kind:'app',id:a.id,name:a.name,address});
  if(!r.ok){ alert(`Address gate refused ${a.name}: ${r.reason}`); return null; }
  const m=store(ADDR); m[a.id]=r.record.address; save(ADDR,m); return r;
}
const NA=globalThis.NativeApps||null;
export const CONN_KEY='computerConnections';
export function connections(){return {p2p:false,p2pByHD:false,local:false,remote:false,localUrl:'http://127.0.0.1:8080',remoteUrl:'',...store(CONN_KEY)};}
function nativeResult(m){ $('#appsNativeStatus')&&($('#appsNativeStatus').textContent=m); }
addEventListener('native-app-result',e=>{nativeResult(e.detail.message); alert(e.detail.message);});
addEventListener('native-piece-picked',e=>{const d=e.detail; if(!d.ok) return alert('Could not add: '+d.message);
  const x=store('computerAppsExtra');
  const a=d.kind==='apk'?{id:d.id,type:'native',name:d.name.replace(/\.apk$/i,''),icon:'📦',package:d.package,localApk:true,status:'ready',address:d.address}
                       :{id:d.id,type:'web',name:d.name.replace(/\.zip$/i,''),icon:'▢',folder:d.folder,entry:d.entry,status:'ready',address:d.address};
  x[a.id]=a; save('computerAppsExtra',x); const m=store(ADDR); m[a.id]=d.address; save(ADDR,m);
  if(d.kind==='apk'&&!d.installed&&confirm(`${a.name} (${d.package}) isn't installed. Install it now?`)) NA.installStored(d.id,d.address);
  load(); });
export function openNative(a,bridge=NA){
  if(!bridge){ alert(`${a.name} is a native Android app (${a.package}). Open this Computer as the Android APK to launch it.`); return false; }
  if(!a.package){ alert(`${a.name} isn't built yet.\n${a.note||''}`); return false; }
  const r=gateApp(a); if(!r) return false;
  if(bridge.isInstalled(a.package)){ const res=JSON.parse(bridge.launch(a.package,r.record.address)); if(!res.ok) alert(res.message); else if(a.opens==='keyboard-setup') setTimeout(()=>bridge.openKeyboardSettings(),400); return res.ok; }
  if(a.localApk){ bridge.installStored(a.id,r.record.address); return true; }
  if(a.apkSource && confirm(`${a.name} isn't installed. Download and install it from\n${a.apkSource}?\n(Or Cancel and use + Add app to pick the APK file.)`)){ bridge.installFromUrl(a.apkSource,a.id,r.record.address); nativeResult(`Downloading ${a.name}…`); return true; }
  if(!a.apkSource) alert(`${a.name} isn't installed. Use + Add app and pick its APK file.`);
  return false;
}
function open(a){
  if(a.type==='native') return openNative(a);
  if(a.status!=='ready'){ alert(`${a.name} isn't built as a web app yet.\n${a.note||''}`); return; }
  const r=gateApp(a); if(!r) return;
  const edits=store(EDITS)[a.id]||{}, base=/^https?:/.test(a.folder)?`${a.folder}/`:`${location.href.replace(/[^/]*$/,'')}${a.folder}/`;
  const f=$('#appFrame'); f.removeAttribute('srcdoc'); f.src='about:blank';
  if(edits[a.entry]!=null) f.srcdoc=edits[a.entry].replace(/<head([^>]*)>/i,`<head$1><base href="${base}">`); else f.src=`${a.folder}/${a.entry}`;
  $('#appTitle').textContent=`${a.icon} ${a.name} · ${r.record.address} · ${r.record.gate||''}`; $('#appOverlay').hidden=false;
  f.onload=()=>{ guardUploads(f,a); if(a.id==='paper'){ mobileLayer(f,'mobile/paper-mobile.css'); neuralNotice(f); } };
}
// Phone-first layout layer, added from outside so the app's own files stay unchanged.
function mobileLayer(f,href){ try{ const d=f.contentDocument; if(!d||d.getElementById('computerMobileLayer')) return; const l=d.createElement('link'); l.id='computerMobileLayer'; l.rel='stylesheet'; l.href=new URL(href,location.href).href; d.head.appendChild(l); }catch{} }
// Every file a hosted app takes in passes the address gate first (the app itself is not modified).
// Capture-phase listener on the app's document runs before the app's own upload handler; refused files never reach it.
export function guardUploads(f,a){
  let doc; try{ doc=f.contentDocument; }catch{ return; } if(!doc||doc.__gateGuard) return; doc.__gateGuard=true;
  doc.addEventListener('change',e=>{
    const input=e.target; if(!input||input.type!=='file'||!input.files?.length) return;
    const address=addrFor(a);
    for(const file of input.files){ const g=admit({kind:'upload',id:`${a.id}/${file.name}`,name:file.name,address});
      if(!g.ok){ e.stopImmediatePropagation(); e.stopPropagation(); input.value=''; alert(`Address gate refused upload ${file.name}: ${g.reason}`); render(); return; } }
    render();
  },true);
}
// TinyLlama needs WebGPU with shader-f16. Say plainly when this phone lacks it; everything else in Paper still works.
async function neuralNotice(f){
  let ok=false, why='this WebView has no WebGPU';
  try{ const gpu=f.contentWindow?.navigator?.gpu; if(gpu){ const ad=await gpu.requestAdapter(); if(!ad) why='no WebGPU adapter'; else if(!ad.features.has('shader-f16')) why='WebGPU is present but without shader-f16'; else ok=true; } }catch(e){ why=e.message; }
  const t=$('#appTitle'); if(t) t.textContent+= ok?' · TinyLlama: available (first load downloads weights)':` · TinyLlama off on this phone (${why}); the rest of Paper works offline`;
}
function render(){
  const addrs=store(ADDR);
  $('#appsGrid').innerHTML=apps.map(a=>`<div class="app-tile ${a.status!=='ready'?'pending':''}"><button class="app-open" data-id="${esc(a.id)}"><span class="app-icon">${esc(a.icon)}</span><strong>${esc(a.name)}</strong><small>${a.status!=='ready'?'not built yet':a.type==='native'?(NA?(NA.isInstalled(a.package)?'native · installed':'native · tap to install'):'native app'):esc(addrs[a.id]||'no address yet')}</small></button>${a.type==='native'?'':`<button class="app-edit" data-id="${esc(a.id)}" title="Edit files">✎</button>`}</div>`).join('');
  $('#appsLedger').textContent=ledger().slice(0,20).map(x=>`${x.at.slice(0,19)}  ${x.kind}  ${x.name}  → ${x.address} (${x.gate||'?'})`).join('\n')||'Nothing has passed the gate yet.';
}
async function edit(a){
  const ed=$('#appEditor'); ed.hidden=false; $('#edTitle').textContent=`Edit ${a.name} (${a.folder})`;
  $('#edFile').innerHTML=(a.files||[]).map(f=>`<option>${esc(f)}</option>`).join('')||'<option>index.html</option>';
  const show=async()=>{const f=$('#edFile').value, e=store(EDITS)[a.id]||{}; $('#edText').value=e[f]??await fetch(`${a.folder}/${f}`).then(r=>r.ok?r.text():'').catch(()=>'');};
  $('#edFile').onchange=show; await show();
  $('#edSave').onclick=()=>{const g=admit({kind:'file',id:`${a.id}/${$('#edFile').value}`,name:$('#edFile').value,address:addrFor(a)||prompt('Address for this file?')}); if(!g.ok) return alert('Address gate refused: '+g.reason); const all=store(EDITS); (all[a.id]??={})[$('#edFile').value]=$('#edText').value; save(EDITS,all); $('#edStatus').textContent=`Saved in this browser through gate ${g.record.address}. Use Export to keep it.`; render();};
  $('#edRevert').onclick=()=>{const all=store(EDITS); if(all[a.id]) delete all[a.id][$('#edFile').value]; save(EDITS,all); show();};
  $('#edExport').onclick=()=>{const blob=new Blob([JSON.stringify({app:a.id,folder:a.folder,address:addrFor(a),edits:store(EDITS)[a.id]||{}},null,1)],{type:'application/json'}); const l=document.createElement('a'); l.href=URL.createObjectURL(blob); l.download=`${a.id}-edits.json`; l.click();};
  $('#edImport').onchange=async ev=>{const file=ev.target.files[0]; if(!file) return; const g=admit({kind:'file',id:`${a.id}/${file.name}`,name:file.name,address:addrFor(a)||prompt('Address for this file?')}); if(!g.ok) return alert('Address gate refused: '+g.reason); const text=await file.text(); const all=store(EDITS); all[a.id]??={};
    if(file.name.endsWith('-edits.json')){ try{Object.assign(all[a.id],JSON.parse(text).edits||{})}catch{} } else all[a.id][file.name]=text; save(EDITS,all); $('#edStatus').textContent=`Imported ${file.name} through gate ${g.record.address}.`; render(); show();};
}
document.addEventListener('click',e=>{const o=e.target.closest('.app-open'),d=e.target.closest('.app-edit'); const a=apps.find(x=>x.id===(o||d)?.dataset.id); if(o&&a) open(a); if(d&&a) edit(a);
  if(e.target.id==='appClose'){ $('#appOverlay').hidden=true; $('#appFrame').src='about:blank'; } if(e.target.id==='edClose') $('#appEditor').hidden=true;
  if(e.target.id==='appsAdd'){
    if(NA){ const address=prompt('Address for the new app? (gate.line.color.tone.base). Then pick an .apk or a .zip of a web app folder. The app itself is not changed.'); if(!address) return;
      const g=admit({kind:'app-add',id:'new-app',name:'picked file',address}); if(!g.ok) return alert('Address gate refused: '+g.reason); NA.pickPiece(g.record.address); return; }
    const zipOrFolder=prompt('Add a web app already inside apps/: folder name (e.g. myapp)'); if(!zipOrFolder) return; const id=zipOrFolder.replace(/\W+/g,'-'); const a={id,type:'web',name:zipOrFolder,icon:'▢',folder:`apps/${zipOrFolder}`,entry:'index.html',status:'ready'}; const g=gateApp(a); if(!g) return; const x=store('computerAppsExtra'); x[id]=a; save('computerAppsExtra',x); load(); }
  if(e.target.id==='connSave'){ const c={p2p:$('#connP2p').checked,local:$('#connLocal').checked,remote:$('#connRemote').checked,localUrl:$('#connLocalUrl').value.trim(),remoteUrl:$('#connRemoteUrl').value.trim()}; save(CONN_KEY,c); $('#connStatus').textContent='Saved. All off is fine: everything on this device keeps working.'; }
});
function initConn(){const c=connections(); if(!$('#connP2p')) return; $('#connP2p').checked=c.p2p; $('#connP2pHd').checked=c.p2pByHD; $('#connLocal').checked=c.local; $('#connRemote').checked=c.remote; $('#connLocalUrl').value=c.localUrl; $('#connRemoteUrl').value=c.remoteUrl;}
load(); initConn(); globalThis.ComputerApps={ reload:load, list:()=>apps, connections };

// P2P eligibility. Off by default: with p2pByHD off, everyone can use P2P.
// With it on, the rule comes ONLY from apps/p2p-hd-rules.json (left empty on purpose: Alexis fills it in).
export async function p2pEligible(chart,conn=connections(),rules=null){
  if(!conn.p2p) return {ok:false,reason:'P2P is off'};
  if(!conn.p2pByHD) return {ok:true,reason:'open to everyone'};
  rules??=await fetch('./apps/p2p-hd-rules.json').then(r=>r.json()).catch(()=>({}));
  const need=rules.requireDefinedCenters||[], anyOf=rules.anyDefinedCenters||[];
  if(!need.length&&!anyOf.length) return {ok:true,reason:'HD rule config is empty, so nobody is blocked yet'};
  const def=new Set(chart?.definedCenters||[]);
  if(!need.every(c=>def.has(c))) return {ok:false,reason:'chart is missing required defined centers'};
  if(anyOf.length&&!anyOf.some(c=>def.has(c))) return {ok:false,reason:'chart has none of the listed centers'};
  return {ok:true,reason:'matches HD rule config'};
}
