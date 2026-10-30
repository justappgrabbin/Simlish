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
function open(a){
  if(a.status!=='ready'){ alert(`${a.name} isn't built as a web app yet.\n${a.note||''}`); return; }
  const r=gateApp(a); if(!r) return;
  const edits=store(EDITS)[a.id]||{}, base=`${location.href.replace(/[^/]*$/,'')}${a.folder}/`;
  const f=$('#appFrame'); f.removeAttribute('srcdoc'); f.src='about:blank';
  if(edits[a.entry]!=null) f.srcdoc=edits[a.entry].replace(/<head([^>]*)>/i,`<head$1><base href="${base}">`); else f.src=`${a.folder}/${a.entry}`;
  $('#appTitle').textContent=`${a.icon} ${a.name} · ${r.record.address} · ${r.record.gate||''}`; $('#appOverlay').hidden=false;
}
function render(){
  const addrs=store(ADDR);
  $('#appsGrid').innerHTML=apps.map(a=>`<div class="app-tile ${a.status!=='ready'?'pending':''}"><button class="app-open" data-id="${esc(a.id)}"><span class="app-icon">${esc(a.icon)}</span><strong>${esc(a.name)}</strong><small>${a.status==='ready'?esc(addrs[a.id]||'no address yet'):'not built yet'}</small></button><button class="app-edit" data-id="${esc(a.id)}" title="Edit files">✎</button></div>`).join('');
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
  if(e.target.id==='appsAdd'){ const zipOrFolder=prompt('Add a web app already inside apps/: folder name (e.g. myapp)'); if(!zipOrFolder) return; const id=zipOrFolder.replace(/\W+/g,'-'); const a={id,name:zipOrFolder,icon:'▢',folder:`apps/${zipOrFolder}`,entry:'index.html',status:'ready'}; const g=gateApp(a); if(!g) return; const x=store('computerAppsExtra'); x[id]=a; save('computerAppsExtra',x); load(); }});
load(); globalThis.ComputerApps={ reload:load, list:()=>apps };
