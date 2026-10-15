// Uploader-facing panels added to the studio: personalize, register, and finish an app. Additive; builds its own DOM.
import {personalize,readPersonalization,TARGETS} from './personalize.mjs';
import {bootSimlish} from './boot.mjs';
import {ADDRESS_ORDER} from './address-resolution.mjs';
import {finishApp} from './app-finisher.mjs';
import {storageOutbox} from './supabase-original.mjs';
const env=globalThis.SIMLISH_CONFIG||{},outbox=storageOutbox(localStorage);
const el=(tag,props={},...kids)=>{const e=Object.assign(document.createElement(tag),props);e.append(...kids);return e};
const panel=el('details',{id:'simlishPanel',open:true},el('summary',{textContent:'Make it yours · register · finish an app'}));
const status=el('p',{className:'hint',role:'status',id:'simlishStatus',textContent:'Starting…'});
const name=el('input',{id:'upName',placeholder:'Your name'}),theme=el('input',{id:'upTheme',placeholder:'Your world theme (set once)'});
const target=el('select',{id:'upTarget'});for(const t of TARGETS){const o=el('option',{value:t,textContent:t==='reality'?'reality':t+' (not yet implemented)'});if(t!=='reality')o.disabled=true;target.append(o)}
const places=Object.fromEntries(ADDRESS_ORDER.map(k=>[k,el('input',{id:'addr_'+k,placeholder:k,size:8})]));
const save=el('button',{id:'upSave',textContent:'Save and register'}),appFile=el('input',{type:'file',id:'appFile'}),finish=el('button',{id:'appFinish',textContent:'Finish this app'});
const out=el('div',{id:'finishOut'});
panel.append(el('label',{},'Name',name),el('label',{},'World theme',theme),el('label',{},'Build target',target),
 el('p',{className:'hint',textContent:'Your full address, all 13 places. Anything left empty is kept and marked investigating, not rejected.'}),
 el('div',{style:'display:flex;flex-wrap:wrap;gap:4px'},...ADDRESS_ORDER.map(k=>places[k])),save,
 el('label',{className:'secondary'},'Put in an app (zip or any file)',appFile),finish,out,status);
document.querySelector('aside').append(panel);
function fill(){const p=readPersonalization(localStorage);if(!p)return;name.value=p.name||'';theme.value=p.worldTheme||'';theme.disabled=!!p.worldTheme;target.value=p.target||'reality';for(const k of ADDRESS_ORDER)if(p.address&&p.address[k]!==undefined)places[k].value=p.address[k]}
function addressFromForm(){const a={};for(const k of ADDRESS_ORDER){const v=places[k].value.trim();if(v!=='')a[k]=/^-?\d+$/.test(v)?Number(v):v}return Object.keys(a).length?a:null}
async function boot(){try{const r=await bootSimlish({storage:localStorage,env,outbox});status.textContent='Registration: '+r.registration.status+(r.registration.investigation?' · missing: '+[].concat(r.registration.investigation.missingPlaces).join(', '):'')+' · sync: '+r.sync.status}catch(e){status.textContent='Startup check: '+e.message}}
save.onclick=async()=>{try{personalize({name:name.value,worldTheme:theme.value,address:addressFromForm(),target:target.value},localStorage);fill();await boot()}catch(e){status.textContent=e.message}};
function saveFile(bytes,fileName,type){const blob=new Blob([bytes],{type});if(globalThis.NativeDownloads){const r=new FileReader();r.onload=()=>NativeDownloads.save(fileName,type,String(r.result).split(',')[1]);r.readAsDataURL(blob);return}const a=el('a',{href:URL.createObjectURL(blob),download:fileName});document.body.append(a);a.click();a.remove()}
finish.onclick=async()=>{const f=appFile.files[0];if(!f){status.textContent='Choose an app file first.';return}finish.disabled=true;
 try{const bytes=new Uint8Array(await f.arrayBuffer()),p=readPersonalization(localStorage);
  const r=await finishApp({name:f.name,bytes},{personalization:p,address:p?.address??null,env,outbox,target:p?.target||'reality'});
  out.replaceChildren(el('p',{textContent:r.report.status+' · added '+r.report.addedFiles.length+' files · original: '+r.report.supabase.status}),
   el('button',{textContent:'Download finished app (zip)',onclick:()=>saveFile(r.zip,'finished-'+f.name.replace(/\.zip$/i,'')+'.zip','application/zip')}),
   el('button',{textContent:'Download report',onclick:()=>saveFile(JSON.stringify(r.report,null,2),'finisher-report.json','application/json')}));
  status.textContent=r.report.swarm.phenotype_ga.visible.summary}catch(e){status.textContent='Finisher: '+e.message}finally{finish.disabled=false}};
fill();boot();
