import { bootstrapATO, MorphingWorkspace, ATOEngine, FunctionRegistry } from './ato/index.mjs';
import { AutomataIngestionExecutionKernel } from './automata/ingestion-execution-kernel.mjs';
import { loadTinyLlama, chat as neuralChat, isReady as neuralReady, loadedModelId, MODEL_ID as NEURAL_MODEL_ID, LEGACY_MODEL_ID, MAX_TOKENS as NEURAL_MAX_TOKENS } from './neural.mjs';

if(!globalThis.structuredClone)globalThis.structuredClone=value=>JSON.parse(JSON.stringify(value));
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const id=()=>crypto.randomUUID?.()||`${Date.now()}-${Math.random()}`;
const clone=x=>structuredClone(x);
const key='paper-system-v1';
const templates={portfolio:['Home','Work','About','Contact'],blog:['Home','Articles','About'],shop:['Home','Products','About','Contact'],docs:['Overview','Guide','Reference']};
const defaultDna=()=>({palette:'#a78bfa',density:.5,growth:.6,exploration:.5,cooperation:.5,adaptability:.5});
const blank=()=>({id:id(),name:'Untitled Paper',description:'',pages:[],assets:[],dna:defaultDna(),usage:{},agents:[],source:[],weaves:[]});
let data={schema:1,head:null,versions:[],projects:[blank()],active:null};
let db=null,persistQueue=Promise.resolve();
async function load(){try{db=await new Promise((resolve,reject)=>{const request=indexedDB.open(key,1);request.onupgradeneeded=()=>request.result.createObjectStore('state');request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error)});const saved=await new Promise((resolve,reject)=>{const tx=db.transaction('state','readonly'),q=tx.objectStore('state').get('current');q.onsuccess=()=>resolve(q.result);q.onerror=()=>reject(q.error)});if(saved?.schema===1&&Array.isArray(saved.versions))data=saved}catch(e){message='Device storage unavailable; edits remain in memory until exported.'}if(!data.active)data.active=data.projects[0]?.id}

let view='paper',selectedPage=null,message='',receipt=null;
const neural={status:'idle',progress:'',prompt:'',output:'',includeContext:true,busy:false,error:''};
const app=()=>data.projects.find(p=>p.id===data.active)||data.projects[0];
const save=()=>{if(!db)return;const snapshot=clone(data);persistQueue=persistQueue.then(()=>new Promise((resolve,reject)=>{const tx=db.transaction('state','readwrite');tx.objectStore('state').put(snapshot,'current');tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error)})).catch(e=>{message=`Device storage could not save: ${e.message}`});return persistQueue};
const commit=(label,mutate,parent=data.head,evidence=null)=>{const next=clone(app());mutate(next);const ix=data.projects.findIndex(p=>p.id===next.id);data.projects[ix]=next;const v={id:id(),parent,time:Date.now(),label,project:clone(next),evidence};data.versions.push(v);data.head=v.id;save();render()};
const ato=bootstrapATO();const generationEngine=new ATOEngine({registry:new FunctionRegistry().register('paper.infer-pages',infer)});const workspace=new MorphingWorkspace({id:'paper',views:[{id:'paper',type:'editor',state:{}}],focus:'paper'});const kernel=new AutomataIngestionExecutionKernel();
const setView=v=>{view=v;try{if(workspace.current.views.some(x=>x.id==='view'))workspace.morph([{type:'replace',id:'view',view:{id:'view',type:v,state:{}}}],{rationale:`Open ${v}`,automatons:['conversation']});else workspace.morph([{type:'open',view:{id:'view',type:v,state:{}}}],{rationale:`Open ${v}`,automatons:['conversation']})}catch{}render()};
const page=(title,body='',parent=null)=>({id:id(),title,body,parent,kind:'page'});
const slug=(s)=>String(s).toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'')||'page';
function infer(desc){
  const d=String(desc||'');
  let list=['Home'];
  if(/garden|plant|flower|landscape|botanic/i.test(d))list.push('Garden','Visit','About');
  if(/portfolio|gallery|photograph|photo|work|artist|design/i.test(d))list.push('Work','Gallery','About','Contact');
  if(/shop|store|product|ecommerce|sell/i.test(d))list.push('Products','Cart','About','Contact');
  if(/blog|article|journal|writing|newsletter/i.test(d))list.push('Articles','About');
  if(/docs|guide|manual|reference|documentation/i.test(d))list=['Overview','Guide','Reference','About'];
  if(/restaurant|cafe|food|menu|kitchen|bakery/i.test(d))list.push('Menu','About','Contact');
  if(/music|band|album|concert|podcast/i.test(d))list.push('Music','About','Contact');
  if(/event|wedding|conference|meetup/i.test(d))list.push('Events','About','Contact');
  if(/school|course|learn|class|workshop/i.test(d))list.push('Courses','About','Contact');
  if(/about|profile|bio/i.test(d))list.push('About');
  if(/contact|booking|message|hire|email/i.test(d))list.push('Contact');
  if(list.length===1){
    if(/\b(site|page|web|project)\b/i.test(d))list.push('About','Contact');
    else list.push('About');
  }
  return [...new Set(list)];
}
function stubBody(title,desc,name){
  const d=String(desc||'').trim();
  const n=String(name||'this project').trim()||'this project';
  if(title==='Home')return d||`Welcome to ${n}.`;
  if(/about/i.test(title))return d?`About ${n}.\n\n${d}`:`The story behind ${n}.`;
  if(/contact|visit|booking/i.test(title))return `Get in touch with ${n}.\n\nShare a note, booking request, or question.`;
  if(/gallery|work|portfolio/i.test(title))return `Selected work from ${n}. Upload images in Files to fill this space.`;
  if(/product/i.test(title))return `Products and offerings from ${n}.`;
  if(/article|blog/i.test(title))return `Writing and updates from ${n}.`;
  if(/garden/i.test(title))return d||`A living garden space for ${n}.`;
  if(/menu/i.test(title))return `What ${n} is serving.`;
  if(/music|event|course|guide|reference|overview|cart/i.test(title))return `${title} for ${n}.`;
  return `${title} — part of ${n}.`;
}
function stripTags(html){
  return String(html||'')
    .replace(/<script[\s\S]*?<\/script>/gi,' ')
    .replace(/<style[\s\S]*?<\/style>/gi,' ')
    .replace(/<[^>]+>/g,' ')
    .replace(/&nbsp;/g,' ')
    .replace(/&amp;/g,'&')
    .replace(/&lt;/g,'<')
    .replace(/&gt;/g,'>')
    .replace(/\s+/g,' ')
    .trim();
}
function titleFromFilename(name){
  const base=String(name||'Page').replace(/\.[^.]+$/,'').replace(/[_-]+/g,' ').trim();
  if(!base)return 'Imported page';
  return base.replace(/\b\w/g,c=>c.toUpperCase());
}
function humanReceiptSummary(receipt,extras=[]){
  const kind=receipt?.artifact?.kind||'file';
  const path=receipt?.execution?.path||'';
  const ok=receipt?.execution?.ok;
  const lines=[`Kind: ${kind}`,`Path: ${path}`,ok===false?`Status: needs attention`:'Status: ok',...extras];
  if(receipt?.address?.id)lines.push(`Address: ${receipt.address.id.slice(0,24)}…`);
  return lines.join('\n');
}
function tree(p){const branches=p.pages.filter(x=>!x.parent);return `<div class="tree"><span class="node root">🌱 ${esc(p.name)}</span>${branches.length?`<ul>${branches.map(b=>`<li><span class="node">📄 ${esc(b.title)}</span>${p.pages.some(x=>x.parent===b.id)?`<ul>${p.pages.filter(x=>x.parent===b.id).map(x=>`<li><span class="node leaf">🍃 ${esc(x.title)}</span></li>`).join('')}</ul>`:''}</li>`).join('')}</ul>`:'<p class="muted">Plant a page to grow the first branch.</p>'}</div>`}
function suggestions(p){const s=[];if(p.pages.length===0)s.push(['Plant your first page','page']);if(p.pages.length>=2&&!p.pages.some(x=>/contact/i.test(x.title)))s.push(['Add a contact page','contact']);if(p.assets.length&&!p.pages.some(x=>/gallery|work/i.test(x.title)))s.push(['Give uploads a gallery','gallery']);if((p.usage.preview||0)>2&&!p.pages.some(x=>/about/i.test(x.title)))s.push(['Introduce the project with an About page','about']);return s}
function siteHTML(p,focus=null){const all=p.pages.length?p.pages:[page('Home',p.description)];const pages=focus?[focus]:all;const color=/^#[0-9a-f]{6}$/i.test(p.dna.palette)?p.dna.palette:'#a78bfa';const nav=all.map((x,i)=>`<a href="${focus?'../':''}pages/${slug(x.title)}-${x.id.slice(0,6)}.html">${esc(x.title)}</a>`).join('');const imgs=p.assets.filter(a=>a.type&&a.type.startsWith('image/'));const sections=pages.map((x,i)=>{const isGallery=/gallery|work|portfolio|garden/i.test(x.title);const shown=isGallery?imgs:imgs.slice(0,3);return `<section id="p${i}"><div class="tag">${String(i+1).padStart(2,'0')} / ${esc(p.name)}</div><h2>${esc(x.title)}</h2><p>${esc(x.body||p.description||'Your story begins here.')}</p>${shown.map(a=>`<img alt="${esc(a.name)}" src="${a.data}">`).join('')}</section>`}).join('');return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="${esc(p.description)}"><title>${esc(p.name)}</title><style>*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#10121a;color:#f8f7ff;font:18px/1.65 system-ui,sans-serif}header{position:sticky;top:0;background:#141722e8;backdrop-filter:blur(8px);padding:14px max(20px,5vw);display:flex;justify-content:space-between;gap:18px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #424657}nav{display:flex;gap:18px;flex-wrap:wrap}a{color:#eee;text-decoration:none}a:hover{color:${color}}strong,.tag{color:${color}}section{min-height:60vh;max-width:950px;margin:auto;padding:clamp(60px,8vw,110px) 20px;border-bottom:1px solid #363847}h1{font-size:clamp(2rem,5vw,4rem);margin:0}h2{font-size:clamp(2rem,4vw,3.5rem);line-height:1.12;margin:12px 0}.tag{font-size:.8rem;text-transform:uppercase;letter-spacing:.17em}p{max-width:66ch;color:#c4c7d4;white-space:pre-wrap}img{max-width:100%;max-height:420px;margin:15px 12px 0 0;border-radius:14px}footer{padding:36px;text-align:center;color:#999}</style></head><body><header><strong>${esc(p.name)}</strong><nav>${nav}</nav></header><main><section><div class="tag">Paper / living site</div><h1>${esc(p.name)}</h1><p>${esc(p.description)}</p></section>${sections}</main><footer>${esc(p.name)}</footer></body></html>`}
function download(blob,name){if(window.PaperAndroid){const reader=new FileReader();reader.onload=()=>window.PaperAndroid.save(name,String(reader.result).split(',')[1]);reader.readAsDataURL(blob);return}const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),30000)}
async function exportSite(){const p=app(),zip=new JSZip();zip.file('index.html',siteHTML(p));for(const item of p.pages)zip.file(`pages/${slug(item.title)}-${item.id.slice(0,6)}.html`,siteHTML(p,item));zip.file('paper-project.json',JSON.stringify({schema:1,project:p,versions:data.versions.filter(v=>v.project.id===p.id)},null,2));for(const a of p.assets)if(a.data)zip.file('assets/'+a.name,a.data.split(',')[1],{base64:true});download(await zip.generateAsync({type:'blob'}),slug(p.name)+'.zip')}

function projectContext(p){
  const chunks=[];
  if(p.name)chunks.push(`Project: ${p.name}`);
  if(p.description)chunks.push(`Description: ${p.description}`);
  for(const page of p.pages||[]){
    const body=(page.body||'').trim();
    chunks.push(`Page "${page.title}"${body?`:\n${body}`:''}`);
  }
  for(const s of p.source||[]){
    if(!s.content)continue;
    const text=String(s.content).slice(0,4000);
    chunks.push(`Source file ${s.name} (${s.kind}):\n${text}`);
  }
  return chunks.join('\n\n').slice(0,12000);
}
function neuralStatusLabel(){
  if(neuralReady())return `Ready · ${loadedModelId()||NEURAL_MODEL_ID}`;
  if(neural.busy&&neural.status==='loading')return neural.progress||'Loading TinyLlama…';
  if(neural.busy&&neural.status==='running')return 'Running…';
  if(neural.error)return neural.error;
  return 'Not loaded yet';
}
async function ensureTinyLlama(){
  if(neuralReady())return;
  neural.busy=true;neural.status='loading';neural.error='';neural.progress='Starting TinyLlama download…';
  message='Loading TinyLlama in this browser (WebGPU)…';
  render();
  try{
    await loadTinyLlama(text=>{neural.progress=text;const el=$('#neural-progress');if(el)el.textContent=text;else{message=text}});
    neural.status='ready';neural.progress='TinyLlama ready.';neural.busy=false;
    message=`TinyLlama ready (${loadedModelId()||NEURAL_MODEL_ID}).`;
  }catch(err){
    neural.busy=false;neural.status='error';neural.error=err.message||String(err);
    message=`TinyLlama load failed: ${neural.error}`;
    throw err;
  }finally{render()}
}
async function runNeuralPrompt(rawPrompt,{useContext=false}={}){
  const p=app();
  let prompt=String(rawPrompt||'').trim();
  if(!prompt)throw Error('Enter a prompt.');
  if(useContext){
    const ctx=projectContext(p);
    if(ctx)prompt=`Context from this Paper project:\\n\\n${ctx}\\n\\n---\\n\\nUser request:\\n${prompt}`;
  }
  await ensureTinyLlama();
  neural.busy=true;neural.status='running';neural.error='';message='TinyLlama is generating…';render();
  try{
    const out=await neuralChat(prompt,{max_tokens:NEURAL_MAX_TOKENS});
    neural.output=out;neural.status='ready';neural.busy=false;
    message='TinyLlama reply ready.';
    return out;
  }catch(err){
    neural.busy=false;neural.status='error';neural.error=err.message||String(err);
    message=`TinyLlama run failed: ${neural.error}`;
    throw err;
  }finally{render()}
}
function render(){const p=app(),nav=[['paper','📄 Paper'],['seed','🌱 Seed'],['loom','🧵 Loom'],['cardstock','▣ Cardstock'],['papermate','🧬 Papermate'],['notebook','📓 Notebook'],['files','📦 Files'],['neural','🧠 Neural'],['preview','◩ Preview']];$('#app').innerHTML=`<div class="shell"><aside class="sidebar"><div class="brand">▤ Paper<i>.</i></div><nav class="nav" aria-label="Paper tools">${nav.map(([k,n])=>`<button data-view="${k}" class="${view===k?'active':''}">${n}</button>`).join('')}</nav><div class="sidefoot">Local browser runtime<br>${ato.tools.length} ATO tools mounted<br>${neuralReady()?`TinyLlama ready`:`TinyLlama optional`}<br>No backend</div></aside><main class="main"><div class="top"><div><div class="eyebrow">${esc(view)} / ${esc(p.name)}</div><h1>${{paper:'Build a living site',seed:'Grow the structure',loom:'Weave projects',cardstock:'Reusable forms',papermate:'Agent DNA',notebook:'Landed history',files:'Ingest & execute',neural:'TinyLlama in the browser',preview:'Live output'}[view]}</h1><p class="sub">${{paper:'Describe, edit and export a real multi page site.',seed:'Every page and section becomes a branch.',loom:'Compose existing pages into one descendant.',cardstock:'Start from a useful scaffold.',papermate:'Recombine traits and keep parent lineage.',notebook:'Branch from any earlier state without erasing it.',files:'Inspect uploads through the browser execution kernel.',neural:'WebLLM TinyLlama — downloads once, runs locally on WebGPU. No backend.',preview:'Review exactly what the static export will show.'}[view]}</p></div><div class="row"><select id="project" aria-label="Active project">${data.projects.map(x=>`<option value="${x.id}" ${x.id===p.id?'selected':''}>${esc(x.name)}</option>`).join('')}</select><button class="btn" data-action="new-project">New project</button><button class="btn mint" data-action="export">Export site</button></div></div>${message?`<div class="notice" role="status">${esc(message)}</div>`:''}<div id="content">${content(view,p)}</div></main></div>`;if(view==='preview'){$('#preview-frame').srcdoc=siteHTML(p)}bind()}
function content(v,p){if(v==='paper')return `<div class="grid"><div class="card span8"><h2>Describe your Paper</h2><form id="describe"><label for="name">Site name</label><input id="name" value="${esc(p.name)}" required maxlength="80"><label for="description">What should it be?</label><textarea id="description" placeholder="A photographer portfolio with a gallery, about page and contact page">${esc(p.description)}</textarea><div class="row"><button class="btn primary">Generate pages</button><button type="button" class="btn" data-action="draft-neural" ${neuralReady()?'':'title="Load TinyLlama in Neural first (or click — it will load)"'}>Draft with TinyLlama</button><span class="hint">Generate uses local structure. Draft fills page bodies with TinyLlama once loaded.</span></div></form></div><div class="card span4"><h2>Current organism</h2><div class="kpis"><div class="kpi"><strong>${p.pages.length}</strong><span>pages</span></div><div class="kpi"><strong>${p.assets.length}</strong><span>files</span></div><div class="kpi"><strong>${data.versions.filter(x=>x.project.id===p.id).length}</strong><span>versions</span></div></div><p class="hint">Every edit saves a new version with a parent address.</p></div><div class="card span12"><h2>Pages</h2><div class="page-list">${p.pages.length?p.pages.map(x=>`<div class="item"><span><b>${esc(x.title)}</b> ${x.parent?'<span class="badge">section</span>':''}<br><span class="statline">${esc((x.body||'').slice(0,90))}</span></span><div class="row"><button class="btn small" data-action="edit-page" data-id="${x.id}">Edit</button><button class="btn small" data-action="branch-page" data-id="${x.id}">Add section</button></div></div>`).join(''):'<div class="empty">No pages yet. Generate from your description or plant one in Seed.</div>'}</div>${selectedPage&&p.pages.some(x=>x.id===selectedPage)?`<form id="page-form" class="card" style="margin-top:16px"><h3>Edit page</h3><input name="title" aria-label="Page title" value="${esc(p.pages.find(x=>x.id===selectedPage).title)}" required><textarea name="body" aria-label="Page content">${esc(p.pages.find(x=>x.id===selectedPage).body)}</textarea><button class="btn primary">Save descendant</button></form>`:''}</div></div>`;
if(v==='seed')return `<div class="grid"><div class="card span8"><h2>Living tree</h2>${tree(p)}</div><div class="card span4"><h2>Grow a branch</h2><form id="grow"><label for="grow-title">Page or section title</label><input id="grow-title" required><label for="grow-parent">Grows from</label><select id="grow-parent"><option value="">Seed (new page)</option>${p.pages.filter(x=>!x.parent).map(x=>`<option value="${x.id}">${esc(x.title)}</option>`).join('')}</select><label for="grow-body">Content</label><textarea id="grow-body"></textarea><button class="btn primary">Grow</button></form><h3 style="margin-top:28px">Growth signals</h3>${suggestions(p).map(([s,k])=>`<div class="feature">${esc(s)} <button class="btn small" data-action="suggest" data-kind="${k}">Apply</button></div>`).join('')||'<p class="muted">The current structure has its basic pages.</p>'}</div></div>`;
if(v==='cardstock')return `<div class="grid">${Object.entries(templates).map(([name,pages])=>`<div class="card span6"><div class="eyebrow">Cardstock / ${name}</div><h2>${name[0].toUpperCase()+name.slice(1)}</h2><p>${pages.join(' · ')}</p><button class="btn primary" data-action="template" data-kind="${name}">Grow from template</button></div>`).join('')}</div>`;
if(v==='loom')return `<div class="grid"><div class="card span6"><h2>Compose another Paper</h2><p class="muted">Copy its pages as linked descendants with source identifiers retained.</p><select id="loom-source" aria-label="Source project"><option value="">Select a project</option>${data.projects.filter(x=>x.id!==p.id).map(x=>`<option value="${x.id}">${esc(x.name)} (${x.pages.length} pages)</option>`).join('')}</select><p><button class="btn primary" data-action="weave">Weave pages</button></p></div><div class="card span6"><h2>Woven sources</h2>${p.weaves.length?p.weaves.map(x=>`<div class="item">${esc(x.name)} <span class="badge">${x.count} pages</span></div>`).join(''):'<div class="empty">Create another project, then bring its pages here.</div>'}</div></div>`;
if(v==='papermate')return `<div class="grid"><div class="card span6"><h2>Heritable traits</h2><p class="muted">These govern local suggestions and the exported accent. Offspring uses bounded crossover and mutation.</p>${Object.entries(p.dna).filter(([k])=>k!=='palette').map(([k,val])=>`<div class="gene"><label for="g-${k}">${k}</label><input type="range" id="g-${k}" min="0" max="1" step="0.01" value="${val}" data-gene="${k}"><span>${Math.round(val*100)}</span></div>`).join('')}<label for="palette">Accent</label><input id="palette" type="color" value="${p.dna.palette}"><p><button class="btn" data-action="save-dna">Save DNA</button></p></div><div class="card span6"><h2>Recombine agents</h2><p class="muted">Add a parent agent from this project’s DNA, then breed two agents. Every birth records its parents and mutations.</p><button class="btn" data-action="spawn">Create agent</button><div class="page-list" style="margin-top:14px">${p.agents.map(a=>`<div class="item"><span>${esc(a.name)} <span class="badge">generation ${a.generation}</span><br><span class="statline">${a.parents.length?'Parents '+a.parents.map(x=>x.slice(0,8)).join(' × '):'Founder'} · ${a.mutations.length} mutations</span></span><input type="checkbox" data-mate="${a.id}" aria-label="Select ${esc(a.name)}" style="width:auto"></div>`).join('')}</div><p><button class="btn primary" data-action="mate" ${p.agents.length<2?'disabled':''}>Breed selected agents</button></p></div></div>`;
if(v==='notebook')return `<div class="grid"><div class="card span8"><h2>Lineage</h2><div class="timeline">${data.versions.filter(x=>x.project.id===p.id).slice().reverse().map(x=>`<div class="item"><span><b>${esc(x.label)}</b> ${x.id===data.head?'<span class="badge">current</span>':''}<br><span class="statline">${new Date(x.time).toLocaleString()} · ${x.project.pages.length} pages · parent ${x.parent?.slice(0,8)||'seed'}</span></span><button class="btn small" data-action="fork" data-id="${x.id}">Branch here</button></div>`).join('')}</div></div><div class="card span4"><h2>Law of history</h2><p>Earlier versions stay landed. Branching creates a new descendant from the chosen state and retains every previous branch.</p><button class="btn" data-action="backup">Download complete history</button></div></div>`;
if(v==='files')return `<div class="grid"><div class="card span6"><h2>Upload a source</h2><p>ZIP, HTML, JavaScript, JSON, text and images. The automata kernel compiles, observes and addresses supported artifacts locally.</p><button class="btn primary" data-action="upload">Choose files</button><p class="hint">Imported code is inspected. Self-contained JavaScript can run only through the sandbox after you choose Run.</p></div><div class="card span6"><h2>Sources</h2><div class="files">${p.source.map(s=>`<div class="item"><span>${esc(s.name)}<br><span class="statline">${esc(s.kind)} · ${s.size} bytes · ${s.address?.slice(0,18)||'unaddressed'}</span></span>${s.kind==='javascript'?`<button class="btn small" data-action="run" data-id="${s.id}">Run</button>`:''}</div>`).join('')||'<div class="empty">Nothing imported yet.</div>'}</div></div>${receipt?`<div class="card span12"><h2>Import result</h2><pre class="receipt-summary">${esc(receipt.summary||humanReceiptSummary(receipt))}</pre><details class="receipt-details"><summary>Technical receipt</summary><pre class="code">${esc(JSON.stringify(receipt.raw||receipt,null,2))}</pre></details></div>`:''}</div>`;
if(v==='neural')return `<div class="grid"><div class="card span8"><h2>TinyLlama Foundry</h2><p class="muted">Same path as the original Paper synth: WebLLM in this tab, weights in the browser, no API key and no server chat.</p><p class="statline">Model: <code>${esc(loadedModelId()||NEURAL_MODEL_ID)}</code> · fallback id <code>${esc(LEGACY_MODEL_ID)}</code> · max_tokens ${NEURAL_MAX_TOKENS}</p><div class="row" style="margin:14px 0"><button class="btn primary" data-action="load-neural" ${neural.busy?'disabled':''}>${neuralReady()?'Reload TinyLlama':'Load TinyLlama'}</button><span class="badge">${esc(neuralStatusLabel())}</span></div><p id="neural-progress" class="hint">${esc(neural.progress||(neuralReady()?'Model cached in this browser.':'First load downloads weights over the network into browser cache (WebGPU).'))}</p><label for="neural-prompt">Prompt</label><textarea id="neural-prompt" rows="8" placeholder="Ask TinyLlama anything about this Paper, or draft copy…">${esc(neural.prompt)}</textarea><label class="row" style="font-weight:500;gap:8px;margin:10px 0"><input type="checkbox" id="neural-context" ${neural.includeContext?'checked':''} style="width:auto"> Include project pages &amp; uploaded source text in the prompt</label><div class="row"><button class="btn mint" data-action="run-neural" ${neural.busy?'disabled':''}>Run</button></div></div><div class="card span4"><h2>Output</h2><pre class="code" id="neural-output" style="max-height:480px;min-height:220px">${esc(neural.output||'Reply appears here after Run.')}</pre><p class="hint">All inference stays on this device. ATO / Klein stay mounted beside Neural — Neural is the talking path.</p></div></div>`;
if(v==='preview')return `<div class="card"><div class="row" style="justify-content:space-between"><h2>Generated static site</h2><button class="btn mint" data-action="export">Download ZIP</button></div><iframe title="Paper site preview" id="preview-frame" class="preview" sandbox="allow-scripts"></iframe></div>`;return ''}
function bind(){document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(b.dataset.view));$('#project').onchange=e=>{data.active=e.target.value;data.head=data.versions.filter(x=>x.project.id===data.active).at(-1)?.id||null;selectedPage=null;save();render()};$('#describe')?.addEventListener('submit',async e=>{e.preventDefault();try{const name=$('#name').value.trim(),desc=$('#description').value.trim();const compiled=generationEngine.compile({address:{mode:'macro',gate:63,line:1,color:1,tone:1,base:1},purpose:'Infer Paper page structure',graph:{nodes:[{id:'description',kind:'source'},{id:'pages',kind:'sink'}],edges:[{from:'description',to:'pages',functionId:'paper.infer-pages',fromPort:{type:'json'},toPort:{type:'json'}}]}});const result=await compiled.execute(desc);const titles=result.outputs.pages;commit('Paper generated',p=>{p.name=name;p.description=desc;const byTitle=new Map(p.pages.map(x=>[x.title,x]));for(const title of titles){if(byTitle.has(title)){const existing=byTitle.get(title);if(!existing.body)existing.body=stubBody(title,desc,name)}else p.pages.push(page(title,stubBody(title,desc,name)))}},data.head,{engine:'ATO Core',stages:compiled.trace.map(x=>x.stage),address:compiled.address});message=`Generated ${titles.length} pages: ${titles.join(', ')}`;setView('preview')}catch(err){message=err.message;render()}});$('#grow')?.addEventListener('submit',e=>{e.preventDefault();const title=$('#grow-title').value.trim(),body=$('#grow-body').value,parent=$('#grow-parent').value||null;commit(`Grew ${title}`,p=>p.pages.push(page(title,body,parent)))});$('#page-form')?.addEventListener('submit',e=>{e.preventDefault();const form=new FormData(e.target);commit(`Edited ${form.get('title')}`,p=>{const x=p.pages.find(y=>y.id===selectedPage);x.title=String(form.get('title')).trim();x.body=String(form.get('body'))});selectedPage=null});document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>action(b));document.querySelectorAll('[data-gene]').forEach(r=>r.oninput=()=>r.nextElementSibling.textContent=Math.round(+r.value*100));
$('#neural-prompt')?.addEventListener('input',e=>{neural.prompt=e.target.value});
$('#neural-context')?.addEventListener('change',e=>{neural.includeContext=!!e.target.checked});
}
async function action(b){const p=app(),a=b.dataset.action;message='';try{if(a==='new-project'){const n=blank();data.projects.push(n);data.active=n.id;data.head=null;commit('Seed planted',()=>{});setView('paper')}else if(a==='export')await exportSite();else if(a==='edit-page'){selectedPage=b.dataset.id;render();$('#page-form')?.scrollIntoView({behavior:'smooth'})}else if(a==='branch-page'){const x=p.pages.find(y=>y.id===b.dataset.id);commit(`Section under ${x.title}`,q=>q.pages.push(page('New section','',x.id)));setView('seed')}else if(a==='suggest'){const titles={page:'Home',contact:'Contact',gallery:'Gallery',about:'About'};commit(`Growth: ${titles[b.dataset.kind]}`,q=>q.pages.push(page(titles[b.dataset.kind],stubBody(titles[b.dataset.kind],q.description,q.name))))}else if(a==='template'){const t=b.dataset.kind;commit(`Cardstock: ${t}`,q=>{const existing=new Set(q.pages.map(x=>x.title));for(const title of templates[t])if(!existing.has(title))q.pages.push(page(title,stubBody(title,q.description,q.name)))});setView('seed')}else if(a==='weave'){const source=data.projects.find(x=>x.id===$('#loom-source').value);if(!source)throw Error('Choose a source project first.');commit(`Wove ${source.name}`,q=>{for(const x of source.pages)q.pages.push({...clone(x),id:id(),sourceId:x.id,parent:null,title:q.pages.some(z=>z.title===x.title)?`${source.name}: ${x.title}`:x.title});q.weaves.push({id:source.id,name:source.name,count:source.pages.length})})}else if(a==='save-dna')commit('DNA tuned',q=>{for(const k of ['density','growth','exploration','cooperation','adaptability'])q.dna[k]=+$('#g-'+k).value;q.dna.palette=$('#palette').value});else if(a==='spawn')commit('Agent founded',q=>q.agents.push({id:id(),name:`Agent ${q.agents.length+1}`,generation:0,parents:[],genes:clone(q.dna),mutations:[]}));else if(a==='mate'){const selected=[...document.querySelectorAll('[data-mate]:checked')].map(x=>x.dataset.mate);if(selected.length!==2)throw Error('Select exactly two agents.');const [x,y]=selected.map(id=>p.agents.find(z=>z.id===id));if(x.id===y.id)throw Error('Select different agents.');const genes={...x.genes},mutations=[];for(const k of ['density','growth','exploration','cooperation','adaptability']){genes[k]=Math.random()<.5?x.genes[k]:y.genes[k];if(Math.random()<.15){const old=genes[k];genes[k]=Math.max(0,Math.min(1,old+(Math.random()-.5)*.2));mutations.push({gene:k,from:old,to:genes[k]})}}genes.palette=Math.random()<.5?x.genes.palette:y.genes.palette;commit('Agent offspring born',q=>q.agents.push({id:id(),name:`Agent ${q.agents.length+1}`,generation:Math.max(x.generation,y.generation)+1,parents:[x.id,y.id],genes,mutations}))}else if(a==='fork'){const ancestor=data.versions.find(x=>x.id===b.dataset.id);const next=clone(ancestor.project);const ix=data.projects.findIndex(x=>x.id===p.id);data.projects[ix]=next;commit(`Branch from ${ancestor.label}`,()=>{},ancestor.id)}else if(a==='backup')download(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),'paper-complete-history.json');else if(a==='upload')$('#file-input').click();else if(a==='run'){const s=p.source.find(x=>x.id===b.dataset.id);if(!s?.content)throw Error('This source is not stored as executable text.');const r=await kernel.process({name:s.name,content:s.content},{timeoutMs:2500});const summary=r.execution.ok?'Code executed in the isolated browser frame.':`Execution needs attention: ${r.execution.error}`;receipt={summary,artifact:r.artifact,fields:r.fields?.counts,address:r.address,execution:r.execution,raw:r};message=summary;render()}else if(a==='load-neural'){await ensureTinyLlama()}else if(a==='run-neural'){neural.prompt=$('#neural-prompt')?.value??neural.prompt;neural.includeContext=!!$('#neural-context')?.checked;await runNeuralPrompt(neural.prompt,{useContext:neural.includeContext})}else if(a==='draft-neural'){const desc=($('#description')?.value??p.description??'').trim()||p.name||'this site';neural.prompt=`Write clear website page copy for "${p.name||'Untitled'}". Description: ${desc}. Return plain text body copy suitable for a Home page (2–4 short paragraphs).`;const out=await runNeuralPrompt(neural.prompt,{useContext:true});const body=String(out||'').trim();if(!body)throw Error('TinyLlama returned empty draft.');commit('Drafted with TinyLlama',q=>{const home=q.pages.find(x=>/^home$/i.test(x.title));if(home)home.body=body;else q.pages.unshift(page('Home',body));if(!q.description)q.description=desc;});message='Home page drafted with TinyLlama.';setView('paper')}}catch(e){message=e.message;render()}}
$('#file-input').onchange=async e=>{
  for(const f of [...e.target.files]){
    try{
      if(f.size>8*1024*1024)throw Error(`${f.name} exceeds the 8 MB local import limit.`);
      message=`Importing ${f.name}…`;
      const lower=f.name.toLowerCase();
      const isJs=/\.(js|mjs|cjs|jsx|ts|tsx)$/i.test(lower);
      const isJson=/\.json$/i.test(lower)||f.type.includes('json');
      const isHtml=/\.(html|htm)$/i.test(lower)||f.type==='text/html';
      const isMd=/\.(md|markdown)$/i.test(lower);
      const isTxt=/\.txt$/i.test(lower)||f.type==='text/plain';
      const isCss=/\.css$/i.test(lower)||f.type==='text/css';
      const isImage=(f.type&&f.type.startsWith('image/'))||/\.(png|jpe?g|gif|webp|svg|bmp|ico|avif)$/i.test(lower);
      // JS: compile only on upload; Run button executes. JSON/html/md/txt/css/image: execute/ingest.
      const shouldExecute=!isJs;
      const r=await kernel.process(f,{execute:shouldExecute,timeoutMs:2500});
      let content='';
      if(isJs||isJson||isHtml||isMd||isTxt||isCss)content=await f.text();
      let asset=null;
      if(isImage)asset=await new Promise((ok,no)=>{const reader=new FileReader();reader.onload=()=>ok(reader.result);reader.onerror=no;reader.readAsDataURL(f)});
      const kind=r.artifact?.kind||(isHtml?'html':isMd?'markdown':isCss?'css':isImage?'image':isJs?'javascript':isJson?'json':'text');
      const record={id:id(),name:f.name,kind,size:f.size,address:r.address.id,content};
      const extras=[];
      commit(`Imported ${f.name}`,p=>{
        p.source.push(record);
        if(asset){
          p.assets.push({id:id(),name:f.name,type:f.type||'image/*',data:asset});
          extras.push('Imported image into assets');
          if(!p.pages.some(x=>/gallery/i.test(x.title))){
            p.pages.push(page('Gallery','Images you upload show up here.'));
            extras.push('Added a Gallery page');
          }
        }
        if(isHtml||isMd||isTxt){
          let title=titleFromFilename(f.name);
          let body=content;
          if(isHtml){
            const tm=content.match(/<title[^>]*>([^<]*)<\/title>/i);
            if(tm?.[1]?.trim())title=tm[1].trim();
            body=stripTags(content);
          }else if(isMd){
            const hm=content.match(/^#\s+(.+)$/m);
            if(hm?.[1]?.trim())title=hm[1].trim();
            body=content.replace(/^#\s+.+$/m,'').trim();
          }
          if(!body)body=`Imported from ${f.name}`;
          const existing=p.pages.find(x=>x.title===title);
          if(existing)existing.body=body;
          else p.pages.push(page(title,body));
          extras.push(`Imported ${f.name} as a page ('${title}')`);
        }else if(isCss){
          extras.push('Imported CSS into Sources (available for export projects)');
        }else if(isJson){
          extras.push(r.execution?.ok?'JSON parsed and retained':'JSON kept; parse needs attention');
        }else if(isJs){
          extras.push('JS ready — press Run');
        }else if(!isImage){
          extras.push(`${f.name} retained in Sources`);
        }
      });
      const summary=extras.length?extras.join('\n'):humanReceiptSummary(r);
      receipt={
        summary,
        artifact:r.artifact,
        fields:r.fields?.counts,
        address:r.address,
        execution:r.execution,
        raw:{artifact:r.artifact,fields:r.fields?.counts,address:r.address,execution:r.execution,note:r.note},
      };
      message=extras[0]||`${f.name} imported.`;
      render();
    }catch(err){message=err.message;render()}
  }
  e.target.value='';
};
await load();if(!data.versions.length){const p=app();data.versions.push({id:id(),parent:null,time:Date.now(),label:'Seed planted',project:clone(p)});data.head=data.versions[0].id;save()}render();

if(document.modelContext?.registerTool){
  const register=tool=>{try{Promise.resolve(document.modelContext.registerTool(tool)).catch(()=>{})}catch{}};
  register({name:'paper_read_project',title:'Read Paper project',description:'Read the active local Paper project and its page tree.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true},execute(){const p=app();return {name:p.name,description:p.description,pages:p.pages.map(x=>({id:x.id,title:x.title,parent:x.parent})),version:data.head}}});
  register({name:'paper_create_page',title:'Create Paper page',description:'Grow a page in the active Paper project and save a new lineage version.',inputSchema:{type:'object',properties:{title:{type:'string'},body:{type:'string'},parent:{type:'string'}},required:['title'],additionalProperties:false},annotations:{readOnlyHint:false},execute(input){if(typeof input?.title!=='string'||!input.title.trim()||input.title.length>80)throw Error('A title of 1–80 characters is required.');if(input.parent&&!app().pages.some(x=>x.id===input.parent))throw Error('Parent page not found.');const created=page(input.title.trim(),String(input.body||''),input.parent||null);commit(`Grew ${created.title}`,p=>p.pages.push(created));return {id:created.id,version:data.head,title:created.title}}});
}
