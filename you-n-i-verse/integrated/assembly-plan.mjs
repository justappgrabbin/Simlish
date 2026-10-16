// What the swarm assembles, one unit per cycle. Every unit is built from the primitive layers that already exist (primitive-functions.json clean layer + primitive-attributes.json).
// The pages hold phrase/form + function names + qualities; citations stay in primitive-sources.json (shown on request, never embedded here).
// The 'finish' unit is her existing finishApp (app-finisher.mjs): it assesses the assembled app against her checklist and adds the missing pieces beside the files, never over them.
import {finishApp} from '../app-finisher.mjs';
export const DIMENSIONS=['movement','evolution','being','design','space'];
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const css='body{font:15px system-ui;background:#0d1117;color:#e6edf3;margin:1rem;max-width:46rem}a{color:#8ab4ff}table{border-collapse:collapse;width:100%}td,th{border:1px solid #30363d;padding:.3rem;text-align:left;vertical-align:top}code{color:#9ad}';
const page=(title,body)=>'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title><style>'+css+'</style><body>'+body+'</body>';
const byId=p=>new Map(p.map(x=>[x.id,x]));
export function buildPlan({data,tools}){
 const P=byId(data.clean.primitives),A=data.attrs,units=[];
 const gatesOf=d=>data.clean.gatesByDoc3Base[d]||[];
 units.push({id:'skills',title:'skills: one manifest per tool the swarm may use',needs:[],files:()=>tools.map(t=>'app/skills/'+t.name+'.json'),
  act:async()=>tools.map(t=>({path:'app/skills/'+t.name+'.json',data:JSON.stringify({name:t.name,description:t.description,route:t.route,pure:true,network:false},null,1)})),
  observe:async({workspace})=>({ok:tools.every(t=>workspace.has('app/skills/'+t.name+'.json')),detail:tools.length+' skill manifests'})});
 for(const d of DIMENSIONS)units.push({id:'dim-'+d,title:'dimension page: '+d,needs:['skills'],files:()=>['app/pages/dimension-'+d+'.html'],
  act:async()=>{const dp=P.get('dim.'+d),chain=data.clean.primitives.filter(p=>p.kind==='chain-line'&&(p.ctx||[]).some(c=>c.dim===d||c.dimension===d)),gs=gatesOf(d);
   return [{path:'app/pages/dimension-'+d+'.html',data:page(dp.form,'<p><a href="../index.html">home</a></p><h1>'+esc(dp.form)+' <small>'+esc(dp.symbol)+'</small></h1><p>functions: '+(dp.fn.map(f=>'<code>'+esc(f)+'</code>').join(' ')||'none')+'</p>'+(dp.gap.length?'<p>open: '+dp.gap.map(esc).join('; ')+'</p>':'')+
    '<h2>gates placed here by her doc3 anchor map ('+gs.length+')</h2><p>'+gs.map(g=>'<a href="gate-'+g+'.html">G'+g+'</a>').join(' ')+'</p><h2>chain lines</h2><ul>'+chain.map(c=>'<li>'+esc(c.form)+'</li>').join('')+'</ul>')}]},
  observe:async({workspace})=>{const t=workspace.text('app/pages/dimension-'+d+'.html')||'';return {ok:gatesOf(d).every(g=>t.includes('>G'+g+'<')),detail:gatesOf(d).length+' gates linked'}}});
 for(let g=1;g<=64;g++){const gp=P.get('gate.'+g),dim=gp.ctx?.[0]?.dim||'movement';
  units.push({id:'gate-'+g,title:'gate page: G'+g+' '+gp.form,needs:['dim-'+dim],files:()=>['app/pages/gate-'+g+'.html'],
   act:async()=>{const at=A['gate.'+g],rows=[1,2,3,4,5,6].map(l=>{const lp=P.get('gate.'+g+'.line.'+l),la=A['gate.'+g+'.line.'+l];return '<tr><td>'+l+'</td><td>'+esc(lp.form)+'</td><td>'+esc(la.keynote)+'</td><td>'+esc(la.colorOfLine)+'</td><td>'+lp.fn.map(f=>'<code>'+esc(f)+'</code>').join(' ')+'</td></tr>'}).join('');
    return [{path:'app/pages/gate-'+g+'.html',data:page('G'+g+' '+gp.form,'<p><a href="dimension-'+dim+'.html">'+esc(dim)+'</a> · <a href="../index.html">home</a></p><h1>G'+g+' '+esc(gp.form)+'</h1><p>bits <code>'+esc(at.bits)+'</code> · macro '+at.macro+' · complement macro '+at.complement+' · center '+esc(at.centerSG)+' · '+esc(at.doc3.archetype)+'</p><p>functions: '+gp.fn.map(f=>'<code>'+esc(f)+'</code>').join(' ')+'</p><table><tr><th>line</th><th>form</th><th>keynote</th><th>color</th><th>functions</th></tr>'+rows+'</table>')}]},
   observe:async({workspace})=>{const t=workspace.text('app/pages/gate-'+g+'.html')||'';return {ok:t.includes('G'+g+' ')&&(t.match(/<tr><td>\d<\/td>/g)||[]).length===6,detail:'6 lines'}}})}
 units.push({id:'index',title:'app home page linking every assembled page',files:()=>['app/index.html'],needs:[...DIMENSIONS.map(d=>'dim-'+d),...Array.from({length:64},(_,i)=>'gate-'+(i+1))],
  act:async({workspace})=>{const gl=workspace.list('app/pages/gate-').length;return [{path:'app/index.html',data:page('Synthia assembled app','<h1>Assembled by the swarm</h1><p>'+gl+' gate pages, '+DIMENSIONS.length+' dimension pages.</p><ul>'+DIMENSIONS.map(d=>'<li><a href="pages/dimension-'+d+'.html">'+d+'</a></li>').join('')+'</ul>')}]},
  observe:async({workspace})=>{const t=workspace.text('app/index.html')||'';return {ok:DIMENSIONS.every(d=>t.includes('dimension-'+d+'.html')),detail:'links to all dimensions'}}});
 units.push({id:'finish',title:'finish: her checklist (entry, manifest, address, scene, reality, original preserved)',needs:['index'],
  act:async({workspace})=>{const files=workspace.list('app/').filter(f=>!f.path.startsWith('app/simlish/')).map(f=>({path:f.path.slice(4),data:workspace.read(f.path)}));
   const r=await finishApp({name:'synthia-assembled-app',files},{env:{}});globalThis.__lastFinish=r.report;
   return r.files.filter(f=>f.path.startsWith('simlish/')).map(f=>({path:'app/'+f.path,data:f.data}))},
  observe:async({workspace})=>{const r=JSON.parse(workspace.text('app/simlish/finisher-report.json')||'{}');return {ok:Array.isArray(r.missingAfter)&&r.missingAfter.every(x=>x==='address-registration'),detail:'missing before: '+(r.missingBefore||[]).join(', ')+' / after: '+((r.missingAfter||[]).join(', ')||'none')+' ('+r.status+')'}}});
 units.push({id:'soul',title:'SOUL.md: the swarm writes what it assembled (self-authored, writable)',needs:['finish'],files:()=>['app/SOUL.md'],
  act:async({workspace,state})=>{const done=Object.values(state).filter(s=>s.status==='done').length;return [{path:'app/SOUL.md',data:'# SOUL (self-authored, may change)\nThis is the log the swarm writes about itself. It states what exists, not what it feels.\n- units verified: '+done+' of '+Object.keys(state).length+'\n- files in workspace: '+workspace.size+'\n- rules: see the protected constitution; originals are never modified.\n'}]},
  observe:async({workspace})=>({ok:/units verified/.test(workspace.text('app/SOUL.md')||''),detail:'SOUL.md written'})});
 return units;
}
