import {RuleVerification} from './rule-verification.mjs';
const key='you-n-i-verse-verification-v1',$=id=>document.getElementById(id);let ledger;
try{const saved=localStorage.getItem(key);ledger=new RuleVerification(saved?JSON.parse(saved):undefined);}catch(e){ledger=new RuleVerification();$('status').textContent='Previous records could not load: '+e.message;}
if(!ledger.candidates.size){
 ledger.addEvidence({id:'page5',source:'User supplied screenshot, printed page 5',text:'Movement is Energy. Energy is Creation. Creation is Seeing. Seeing is Landscape. Landscape is Environment.'});
 ledger.addCandidate({id:'gate-arc',kind:'formal',claim:'360 divided by 64 equals 5.625 degrees',check:{type:'arithmetic',operation:'quotient',values:[360,64],expected:5.625}});
 ledger.addCandidate({id:'display-binary',kind:'formal',claim:'Displayed binary 100010 has decimal value 34',check:{type:'binary-decimal',bits:'100010',expected:34}});
 ledger.addCandidate({id:'chain-transcription',kind:'source',claim:'Movement is Energy is printed in the supplied source',exactText:'Movement is Energy',evidenceIds:['page5']});
 for(const [id,claim] of [['causal-chain-placement','Movement relationships determine swarm placement'],['gate-anchor-placement','Gate anchors determine entry placement'],['line-trajectory','Line trajectory determines swarm action'],['resonance-weight','Resonance doubles causal output weight']])ledger.addCandidate({id,kind:'empirical',claim});
}
function save(){try{localStorage.setItem(key,JSON.stringify(ledger.export()));return true;}catch(e){$('status').textContent='Storage unavailable: '+e.message+'. Export records to retain this investigation.';return false;}}
function render(){$('findings').replaceChildren();for(const r of ledger.candidates.values()){const a=document.createElement('article'),h=document.createElement('h2'),p=document.createElement('p'),pre=document.createElement('pre');h.textContent=r.claim;const f=ledger.inspect(r.id);p.textContent=f.status+' · '+(f.scope??'not checked');pre.textContent=JSON.stringify({...f,locatedEvidence:ledger.search(r.id).map(e=>({id:e.id,source:e.source,text:e.text}))},null,2);a.append(h,p,pre);$('findings').append(a);}}
$('check').onclick=()=>{for(const id of ledger.candidates.keys())ledger.verify(id);if(save())$('status').textContent='Checks completed and saved. Unverified rules remain blocked.';render();};
$('add').onclick=()=>{try{const r=JSON.parse($('record').value);if(r.version===1&&Array.isArray(r.candidates))ledger=new RuleVerification(r);else if(r.kind)ledger.addCandidate(r);else ledger.addEvidence(r);save();render();}catch(e){$('status').textContent=e.message;}};
$('export').onclick=()=>{const blob=new Blob([JSON.stringify(ledger.export(),null,2)],{type:'application/json'});if(window.NativeDownloads?.save){const reader=new FileReader();reader.onload=()=>NativeDownloads.save('rule-investigation.json','application/json',reader.result.split(',')[1]);reader.readAsDataURL(blob);}else{const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='rule-investigation.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);}};
render();
