// Candidate claims never become findings merely because an author supplied them.
export class RuleVerification {
 constructor(snapshot={version:1,candidates:[],evidence:[],findings:[]}){if(snapshot.version!==1)throw Error('Unsupported verification store');this.candidates=new Map();this.evidence=new Map();this.findings=new Map();for(const x of snapshot.candidates)this.addCandidate(x);for(const x of snapshot.evidence)this.addEvidence(x);for(const x of snapshot.findings)if(this.candidates.has(x.ruleId))this.findings.set(x.ruleId,{...x,status:'needs-recheck',reason:'Persisted findings require reproducible recheck'});}
 addCandidate(rule){if(!rule?.id||typeof rule.claim!=='string'||!rule.claim.trim()||!['formal','source','empirical'].includes(rule.kind))throw Error('Candidate needs ID, claim and verification kind');const prior=this.candidates.get(rule.id);if(prior&&JSON.stringify(prior)!==JSON.stringify(rule))this.findings.delete(rule.id);this.candidates.set(rule.id,structuredClone(rule));return this.inspect(rule.id);}
 addEvidence(item){if(!item?.id||!item.source||typeof item.text!=='string')throw Error('Evidence needs ID, source and text');this.evidence.set(item.id,structuredClone(item));this.findings.clear();}
 search(id){const rule=this.candidates.get(id);if(!rule)throw Error('Unknown candidate');const words=(rule.claim.toLowerCase().match(/[a-z]{3,}/g)||[]);return [...this.evidence.values()].map(e=>({...e,matches:words.filter(w=>e.text.toLowerCase().includes(w)).length})).filter(e=>e.matches).sort((a,b)=>b.matches-a.matches);}
 verify(id){const r=this.candidates.get(id);if(!r)throw Error('Unknown candidate');const checks=[],found=this.search(id);let status='unverified',reason='No applicable reproducible verifier';
 if(r.kind==='formal'){
  const c=r.check;let actual;
  if(c?.type==='arithmetic'&&Array.isArray(c.values)&&c.values.length&&c.values.every(Number.isFinite)&&Number.isFinite(c.expected)){
   if(c.operation==='sum')actual=c.values.reduce((a,b)=>a+b,0);
   if(c.operation==='product')actual=c.values.reduce((a,b)=>a*b,1);
   if(c.operation==='quotient'&&c.values.length===2&&c.values[1]!==0)actual=c.values[0]/c.values[1];
  }else if(c?.type==='binary-decimal'&&/^[01]{1,32}$/.test(c.bits)&&Number.isInteger(c.expected))actual=parseInt(c.bits,2);
  if(Number.isFinite(actual)){const pass=actual===c.expected;checks.push({type:c.type,actual,expected:c.expected,pass});status=pass?'verified':'refuted';reason=pass?'Reproduced formal calculation only':'Formal calculation contradicts claim';}
 }else if(r.kind==='source'){
  const ids=r.evidenceIds??[],records=ids.map(x=>this.evidence.get(x));
  if(ids.length&&records.every(Boolean)&&typeof r.exactText==='string'&&r.exactText.length){const pass=records.every(e=>e.text.includes(r.exactText));checks.push({type:'source-text-match',evidenceIds:ids,pass});status=pass?'supported':'contradicted';reason='Source comparison checks transcription, not factual or causal truth';}
 }
 const finding={ruleId:id,status,scope:r.kind==='formal'?'formal':r.kind==='source'?'transcription':'empirical',reason,checks,evidenceFound:found.map(x=>x.id),next:status==='unverified'?(found.length?'Review located evidence and design an independent measurement':'Seek evidence and specify a reproducible test'):null};this.findings.set(id,finding);return structuredClone(finding);
 }
 inspect(id){if(!this.candidates.has(id))throw Error('Unknown candidate');return structuredClone(this.findings.get(id)??{ruleId:id,status:'unverified',scope:null});}
 canApply(id,scope){const f=this.findings.get(id);return !!f&&f.status==='verified'&&f.scope===scope;}
 export(){return structuredClone({version:1,candidates:[...this.candidates.values()],evidence:[...this.evidence.values()],findings:[...this.findings.values()]});}
}
