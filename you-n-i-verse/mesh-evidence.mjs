import {RuleVerification} from './rule-verification.mjs';
// Import observations/assertions as candidates; never execute donor rule programs.
export async function importMeshEvidence(ledger,checkpoint){
 if(checkpoint?.kind!=='messy-mesh-checkpoint'||checkpoint.format!==1||checkpoint.mesh?.version!==1||!Array.isArray(checkpoint.mesh.triples))throw Error('Expected a MESSY mesh checkpoint');
 const encoded=JSON.stringify(checkpoint);if(encoded.length>5_000_000||checkpoint.mesh.triples.length>10000)throw Error('Mesh checkpoint exceeds import limit');
 const digest=[...new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(encoded)))].map(x=>x.toString(16).padStart(2,'0')).join('');
 const staged=new RuleVerification(ledger.export());let candidates=0;
 for(const [i,t] of checkpoint.mesh.triples.entries()){
  if(!t||!['subject','relation','object'].every(k=>typeof t[k]==='string'&&t[k].trim()&&t[k].length<=180))throw Error('Invalid mesh assertion at index '+i);
  const id=`mesh:${digest}:${i}`,claim=`${t.subject} ${t.relation} ${t.object}`;
  staged.addEvidence({id,source:`MESSY checkpoint sha256:${digest}`,text:claim,assertion:structuredClone(t),active:t.deletedAt==null,provenance:t.provenance??null,evidenceType:'imported-assertion-not-independent-measurement'});
  if(t.deletedAt==null){staged.addCandidate({id,kind:'empirical',claim,evidenceIds:[id],origin:'MESSY checkpoint'});candidates++;}
 }
 return {ledger:staged,digest,assertions:checkpoint.mesh.triples.length,candidates,executableRulesImported:0};
}
