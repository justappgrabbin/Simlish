import assert from 'node:assert/strict';
import {state,encode,decode,parse,code,fromIndex,indexOf,fromDegree,POSITIONS,S} from './resolver.mjs';
const samples=['1.1.1.1.1','18.5.3.4.2','25.2.6.6.5','64.6.6.6.5','3.4.2.1.4'];
for(const s of samples){
  const st=state(s),e=st.encodings,a=parse(s);
  for(const k of ['B','H','A','U','Hx'])assert.equal(decode[k](e[k]),s,`${k} round-trip ${s}`);
  assert.equal(decode.G(e.G,a),s,'G round-trip');
  assert.equal(code(fromIndex(indexOf(a))),s);
  console.log(s,'->',st.gateState.names[0],'|',st.lineState.orbital[0].orbital,'|',st.colorState.names.join('/'),'|',st.toneState.name,'|',st.baseState.name,'| RuleO',JSON.stringify(st.ruleO),'| NATO',e.N.filter(Boolean).slice(0,4).join(' '),'| hex',e.Hx.index);
}
let n=0;for(let i=0;i<69120;i++){const a=fromIndex(i);if(indexOf(a)!==i)throw i;n++}console.log('all',n,'addresses index round-trip');
// degree round trip through gates whose doc positions agree
const ok=Object.entries(POSITIONS).filter(([,p])=>p.start!=null);
for(const [g,p] of ok.slice(0,5)){const r=fromDegree(p.start+0.0001);console.log('degree',p.start.toFixed(4),'->',r.unresolved?r:code(r));}
console.log('gates with agreed position',ok.length,'conflicting',Object.entries(POSITIONS).filter(([,p])=>p.conflict).map(([g])=>g).join(' '));
console.log('PASS');
