import test from 'node:test';import assert from 'node:assert/strict';
import {SwarmComponent,SwarmPair,GenotypeUnit,PhenotypeUnit,INVESTIGATING} from './swarm-nsnf.mjs';
import {sha256} from './sha256.mjs';
import {registrationPair} from './registration-swarm.mjs';
const mem=(f)=>Array.from({length:6},(_,i)=>({id:'m'+i,check:f}));
test('sha256 matches known vectors',()=>{assert.equal(sha256(''),'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');assert.equal(sha256('abc'),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')});
test('PSO is the genotype, GA is the phenotype and cannot exist without a genotype',()=>{
 const pso=new SwarmComponent({id:'g',kind:'PSO',members:mem(i=>i.ok)}),ga=new SwarmComponent({id:'p',kind:'GA',members:mem(()=>true)});
 assert.throws(()=>new GenotypeUnit(ga));assert.throws(()=>new PhenotypeUnit({component:pso,trait:()=>({})}));
 const pu=new PhenotypeUnit({component:ga,trait:g=>({code:g.hexagram_code})});
 assert.throws(()=>pu.express(null,{}),/genotype/);
 const pair=new SwarmPair({id:'x',genotype:pso,phenotype:{component:ga,trait:g=>({code:g.hexagram_code})}});
 const r=pair.forward({ok:true});assert.equal(r.genotype.hexagram_code,'111001');assert.equal(r.phenotype.visible.code,'111001');assert.equal(r.phenotype.genotype_ref.hexagram_code,'111001');
});
test('NSNF terms: component record, event_log and transition; spec gaps are listed as investigating',()=>{
 const pso=new SwarmComponent({id:'g',kind:'PSO',members:mem(i=>i.ok)}),ga=new SwarmComponent({id:'p',kind:'GA',members:mem(()=>true)});
 const pair=new SwarmPair({id:'x',genotype:pso,phenotype:{component:ga,trait:()=>({})}});
 pair.forward({ok:true});pair.forward({ok:false});
 const s=pair.snapshot();
 assert.deepEqual(Object.keys(s.components[0]),['id','gate_id','architecture_type','artifact_hash','parameter_ref','version']);
 assert.equal(s.components[0].gate_id,null);assert.equal(s.event_log.length,2);assert.match(s.event_log[0].payload_hash,/^[0-9a-f]{64}$/);
 assert.equal(s.transitions[0].event_type,'status-change');assert.ok(s.investigating.length>=5&&INVESTIGATING===s.investigating);
});
test('registration pair expresses an investigating phenotype with the missing places',()=>{
 const r=registrationPair().forward({id:'a',version:'1',address:{gate:6}});
 assert.equal(r.status,'investigating');assert.ok(r.phenotype.visible.missing.includes('arcSecond'));assert.equal(r.phenotype.visible.kept,true);
});
