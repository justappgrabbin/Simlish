import test from 'node:test';import assert from 'node:assert/strict';
import {AbcV9,AbcSingleAxis,abcFit,V9_ROLE_SPLIT,ABC_INVESTIGATING} from './abc.mjs';
const sphere=p=>p.reduce((s,v)=>s+(v-0.3)**2,0);
test('ABC v9 (swarm_os_full.py abc_phase): roles employed/onlooker/scout about 0.4/0.4/0.2; scouts restart after 15 steps; deterministic for a seed',()=>{
 const a=new AbcV9({dims:4,objective:sphere,agents:400,seed:5}),n=r=>a.agents.filter(x=>x.role===r).length/400;
 assert.ok(Math.abs(n('employed')-V9_ROLE_SPLIT.employed)<0.08&&Math.abs(n('onlooker')-V9_ROLE_SPLIT.onlooker)<0.08&&Math.abs(n('scout')-V9_ROLE_SPLIT.scout)<0.08);
 const scout=a.agents.find(x=>x.role==='scout');for(let i=0;i<15;i++)a.step();assert.equal(scout.age,15);a.step();assert.equal(scout.age,0,'scout abandoned its position after age>15');
 const b=new AbcV9({dims:4,objective:sphere,agents:40,seed:9}),c=new AbcV9({dims:4,objective:sphere,agents:40,seed:9});for(let i=0;i<10;i++){b.step();c.step()}assert.deepEqual(b.best,c.best);
});
test('both ABC variants improve a plain numeric objective (they are searchers; no claim beyond that)',()=>{
 for(const [C,opt] of [[AbcV9,{agents:40}],[AbcSingleAxis,{agents:12}]]){const s=new C({dims:6,objective:sphere,seed:3,...opt});const first=s.step().score;for(let i=0;i<60;i++)s.step();assert.ok(s.best.score<first,C.name+' improves '+first+' -> '+s.best.score)}
});
test('abcFit has the call shape of swarm-fit.mjs fitParticles and never returns a worse score than the start',()=>{
 for(const variant of ['v9','single-axis']){const r=abcFit({bounds:[[0,63],[0,63]],initial:[10,50],score:x=>(x[0]-20)**2+(x[1]-40)**2,iterations:20,variant});assert.ok(r.score<=r.initialScore);assert.equal(r.workers,false);assert.equal(r.values.length,2)}
 assert.throws(()=>abcFit({bounds:[[0,1]],initial:[0],score:()=>0,variant:'nope'}),RangeError);
 assert.ok(ABC_INVESTIGATING.length>=3);
});
