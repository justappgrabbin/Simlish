// Measures which swarm dynamics can actually reorganize a damaged organization. Nothing here is tuned for a target: same examples->fitness for every method.
// Usage: node integrated/dynamics-experiment.mjs
import './../vendor/kernel.js';
import {TARGETS} from './targets.mjs';
import {fitness,reorganize} from './population.mjs';
import {flipLines,hammingNeighbors} from '../state-space.mjs';
import {fitParticles} from '../swarm-fit.mjs';
import {abcFit} from './abc.mjs';
const bits=(v,n)=>Array.from({length:n},(_,i)=>(v>>i)&1);
const rows={number:[5,10,27].map(v=>bits(v,8)),text:[...'swarm os is alive'].map(c=>bits(c.charCodeAt(0),8)),pattern:Array.from({length:64},()=>[])};
const expect={number:[5,15,42].map(v=>bits(v,8)),text:[...'Swarm Os Is Alive'].map(c=>bits(c.charCodeAt(0),8)),pattern:Array.from({length:64},(_,i)=>[((i&7)&(i>>3))===0?1:0])};
const bnds=a=>a.map(()=>[0,63]),toAtoms=x=>x.map(v=>Math.round(v)&63);
function hamming(atoms,ex){let cur=atoms.slice(),bad=fitness(cur,ex);for(let r=0;r<400&&bad>0;r++){let best=null;for(let i=0;i<cur.length;i++)for(const h of hammingNeighbors(cur[i],1)){const t=cur.slice();t[i]=h.code;const b=fitness(t,ex);if(b<bad&&(!best||b<best.b))best={i,c:h.code,b}}if(!best)break;cur[best.i]=best.c;bad=best.b}return bad}
const methods={
 'hamming-1 only':(a,ex)=>hamming(a,ex),
 'PSO only (swarm-fit.mjs)':async(a,ex)=>{const r=await fitParticles({bounds:bnds(a),initial:a,score:x=>fitness(toAtoms(x),ex),particles:24,iterations:30,seed:719});return fitness(toAtoms(r.values),ex)},
 'ABC v9 only (random start, her abc_phase)':(a,ex)=>{const r=abcFit({bounds:bnds(a),initial:a,score:x=>fitness(toAtoms(x),ex),iterations:30,variant:'v9'});return fitness(toAtoms(r.values),ex)},
 'ABC v9 only (agents seeded around damaged genome, sd 0.02)':(a,ex)=>{const r=abcFit({bounds:bnds(a),initial:a,score:x=>fitness(toAtoms(x),ex),iterations:30,variant:'v9',spread:0.02});return fitness(toAtoms(r.values),ex)},
 'ABC single-axis only (SynthAi morph-search.mjs rule)':(a,ex)=>{const r=abcFit({bounds:bnds(a),initial:a,score:x=>fitness(toAtoms(x),ex),iterations:200,variant:'single-axis'});return fitness(toAtoms(r.values),ex)},
 'existing reorganize (hamming-1 then PSO)':async(a,ex)=>{const r=await reorganize(a,ex);return r.bad}};
const results=[];
for(const k of Object.keys(TARGETS)){
 const ex=[{inputs:rows[k],expect:expect[k]}],good=TARGETS[k].atoms,atoms=[];
 for(let i=8;i<good.length&&atoms.length<2;i+=Math.floor(good.length/3)){for(let l=1;l<=6;l++){const t=good.slice();t[i]=flipLines(t[i],[l]);if(fitness(t,ex)>0){atoms.push([i,l]);break}}}
 for(const n of [1,2]){
  const broken=good.slice();atoms.slice(0,n).forEach(([i,l])=>{broken[i]=flipLines(broken[i],[l])});const b0=fitness(broken,ex);
  const row={target:k,corrupted:n+' atom(s)',badBefore:b0};
  for(const [m,f] of Object.entries(methods))row[m]=await f(broken,ex);
  results.push(row);
 }
}
console.log('wrong output bits AFTER each dynamics (0 = repaired); badBefore = before');
console.table(results);
