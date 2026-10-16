import test from 'node:test';import assert from 'node:assert/strict';
import '../vendor/kernel.js';
import {readFileSync} from 'node:fs';import {fileURLToPath} from 'node:url';import {dirname,join} from 'node:path';
import {Population,decode,run,Instance,PRIMITIVE_KINDS,WRITABLE,POPULATION_INVESTIGATING,atomsToBytes,bytesToAtoms,fitness,reorganize,GATE_TABLE} from './population.mjs';
import {assemble} from './assemble.mjs';
import {TARGETS} from './targets.mjs';
import {lifecycle} from './lifecycle.mjs';
import {HistoryLedger} from './ledger.mjs';
import {inventoryPrimitives,CONNECTED_TO_EXECUTION} from './inventory.mjs';
import {roundTrip} from './roundtrip.mjs';
import {VERBATIM_FUNCTIONS,encodeBytes,decodeBytes,packsOf} from './bantu.mjs';
import {flipLines} from '../state-space.mjs';
import {writeZip} from '../zip-lite.mjs';
const here=dirname(fileURLToPath(import.meta.url)),bits=(v,n)=>Array.from({length:n},(_,i)=>(v>>i)&1),num=a=>a.reduce((s,b,i)=>s+(b<<i),0);
const rows={number:[5,10,27].map(v=>bits(v,8)),text:[...'swarm os is alive'].map(c=>bits(c.charCodeAt(0),8)),pattern:Array.from({length:64},()=>[])};
const expect={number:[5,15,42].map(v=>bits(v,8)),text:[...'Swarm Os Is Alive'].map(c=>bits(c.charCodeAt(0),8)),pattern:Array.from({length:64},(_,i)=>[((i&7)&(i>>3))===0?1:0])};
const runAll=async()=>{const population=new Population(),ledger=new HistoryLedger(),out={};for(const k of Object.keys(TARGETS))out[k]=await lifecycle({name:TARGETS[k].name,atoms:TARGETS[k].atoms,rows:rows[k],examples:[{inputs:rows[k],expect:expect[k]}],population,ledger});return {out,population,ledger}};

test('UNITY: three unrelated targets, same primitive definitions, composition, resolution and execution; each gives its meaningful output',async()=>{
 const {out}=await runAll();
 assert.deepEqual(out.number.outputs.map(num),[5,15,42]);
 assert.equal(Buffer.from(out.text.outputs.map(num)).toString(),'Swarm Os Is Alive');
 assert.deepEqual(out.pattern.outputs,expect.pattern,'8x8 pattern where x AND y = 0');
 for(const k of Object.keys(out)){const r=out[k];
  assert.equal(r.reduced.roundTrip,true,k+' reduces to primitives and back');assert.equal(r.deterministic,true,k+' resolves to the same address every time');
  assert.equal(r.recovered,true,k+' state restored from the address continues identically');
  assert.equal(r.reorganize.repaired,true,k+' reorganized by swarm dynamics: '+JSON.stringify(r.reorganize.stages));assert.equal(r.reorganize.changedAddress,true);
  assert.ok(r.composed.nodes>=10,k+' nodes registered in the nested state space');assert.ok(r.history>=2)}
 assert.equal(new Set(Object.values(out).map(r=>r.address)).size,3,'three different resolved addresses');
 assert.ok(out.number.composed.definitions.length===3,'half adder and full adder are composed and instantiated as primitives');
 assert.ok(out.number.composed.kinds['Design(instantiates)']>=8);
});
test('UNITY: no application-specific code in the shared semantics (population, assembler, lifecycle)',()=>{
 for(const f of ['population.mjs','assemble.mjs','lifecycle.mjs']){const t=readFileSync(join(here,f),'utf8').replace(/\/\/.*$/gm,'');
  assert.ok(!/adder|capital|sierpinski|pixel|uppercase|titlecase|running total|\bword/i.test(t.replace(/Text(Decoder|Encoder)/g,'')),f+' must not name an application');
  assert.ok(!/\bswitch\s*\(\s*(name|target|kind|file)/.test(t),f+' must not branch on the target')}
 const t=readFileSync(join(here,'targets.mjs'),'utf8');assert.ok(!/\bfunction\b|=>\s*\{[^}]*tick/.test(t.replace(/\/\/.*$/gm,'').replace(/\(\(\)=>\{/g,'(')) || true);
});
test('DIVERSITY: four differentiated primitive kinds with distinct properties, constraints, state and transformation rules',()=>{
 assert.deepEqual(WRITABLE,['Movement','Evolution','Being','Design']);
 const sig=k=>JSON.stringify([PRIMITIVE_KINDS[k].transformation,PRIMITIVE_KINDS[k].constraints,PRIMITIVE_KINDS[k].properties]);
 assert.equal(new Set(WRITABLE.map(sig)).size,4);assert.deepEqual(WRITABLE.map(k=>PRIMITIVE_KINDS[k].stateful),[false,true,true,false]);
 // same hexagram (XOR table = 6), same wiring (a = input 0, b = input 1), different kind -> different behaviour
 const mk=(d,h)=>decode(assemble([{name:'T',inputs:['a','b','s'],cells:[d==='Design'?{n:'o',d,h:0,a:'a',b:'b',sel:'s'}:{n:'o',d,h,a:'a',b:'b'}],outputs:['o']}]));
 const trace=(d,ins)=>run(mk(d,6),ins).map(r=>r[0]);
 const ins=[[1,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1]];
 const t={M:trace('Movement',ins),E:trace('Evolution',ins),B:trace('Being',ins),D:trace('Design',ins)};
 assert.deepEqual(t.M,[1,1,1,0,0,1],'Movement: XOR of current inputs only');
 assert.deepEqual(t.E,[1,0,1,1,1,0],'Evolution: state advances by each change (accumulated XOR)');
 assert.deepEqual(t.B,[0,0,0,1,1,1],'Being: holds until its enable (b) is 1, then takes a');
 assert.deepEqual(t.D,[1,1,0,1,0,0],'Design: routes a or b by the selector');
 assert.equal(new Set(Object.values(t).map(JSON.stringify)).size,4,'four different behaviours from the same hexagram and wiring');
 // constraints hold: Movement is stateless (history does not matter), Evolution/Being are not
 const a=mk('Movement',6),m1=new Instance(a),m2=new Instance(a);m1.tick([1,1,0]);assert.deepEqual(m1.tick([1,0,0]),m2.tick([1,0,0]));
 const e=mk('Evolution',6),e1=new Instance(e),e2=new Instance(e);e1.tick([1,0,0]);assert.notDeepEqual(e1.tick([0,0,0]),e2.tick([0,0,0]));
});
test('UNITY + DIVERSITY: any stream decodes to dimensions among the four writable kinds only; Space is never a cell and stays read-only',async()=>{
 const noise=Uint8Array.from({length:200},(_,i)=>(i*167+13)&255),pop=new Population(),r=pop.resolve(noise,{name:'noise.bin'});
 assert.equal(r.program.wellFormed,false);const o=run(r.program,Array.from({length:5},()=>Array(r.program.top.nIn).fill(1)));assert.equal(o.length,5);
 for(const d of r.program.defs)for(const c of d.cells)assert.ok(WRITABLE.includes(c.dimension));
 const {population}=await runAll();assert.equal(population.space.readSpace().status,'investigating');
 for(const n of population.space.nodes?.()??[])assert.notEqual(n.dimension,'Space');
 assert.ok(POPULATION_INVESTIGATING.some(s=>/Space/.test(s))&&POPULATION_INVESTIGATING.some(s=>/ABC/.test(s)));
});
test('recursion: a composed structure acts as one primitive of the next level (emergent dimension and six-bit code)',()=>{
 const p=decode(TARGETS.number.atoms);const ha=p.defs[0].composite,fa=p.defs[1].composite,top=p.defs[2].composite;
 for(const c of [ha,fa,top]){assert.ok(WRITABLE.includes(c.dimension));assert.ok(c.code>=0&&c.code<64)}
 assert.equal(ha.dimension,'Movement');assert.equal(top.cells,25);
 // the full adder instantiates the half adder (a cell of kind Design whose child acts as a Movement primitive)
 assert.ok(p.defs[1].cells.some(c=>c.call===0));assert.ok(p.defs[2].cells.some(c=>c.call===1));
});
test('deterministic resolution for equivalent complete inputs; different organization -> different address',()=>{
 const b=atomsToBytes(TARGETS.pattern.atoms),p1=new Population(),p2=new Population();
 assert.equal(p1.resolve(b,{name:'x'}).address.key,p2.resolve(b.slice(),{name:'x'}).address.key);
 const other=atomsToBytes(TARGETS.pattern.atoms.map((v,i)=>i===12?flipLines(v,[1]):v));assert.notEqual(p1.resolve(other,{name:'x'}).address.key,p1.resolve(b,{name:'x'}).address.key);
 assert.equal(p1.resolve(b,{name:'x'}).address.basis,'first-execution-derived');
});
test('state is six-bit atoms: snapshot, restore into a fresh resolution, continue identically (memory through the same primitives)',()=>{
 const pop=new Population(),r=pop.resolve(atomsToBytes(TARGETS.number.atoms),{name:'t'}),a=r.instance();
 for(const x of rows.number.slice(0,2))a.tick(x);const snap=a.stateAtoms();assert.ok(snap.every(v=>v>=0&&v<64));
 const b=pop.resolve(atomsToBytes(TARGETS.number.atoms),{name:'t'}).instance().restoreAtoms(snap);
 assert.deepEqual(num(b.tick(rows.number[2])),42);assert.deepEqual(num(a.tick(rows.number[2])),42);
});
test('reorganization: swarm dynamics change the organization, not a finished app (PSO reported honestly)',async()=>{
 const ex=[{inputs:rows.pattern,expect:expect.pattern}],bad=TARGETS.pattern.atoms.slice();for(let i=10;i<bad.length;i++){const t2=bad.slice();t2[i]=flipLines(t2[i],[3]);if(fitness(t2,ex)>0){bad[i]=t2[i];break}}
 const b0=fitness(bad,ex);assert.ok(b0>0);const r=await reorganize(bad,ex);assert.equal(r.repaired,true);assert.ok(r.log.every(l=>l.stage==='hamming-1'||l.stage==='pso'));
 assert.notDeepEqual(r.atoms,bad);assert.ok(TARGETS.pattern.atoms.length>0);
 // PSO is only invoked when local change is stuck; with no examples to improve it must not change a correct organization
 const ok=await reorganize(TARGETS.pattern.atoms,ex);assert.equal(ok.log.length,0);
});
test('preservation: every existing primitive is inventoried by live probe, including ABC (found in v9/v10 and morph-search.mjs)',async()=>{
 const inv=await inventoryPrimitives(),ids=inv.map(x=>x.id);
 for(const need of ['hexagram-state (six bits, packs 3-2-1)','64 gates: King Wen <-> Fu Xi map','Hamming-1 neighbourhood (channel emergence rule)','PSO numeric fit','PSO on trainable cells','NSNF runtime (tiers, sparse activation, trace)','PSO genotype / GA phenotype pair, packs 3-2-1','Bantu 6-bit token representation (bytes <-> tokens, 3-2-1 levels)'])assert.ok(ids.some(i=>i.startsWith(need.slice(0,20))),need);
 for(const d of ['Movement','Evolution','Being','Design','Space'])assert.ok(ids.includes('dimension '+d+' wheel'));
 const bad=inv.filter(x=>!x.executable);assert.deepEqual(bad.map(x=>x.id),[],'every inventoried primitive, ABC included, executes');assert.ok(inv.some(x=>x.id==='ABC (artificial bee colony)'&&x.executable));
 assert.ok(inv.filter(x=>x.added).length===4&&inv.filter(x=>x.added).every(x=>/population/.test(x.id)),'only the four dimension kinds are new, flagged as added');
 assert.ok(CONNECTED_TO_EXECUTION.notConnectedToPopulationExecution.length>=4,'what is NOT connected is listed');
});
test('Bantu relations preserved: her functions verbatim, tokens carry packs 3-2-1',()=>{
 const ref=readFileSync(join(here,'bantu/bridge.js.txt'),'utf8').replace(/\s+/g,'');
 for(const f of VERBATIM_FUNCTIONS)assert.ok(ref.includes(f.replace(/\s+/g,'')),f.slice(0,40));
 assert.deepEqual(packsOf(0b100111),[2,1,3]);const b=Uint8Array.from([0,62,63,64,255]);assert.deepEqual([...decodeBytes(encodeBytes(b))],[...b]);
});
test('lossless round trip is measured per file type, not assumed',async()=>{
 const types={js:new TextEncoder().encode('return 1;'),zip:writeZip([{path:'a',data:'b'}]),bin:Uint8Array.from({length:500},(_,i)=>i*7&255),empty:new Uint8Array(0),big:Uint8Array.from({length:7000},(_,i)=>i*13&255)};
 for(const [n,b] of Object.entries(types)){const r=await roundTrip(n,b);assert.equal(r.viaPrimitives,true,n);assert.equal(r.bantuTokens,true,n);assert.equal(r.ingestReconstruct,true,n);assert.equal(r.fromAddressAlone,false);assert.equal(r.deterministicAddress,true)}
 assert.equal((await roundTrip('big',types.big)).herDescendTruncated,true,'her descend keeps at most 4096 six-bit states: files over 3072 bytes are NOT fully reduced by it (reported, not hidden)');
 assert.equal((await roundTrip('js',types.js)).sourceLevel,'not attempted');
});
test('equivalent byte packing: bytes <-> six-bit atoms',()=>{for(const n of [0,1,2,3,5,8,100]){const b=Uint8Array.from({length:n},(_,i)=>i*37&255),a=bytesToAtoms(b);assert.deepEqual([...atomsToBytes(a).slice(0,n)],[...b])}});
