// Runnable demo: THREE unrelated targets through the SAME primitive population (see PRIMITIVES.md). Usage: node integrated/population-demo.mjs
import '../vendor/kernel.js';
import {TARGETS} from './targets.mjs';
import {lifecycle} from './lifecycle.mjs';
import {Population} from './population.mjs';
import {HistoryLedger} from './ledger.mjs';
import {inventoryPrimitives,CONNECTED_TO_EXECUTION} from './inventory.mjs';
import {roundTrip} from './roundtrip.mjs';
import {atomsToBytes} from './population.mjs';
import {writeZip} from '../zip-lite.mjs';
const bits=(v,n)=>Array.from({length:n},(_,i)=>(v>>i)&1),num=a=>a.reduce((s,b,i)=>s+(b<<i),0),te=new TextEncoder();
// Presentation only (outside the primitive semantics, listed as a limitation): bits -> number / text / picture.
const numberRows=[5,10,27].map(v=>bits(v,8)),numberExpect=[5,15,42].map(v=>bits(v,8));
const sentence='swarm os is alive',textRows=[...sentence].map(c=>bits(c.charCodeAt(0),8)),textExpect=[...'Swarm Os Is Alive'].map(c=>bits(c.charCodeAt(0),8));
const patternRows=Array.from({length:64},()=>[]),patternExpect=Array.from({length:64},(_,i)=>[((i&7)&(i>>3))===0?1:0]);
const pop=new Population(),ledger=new HistoryLedger(),sayy=(t,o)=>console.log('\n== '+t+(o===undefined?'':'\n'+(typeof o==='string'?o:JSON.stringify(o)))) ;
const inv=await inventoryPrimitives();
sayy('existing primitives inventoried (executable by live probe): '+inv.filter(x=>x.executable).length+'/'+inv.length,inv.filter(x=>!x.executable).map(x=>x.id+' -> '+x.evidence).join('; '));
const runs={number:{rows:numberRows,examples:[{inputs:numberRows,expect:numberExpect}]},text:{rows:textRows,examples:[{inputs:textRows,expect:textExpect}]},pattern:{rows:patternRows,examples:[{inputs:patternRows,expect:patternExpect}]}};
const results={};
for(const [k,t] of Object.entries(TARGETS)){const r=await lifecycle({name:t.name,atoms:t.atoms,rows:runs[k].rows,examples:runs[k].examples,population:pop,ledger});results[k]=r;
 const o=r.outputs;
 const shown=k==='number'?o.map(num).join(', ')+' (running totals of 5, 10, 27)':k==='text'?'"'+Buffer.from(o.map(num)).toString()+'"':'\n'+Array.from({length:8},(_,y)=>o.slice(y*8,y*8+8).map(v=>v[0]?'#':'.').join('')).join('\n');
 sayy(k.toUpperCase()+'  '+t.name+'  address '+r.address.slice(0,56)+'...',shown);
 console.log('reduced',JSON.stringify(r.reduced),'| composed',JSON.stringify(r.composed.kinds),'defs',r.composed.definitions.length,'nodes',r.composed.nodes,'| deterministic',r.deterministic,'| state recovered from address',r.recovered);
 console.log('reorganize: corrupted atom',r.reorganize.corruptedAtom,'bad',r.reorganize.badBefore,'->',r.reorganize.badAfter,'repaired',r.reorganize.repaired,'stages',JSON.stringify(r.reorganize.stages),'pso-alone',JSON.stringify(r.reorganize.psoAlone),'new address differs',r.reorganize.changedAddress);}
const types={'a.js':te.encode('return 1+1;'),'d.json':te.encode('{"a":[1,2]}'),'s.css':te.encode('a{b:c}'),'p.html':te.encode('<h1>x</h1>'),'n.md':te.encode('# t\nwords'),'t.py':te.encode('print(1)'),
 'add.wasm':Uint8Array.from([0,0x61,0x73,0x6d,1,0,0,0]),'z.zip':writeZip([{path:'a.js',data:'return 1'}]),'small.bin':Uint8Array.from({length:600},(_,i)=>(i*131+7)&255),'big.bin':Uint8Array.from({length:7000},(_,i)=>(i*131+7)&255),'running-total.prim':atomsToBytes(TARGETS.number.atoms)};
const rt=[];for(const [n,b] of Object.entries(types))rt.push(await roundTrip(n,b));
sayy('lossless round trip per file type (bytes->primitives->bytes | Bantu tokens | her ingest reconstruct | her descend not truncated | address alone | source level)');
console.table(rt.map(r=>({file:r.name,bytes:r.bytes,viaPrimitives:r.viaPrimitives,bantu:r.bantuTokens,ingestRebuild:r.ingestReconstruct,herDescendComplete:!r.herDescendTruncated,fromAddressAlone:r.fromAddressAlone,sourceLevel:r.sourceLevel})));
console.log('\nnot connected to the population\'s execution:',CONNECTED_TO_EXECUTION.notConnectedToPopulationExecution.join(' | '));
