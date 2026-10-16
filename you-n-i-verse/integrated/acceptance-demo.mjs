// Executable evidence for the scorecard (integrated/SCORECARD.md). Prints measurements only; it decides nothing. Usage: node integrated/acceptance-demo.mjs
// Every probe calls the existing code; nothing here is part of the primitive semantics.
import '../vendor/kernel.js';
import {spawnSync} from 'node:child_process';
import {TARGETS} from './targets.mjs';
import {lifecycle} from './lifecycle.mjs';
import {Population,decode,run,Instance,atomsToBytes,bytesToAtoms,fitness,reorganize,WRITABLE} from './population.mjs';
import {assemble} from './assemble.mjs';
import {HistoryLedger} from './ledger.mjs';
import {finishApp} from '../app-finisher.mjs';
import {personalize} from '../personalize.mjs';
import {IntegratedSwarm} from './integrated-swarm.mjs';
const bits=(v,n)=>Array.from({length:n},(_,i)=>(v>>i)&1),num=a=>a.reduce((s,b,i)=>s+(b<<i),0);
const rows={number:[5,10,27].map(v=>bits(v,8)),text:[...'swarm os is alive'].map(c=>bits(c.charCodeAt(0),8)),pattern:Array.from({length:64},()=>[])};
const expect={number:[5,15,42].map(v=>bits(v,8)),text:[...'Swarm Os Is Alive'].map(c=>bits(c.charCodeAt(0),8)),pattern:Array.from({length:64},(_,i)=>[((i&7)&(i>>3))===0?1:0])};
const say=(k,v)=>console.log('\n### '+k+'\n'+(typeof v==='string'?v:JSON.stringify(v,null,1)));
const pop=new Population(),ledger=new HistoryLedger(),L={};
for(const k of Object.keys(TARGETS))L[k]=await lifecycle({name:TARGETS[k].name,atoms:TARGETS[k].atoms,rows:rows[k],examples:[{inputs:rows[k],expect:expect[k]}],population:pop,ledger});
say('1 UNITY / 5 EXECUTION / 3 COMPOSITION',Object.fromEntries(Object.entries(L).map(([k,r])=>[k,{output:k==='number'?r.outputs.map(num):k==='text'?Buffer.from(r.outputs.map(num)).toString():'8x8 rows '+Array.from({length:8},(_,y)=>r.outputs.slice(y*8,y*8+8).map(v=>v[0]?'#':'.').join('')).join('|'),definitions:r.composed.definitions.length,kinds:r.composed.kinds,nodes:r.composed.nodes}])));
// 2/8 DIVERSITY + INTEGRITY: how many distinct behaviours do the 64 hexagram values H give each kind? (64 gates -> ?)
const probeIns=[];for(let t=0;t<24;t++)probeIns.push([t&1,(t>>1)&1,(t>>2)&1]);
const behaviours=kind=>{const seen=new Set();for(let H=0;H<64;H++){const cell=kind==='Design'?{n:'o',d:kind,h:H&31,a:'a',b:'b',sel:'s'}:{n:'o',d:kind,h:H,a:'a',b:'b'};
  const p=decode(assemble([{name:'T',inputs:['a','b','s'],cells:[cell],outputs:['o']}]));seen.add(JSON.stringify(run(p,probeIns)))}return seen.size};
say('2/8 distinct behaviours among the 64 hexagram values (H) per primitive kind',Object.fromEntries(WRITABLE.map(k=>[k,behaviours(k)+' of 64'])));
// 4 RESOLUTION
const b=atomsToBytes(TARGETS.pattern.atoms);const k1=new Population().resolve(b,{name:'x'}).address.key,k2=new Population().resolve(b.slice(),{name:'x'}).address.key,k3=new Population().resolve(b,{name:'a-different-name'}).address.key;
say('4 RESOLUTION',{sameBytesSameName_freshPopulations:k1===k2,sameBytesDifferentName:k1===k3,note:'false means the artifact NAME takes part in the address'});
// 6 MEMORY: state atoms through JSON, restored into a fresh process
{const t=TARGETS.number,res=pop.resolve(atomsToBytes(t.atoms),{name:'m'}),a=res.instance();for(const r of rows.number.slice(0,2))a.tick(r);
 const payload=JSON.stringify({bytes:[...atomsToBytes(t.atoms)],state:a.stateAtoms(),next:rows.number[2]});
 const child=spawnSync(process.execPath,['--input-type=module','-e',`import './vendor/kernel.js';import {Population} from './integrated/population.mjs';const p=JSON.parse(process.argv[1]);const i=new Population().resolve(Uint8Array.from(p.bytes),{name:'m'}).instance().restoreAtoms(p.state);console.log(JSON.stringify(i.tick(p.next)))`,payload],{cwd:new URL('..',import.meta.url).pathname,encoding:'utf8'});
 say('6 MEMORY (state serialised as JSON, restored in a NEW node process, continues)',{continuedOutputBits:child.stdout.trim(),expected42:JSON.stringify(bits(42,8)),equal:child.stdout.trim()===JSON.stringify(bits(42,8)),stderr:child.stderr.slice(0,200),durableStoreWiredInPopulation:false});}
// 11 GAP FINISHING
{const good=TARGETS.number.atoms,ex=[{inputs:rows.number,expect:expect.number}];
 const cut=good.slice(0,good.length-12);const before=fitness(cut,ex);const r=await reorganize(cut,ex);
 say('11a incomplete primitive organisation (last 12 of '+good.length+' atoms missing)',{badBitsBefore:before,afterReorganize:r.bad,repaired:r.repaired,note:'missing atoms decode as 0 (decode is total); nothing infers the missing structure'});
 const files=[{path:'index.html',data:'<h1>hello</h1>'}];const fin=await finishApp({name:'half.zip',files},{personalization:{name:'Alexis',worldTheme:'forest',target:'reality'},kernel:undefined});
 say('11b finishApp on an incomplete app (one html file)',{status:fin.status??fin.assessment?.status,added:(fin.added||fin.addedPaths||[]).map(f=>f.path??f),keys:Object.keys(fin).slice(0,12),targetsImplemented:Object.fromEntries(['reality','game','2d'].map(t=>[t,t==='reality']))});}
// 12 PERSONALIZATION
{const A={name:'Ada',worldTheme:'forest',target:'reality',address:{planetary:'Sun',dimension:'Being',zodiac:'Virgo',gate:6,line:4,color:4,tone:4,base:2,degree:25,minute:59,second:31,arcSecond:92,house:5}};
 const B={...A,name:'Bo',worldTheme:'ocean',address:{...A.address,dimension:'Movement',gate:1,line:1}};
 const pa=personalize(A),pb=personalize(B),bytes=new TextEncoder().encode('return 1+1;');
 const sa=await new IntegratedSwarm().execute({name:'a.js',bytes},{personalization:pa}),sb=await new IntegratedSwarm().execute({name:'a.js',bytes},{personalization:pb}),s0=await new IntegratedSwarm().execute({name:'a.js',bytes});
 say('12 PERSONALIZATION (same file, three users: Ada, Bo, none)',{resolvedKey:[sa.address.resolved.basis,sb.address.resolved.basis,s0.address.resolved.basis],differentResolved:sa.address.resolved.key!==sb.address.resolved.key,appCodes:[sa.app.code,sb.app.code,s0.app.code],placement:[sa.swarm.placement?.label,sb.swarm.placement?.label,s0.swarm.placement?.label],executionSameOutput:JSON.stringify(sa.execution.returnValue)===JSON.stringify(sb.execution.returnValue),phenotypeTitles:[pa.swarm.phenotype.title,pb.swarm.phenotype.title],populationTakesPersonalizationInput:false});}
