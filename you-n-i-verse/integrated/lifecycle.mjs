// The same lifecycle for every target; no target-specific branch exists here: reduce -> resolve/address -> compose -> execute -> remember/recover -> reorganize.
import {Population,Instance,run,decode,atomsToBytes,bytesToAtoms,fitness,reorganize,flipLines} from './population.mjs';
import {HistoryLedger} from './ledger.mjs';
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
export async function lifecycle({name,atoms,rows,examples,population=new Population(),ledger=new HistoryLedger(),corruptAfter=8}){
 const bytes=atomsToBytes(atoms),res=population.resolve(bytes,{name});                          // reduce (her descend) + resolve (her resolver) + nodes in the state space
 const reduced={bytes:bytes.length,primitives:res.primitives,roundTrip:same(bytesToAtoms(bytes).slice(0,atoms.length),atoms)};
 const out=run(res.program,rows);                                                                 // execute
 const composed={definitions:res.program.defs.map(d=>({index:d.index,cells:d.cells.length,actsAs:{dimension:d.composite.dimension,code:d.composite.code}})),nodes:res.nodes.length,kinds:res.program.defs.flatMap(d=>d.cells).reduce((a,c)=>(a[c.call===null?c.dimension:'Design(instantiates)']=(a[c.call===null?c.dimension:'Design(instantiates)']||0)+1,a),{})};
 // remember + recover: state is six-bit atoms landed at the resolved address; a fresh resolution restores it and continues identically
 const half=Math.floor(rows.length/2),a=res.instance();for(const r of rows.slice(0,half))a.tick(r);
 ledger.append(res.address.key,{name,kind:'state',ticks:half,state:a.stateAtoms()});
 const b=population.resolve(bytes,{name}).instance(),saved=ledger.recall(res.address.key).filter(e=>e.kind==='state').pop();b.restoreAtoms(saved.state);
 const resumed=rows.slice(half).map(r=>b.tick(r)),recovered=same(resumed,out.slice(half));
 const again=population.resolve(bytes,{name});const deterministic=again.address.key===res.address.key;
 // reorganize: the swarm changes the organization (the stream), not a finished app
 let at=-1;for(let i=corruptAfter;i<atoms.length&&at<0;i++)for(let l=1;l<=6&&at<0;l++){const t=atoms.slice();t[i]=flipLines(t[i],[l]);if(fitness(t,examples)>0)at=i*8+l}
 const idx=Math.floor(at/8),line=at%8,broken=atoms.slice();broken[idx]=flipLines(broken[idx],[line]);
 const brokenBad=fitness(broken,examples),r=await reorganize(broken,examples);
 const newRes=population.resolve(atomsToBytes(r.atoms),{name:name+' (reorganized)'});
 ledger.append(res.address.key,{name,kind:'reorganized-to',to:newRes.address.key,corrupted:{atom:idx,line},steps:r.log.length});
 ledger.append(newRes.address.key,{name,kind:'reorganized-from',from:res.address.key});
 return {name,reduced,address:res.address.key,deterministic,composed,outputs:out,recovered,
  reorganize:{corruptedAtom:idx,line,badBefore:brokenBad,badAfter:r.bad,repaired:r.repaired,stages:r.log.map(l=>l.stage).reduce((m,s)=>(m[s]=(m[s]||0)+1,m),{}),psoAlone:r.log.find(l=>l.stage==='pso')||null,backToOriginal:same(r.atoms,atoms),newAddress:newRes.address.key,changedAddress:newRes.address.key!==res.address.key},
  history:ledger.recall(res.address.key).length};
}
