// Authoring tool, not an interpreter: turns a named netlist into the six-bit stream that population.mjs decodes. It adds no behaviour.
// A cell is {n:name, d:'Movement'|'Evolution'|'Being'|'Design', h:gate-name-or-number, a,b,(sel)|en} or {n:name, call:'defName', args:[...], outs:[names]}.
// A reference may name a signal that appears LATER: it then reads the previous tick (feedback/memory).
import {GATE_TABLE,WRITABLE} from './population.mjs';
export function assemble(defs){
 const index=new Map(defs.map((d,i)=>[d.name,i])),atoms=[];
 if(defs.length>8)throw new RangeError('at most 8 definitions');
 atoms.push(defs.length-1);
 defs.forEach((def,di)=>{
  const sig=new Map();let slot=0;for(const i of def.inputs||[])sig.set(i,slot++);
  const layout=[];for(const c of def.cells){layout.push({c,slot});if(c.call!==undefined){const callee=defs[index.get(c.call)];if(index.get(c.call)>=di)throw new RangeError('definitions may only instantiate earlier ones');(c.outs||[]).forEach((o,k)=>sig.set(c.n+'.'+o,slot+k));slot+=callee.outputs.length}else{sig.set(c.n,slot);slot++}}
  const ref=n=>{if(!sig.has(n))throw new RangeError('unknown signal '+n+' in '+def.name);return sig.get(n)};
  atoms.push((def.inputs||[]).length,def.cells.length,def.outputs.length);
  for(const {c} of layout){
   if(c.call!==undefined){const k=index.get(c.call);atoms.push(WRITABLE.indexOf('Design'),32|k,...c.args.map(ref));continue}
   atoms.push(WRITABLE.indexOf(c.d));const H=typeof c.h==='number'?c.h:GATE_TABLE[c.h];if(H===undefined)throw new RangeError('unknown gate '+c.h);atoms.push(H&63);
   if(c.d==='Design')atoms.push(ref(c.a),ref(c.b),ref(c.sel));else atoms.push(ref(c.a),ref(c.b??c.en));
  }
  for(const o of def.outputs)atoms.push(ref(o));
 });
 return atoms;
}
