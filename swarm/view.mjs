import { SwarmWorld } from './world.mjs';
import { letterState, resolveExpressionColor } from './letter-expression.mjs';
const canvas = document.querySelector('canvas'), ctx = canvas.getContext('2d');
let world, definitions = false;
function assemble(text) {
 const words = text.trim().split(/\s+/u).filter(Boolean);
 if (!words.length || [...text].length > 72) throw new Error('Enter a phrase of 1–72 characters.');
 world = new SwarmWorld(); world.requestFlavor(document.querySelector('#flavor').value);
 let offset = 0;
 const wordIds = [];
 for (const [wordIndex, word] of words.entries()) {
  const members = [];
  for (const glyph of [...word]) {
   const id = 'p'+offset, x = 1+(offset%12)*.65, y = 1.5+Math.floor(offset/12)*1;
   world.addParticle({id,x,y,bit:offset%2,qualities:{role:'constituent',glyph},address:null});
   if (members.length) world.connect(members.at(-1),id,{context:'word:'+wordIndex});
   members.push(id); offset++;
  }
  const id = 'word:'+wordIndex;world.compose(id,members,'o_sequence');wordIds.push(id);
 }
 world.compose('phrase',wordIds,'o_sequence');
 window.swarmWorld = world;
 // Host integration supplies the actual address-derived color and provenance.
 window.resolveSwarmColor = (expression,resolution) => resolveExpressionColor(world,expression,resolution);
 document.querySelector('#details').textContent='Color awaits a resolved address. Outlined letters have strained or broken relations.';
}
document.querySelector('#assemble').onsubmit=e=>{
 e.preventDefault();try{assemble(document.querySelector('#phrase').value);}catch(error){document.querySelector('#details').textContent=error.message;}
};
document.querySelector('#flavor').onchange=e=>world.requestFlavor(e.target.value);
document.querySelector('#definition').onclick=()=>{definitions=!definitions;};
document.querySelector('#export').onclick=()=>{
 const url=URL.createObjectURL(new Blob([JSON.stringify(world.snapshot(),null,2)],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='simlish-world.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
canvas.onclick=e=>{
 const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*10,y=(e.clientY-r.top)/r.height*10;
 const nearest=[...world.particles.values()].sort((a,b)=>Math.hypot(a.x-x,a.y-y)-Math.hypot(b.x-x,b.y-y))[0];
 world.impulse(nearest.id,(x-nearest.x)*4,-8);
};
assemble(document.querySelector('#phrase').value);
function frame() {
 world.step();world.step();ctx.clearRect(0,0,680,680);
 for(const edge of world.edges){
  const a=world.particles.get(edge.source),b=world.particles.get(edge.target);
  ctx.strokeStyle=edge.active?'#719ba6':'#e3e9ed';ctx.setLineDash(edge.active?[]:[4,6]);
  ctx.beginPath();ctx.moveTo(a.x*68,a.y*68);ctx.lineTo(b.x*68,b.y*68);ctx.stroke();
 }
 ctx.setLineDash([]);ctx.textAlign='center';ctx.textBaseline='middle';
 let detached=0;
 for(const p of world.particles.values()){
  const expression=letterState(world,p.id);if(expression.state==='separated')detached++;
  ctx.font='bold 22px system-ui';ctx.fillStyle=p.expressionColor?.color ?? '#edf5f7';
  ctx.fillText(expression.glyph,p.x*68,p.y*68);
  if(expression.state!=='connected'){
   ctx.strokeStyle='#ffffff';ctx.setLineDash(expression.state==='separated'?[3,3]:[]);
   ctx.strokeRect(p.x*68-14,p.y*68-17,28,34);ctx.setLineDash([]);
  }
  if(definitions){ctx.font='10px system-ui';ctx.fillText(p.id+' · '+expression.state,p.x*68,p.y*68+25);}
 }
 document.querySelector('#status').textContent=world.composites.size+' compositions · '+detached+' letters with broken relations · '+world.history.length+' recorded events';
 requestAnimationFrame(frame);
}
frame();
