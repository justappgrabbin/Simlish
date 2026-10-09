import { SwarmWorld } from './world.mjs';
const world = new SwarmWorld(), canvas = document.querySelector('canvas'), ctx = canvas.getContext('2d');
let definitions = false;
for(let i=0;i<24;i++) world.addParticle({id:'p'+i,x:3+(i%6)*.6,y:3+Math.floor(i/6)*.6,
 bit:i%2,qualities:{role:'constituent'},address:null});
for(let i=0;i<24;i++) {
 if(i%6<5) world.connect('p'+i,'p'+(i+1));
 if(i<18) world.connect('p'+i,'p'+(i+6));
}
world.compose('body',[...world.particles.keys()]);
window.swarmWorld=world;
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
function frame() {
 world.step();world.step();ctx.clearRect(0,0,680,680);
 for(const e of world.edges.filter(e=>e.active)){
  const a=world.particles.get(e.source),b=world.particles.get(e.target);
  ctx.strokeStyle='#719ba6';ctx.beginPath();ctx.moveTo(a.x*68,a.y*68);ctx.lineTo(b.x*68,b.y*68);ctx.stroke();
 }
 for(const p of world.particles.values()){
  ctx.fillStyle=p.bit?'#ffa66d':'#83d9ef';ctx.beginPath();ctx.arc(p.x*68,p.y*68,7,0,Math.PI*2);ctx.fill();
  if(definitions){ctx.fillStyle='white';ctx.fillText(p.id+':'+p.bit,p.x*68+8,p.y*68);}
 }
 document.querySelector('#status').textContent=world.edges.filter(e=>e.active).length+' active bonds · '+world.history.length+' recorded events';
 requestAnimationFrame(frame);
}
frame();
