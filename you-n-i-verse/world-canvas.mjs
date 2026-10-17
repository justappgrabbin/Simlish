// Draw only actual engine snapshots. No decorative "morph completed" animation.
const label = object => object.name + (object.morphState ? ' · form '+object.morphState : '');
const size = value => Math.max(.12,Math.min(3,Number(value)||.12));
const lerp = (a,b,t) => a+(b-a)*t;

export function drawWorld(canvas,state,{selected='chair'}={}) {
  if(!canvas||!state?.objects)throw new Error('A world canvas and real world snapshot are required');
  const ctx=canvas.getContext('2d');
  if(!ctx)throw new Error('Canvas rendering unavailable');
  const w=720,h=390;
  canvas.width=w;canvas.height=h;
  ctx.clearRect(0,0,w,h);
  const bg=ctx.createLinearGradient(0,0,w,h);
  bg.addColorStop(0,'#090f22');bg.addColorStop(1,'#241436');
  ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);
  ctx.strokeStyle='#f3bdff15';ctx.lineWidth=1;
  for(let a=0;a<=10;a++) {
    const x=38+a*64, y=32+a*32;
    ctx.beginPath();ctx.moveTo(x,26);ctx.lineTo(x,h-38);ctx.stroke();
    ctx.beginPath();ctx.moveTo(35,y);ctx.lineTo(w-35,y);ctx.stroke();
  }
  const point=object=>{
    const x=object.particle?.x ?? object.x+5;
    const z=object.particle?.y ?? object.z+5;
    return {x:40+Math.max(0,Math.min(10,x))*64,y:35+Math.max(0,Math.min(10,z))*30};
  };
  const visible=state.objects.filter(o=>!o.removed);
  const lookup=new Map(visible.map(o=>[o.id,o]));
  for(const c of state.composites||[]) {
    const members=c.members.map(id=>lookup.get(id)).filter(Boolean);
    if(members.length<2)continue;
    ctx.beginPath();members.forEach((o,i)=>{const p=point(o);if(i===0)ctx.moveTo(p.x,p.y);else ctx.lineTo(p.x,p.y)});
    ctx.strokeStyle='#ffd488';ctx.lineWidth=3;ctx.stroke();
  }
  for(const o of visible) {
    const p=point(o),scale=17*size(o.sx)*(1+o.morphState*.12), tall=28*size(o.sy)*(1+o.morphState*.18),depth=12*size(o.sz);
    const isSelected=o.id===selected;
    ctx.save();ctx.translate(p.x,p.y);
    ctx.shadowColor=isSelected?'#ffdca5':'#000000';ctx.shadowBlur=isSelected?18:7;
    ctx.lineWidth=isSelected?3:1.4;ctx.strokeStyle=isSelected?'#ffe1ac':'#f8dcff88';
    switch(o.type) {
      case 'chair':
        ctx.fillStyle='#c284da';ctx.fillRect(-scale/2,-tall*.42,scale,tall*.35);
        ctx.fillStyle='#855fa3';ctx.fillRect(-scale/2,-tall,scale*.18,tall*.58);
        ctx.fillRect(-scale*.41,-tall*.08,scale*.12,tall*.6);
        ctx.fillRect(scale*.29,-tall*.08,scale*.12,tall*.6);
        break;
      case 'table':
        ctx.fillStyle='#cb8dba';ctx.fillRect(-scale,-tall*.5,scale*2,Math.max(6,depth*.35));
        ctx.fillStyle='#936f99';ctx.fillRect(-scale*.78,-tall*.38,6,tall*.75);
        ctx.fillRect(scale*.7,-tall*.38,6,tall*.75);
        break;
      case 'plant':
        ctx.fillStyle='#83b985';ctx.fillRect(-3,-tall,6,tall);
        ctx.beginPath();ctx.ellipse(-8,-tall*.65,13,7,-.5,0,Math.PI*2);ctx.fill();
        ctx.beginPath();ctx.ellipse(10,-tall*.85,14,7,.5,0,Math.PI*2);ctx.fill();
        ctx.fillStyle='#985c73';ctx.fillRect(-10,0,20,13);
        break;
      case 'door':
        ctx.fillStyle='#a16dc9';ctx.fillRect(-scale/2,-tall,scale,tall);
        ctx.strokeRect(-scale/2,-tall,scale,tall);
        ctx.fillStyle='#ffe4ac';ctx.fillRect(scale*.2,-tall*.5,4,4);
        break;
      case 'food':
        ctx.fillStyle='#ee9fb2';ctx.beginPath();ctx.ellipse(0,-8,scale,depth,0,0,Math.PI*2);ctx.fill();
        ctx.stroke();
        break;
      case 'beacon':
        ctx.fillStyle='#ffc875';ctx.beginPath();ctx.moveTo(0,-tall);ctx.lineTo(scale,-tall*.28);
        ctx.lineTo(0,0);ctx.lineTo(-scale,-tall*.28);ctx.closePath();ctx.fill();ctx.stroke();
        break;
      default:
        ctx.fillStyle='#ae9ed9';ctx.fillRect(-scale/2,-tall,scale,tall);
    }
    ctx.shadowBlur=0;
    ctx.textAlign='center';ctx.font='12px system-ui,sans-serif';ctx.fillStyle='#fff7fc';
    ctx.fillText(label(o),0,26);
    ctx.restore();
  }
  ctx.fillStyle='#d6b9eb';ctx.font='13px system-ui,sans-serif';ctx.textAlign='left';
  ctx.fillText('SIMLISH · '+visible.length+' objects · '+(state.composites?.length||0)+' compositions',16,20);
  ctx.fillText('World events '+state.events+' · swarm ticks '+state.physicsTick+' · ATO revisions '+state.formationRevisions,16,h-11);
  return {visibleObjects:visible.length,formationRevisions:state.formationRevisions};
}
