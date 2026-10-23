/** Visualize observed/supplied relationships; appearance is a projection, not a gate address. */
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
export function cloudScene(swarm,{width=720,height=480}={}){
  if(!Number.isFinite(width)||!Number.isFinite(height)||width<100||height<100)throw new RangeError('valid cloud viewport required');
  const positions=swarm.positions(),ids=new Set(positions.map(p=>p.id));
  const nodes=positions.map(p=>({id:p.id,label:String(swarm.nodes.get(p.id)?.value??p.id),x:Math.round(width/2+p.x*(width*.4)),y:Math.round(height/2-p.y*(height*.4))}));
  const byId=new Map(nodes.map(n=>[n.id,n]));
  const edges=[...swarm.edges.values()].filter(e=>ids.has(e.source)&&ids.has(e.target)&&e.relation!=='constituent-of').map(e=>({id:e.id,from:e.source,to:e.target,predicate:e.relation,provenance:e.provenance??null}));
  return {schema:'synthia.semantic-cloud.v1',width,height,nodes,edges,identityPolicy:'positions are visual placement; semantic edges only from supplied provenance',byId};
}
export function cloudSVG(scene,{accent='#b879ff',connection='#20dacb'}={}){
  const check=s=>/^#[a-f\d]{6}$/i.test(s);if(!check(accent)||!check(connection))throw new Error('invalid palette');
  const lines=scene.edges.map(e=>{const a=scene.byId.get(e.from),b=scene.byId.get(e.to);return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${connection}" stroke-opacity=".55" stroke-width="2"/>`;}).join('');
  const items=scene.nodes.map(n=>`<g><circle cx="${n.x}" cy="${n.y}" r="14" fill="${accent}" stroke="#fcde93" stroke-width="1.5"/><text x="${n.x}" y="${n.y+29}" fill="#eee7ff" text-anchor="middle" font-size="12">${esc(n.label.slice(0,36))}</text></g>`).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${scene.width}" height="${scene.height}" viewBox="0 0 ${scene.width} ${scene.height}" role="img" aria-label="Provenance-grounded semantic cloud"><rect width="100%" height="100%" fill="#130d22"/>${lines}${items}</svg>`;
}
