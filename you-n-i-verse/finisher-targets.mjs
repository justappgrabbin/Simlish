// Target adapters: how a finished app is translated. 'reality' is implemented (scene events + reality readiness).
// 'game' and '2d' are NOT YET IMPLEMENTED stubs so they can be added later without changing the finisher.
import {interpretScene} from './scene-events.mjs';
import {inspectRealityOrder,executeRealityOrder} from './reality-execution.mjs';
export const SCENE_WORLDS=Object.freeze(['cosmic','forest','ocean','desert']);
export const realityTarget={id:'reality',implemented:true,
 async translate(app){
  const theme=String(app.personalization?.worldTheme||'').toLowerCase(),world=SCENE_WORLDS.includes(theme)?theme:null;
  let scene;
  if(!world)scene={status:'investigating',reason:theme?'World theme "'+app.personalization.worldTheme+'" is not one of the four scene worlds (cosmic, forest, ocean, desert); other worlds are not mapped yet':'No world theme set yet'};
  else{const r=await interpretScene('you enters '+world,{addressContext:app.address?{leadingDimension:app.address.dimension,coordinates:app.address,evidence:{source:app.source}}:null,kernel:app.kernel});
   scene=r.ok?{status:'mapped',event:r.event}:{status:'investigating',reason:r.reason,needs:r.needs}}
  const order={semanticAddress:scene.status==='mapped'?{leadingDimension:app.address.dimension,evidence:{source:app.source}}:null,
   scene:{objects:app.surfaces.map(p=>({id:p,kind:'app-surface'}))},assets:[],camera:null,lighting:null,renderer:null,quality:{referenceImages:[]}};
  const readiness=inspectRealityOrder(order),execution=await executeRealityOrder(order,{approve:()=>false});
  return {target:'reality',scene,readiness:{ready:readiness.ready,missing:readiness.missing},execution:{status:execution.status},
   note:'Mapped into scene events and a reality order; nothing is rendered until geometry, material, camera, lighting, renderer and reference images exist and a person approves.'};
 }};
const notYet=id=>({id,implemented:false,async translate(){return {target:id,status:'not-implemented',note:'Planned extension target; no translation exists yet.'}}});
export const gameTarget=notYet('game'),twoDTarget=notYet('2d');
export const TARGET_ADAPTERS=Object.freeze({reality:realityTarget,game:gameTarget,'2d':twoDTarget});
