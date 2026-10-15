// Reality execution readiness: inspectable capacity and bounded swarm work orders.
// A large state space is not a substitute for geometry, assets or rendering.
export const STATE_SPACE=Object.freeze({dimensions:5,centersPerDimension:9,gatesPerCenter:64,lines:6,colors:6,tones:6,bases:5,emergentCenterStates:81});
export function stateCapacity(){
 const primary=STATE_SPACE.dimensions*STATE_SPACE.centersPerDimension*STATE_SPACE.gatesPerCenter;
 const configurations=BigInt(primary)*6n*6n*6n*5n;
 return {primary,configurations:configurations.toString(),emergentCenterStates:81,
   note:'Combinatorial capacity is not a claim of trained coverage or visual quality'};
}
export function inspectRealityOrder(order){
 const missing=[];
 if(!order?.semanticAddress?.leadingDimension||!order?.semanticAddress?.evidence?.source)missing.push('grounded semantic address');
 if(!order?.scene?.objects?.length)missing.push('scene objects');
 if(!order?.assets?.some(a=>a?.geometry&&a?.material))missing.push('geometry and material assets');
 if(!order?.camera)missing.push('camera');
 if(!order?.lighting)missing.push('lighting');
 if(!order?.renderer?.render)missing.push('render implementation');
 if(!order?.quality?.referenceImages?.length)missing.push('reference images for fidelity evaluation');
 return {ready:missing.length===0,missing,capacity:stateCapacity()};
}
export async function executeRealityOrder(order,{approve=()=>false}={}){
 const inspection=inspectRealityOrder(order);
 if(!inspection.ready)return {status:'blocked',inspection};
 if(!approve(order))return {status:'awaiting-approval',inspection};
 const receipt=await order.renderer.render({scene:order.scene,assets:order.assets,camera:order.camera,lighting:order.lighting});
 if(!receipt||!receipt.output)throw new Error('Renderer returned no output');
 return {status:'rendered-unverified',receipt,inspection,quality:'not certified without image comparison'};
}
