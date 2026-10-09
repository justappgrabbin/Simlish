export const clamp=(n,min,max)=>Math.max(min,Math.min(max,n));
export function inverseRotate(x,y,degrees){const a=degrees*Math.PI/180;return {x:x*Math.cos(a)+y*Math.sin(a),y:-x*Math.sin(a)+y*Math.cos(a)}}
export function photoGesture(state,{dx=0,dy=0,ratio=1,angle=0,focal={x:0,y:0}},faceWidth){
 if(!Number.isFinite(faceWidth)||faceWidth<=0)throw Error('Invalid face size');
 const zoom=clamp(state.zoom*ratio,.3,4),actualRatio=zoom/state.zoom;
 const f=inverseRotate(focal.x/faceWidth,focal.y/faceWidth,state.rotation),d=inverseRotate(dx/faceWidth,dy/faceWidth,state.rotation);
 return {...state,zoom,rotation:clamp(state.rotation+angle,-30,30),pan:{x:actualRatio*state.pan.x+(1-actualRatio)*f.x+d.x,y:actualRatio*state.pan.y+(1-actualRatio)*f.y+d.y}};
}
export function circleGesture(state,{dx=0,dy=0,ratio=1},width=380,height=870){return {...state,faceX:clamp(state.faceX+dx/width,0,1),faceY:clamp(state.faceY+dy/height,0,1),faceSize:clamp(state.faceSize*ratio,.08,.5)}}
export function pairMetrics(points){const [a,b]=points;return {x:(a.x+b.x)/2,y:(a.y+b.y)/2,distance:Math.max(1,Math.hypot(b.x-a.x,b.y-a.y)),angle:Math.atan2(b.y-a.y,b.x-a.x)*180/Math.PI}}
