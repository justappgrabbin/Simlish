// Regularized radial scaling response adapted from svMorph's expansion kernel.
// See vendor/svmorph/LICENSE and MORPH-SOURCES.md for attribution.
export function radialScale(points,center,radius,strength=0){
 if(!Number.isFinite(radius)||radius<=0||!Number.isFinite(strength))throw Error('Invalid sculpt parameters');
 const epsilon=.6;
 return points.map(p=>{
  const x=(p.x-center.x)/radius,y=(p.y-center.y)/radius,r2=x*x+y*y+epsilon*epsilon;
  const response=(1/Math.pow(r2,1.5)+3*epsilon*epsilon/(2*Math.pow(r2,2.5)));
  const scale=1+strength*.035*response;
  return {...p,x:center.x+x*radius*scale,y:center.y+y*radius*scale};
 });
}
export function morphGeometry(source,target,oval,{shape=1,sculpt=0}={}){
 if(!Number.isFinite(shape)||shape<0||shape>1||!Number.isFinite(sculpt)||Math.abs(sculpt)>1)throw Error('Morph controls out of range');
 const bounds=points=>{const selected=oval.map(i=>points[i]);return {x:Math.min(...selected.map(p=>p.x)),y:Math.min(...selected.map(p=>p.y)),w:Math.max(...selected.map(p=>p.x))-Math.min(...selected.map(p=>p.x)),h:Math.max(...selected.map(p=>p.y))-Math.min(...selected.map(p=>p.y))};};
 const s=bounds(source),t=bounds(target);if(s.w<=0||s.h<=0||t.w<=0||t.h<=0)throw Error('Degenerate face geometry');
 const points=target.map((p,i)=>{
  const fitted={x:t.x+(source[i].x-s.x)/s.w*t.w,y:t.y+(source[i].y-s.y)/s.h*t.h};
  return {x:fitted.x+(p.x-fitted.x)*shape,y:fitted.y+(p.y-fitted.y)*shape};
 });
 return radialScale(points,{x:t.x+t.w/2,y:t.y+t.h/2},Math.max(t.w,t.h)/2,sculpt);
}
