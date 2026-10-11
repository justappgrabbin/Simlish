import {affineTriangle} from './photo-morph.mjs';
export function matchGrid(a,b,width,height,{columns=8,rows=12,radius=12}={}){
 const gray=(data,x,y)=>{const i=(y*width+x)*4;return (data[i]+data[i+1]+data[i+2])/3};
 const source=[],target=[],triangles=[];
 for(let y=0;y<=rows;y++)for(let x=0;x<=columns;x++){
  const px=Math.round(x*(width-1)/columns),py=Math.round(y*(height-1)/rows);let best=Infinity,dx=0,dy=0;
  if(x&&x<columns&&y&&y<rows)for(let v=-radius;v<=radius;v+=2)for(let u=-radius;u<=radius;u+=2){
   if(px+u<2||px+u>=width-2||py+v<2||py+v>=height-2)continue;
   let score=(u*u+v*v)*.4;
   for(let sy=-2;sy<=2;sy++)for(let sx=-2;sx<=2;sx++){const d=gray(a,px+sx,py+sy)-gray(b,px+u+sx,py+v+sy);score+=d*d}
   if(score<best){best=score;dx=u;dy=v}
  }
  source.push({x:px,y:py});target.push({x:px+dx,y:py+dy});
 }
 for(let y=0;y<rows;y++)for(let x=0;x<columns;x++){const i=y*(columns+1)+x;triangles.push([i,i+1,i+columns+1],[i+1,i+columns+2,i+columns+1])}
 // Reject folded cells rather than drawing inverted surfaces.
 const area=([a,b,c])=>(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
 let safe=false;for(let attempt=0;attempt<7;attempt++){safe=triangles.every(ids=>area(ids.map(i=>source[i]))*area(ids.map(i=>target[i]))>0);if(safe)break;target.forEach((p,i)=>{p.x=(p.x+source[i].x)/2;p.y=(p.y+source[i].y)/2})}
 return {source,target:safe?target:source.map(p=>({...p})),triangles,fallback:!safe};
}
export function intermediatePoints(mesh,t){return mesh.source.map((p,i)=>({x:p.x+(mesh.target[i].x-p.x)*t,y:p.y+(mesh.target[i].y-p.y)*t}))}
function warp(image,points,dest,triangles){
 const out=document.createElement('canvas');out.width=image.width;out.height=image.height;const c=out.getContext('2d');
 for(const ids of triangles){const d=ids.map(i=>dest[i]),matrix=affineTriangle(ids.map(i=>points[i]),d);if(!matrix)continue;c.save();c.beginPath();d.forEach((p,i)=>i?c.lineTo(p.x,p.y):c.moveTo(p.x,p.y));c.closePath();c.clip();c.setTransform(...matrix);c.drawImage(image,0,0);c.restore()}
 return out;
}
export async function expandMotion(sheet,grid,{steps=6,cancelled=()=>false,onProgress=()=>{}}={}){
 const count=grid.count*steps,cols=Math.min(8,count),rows=Math.ceil(count/cols),w=grid.frameWidth,h=grid.frameHeight;
 if(cols*rows*w*h>16000000)throw Error('Interpolated sheet exceeds the memory limit. Use smaller frames.');
 const out=document.createElement('canvas');out.width=cols*w;out.height=rows*h;const ctx=out.getContext('2d');let fallbacks=0;
 const cell=i=>{const c=document.createElement('canvas');c.width=w;c.height=h;c.getContext('2d').drawImage(sheet,i%grid.cols*w,Math.floor(i/grid.cols)*h,w,h,0,0,w,h);return c};
 for(let i=0;i<grid.count;i++){
  if(cancelled())throw Error('Animation preparation cancelled.');
  const a=cell(i),b=cell((i+1)%grid.count),mesh=matchGrid(a.getContext('2d').getImageData(0,0,w,h).data,b.getContext('2d').getImageData(0,0,w,h).data,w,h);if(mesh.fallback)fallbacks++;
  for(let j=0;j<steps;j++){const n=i*steps+j,x=n%cols*w,y=Math.floor(n/cols)*h,t=j/steps;
   if(j===0)ctx.drawImage(a,x,y);else{const dest=intermediatePoints(mesh,t),wa=warp(a,mesh.source,dest,mesh.triangles),wb=warp(b,mesh.target,dest,mesh.triangles);ctx.globalAlpha=1;ctx.drawImage(wa,x,y);ctx.globalAlpha=t;ctx.drawImage(wb,x,y);ctx.globalAlpha=1;}
  }
  onProgress(i+1,grid.count);await new Promise(resolve=>setTimeout(resolve,0));
 }
 if(cancelled())throw Error('Animation preparation cancelled.');
 return {canvas:out,grid:{cols,rows,frameWidth:w,frameHeight:h,count},method:'block-matched-triangle-warp',fallbacks,keyframes:grid.count,steps};
}
