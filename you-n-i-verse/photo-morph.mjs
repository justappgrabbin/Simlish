export function affineTriangle(source,target) {
 const [a,b,c]=source,[p,q,r]=target;
 const det=(b.x-a.x)*(c.y-a.y)-(c.x-a.x)*(b.y-a.y);
 if(Math.abs(det)<1e-8)return null;
 const u=(c.y-a.y)/det,v=-(b.y-a.y)/det,w=-(c.x-a.x)/det,z=(b.x-a.x)/det;
 const A=(q.x-p.x)*u+(r.x-p.x)*v,B=(q.y-p.y)*u+(r.y-p.y)*v;
 const C=(q.x-p.x)*w+(r.x-p.x)*z,D=(q.y-p.y)*w+(r.y-p.y)*z;
 return [A,B,C,D,p.x-A*a.x-C*a.y,p.y-B*a.x-D*a.y];
}
const oval=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
let detectorPromise;
async function detector(){
 if(!detectorPromise)detectorPromise=(async()=>{
  const {FilesetResolver,FaceLandmarker}=await import('./vendor/vision/vision_bundle.mjs');
  const files=await FilesetResolver.forVisionTasks(new URL('./vendor/vision/wasm',import.meta.url).href);
  return FaceLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:new URL('./vendor/vision/face_landmarker.task',import.meta.url).href,delegate:'CPU'},runningMode:'IMAGE',numFaces:2,minFaceDetectionConfidence:.35,minFacePresenceConfidence:.35});
 })().catch(error=>{detectorPromise=null;throw error});
 return detectorPromise;
}
function surface(image){const c=document.createElement('canvas');const w=image.naturalWidth||image.width,h=image.naturalHeight||image.height;const scale=Math.min(1,1200/Math.max(w,h));c.width=Math.round(w*scale);c.height=Math.round(h*scale);c.getContext('2d').drawImage(image,0,0,c.width,c.height);return c;}
async function triangulator(){if(globalThis.Delaunator)return globalThis.Delaunator;await new Promise((resolve,reject)=>{const script=document.createElement('script');script.src=new URL('./vendor/vision/delaunator.min.js',import.meta.url).href;script.onload=resolve;script.onerror=()=>reject(Error('Could not load local face triangulation'));document.head.append(script)});return globalThis.Delaunator;}
function detectPortrait(model,image){
 let faces=model.detect(image).faceLandmarks;
 if(faces.length!==0)return faces;
 // The supplied full-body target has a small head. Inspect its upper region
 // at full resolution and translate the landmarks back to image coordinates.
 for(const fraction of [.45,.3]){
  const region=document.createElement('canvas');region.width=image.width*3;region.height=Math.round(image.height*fraction)*3;
  region.getContext('2d').drawImage(image,0,0,image.width*3,image.height*3);
  faces=model.detect(region).faceLandmarks;
  if(faces.length)return faces.map(face=>face.map(p=>({...p,y:p.y*fraction})));
 }
 return faces;
}
export async function morphPhoto(photo,target){
 const source=surface(photo),body=surface(target),model=await detector();
 const sourceFaces=detectPortrait(model,source),targetFaces=detectPortrait(model,body);
 if(sourceFaces.length!==1)throw Error(`Found ${sourceFaces.length} faces in your photo. Use a photo with one visible face.`);
 if(targetFaces.length!==1)throw Error(`Found ${targetFaces.length} faces in the target. Choose a target image with one visible face.`);
 const points=(landmarks,c)=>landmarks.slice(0,468).map(p=>({x:p.x*c.width,y:p.y*c.height}));
 const s=points(sourceFaces[0],source),t=points(targetFaces[0],body),Delaunator=await triangulator();
 const triangles=Delaunator.from(t,p=>p.x,p=>p.y).triangles;
 const warped=document.createElement('canvas');warped.width=body.width;warped.height=body.height;const ctx=warped.getContext('2d');
 for(let i=0;i<triangles.length;i+=3){const ids=[triangles[i],triangles[i+1],triangles[i+2]],dest=ids.map(j=>t[j]),matrix=affineTriangle(ids.map(j=>s[j]),dest);if(!matrix)continue;
  const cx=dest.reduce((v,p)=>v+p.x,0)/3,cy=dest.reduce((v,p)=>v+p.y,0)/3;
  ctx.save();ctx.beginPath();dest.forEach((p,j)=>{const dx=p.x-cx,dy=p.y-cy,len=Math.hypot(dx,dy)||1;const x=p.x+dx/len*.6,y=p.y+dy/len*.6;j?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.closePath();ctx.clip();ctx.setTransform(...matrix);ctx.drawImage(source,0,0);ctx.restore();
 }
 const mask=document.createElement('canvas');mask.width=body.width;mask.height=body.height;const mc=mask.getContext('2d');
 const center={x:oval.reduce((a,j)=>a+t[j].x,0)/oval.length,y:oval.reduce((a,j)=>a+t[j].y,0)/oval.length};
 mc.fillStyle='white';mc.beginPath();oval.forEach((j,i)=>{const x=center.x+(t[j].x-center.x)*.96,y=center.y+(t[j].y-center.y)*.96;i?mc.lineTo(x,y):mc.moveTo(x,y)});mc.closePath();mc.fill();
 const pixels=ctx.getImageData(0,0,body.width,body.height),base=body.getContext('2d').getImageData(0,0,body.width,body.height),m=mc.getImageData(0,0,body.width,body.height);
 const sums=[0,0,0],targetSums=[0,0,0];let count=0;
 for(let i=0;i<pixels.data.length;i+=4)if(m.data[i+3]>200&&pixels.data[i+3]>200){for(let c=0;c<3;c++){sums[c]+=pixels.data[i+c];targetSums[c]+=base.data[i+c]}count++;}
 if(!count)throw Error('No usable face surface found.');
 const correction=sums.map((sum,c)=>(targetSums[c]-sum)/count*.6);
 for(let i=0;i<pixels.data.length;i+=4)for(let c=0;c<3;c++)pixels.data[i+c]=Math.max(0,Math.min(255,pixels.data[i+c]+correction[c]));ctx.putImageData(pixels,0,0);
 const feather=document.createElement('canvas');feather.width=body.width;feather.height=body.height;const fc=feather.getContext('2d');fc.filter=`blur(${Math.max(2,Math.sqrt(count)*.035)}px)`;fc.drawImage(mask,0,0);
 ctx.globalCompositeOperation='destination-in';ctx.drawImage(feather,0,0);
 const result=document.createElement('canvas');result.width=body.width;result.height=body.height;result.getContext('2d').drawImage(body,0,0);result.getContext('2d').drawImage(warped,0,0);
 return {canvas:result,receipt:{method:'local-neural-landmarks/triangle-warp/color-match/feather',landmarks:468,triangles:triangles.length/3,sourceFaces:1,targetFaces:1,network:false,trainedHere:false}};
}
