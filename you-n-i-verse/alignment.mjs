export function faceCrop(box,width,height){
 const {x,y,width:w,height:h}=box;if(![x,y,w,h,width,height].every(Number.isFinite)||w<=0||h<=0||width<=0||height<=0)throw Error('Invalid face bounds');
 const left=Math.max(0,x-w*.08),top=Math.max(0,y-h*.04),right=Math.min(width,x+w*1.08),bottom=Math.min(height,y+h*1.04);if(right<=left||bottom<=top)throw Error('Face outside image');return {x:left,y:top,width:right-left,height:bottom-top};
}
export async function detectFace(image,Detector=globalThis.FaceDetector){if(!Detector)return {status:'unsupported'};try{const faces=await new Detector({fastMode:false,maxDetectedFaces:2}).detect(image);if(faces.length!==1)return {status:faces.length?'multiple':'none'};return {status:'detected',crop:faceCrop(faces[0].boundingBox,image.naturalWidth,image.naturalHeight)}}catch{return {status:'failed'}}}
