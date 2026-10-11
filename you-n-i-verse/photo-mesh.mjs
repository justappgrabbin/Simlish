// A landmark-derived visible face surface, not a reconstructed complete head/body.
export function faceGeometry(landmarks,triangles){
 if(landmarks.length<468)throw Error('468 face landmarks required');
 const positions=new Float32Array(468*3),uv=new Float32Array(468*2),style=new Float32Array(468*3),mouth=new Float32Array(468*3);
 const lowerLip=new Set([14,17,87,178,88,95,318,402,317,84,181,91,146,314,405,321,375]);
 for(let i=0;i<468;i++){
  const p=landmarks[i];if(![p.x,p.y,p.z].every(Number.isFinite))throw Error('Nonfinite landmark');
  const x=(p.x-.5)*2,y=(.5-p.y)*2,z=-p.z*2;
  positions.set([x,y,z],i*3);uv.set([p.x,p.y],i*2);
  style.set([x*.18,-y*.06,z*.12],i*3);
  if(lowerLip.has(i))mouth[i*3+1]=-.015;
 }
 const indices=Uint16Array.from(triangles);if(indices.length%3||indices.some(i=>i>=468))throw Error('Invalid triangle indices');
 const normals=new Float32Array(positions.length);
 for(let i=0;i<indices.length;i+=3){
  let a=indices[i]*3,b=indices[i+1]*3,c=indices[i+2]*3;
  const ax=positions[b]-positions[a],ay=positions[b+1]-positions[a+1],az=positions[b+2]-positions[a+2],bx=positions[c]-positions[a],by=positions[c+1]-positions[a+1],bz=positions[c+2]-positions[a+2];
  let nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx;
  if(nz<0){[indices[i+1],indices[i+2]]=[indices[i+2],indices[i+1]];nx=-nx;ny=-ny;nz=-nz;}
  for(const j of [a,b,c]){normals[j]+=nx;normals[j+1]+=ny;normals[j+2]+=nz;}
 }
 for(let i=0;i<normals.length;i+=3){const len=Math.hypot(normals[i],normals[i+1],normals[i+2])||1;normals[i]/=len;normals[i+1]/=len;normals[i+2]/=len;}
 return {positions,uv,indices,style,mouth,normals};
}
export function encodeFaceGLB(geometry,png,{styleWeight=0}={}){
 if(!Number.isFinite(styleWeight)||styleWeight<0||styleWeight>1)throw Error("Invalid style weight");
 const chunks=[],bufferViews=[],accessors=[];let offset=0;
 const bytes=value=>value instanceof Uint8Array?value:new Uint8Array(value.buffer,value.byteOffset,value.byteLength);
 function view(value){const b=bytes(value),padded=new Uint8Array(Math.ceil(b.length/4)*4);padded.set(b);const id=bufferViews.length;bufferViews.push({buffer:0,byteOffset:offset,byteLength:b.length});chunks.push(padded);offset+=padded.length;return id;}
 function accessor(value,type,componentType,bounds=false){const width={SCALAR:1,VEC2:2,VEC3:3}[type],a={bufferView:view(value),componentType,count:value.length/width,type};if(bounds){a.min=Array(width).fill(Infinity);a.max=Array(width).fill(-Infinity);for(let i=0;i<value.length;i++){a.min[i%width]=Math.min(a.min[i%width],value[i]);a.max[i%width]=Math.max(a.max[i%width],value[i]);}}accessors.push(a);return accessors.length-1;}
 const position=accessor(geometry.positions,'VEC3',5126,true),uv=accessor(geometry.uv,'VEC2',5126),indices=accessor(geometry.indices,'SCALAR',5123),style=accessor(geometry.style,'VEC3',5126,true),mouth=accessor(geometry.mouth,'VEC3',5126,true);
 const normal=accessor(geometry.normals,'VEC3',5126);
 const image=view(png),times=accessor(new Float32Array([0,1,2]),'SCALAR',5126,true),weights=accessor(new Float32Array([styleWeight,0,styleWeight,1,styleWeight,0]),'SCALAR',5126);
 const gltf={asset:{version:'2.0',generator:'YOU-N-I-VERSE landmark face surface',extras:{scope:'visible-face-surface',estimatedDepth:true,completeHead:false,completeBody:false}},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0,name:'Photo face surface'}],meshes:[{weights:[styleWeight,0],extras:{targetNames:['Stylize','MouthOpen']},primitives:[{attributes:{POSITION:position,NORMAL:normal,TEXCOORD_0:uv},indices,material:0,targets:[{POSITION:style},{POSITION:mouth}]}]}],materials:[{doubleSided:true,pbrMetallicRoughness:{baseColorTexture:{index:0},metallicFactor:0,roughnessFactor:1}}],textures:[{source:0}],images:[{bufferView:image,mimeType:'image/png'}],animations:[{name:'Mouth expression demonstration',samplers:[{input:times,output:weights,interpolation:'LINEAR'}],channels:[{sampler:0,target:{node:0,path:'weights'}}]}],buffers:[{byteLength:offset}],bufferViews,accessors};
 const json=new TextEncoder().encode(JSON.stringify(gltf)),jsonSize=Math.ceil(json.length/4)*4,total=12+8+jsonSize+8+offset,out=new Uint8Array(total),dv=new DataView(out.buffer);
 dv.setUint32(0,0x46546c67,true);dv.setUint32(4,2,true);dv.setUint32(8,total,true);dv.setUint32(12,jsonSize,true);dv.setUint32(16,0x4e4f534a,true);out.fill(32,20,20+jsonSize);out.set(json,20);
 const binaryStart=20+jsonSize;dv.setUint32(binaryStart,offset,true);dv.setUint32(binaryStart+4,0x004e4942,true);let cursor=binaryStart+8;for(const chunk of chunks){out.set(chunk,cursor);cursor+=chunk.length;}return out;
}
