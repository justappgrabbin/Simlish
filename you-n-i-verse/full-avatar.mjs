import {addProportionMorphs} from './body-proportions.mjs';
import * as THREE from './vendor/three/three.module.js';
import {GLTFLoader} from './vendor/three/GLTFLoader.js';
import {reconstructFaceSurface} from './photo-morph.mjs';
// MHR's imported rig is Y-up, in metres. The face photo supplies visible surface
// geometry only; body, skull, and unseen anatomy remain inferred template geometry.
export async function buildFullAvatar(photo,identity,plan){
 if(!photo)throw Error('Choose your photo first.');
 const loader=new GLTFLoader();
 const [body,faceResult]=await Promise.all([loader.loadAsync('./vendor/mhr/body-lod4.glb'),reconstructFaceSurface(photo)]);
 const face=await loader.parseAsync(faceResult.bytes.buffer,'');
 const model=new THREE.Group();model.name='Persistent_photo_character';model.add(body.scene);model.updateMatrixWorld(true);
 let head;model.traverse(node=>{if(node.isBone&&node.name==='c_head')head=node;});
 if(!head)throw Error('Detailed body is missing its head attachment.');
 let surface;face.scene.traverse(node=>{if(node.isMesh)surface=node;});
 const uv=surface.geometry.attributes.uv,sample=document.createElement('canvas');sample.width=sample.height=1;const ctx=sample.getContext('2d');ctx.drawImage(photo,uv.getX(4)*(photo.naturalWidth||photo.width),uv.getY(4)*(photo.naturalHeight||photo.height),1,1,0,0,1,1);const rgba=ctx.getImageData(0,0,1,1).data,skin=new THREE.Color('rgb('+rgba[0]+','+rgba[1]+','+rgba[2]+')');
 const headPosition=head.getWorldPosition(new THREE.Vector3());
 const faceBox=new THREE.Box3().setFromObject(face.scene),size=faceBox.getSize(new THREE.Vector3()),center=faceBox.getCenter(new THREE.Vector3());
 const scale=.175/size.y;
 face.scene.scale.setScalar(scale);
 const anchor=new THREE.Vector3(headPosition.x,headPosition.y+.065,headPosition.z+.112);
 face.scene.position.copy(anchor).addScaledVector(center,-scale);
 model.add(face.scene);model.updateMatrixWorld(true);head.attach(face.scene);
 // Remove only the template's front face so its neutral features cannot show
 // through the photo. Keep the rear skull, ears and neck on the detailed rig.
 model.traverse(node=>{
  if(!node.isSkinnedMesh)return;
  node.skeleton.update();const geometry=node.geometry,index=geometry.index,position=geometry.attributes.position;
  const kept=[],point=new THREE.Vector3();
  for(let i=0;i<index.count;i+=3){
   const ids=[index.getX(i),index.getX(i+1),index.getX(i+2)];let front=0;
   for(const id of ids){node.getVertexPosition(id,point).applyMatrix4(node.matrixWorld);if(point.y>headPosition.y-.01&&point.z>headPosition.z+.075)front++;}
   if(front<3)kept.push(...ids);
  }
  geometry.setIndex(kept);geometry.computeBoundingBox();addProportionMorphs(node);
  const slots={top:[],bottom:[],shoes:[]};
  const colors=new Float32Array(position.count*3),shirt=new THREE.Color('#344f78'),pants=new THREE.Color('#202c45'),shoe=new THREE.Color('#171d29');
  for(let id=0;id<position.count;id++){node.getVertexPosition(id,point).applyMatrix4(node.matrixWorld);const color=point.y<.11?shoe:point.y<.87?pants:point.y<1.44&&(Math.abs(point.x)<.3||point.y>1.27)?shirt:skin;if(color===shirt)slots.top.push(id);else if(color===pants)slots.bottom.push(id);else if(color===shoe)slots.shoes.push(id);color.toArray(colors,id*3);}
  node.userData.appearanceSlots=slots;
  geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));const materials=Array.isArray(node.material)?node.material:[node.material];for(const material of materials){material.color.set(0xffffff);material.vertexColors=true;material.needsUpdate=true;}
 });
 const tracks=[...face.animations[0].tracks];
 const bind=(name,amount)=>{let bone;model.traverse(n=>{if(n.isBone&&n.name===name)bone=n;});if(!bone)return;const rest=bone.quaternion.clone(),axis=new THREE.Vector3(0,1,0),values=[];
  for(const angle of [0,amount,0,-amount,0])values.push(...rest.clone().multiply(new THREE.Quaternion().setFromAxisAngle(axis,angle)).toArray());
  tracks.push(new THREE.QuaternionKeyframeTrack(bone.uuid+'.quaternion',[0,.5,1,1.5,2],values));
 };
 bind('c_head',.06);bind('l_uparm',.1);bind('r_uparm',-.1);bind('l_upleg',.1);bind('r_upleg',-.1);
 model.traverse(node=>{if(node.isBone)node.userData.proportionBaseScale=node.scale.toArray();});
 model.userData={heightBaseScale:1,bodyProportions:plan?.proportions,requestedForm:plan?.description??'',formPlan:plan,sourceIdentity:identity?.id??null,source:'MHR v1.0.1 LOD4 + local photo landmarks',bodyProvenance:'Inferred human template; not recovered body or clothing',faceProvenance:'Photo-textured visible face, estimated depth',completeBody:true};
 return {scene:model,animations:[new THREE.AnimationClip('Photo character movement',2,tracks)]};
}
