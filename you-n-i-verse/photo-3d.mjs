import {planBodyForm} from './body-form.mjs';
import {mountBodyProportions} from './body-proportions.mjs';
import {buildFullAvatar} from './full-avatar.mjs';
import * as THREE from './vendor/three/three.module.js';
import {GLTFExporter} from './vendor/three/GLTFExporter.js';
import {GLTFLoader} from './vendor/three/GLTFLoader.js';
import {reconstructFaceSurface} from './photo-morph.mjs';
export function mountPhoto3D(getPhoto,{getIdentity=async()=>null,onChange=()=>{}}={}){
 const $=id=>document.getElementById(id);let renderer,scene,camera,model,mixer,bytes,buildResult,identity,external=false,clips=[],rotate=false,last=0,drag=null;
 function init(){
  if(renderer)return;renderer=new THREE.WebGLRenderer({canvas:$('face3d'),antialias:true,preserveDrawingBuffer:true});renderer.setSize(400,400,false);renderer.setPixelRatio(Math.min(devicePixelRatio,2));scene=new THREE.Scene();scene.background=new THREE.Color('#161128');camera=new THREE.PerspectiveCamera(35,1,.01,100);camera.position.z=3;
  scene.add(new THREE.HemisphereLight(0xffffff,0x8877bb,2));const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(0,2,4);scene.add(light);
  const canvas=$('face3d');canvas.style.touchAction='none';canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);};canvas.onpointermove=e=>{if(!drag||!model)return;model.rotation.y+=(e.clientX-drag.x)*.01;model.rotation.x+=(e.clientY-drag.y)*.01;drag={x:e.clientX,y:e.clientY};};canvas.onpointerup=()=>drag=null;canvas.onpointercancel=()=>drag=null;
  renderer.setAnimationLoop(now=>{const dt=last?Math.min((now-last)/1000,.1):0;last=now;if(mixer&&rotate){mixer.update(dt);if(!external)model.traverse(node=>{if(node.morphTargetInfluences)node.morphTargetInfluences[0]=Number($("style3d").value);});}renderer.render(scene,camera);});
 }
 function dispose(old){if(!old)return;scene.remove(old);old.traverse(node=>{node.geometry?.dispose();if(node.material){for(const m of (Array.isArray(node.material)?node.material:[node.material])){m.map?.dispose();m.dispose();}}});}
 $('build3d').onclick=async()=>{
  const source=getPhoto();$('build3d').disabled=true;$('meshStatus').textContent='Reading photo landmarks and building the 3D face surface…';
  try{const result=await reconstructFaceSurface(source);if(source!==getPhoto())throw Error('Photo changed; build again.');init();const gltf=await new GLTFLoader().parseAsync(result.bytes.buffer,'');dispose(model);model=gltf.scene;scene.add(model);external=false;$('style3d').disabled=false;bytes=result.bytes;buildResult=result;identity=source;
   model.updateMatrixWorld(true);model.traverse(node=>{if(node.isSkinnedMesh){node.skeleton.update();node.computeBoundingBox();}});const box=new THREE.Box3().setFromObject(model),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());model.position.sub(center);camera.position.z=Math.max(size.x,size.y)*2.2;
   clips=gltf.animations;mixer=new THREE.AnimationMixer(model);mixer.clipAction(clips[0]).play();$('avatarExpressions').replaceChildren();$('bodyProportions').replaceChildren();$('style3d').value=0;$('exportAvatarPNG').disabled=false;$('export3d').disabled=false;$('animate3d').disabled=false;$('meshStatus').textContent='3D face surface ready. Drag to turn. Depth is estimated; back of head and body are not reconstructed.';
  }catch(error){$('meshStatus').textContent=error.message;}finally{$('build3d').disabled=false;}
 };
 $('style3d').oninput=()=>{model?.traverse(node=>{if(node.morphTargetInfluences)node.morphTargetInfluences[0]=Number($('style3d').value);});};
 $('animate3d').onclick=()=>{rotate=!rotate;$('animate3d').textContent=rotate?'Pause animation':'Animate avatar';};
 $('exportAvatarPNG').onclick=async()=>{if(!model)return;try{renderer.render(scene,camera);const blob=await new Promise(resolve=>renderer.domElement.toBlob(resolve,'image/png'));if(!blob)throw Error('Image export failed.');if(window.NativeDownloads?.save){const reader=new FileReader();reader.onload=()=>NativeDownloads.save('you-n-i-verse-character.png','image/png',reader.result.split(',')[1]);reader.readAsDataURL(blob);}else{const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='you-n-i-verse-character.png';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}}catch(error){$('meshStatus').textContent=error.message;}};
 $('export3d').onclick=async()=>{
  try{
  if(!external&&identity!==getPhoto()){$('meshStatus').textContent='Photo changed; rebuild before exporting.';return;}
  if(!external)bytes=buildResult.export({styleWeight:Number($('style3d').value)});
  else bytes=new Uint8Array(await new GLTFExporter().parseAsync(model,{binary:true,animations:clips}));
  const blob=new Blob([bytes],{type:'model/gltf-binary'});
  if(window.NativeDownloads?.save){const reader=new FileReader();reader.onload=()=>NativeDownloads.save('you-n-i-verse-avatar.glb','model/gltf-binary',reader.result.split(',')[1]);reader.readAsDataURL(blob);return;}
  const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='you-n-i-verse-avatar.glb';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(error){$('meshStatus').textContent='GLB export failed: '+error.message;}
 };
 async function openModel(gltf){
  init();dispose(model);model=gltf.scene;let provenance;model.traverse(node=>{if(node.userData.sourceIdentity)provenance=node.userData;});if(provenance)model.userData={...model.userData,...provenance};scene.add(model);external=true;buildResult=null;
  model.updateMatrixWorld(true);model.traverse(node=>{if(node.isSkinnedMesh){node.skeleton.update();node.computeBoundingBox();}});const box=new THREE.Box3().setFromObject(model),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());model.position.sub(center);camera.position.z=Math.max(size.x,size.y)*2.2;camera.far=Math.max(100,camera.position.z*10);camera.updateProjectionMatrix();
  mixer=null;rotate=false;clips=gltf.animations;$('avatarExpressions').replaceChildren();mountBodyProportions(model,$('bodyProportions'),()=>{rotate=false;$('animate3d').textContent='Animate avatar';onChange();});const tracks=[];
  model.traverse(node=>{if(!node.morphTargetDictionary)return;for(const [name,index] of Object.entries(node.morphTargetDictionary)){if(['ChestWidth','WaistWidth','HipWidth'].includes(name))continue;
   const label=document.createElement('label');label.textContent=/^shape_/.test(name)?'Shape '+(Number(name.slice(6))+1):name.replace(/([a-z])([A-Z])/g,'$1 $2');const slider=document.createElement('input');slider.type='range';const bodyShape=/^shape_/.test(name);slider.min=bodyShape?-3:0;slider.max=bodyShape?3:1;slider.step=.01;slider.value=node.morphTargetInfluences[index];slider.oninput=()=>{rotate=false;$('animate3d').textContent='Animate avatar';node.morphTargetInfluences[index]=Number(slider.value);onChange();};label.append(slider);$('avatarExpressions').append(label);
   if(/eyeBlink(Left|Right)/i.test(name))tracks.push(new THREE.NumberKeyframeTrack(node.uuid+'.morphTargetInfluences['+name+']',[0,1,1.15,1.3,3],[0,0,1,0,0]));
   if(name==='jawOpen'||(name==='mouthOpen'&&!('jawOpen' in node.morphTargetDictionary)))tracks.push(new THREE.NumberKeyframeTrack(node.uuid+'.morphTargetInfluences['+name+']',[0,.5,1,1.5,2,3],[0,.2,0,.4,.1,0]));
  }});
  if(!clips.length&&tracks.length)clips=[new THREE.AnimationClip('Expression demonstration',3,tracks)];
  if(clips.length){mixer=new THREE.AnimationMixer(model);mixer.clipAction(clips[0]).play();}
  let requestedForm;model.traverse(node=>{if(node.userData.requestedForm!==undefined)requestedForm=node.userData.requestedForm;});if(requestedForm!==undefined)$('bodyRequest').value=requestedForm;
  $('style3d').disabled=true;$('animate3d').disabled=!mixer;$('animate3d').textContent='Animate avatar';$('exportAvatarPNG').disabled=false;$('export3d').disabled=false;$('meshStatus').textContent='Model-host avatar loaded. Drag to turn, or export GLB.';
 }
 const api={open:async(data,{silent=false}={})=>{bytes=data;const gltf=await new GLTFLoader().parseAsync(data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength),'');await openModel(gltf);if(!silent)onChange();},
 snapshot:async()=>{if(!model||!external)return null;const result=await new GLTFExporter().parseAsync(model,{binary:true,animations:clips});return {data:new Uint8Array(result),identityId:model.userData.sourceIdentity??null};},
 clear:()=>{if(model&&external){dispose(model);model=null;clips=[];mixer=null;$('avatarExpressions').replaceChildren();$('bodyProportions').replaceChildren();$('exportAvatarPNG').disabled=true;$('export3d').disabled=true;$('animate3d').disabled=true;}}};
 async function createFull(){const source=getPhoto();const plan=planBodyForm($('bodyRequest').value);if(!plan.ok){$('meshStatus').textContent=plan.reason;return;}$('buildFull3d').disabled=true;$('meshStatus').textContent='Building your complete character locally…';try{const sourceIdentity=await getIdentity(source);const gltf=await buildFullAvatar(source,sourceIdentity,plan);if(source!==getPhoto())throw Error('Photo changed; build again.');await openModel(gltf);$('meshStatus').textContent='Character ready'+(plan.applied.length?' · '+plan.applied.join(', '):'')+'. Face fitted automatically. Body proportions are generated.';onChange();}catch(error){$('meshStatus').textContent=error.message;}finally{$('buildFull3d').disabled=false;}}
 $('buildFull3d').onclick=createFull;api.createFull=createFull;
 $('import3d').onchange=async event=>{try{const file=event.target.files[0];if(!file)return;if(file.size>50_000_000)throw Error('Use a GLB below 50 MB.');await api.open(new Uint8Array(await file.arrayBuffer()));}catch(error){$('meshStatus').textContent=error.message;}finally{event.target.value='';}};
 return api;
}
