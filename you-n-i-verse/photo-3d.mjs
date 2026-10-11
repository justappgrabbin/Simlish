import * as THREE from './vendor/three/three.module.js';
import {GLTFLoader} from './vendor/three/GLTFLoader.js';
import {reconstructFaceSurface} from './photo-morph.mjs';
export function mountPhoto3D(getPhoto){
 const $=id=>document.getElementById(id);let renderer,scene,camera,model,mixer,bytes,buildResult,identity,rotate=false,last=0,drag=null;
 function init(){
  if(renderer)return;renderer=new THREE.WebGLRenderer({canvas:$('face3d'),antialias:true});renderer.setSize(400,400,false);renderer.setPixelRatio(Math.min(devicePixelRatio,2));scene=new THREE.Scene();scene.background=new THREE.Color('#161128');camera=new THREE.PerspectiveCamera(35,1,.01,100);camera.position.z=3;
  scene.add(new THREE.HemisphereLight(0xffffff,0x8877bb,2));const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(0,2,4);scene.add(light);
  const canvas=$('face3d');canvas.style.touchAction='none';canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);};canvas.onpointermove=e=>{if(!drag||!model)return;model.rotation.y+=(e.clientX-drag.x)*.01;model.rotation.x+=(e.clientY-drag.y)*.01;drag={x:e.clientX,y:e.clientY};};canvas.onpointerup=()=>drag=null;canvas.onpointercancel=()=>drag=null;
  renderer.setAnimationLoop(now=>{const dt=last?Math.min((now-last)/1000,.1):0;last=now;if(mixer&&rotate){mixer.update(dt);model.traverse(node=>{if(node.morphTargetInfluences)node.morphTargetInfluences[0]=Number($("style3d").value);});}renderer.render(scene,camera);});
 }
 function dispose(old){if(!old)return;scene.remove(old);old.traverse(node=>{node.geometry?.dispose();if(node.material){for(const m of (Array.isArray(node.material)?node.material:[node.material])){m.map?.dispose();m.dispose();}}});}
 $('build3d').onclick=async()=>{
  const source=getPhoto();$('build3d').disabled=true;$('meshStatus').textContent='Reading photo landmarks and building the 3D face surface…';
  try{const result=await reconstructFaceSurface(source);if(source!==getPhoto())throw Error('Photo changed; build again.');init();const gltf=await new GLTFLoader().parseAsync(result.bytes.buffer,'');dispose(model);model=gltf.scene;scene.add(model);bytes=result.bytes;buildResult=result;identity=source;
   const box=new THREE.Box3().setFromObject(model),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());model.position.sub(center);camera.position.z=Math.max(size.x,size.y)*2.2;
   mixer=new THREE.AnimationMixer(model);mixer.clipAction(gltf.animations[0]).play();$('style3d').value=0;$('export3d').disabled=false;$('animate3d').disabled=false;$('meshStatus').textContent='3D face surface ready. Drag to turn. Depth is estimated; back of head and body are not reconstructed.';
  }catch(error){$('meshStatus').textContent=error.message;}finally{$('build3d').disabled=false;}
 };
 $('style3d').oninput=()=>{model?.traverse(node=>{if(node.morphTargetInfluences)node.morphTargetInfluences[0]=Number($('style3d').value);});};
 $('animate3d').onclick=()=>{rotate=!rotate;$('animate3d').textContent=rotate?'Pause expression':'Animate expression';};
 $('export3d').onclick=()=>{
  if(identity!==getPhoto()){$('meshStatus').textContent='Photo changed; rebuild before exporting.';return;}
  bytes=buildResult.export({styleWeight:Number($('style3d').value)});
  const blob=new Blob([bytes],{type:'model/gltf-binary'});
  if(window.NativeDownloads?.save){const reader=new FileReader();reader.onload=()=>NativeDownloads.save('you-n-i-verse-face.glb','model/gltf-binary',reader.result.split(',')[1]);reader.readAsDataURL(blob);return;}
  const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='you-n-i-verse-face.glb';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 };
}
