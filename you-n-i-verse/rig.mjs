import {attachPhotoSurface} from './photo-head.mjs';
import {readProject} from './project-store.mjs';
import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {inspectGLB} from './rig-format.mjs';
const $=id=>document.getElementById(id),stage=$('stage'),diagnostics=new URL(location.href).searchParams.get('test')==='1';$('rigDiagnostics').hidden=!diagnostics;
let photoBinding=null;
let renderer,scene,camera,controls,model,mixer,actions={},active=null,clips=[],playing=true,exporting=false,revision=0,last=0,persistence=Promise.resolve();
const colors={cosmic:0x241036,forest:0x163624,ocean:0x123c59,desert:0x5b3728};
function store(mode,fn){return new Promise((resolve,reject)=>{const r=indexedDB.open('you-n-i-verse-rig',1);r.onupgradeneeded=()=>r.result.createObjectStore('rig');r.onerror=()=>reject(r.error);r.onsuccess=()=>{const db=r.result,tx=db.transaction('rig',mode);let result;try{const request=fn(tx.objectStore('rig'));if(request)request.onsuccess=()=>result=request.result}catch(e){db.close();reject(e);return}tx.oncomplete=()=>{db.close();resolve(result)};tx.onerror=tx.onabort=()=>{db.close();reject(tx.error)}}})}
function dispose(root){root?.traverse(o=>{o.geometry?.dispose();for(const m of (Array.isArray(o.material)?o.material:[o.material]))if(m){for(const v of Object.values(m))if(v?.isTexture)v.dispose();m.dispose()}})}
function fit(){const box=new THREE.Box3().setFromObject(model),size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3()),height=Math.max(size.y,size.x,size.z,.5);camera.position.set(center.x,center.y+height*.12,center.z+height*1.9);camera.near=height/100;camera.far=height*100;camera.updateProjectionMatrix();controls.target.copy(center);controls.update()}
function changeClip(name){if(!actions[name])return;const next=actions[name];if(active!==next){active?.fadeOut(.25);next.reset().setEffectiveWeight(1).fadeIn(.25).play();active=next}playing=true;$('pause').textContent='Pause'}
function settings(){return {clip:$('clip').value,speed:Number($('speed').value),world:$('rigWorld').value,photoBound:!!photoBinding,expressions:[...$('expression').querySelectorAll('input')].map(x=>Number(x.value))}}
function saveSettings(){if(!model)return;persistence=persistence.catch(()=>{}).then(async()=>{const old=await store('readonly',s=>s.get('current'));if(old)await store('readwrite',s=>s.put({...old,settings:settings()},'current'))}).catch(e=>$('storageStatus').textContent='Settings could not save: '+e.message)}
async function loadRig(buffer,label,saved={}){
 const ticket=++revision,info=inspectGLB(buffer);$('status').textContent='Opening '+label+'…';
 const gltf=await new GLTFLoader().parseAsync(buffer,'');if(ticket!==revision){dispose(gltf.scene);return false}
 if(model){scene.remove(model);mixer?.stopAllAction();mixer?.uncacheRoot(model);dispose(model)}
 photoBinding=null;model=gltf.scene;scene.add(model);clips=gltf.animations;mixer=new THREE.AnimationMixer(model);actions={};active=null;playing=true;
 $('clip').replaceChildren();clips.forEach((clip,i)=>{const name=clip.name||'Animation '+(i+1);actions[name]=mixer.clipAction(clip);const option=document.createElement('option');option.value=name;option.textContent=name;$('clip').append(option)});
 const first=actions[saved.clip]?saved.clip:(actions.Walking?'Walking':Object.keys(actions)[0]);if(first){$('clip').value=first;changeClip(first)}
 $('clip').disabled=!clips.length;$('pause').disabled=!clips.length;for(const id of ['exportRigSheet','sendRigSheet'])$(id).disabled=!clips.length;
 $('speed').value=saved.speed??1;$('rigWorld').value=colors[saved.world]?saved.world:'cosmic';scene.background=new THREE.Color(colors[$('rigWorld').value]);
 $('expression').replaceChildren();let index=0,bones=0,morphs=0;
 model.traverse(o=>{if(o.isBone)bones++;if(o.morphTargetDictionary)for(const [name,i] of Object.entries(o.morphTargetDictionary)){const label=document.createElement('label');label.textContent=name;const input=document.createElement('input');input.type='range';input.min=0;input.max=1;input.step=.01;input.value=saved.expressions?.[index++]??0;o.morphTargetInfluences[i]=Number(input.value);input.oninput=()=>{o.morphTargetInfluences[i]=Number(input.value);saveSettings()};label.append(input);$('expression').append(label);morphs++}});
 fit();if(saved.photoBound&&diagnostics){try{await bindSavedPhoto(false)}catch(e){$('photoRigStatus').textContent='Photo surface could not restore: '+e.message}}if(ticket!==revision)return false;$('rigInfo').textContent=bones+' bones · '+clips.length+' motion clips · '+morphs+' expression controls';$('status').textContent=label+' ready. Motion is interpolated on the rig during playback.';if(!photoBinding)$('identityStatus').textContent=label==='Demonstration robot'?'This bundled robot tests movement and expressions. It is not your personalized avatar.':'Your imported character supplies its appearance. Photo-to-face texture binding is not connected in this 3D view yet.';return true;
}
async function open(buffer,label,saved){if(await loadRig(buffer,label,saved)){await store('readwrite',s=>s.put({buffer,label,settings:settings()},'current'));$('storageStatus').textContent='Character and settings saved on this device.'}}
async function demo(){const r=await fetch('assets/rigs/RobotExpressive.glb');if(!r.ok)throw Error('Bundled rig could not load');const project=await readProject().catch(()=>null);await open(await r.arrayBuffer(),'Demonstration robot',{world:project?.world??'cosmic'})}
function download(blob,name){if(globalThis.NativeDownloads){const r=new FileReader();r.onload=()=>NativeDownloads.save(name,blob.type,String(r.result).split(',')[1]);r.readAsDataURL(blob)}else{const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),30000)}}
async function spriteSheet(){
 if(!active||exporting)throw Error('Select a movement first.');exporting=true;const oldSize=renderer.getSize(new THREE.Vector2()),aspect=camera.aspect,time=mixer.time,wasPlaying=playing;
 const sheet=document.createElement('canvas');sheet.width=1024;sheet.height=1152;const c=sheet.getContext('2d'),duration=active.getClip().duration;
 try{renderer.setSize(256,384,false);camera.aspect=2/3;camera.updateProjectionMatrix();mixer.stopAllAction();active.reset().setEffectiveWeight(1).play();
 for(let i=0;i<12;i++){mixer.setTime(duration*i/12);renderer.render(scene,camera);c.drawImage(renderer.domElement,i%4*256,Math.floor(i/4)*384);$('exportStatus').textContent='Rendering pose '+(i+1)+' of 12…';await new Promise(resolve=>setTimeout(resolve,0))}
 return sheet;
 }finally{mixer.setTime(time);playing=wasPlaying;camera.aspect=aspect;camera.updateProjectionMatrix();renderer.setSize(oldSize.x,oldSize.y,false);exporting=false}
}
async function exportSheet(send){try{const sheet=await spriteSheet();if(send){await store('readwrite',s=>s.put({image:sheet.toDataURL(),cols:4,rows:3,fps:12,world:$('rigWorld').value,photoApplied:!!photoBinding},'studio-sheet'));location.href='index.html'}else{const blob=await new Promise(r=>sheet.toBlob(r));if(!blob)throw Error('Image export failed');download(blob,'you-n-i-verse-rig-movement.png');$('exportStatus').textContent='Movement sheet exported: 4 columns × 3 rows, 12 poses.'}}catch(e){$('exportStatus').textContent=e.message}}
try{
 renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));stage.append(renderer.domElement);scene=new THREE.Scene();scene.background=new THREE.Color(colors.cosmic);camera=new THREE.PerspectiveCamera(40,1,.01,100);controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
 scene.add(new THREE.HemisphereLight(0xffe7fa,0x34244b,2));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(3,5,4);scene.add(light);
 const resize=()=>{if(exporting)return;const w=stage.clientWidth,h=stage.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()};new ResizeObserver(resize).observe(stage);resize();
 renderer.setAnimationLoop(t=>{const dt=Math.min((t-last)/1000,.05);last=t;if(exporting||document.hidden)return;if(playing)mixer?.update(dt*Number($('speed').value));controls.update();renderer.render(scene,camera)});
 $('clip').onchange=()=>{changeClip($('clip').value);saveSettings()};$('pause').onclick=()=>{playing=!playing;$('pause').textContent=playing?'Pause':'Play'};$('speed').oninput=saveSettings;$('rigWorld').onchange=()=>{scene.background.setHex(colors[$('rigWorld').value]);saveSettings()};
 $('demo').onclick=()=>demo().catch(e=>$('status').textContent=e.message);$('rigFile').onchange=async()=>{try{const file=$('rigFile').files[0];if(file)await open(await file.arrayBuffer(),file.name)}catch(e){$('status').textContent=e.message}};$('exportRigSheet').onclick=()=>exportSheet(false);$('sendRigSheet').onclick=()=>exportSheet(true);
 let saved;try{saved=await store('readonly',s=>s.get('current'))}catch(e){$('storageStatus').textContent='Local storage unavailable: '+e.message}
 if(saved&&(diagnostics||saved.label!=='Demonstration robot'))await loadRig(saved.buffer,saved.label,{...saved.settings,photoBound:diagnostics&&saved.settings?.photoBound});else if(diagnostics)await demo();else{$('status').textContent='Animation stage: waiting for a prepared character. No test body has been substituted.';$('identityStatus').textContent='Use the completed character from the body stage. Your saved photo and image preview remain in the studio.';}
}catch(e){$('status').textContent='3D view could not start: '+e.message+'. The photo studio remains available.'}

async function bindSavedPhoto(persist=true){
 const ticket=revision,project=await readProject();if(!project?.photo)throw Error('Choose and save your photo in the photo studio first.');const image=new Image();image.src=project.photo;await image.decode();if(ticket!==revision)return;
 if(photoBinding){photoBinding.mesh.removeFromParent();dispose(photoBinding.mesh);photoBinding=null}
 const binding=await attachPhotoSurface(model,image);if(ticket!==revision){binding.mesh.removeFromParent();dispose(binding.mesh);return}photoBinding=binding;$('photoRigStatus').textContent='Your photo-textured face is attached to '+binding.receipt.boundBone+'. It follows movement and is included in sprite exports. This is a frontal surface, not a complete reconstructed head.';$('identityStatus').textContent='Your saved photo supplies the face surface. The rig supplies the rest of the character.';if(persist)saveSettings();
}
$('bindPhoto').onclick=async()=>{const button=$('bindPhoto');button.disabled=true;try{await bindSavedPhoto()}catch(e){$('photoRigStatus').textContent=e.message}finally{button.disabled=false}};
