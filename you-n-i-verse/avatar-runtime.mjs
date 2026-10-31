import * as THREE from './vendor/three/three.module.js';
import {GLTFLoader} from './vendor/three/addons/loaders/GLTFLoader.js';
import {readProject} from './project-store.mjs';
import {selectCharacterSource} from './avatar-selection.mjs';

function image(url) {
  return new Promise((resolve,reject)=>{
    const i=new Image();
    i.onload=()=>resolve(i);
    i.onerror=()=>reject(Error('Cannot open character image'));
    i.src=url;
  });
}
function readRig() {
  return new Promise(resolve=>{
    if(!globalThis.indexedDB){resolve(null);return;}
    let db;
    try {
      const req=indexedDB.open('you-n-i-verse-rig',1);
      req.onupgradeneeded=()=>{if(!req.result.objectStoreNames.contains('rig'))req.result.createObjectStore('rig');};
      req.onerror=req.onblocked=()=>resolve(null);
      req.onsuccess=()=>{
        db=req.result;
        let result=null;
        const tx=db.transaction('rig','readonly'),r=tx.objectStore('rig').get('current');
        r.onsuccess=()=>{result=r.result;};
        tx.oncomplete=()=>{db.close();resolve(result);};
        tx.onerror=tx.onabort=()=>{db.close();resolve(null);};
      };
    }catch{db?.close();resolve(null);}
  });
}
function spriteCharacter(player,source,options={}){
  const canvas=document.createElement('canvas'),w=options.width??260,h=options.height??580;
  canvas.width=w;canvas.height=h;
  const ctx=canvas.getContext('2d',{alpha:true});
  let lastFrame=-1;
  function draw(frame){
    ctx.clearRect(0,0,w,h);
    if(options.atlas&&options.grid){
      const g=options.grid,fw=options.atlas.width/g.cols,fh=options.atlas.height/g.rows;
      const n=Math.min(frame,g.count-1);
      ctx.drawImage(options.atlas,(n%g.cols)*fw,Math.floor(n/g.cols)*fh,fw,fh,0,0,w,h);
    }else if(options.crop){
      const [x,y,sw,sh]=options.crop;ctx.drawImage(source,x,y,sw,sh,0,0,w,h);
    }else ctx.drawImage(source,0,0,w,h);
  }
  draw(0);
  const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
  texture.minFilter=THREE.LinearFilter;
  const material=new THREE.SpriteMaterial({map:texture,transparent:true,alphaTest:.03,depthWrite:false});
  const sprite=new THREE.Sprite(material);
  sprite.position.y=1.65;sprite.scale.set(w/h*3.3,3.3,1);player.add(sprite);
  return {update(now,moving){
    const grid=options.grid;
    if(!grid||!options.atlas)return;
    const frame=moving?Math.floor(now/95)%grid.count:0;
    if(frame===lastFrame)return;
    draw(frame);texture.needsUpdate=true;lastFrame=frame;
  }};
}
async function importRig(player,rig){
  const raw=rig.buffer;
  const buffer=raw instanceof ArrayBuffer?raw:raw?.buffer;
  if(!(buffer instanceof ArrayBuffer))throw Error('Saved 3D model has no usable buffer');
  const gltf=await new GLTFLoader().parseAsync(buffer,'');
  const object=gltf.scene;
  const bounds=new THREE.Box3().setFromObject(object);
  const height=bounds.getSize(new THREE.Vector3()).y;
  if(!Number.isFinite(height)||height<=0)throw Error('Saved 3D model has no measurable body');
  const scale=Math.min(30,3.15/height);object.scale.multiplyScalar(scale);
  const box=new THREE.Box3().setFromObject(object);
  const center=box.getCenter(new THREE.Vector3());
  object.position.set(-center.x,-box.min.y,-center.z);
  player.add(object);
  const clips=gltf.animations;
  if(!clips.length)return {update(){}};
  const mixer=new THREE.AnimationMixer(object);
  const byName=part=>clips.find(clip=>clip.name.toLowerCase().includes(part));
  const idle=byName('idle')??clips[0],walk=byName('walk')??byName('run')??idle;
  const actions=new Map(clips.map(clip=>[clip,mixer.clipAction(clip)]));
  let selected=null;
  return {update(now,moving,dt){
    const next=actions.get(moving?walk:idle);
    if(next!==selected){selected?.fadeOut(.18);next.reset().fadeIn(.18).play();selected=next;}
    mixer.update(dt*(Number(rig.settings?.speed)||1));
  }};
}
export async function loadCharacterIntoWorld(player,onStatus=()=>{}){
  let project=null,readError=null;
  try{project=await readProject();}catch(error){readError=error.message;}
  const rig=await readRig();
  const selected=selectCharacterSource(project,rig);
  let controller;
  try{
    if(selected.kind==='morph-image'){
      const img=await image(project.gameAvatar);
      let atlas=null,grid=null;
      const g=project.animationReceipt?.grid;
      if(project.gameAvatarMethod==='personalized-motion-sheet'&&project.morphedSprites&&g&&
        Number.isInteger(g.cols)&&Number.isInteger(g.rows)&&Number.isInteger(g.count)&&g.count>1&&g.count<=256){
        try{atlas=await image(project.morphedSprites);if(atlas.width%g.cols||atlas.height%g.rows||g.count>g.cols*g.rows)atlas=null;else grid=g;}
        catch{atlas=null;}
      }
      controller=spriteCharacter(player,img,{atlas,grid});
    }else if(selected.kind==='custom-3d'){
      controller=await importRig(player,rig);
    }else {
      const board=await image('assets/avatar-states.jpg');
      controller=spriteCharacter(player,board,{crop:[600,170,100,180]});
    }
  }catch(error){
    onStatus('Character could not load: '+error.message);
    throw error; // Do not silently turn a failed personal character into a fake success.
  }
  const note=readError?('Local project storage unavailable: '+readError+'. '):'';
  onStatus(note+selected.label+(selected.kind==='morph-image'?' · Imported GLB is required for actual 3D body animation.':''));
  return {source:selected,update:controller.update};
}
