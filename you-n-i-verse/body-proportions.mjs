import * as THREE from './vendor/three/three.module.js';
export const bodyControls=[['height','Height',.8,1.2],['shoulders','Shoulders',.8,1.4],['chest','Chest',.8,1.6],['waist','Waist',.6,1.4],['hips','Hips',.8,1.8],['uArm','Upper arms',.8,1.6],['fArm','Forearms',.8,1.6],['thigh','Thighs',.8,1.8],['calf','Calves',.8,1.6]];
const boneControls={shoulders:['l_clavicle','r_clavicle'],uArm:['l_uparm','r_uparm'],fArm:['l_lowarm','r_lowarm'],thigh:['l_upleg','r_upleg'],calf:['l_lowleg','r_lowleg']};
const surfaceControls={chest:'ChestWidth',waist:'WaistWidth',hips:'HipWidth'};
export function addProportionMorphs(mesh){
 const geometry=mesh.geometry,position=geometry.attributes.position,indices=geometry.attributes.skinIndex,weights=geometry.attributes.skinWeight;
 const deltas=Object.fromEntries(Object.values(surfaceControls).map(name=>[name,new Float32Array(position.count*3)]));
 const point=new THREE.Vector3(),changed=new THREE.Vector3(),blended=new THREE.Matrix4(),part=new THREE.Matrix4(),world=new THREE.Matrix4();
 const regions={ChestWidth:[1.24,.2],WaistWidth:[1.02,.17],HipWidth:[.84,.18]};
 for(let i=0;i<position.count;i++){
  blended.elements.fill(0);
  for(let k=0;k<4;k++){const index=indices.getComponent(i,k),weight=weights.getComponent(i,k);part.multiplyMatrices(mesh.skeleton.bones[index].matrixWorld,mesh.skeleton.boneInverses[index]);for(let n=0;n<16;n++)blended.elements[n]+=part.elements[n]*weight;}
  world.copy(mesh.matrixWorld).multiply(mesh.bindMatrixInverse).multiply(blended).multiply(mesh.bindMatrix);const inverse=world.clone().invert();point.fromBufferAttribute(position,i).applyMatrix4(world);
  for(const [name,[center,radius]] of Object.entries(regions)){
   const distance=Math.abs(point.y-center)/radius,fade=Math.max(0,1-distance*distance),trunk=Math.max(0,1-Math.max(0,Math.abs(point.x)-.18)/.1);
   changed.copy(point);changed.x+=point.x*fade*trunk;changed.z+=point.z*fade*trunk*.35;changed.applyMatrix4(inverse).sub(new THREE.Vector3().fromBufferAttribute(position,i));changed.toArray(deltas[name],i*3);
  }
 }
 const names=Object.keys(mesh.morphTargetDictionary);for(const [name,delta] of Object.entries(deltas)){names.push(name);geometry.morphAttributes.position.push(new THREE.BufferAttribute(delta,3));if(geometry.morphAttributes.normal)geometry.morphAttributes.normal.push(new THREE.BufferAttribute(new Float32Array(delta.length),3));}
 mesh.updateMorphTargets();mesh.morphTargetDictionary=Object.fromEntries(names.map((name,index)=>[name,index]));
}
export function mountBodyProportions(model,container,onChange){
 container.replaceChildren();let root;model.traverse(node=>{if(node.name==='Persistent_photo_character')root=node;});if(!root)return;
 const state=root.userData.bodyProportions??Object.fromEntries(bodyControls.map(([key])=>[key,1]));root.userData.bodyProportions=state;
 root.userData.heightBaseScale??=root.scale.y/(state.height||1);
 const apply=()=>{
  root.scale.y=root.userData.heightBaseScale*state.height;
  model.traverse(node=>{
   if(node.isBone){for(const [key,names] of Object.entries(boneControls)){if(names.includes(node.name)){node.userData.proportionBaseScale??=node.scale.toArray().map(x=>x/(state[key]||1));node.scale.fromArray(node.userData.proportionBaseScale).multiplyScalar(state[key]);}}}
   if(node.morphTargetDictionary){for(const [key,name] of Object.entries(surfaceControls)){const index=node.morphTargetDictionary[name];if(index!==undefined)node.morphTargetInfluences[index]=state[key]-1;}}
  });
 };
 // Record neutral/restored scales before any slider changes.
 apply();
 for(const [key,title,min,max] of bodyControls){const label=document.createElement('label');label.textContent=title;const slider=document.createElement('input');slider.type='range';slider.min=min;slider.max=max;slider.step=.01;slider.value=state[key];slider.dataset.bodyControl=key;slider.oninput=()=>{state[key]=Number(slider.value);apply();onChange();};label.append(slider);container.append(label);}
}
