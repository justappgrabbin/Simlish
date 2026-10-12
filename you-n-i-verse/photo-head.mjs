import * as THREE from 'three';
import {portraitSurface} from './photo-morph.mjs';
export async function attachPhotoSurface(model,photo){
 const portrait=await portraitSurface(photo);model.updateMatrixWorld(true);
 let bone;model.traverse(o=>{if(!bone&&o.isBone&&/^(head|head_01|mixamorighead)$/i.test(o.name.replace(/[^a-z0-9_]/gi,'')))bone=o});
 if(!bone)throw Error('This rig needs a named Head bone for the photo surface.');
 const box=new THREE.Box3();bone.traverse(o=>{if(o.isMesh)box.union(new THREE.Box3().setFromObject(o))});
 if(box.isEmpty())throw Error('This head bone has no attached head surface. A rig-specific face binding is needed.');
 const size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3()),p=portrait.points;
 const minX=Math.min(...p.map(v=>v.x)),maxX=Math.max(...p.map(v=>v.x)),minY=Math.min(...p.map(v=>v.y)),maxY=Math.max(...p.map(v=>v.y)),midX=(minX+maxX)/2,midY=(minY+maxY)/2;
 const scale=Math.min(size.x*.9/(maxX-minX),size.y*.88/((maxY-minY)*portrait.image.height/portrait.image.width));
 const inverse=bone.matrixWorld.clone().invert(),verts=[],uv=[];
 for(const v of p){const pos=new THREE.Vector3(center.x+(v.x-midX)*scale,center.y-(v.y-midY)*scale*portrait.image.height/portrait.image.width,box.max.z-v.z*scale*.45+.01*size.z).applyMatrix4(inverse);verts.push(pos.x,pos.y,pos.z);uv.push(v.x,1-v.y)}
 const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(verts,3));geometry.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));geometry.setIndex(portrait.triangles);geometry.computeVertexNormals();
 const texture=new THREE.CanvasTexture(portrait.image);texture.colorSpace=THREE.SRGBColorSpace;
 const material=new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide});const mesh=new THREE.Mesh(geometry,material);mesh.name='UserPhotoSurface';bone.add(mesh);
 return {mesh,receipt:{method:portrait.method,landmarks:468,boundBone:bone.name,unseenViewsReconstructed:false,network:false}};
}
