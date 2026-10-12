import test from 'node:test';import assert from 'node:assert/strict';import {faceGeometry,encodeFaceGLB} from './photo-mesh.mjs';
const landmarks=Array.from({length:468},(_,i)=>({x:(i%26)/26,y:Math.floor(i/26)/18,z:Math.sin(i)*.05}));
function read(bytes){const view=new DataView(bytes.buffer);assert.equal(view.getUint32(0,true),0x46546c67);assert.equal(view.getUint32(8,true),bytes.length);const size=view.getUint32(12,true);return JSON.parse(new TextDecoder().decode(bytes.slice(20,20+size)));}
test('face surface exports valid aligned GLB with texture, morph targets and animation',()=>{
 const mesh=faceGeometry(landmarks,[0,1,26]);const bytes=encodeFaceGLB(mesh,new Uint8Array([137,80,78,71]),{styleWeight:.8}),gltf=read(bytes);
 assert.equal(gltf.accessors[0].count,468);assert.equal(gltf.meshes[0].primitives[0].targets.length,2);assert.deepEqual(gltf.meshes[0].weights,[.8,0]);assert.equal(gltf.animations[0].channels[0].target.path,'weights');assert.equal(gltf.asset.extras.completeHead,false);for(const v of gltf.bufferViews)assert.equal(v.byteOffset%4,0);
 const output=gltf.accessors[gltf.animations[0].samplers[0].output];assert.equal(output.count,6);
});
test('bad topology and invalid photo geometry cannot produce a mesh',()=>{
 assert.throws(()=>faceGeometry(landmarks,[0,1,999]),/indices/);assert.throws(()=>faceGeometry([{x:0,y:0,z:0}],[]),/468/);const bad=landmarks.map(p=>({...p}));bad[2].z=NaN;assert.throws(()=>faceGeometry(bad,[0,1,2]),/Nonfinite/);
});
