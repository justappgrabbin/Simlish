import test from 'node:test';
import assert from 'node:assert/strict';
import {selectCharacterSource} from './avatar-selection.mjs';

test('an explicit saved Morph World result wins over old unrelated rig',()=>{
 const p={photo:'data:image/png;base64,123',gameAvatar:'data:image/png;base64,abc',gameAvatarMethod:'manual-face-overlay'};
 assert.equal(selectCharacterSource(p,{label:'Other model',buffer:new Uint8Array([1])}).kind,'morph-image');
});
test('custom 3D model is used when no saved character image exists',()=>{
 assert.equal(selectCharacterSource(null,{label:'My custom GLB',buffer:new Uint8Array([1])}).kind,'custom-3d');
});
test('never load the diagnostic robot as an identity',()=>{
 assert.equal(selectCharacterSource(null,{label:'Demonstration robot',buffer:new Uint8Array([1])}).kind,'reference-art');
});
test('unrendered or invalid characters do not claim to be personalized',()=>{
 assert.equal(selectCharacterSource({photo:'data:image/png;base64,a'},null).kind,'reference-art');
 assert.equal(selectCharacterSource({photo:'x',gameAvatar:'https://example.com/face.png'},null).kind,'reference-art');
});
