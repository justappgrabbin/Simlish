import test from 'node:test';import assert from 'node:assert/strict';
import {PhotoIdentityStore,validatePhotoIdentity} from './photo-identity.mjs';
const source='data:image/png;base64,AQID',other='data:image/png;base64,BAUG';
test('source identity survives reload/export and serializes the photo only once',async()=>{
 const store=new PhotoIdentityStore(),photo={},replacement={};let calls=0;
 const [a,b]=await Promise.all([store.capture(photo,()=>{calls++;return source}),store.capture(photo,()=>{calls++;return other})]);
 assert.equal(a,b);assert.equal(calls,1);
 const saved=JSON.parse(JSON.stringify(a));
 const restored=await new PhotoIdentityStore().capture(replacement,()=>{throw Error('Do not re-encode saved source')},{encoded:saved.photo,expected:saved.identity});
 assert.deepEqual(restored,saved);assert(Object.isFrozen(a.identity));
 const changed=await store.capture({},()=>other);assert.notEqual(changed.identity.id,a.identity.id);assert.equal(store.get(photo).photo,source);
});
test('mismatched saved photo cannot silently reuse another identity',async()=>{
 const store=new PhotoIdentityStore();const original=await store.capture({},()=>source);const replacement={};
 await assert.rejects(store.capture(replacement,()=>other,{expected:original.identity}),/does not match/);
 assert.equal(store.get(replacement),undefined);
 assert.throws(()=>validatePhotoIdentity({...original.identity,id:'photo:wrong'}));
});
