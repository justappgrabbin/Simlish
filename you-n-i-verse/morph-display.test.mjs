import test from 'node:test';import assert from 'node:assert/strict';
import {selectMorphPreview} from './morph-display.mjs';
const rawBody={},processed={},rawSheet={},personalized={};
test('unprocessed motion never hides a personalized body preview',()=>{
 const r=selectMorphPreview({body:rawBody,morphedBody:processed,sprites:rawSheet});
 assert.equal(r.image,processed);assert.equal(r.kind,'body');assert.equal(r.needsManualFace,false);
});
test('face overlay stays visible if the automatic morph or sheet process fails',()=>{
 const r=selectMorphPreview({body:rawBody,sprites:rawSheet});
 assert.equal(r.image,rawBody);assert.equal(r.needsManualFace,true);
});
test('fully personalized motion has priority when available',()=>{
 const r=selectMorphPreview({body:rawBody,sprites:rawSheet,morphedSprites:personalized});
 assert.equal(r.image,personalized);assert.equal(r.needsManualFace,false);
});
test('missing assets are identified instead of pretending a character exists',()=>{
 assert.equal(selectMorphPreview().kind,'empty');
});
