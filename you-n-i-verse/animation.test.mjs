import test from 'node:test';
import assert from 'node:assert/strict';
import {sheetGrid,personalizeFrames} from './animation.mjs';
test('grid rejects invalid sizes before allocating animation memory',()=>{
 assert.equal(sheetGrid(640,128,5,1).count,5);
 for(const args of [[641,128,5,1],[640,128,0,1],[640,128,1.5,1],[8192,8192,1,1]])assert.throws(()=>sheetGrid(...args));
});
test('failed or cancelled frame never returns a partly personalized sheet',async()=>{
 globalThis.document={createElement:()=>({width:0,height:0,getContext:()=>({drawImage(){}})})};
 const grid=sheetGrid(20,10,2,1);let calls=0;
 await assert.rejects(personalizeFrames({}, {width:20,height:10},grid,{morph:async()=>{if(++calls===2)throw Error('no face');return {canvas:{width:10,height:10},receipt:{}}}}),/Frame 2.*No partial sheet/);
 calls=0;await assert.rejects(personalizeFrames({}, {width:20,height:10},grid,{cancelled:()=>calls===1,morph:async()=>{calls++;return {canvas:{},receipt:{}}}}),/cancelled/);
 delete globalThis.document;
});
