import {test} from 'node:test';
import assert from 'node:assert/strict';
import {MorphLibrary} from './morph-library.mjs';
test('all forms share one source and cached results; source replacement invalidates',async()=>{
 const calls=[];const library=new MorphLibrary(async(source,target)=>{calls.push([source,target]);return {canvas:target};});
 const photo={};library.setSource(photo);library.add('a','Any form',{});library.add('b','Another form',{});
 assert.equal((await library.prepareAll()).length,2);assert.ok(calls.every(([s])=>s===photo));
 await library.prepare('a');assert.equal(calls.length,2);library.setSource({});await library.prepare('a');assert.equal(calls.length,3);
});
test('failed targets do not stop other forms',async()=>{
 const library=new MorphLibrary(async(_,target)=>{if(target.bad)throw Error('No face');return {canvas:target};});library.setSource({});library.add('bad','Bad',{bad:true});library.add('ok','OK',{});
 const results=await library.prepareAll();assert.ok(results[0].error);assert.ok(results[1].result);
});
test('old source result cannot enter the new identity cache',async()=>{
 let finish;const library=new MorphLibrary(()=>new Promise(resolve=>finish=resolve));library.setSource({});library.add('a','A',{});
 const pending=library.prepare('a');await Promise.resolve();library.setSource({});finish({canvas:{}});
 await assert.rejects(pending,/changed/);assert.equal(library.targets.get('a').result,null);
});
